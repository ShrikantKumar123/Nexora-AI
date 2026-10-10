from pathlib import Path
from typing import List, Optional
import hashlib
import os

import certifi
from dotenv import load_dotenv

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# ============================================================
# Configuration
# ============================================================

load_dotenv()

# Configure SSL certificates for HTTPS requests.
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

PROJECT_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = PROJECT_DIR / "uploads"
CHROMA_DIR = PROJECT_DIR / "chroma_db"

COLLECTION_NAME = "nexora_ai_documents"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

DEFAULT_TOP_K = 4

# This is a starting value, not a universal threshold.
# Tune it using your own retrieval evaluation data.
DEFAULT_RELEVANCE_THRESHOLD = 0.20

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".txts",
    ".tsv",
    ".csv",
}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Embedding Model
# ============================================================

# Set GOOGLE_API_KEY in your .env file.
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)


# ============================================================
# Chroma Vector Database
# ============================================================

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)


# ============================================================
# File Validation
# ============================================================

def validate_file(file_path: str) -> Path:
    """
    Validate file existence and supported extension.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not path.is_file():
        raise ValueError(
            "The provided path is not a file."
        )

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(
            sorted(SUPPORTED_EXTENSIONS)
        )

        raise ValueError(
            f"Unsupported file type. Supported formats: {supported}"
        )

    return path


# ============================================================
# Read Documents
# ============================================================

def read_file_documents(file_path: str) -> List[Document]:
    """
    Read supported files and return LangChain Documents.

    PDF:
        Extracts each page separately and preserves page numbers.

    TXT, TXTs, TSV, CSV:
        Reads the content as text.
    """

    path = validate_file(file_path)

    extension = path.suffix.lower()
    filename = path.name

    documents = []

    if extension == ".pdf":
        reader = PdfReader(str(path))

        if reader.is_encrypted:
            try:
                result = reader.decrypt("")
                if not result:
                    raise ValueError(
                        f"PDF is password-protected: {filename}"
                    )
            except Exception as exc:
                raise ValueError(
                    f"Unable to read encrypted PDF: {filename}"
                ) from exc

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            page_text = page.extract_text() or ""

            if page_text.strip():
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": filename,
                            "page": page_number,
                            "file_type": "pdf",
                        },
                    )
                )

    elif extension in {
        ".txt",
        ".txts",
        ".tsv",
        ".csv",
    }:
        text = path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "file_type": extension.lstrip("."),
                    },
                )
            )

    if not documents:
        raise ValueError(
            f"No readable text was extracted from {filename}. "
            "If this is a scanned PDF, OCR may be required."
        )

    return documents


def read_file_text(file_path: str) -> str:
    """
    Read a file and return all extracted text as one string.
    """

    documents = read_file_documents(file_path)

    return "\n\n".join(
        document.page_content
        for document in documents
    )


# ============================================================
# Document Chunking
# ============================================================

def split_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Split documents into smaller chunks while retaining metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    return splitter.split_documents(documents)


# ============================================================
# Generate Stable Chunk IDs
# ============================================================

def create_chunk_id(
    thread_id: str,
    source: str,
    page: Optional[int],
    start_index: Optional[int],
    content: str,
) -> str:
    """
    Generate a deterministic ID for a document chunk.
    """

    unique_content = (
        f"{thread_id}|{source}|{page}|"
        f"{start_index}|{content}"
    )

    return hashlib.sha256(
        unique_content.encode("utf-8")
    ).hexdigest()


# ============================================================
# Add or Replace a Document
# ============================================================

def add_document_to_rag(
    file_path: str,
    thread_id: str,
) -> dict:
    """
    Extract, chunk, and index a file in ChromaDB.

    Uploading the same filename again in the same thread
    replaces the existing indexed chunks for that file.
    """

    if not thread_id or not thread_id.strip():
        raise ValueError("A valid thread_id is required.")

    path = validate_file(file_path)
    filename = path.name

    # 1. Extract source documents.
    source_documents = read_file_documents(
        str(path)
    )

    # 2. Split into chunks.
    chunks = split_documents(source_documents)

    if not chunks:
        raise ValueError(
            f"No chunks were generated for {filename}."
        )

    indexed_documents = []
    chunk_ids = []

    # 3. Add metadata and deterministic IDs.
    for chunk in chunks:
        chunk.page_content = chunk.page_content.strip()

        metadata = dict(chunk.metadata)
        metadata["thread_id"] = thread_id
        metadata["source"] = filename

        page = metadata.get("page")
        start_index = metadata.get("start_index")

        chunk.metadata = metadata

        chunk_id = create_chunk_id(
            thread_id=thread_id,
            source=filename,
            page=page,
            start_index=start_index,
            content=chunk.page_content,
        )

        indexed_documents.append(chunk)
        chunk_ids.append(chunk_id)

    # 4. Remove previous chunks for this file and thread.
    # This also prevents stale content after re-upload.
    vectorstore.delete(
        where={
            "$and": [
                {"thread_id": thread_id},
                {"source": filename},
            ]
        }
    )

    # 5. Store the new chunks and generate embeddings.
    vectorstore.add_documents(
        documents=indexed_documents,
        ids=chunk_ids,
    )

    return {
        "filename": filename,
        "thread_id": thread_id,
        "chunks": len(indexed_documents),
        "status": "success",
        "message": f"{filename} indexed successfully.",
    }


# ============================================================
# Retrieve Relevant Documents
# ============================================================

def retrieve_from_rag(
    query: str,
    thread_id: str,
    k: int = DEFAULT_TOP_K,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    search_type: str = "similarity",
) -> str:
    """
    Search indexed documents belonging to a specific thread.

    search_type="similarity":
        Finds chunks most similar to the query and applies
        a relevance-score threshold.

    search_type="mmr":
        Finds relevant but diverse chunks.

    Returns source information and extracted content.
    """

    if not thread_id or not thread_id.strip():
        raise ValueError("A valid thread_id is required.")

    if not query or not query.strip():
        return "Please provide a question to search for."

    if k < 1:
        raise ValueError("k must be at least 1.")

    if search_type not in {"similarity", "mmr"}:
        raise ValueError(
            "search_type must be 'similarity' or 'mmr'."
        )

    search_filter = {
        "thread_id": thread_id,
    }

    # --------------------------------------------------------
    # Similarity Search
    # --------------------------------------------------------

    if search_type == "similarity":
        scored_documents = (
            vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=search_filter,
            )
        )

        results = [
            (document, score)
            for document, score in scored_documents
            if score >= relevance_threshold
        ]

    # --------------------------------------------------------
    # Maximal Marginal Relevance (MMR) Search
    # --------------------------------------------------------

    else:
        documents = (
            vectorstore.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=max(k * 4, 20),
                filter=search_filter,
            )
        )

        results = [
            (document, None)
            for document in documents
        ]

    if not results:
        return (
            "No sufficiently relevant information was found "
            "in the documents uploaded to this conversation."
        )

    # --------------------------------------------------------
    # Format Search Results
    # --------------------------------------------------------

    formatted_results = []

    for index, (document, score) in enumerate(
        results,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            "unknown",
        )

        page = document.metadata.get("page")

        source_details = f"Source: {source}"

        if page is not None:
            source_details += f", Page: {page}"

        if score is not None:
            source_details += (
                f", Relevance Score: {score:.3f}"
            )

        formatted_results.append(
            f"[Result {index} | {source_details}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(formatted_results)


# ============================================================
# Delete a Document
# ============================================================

def delete_document_from_rag(
    filename: str,
    thread_id: str,
) -> dict:
    """
    Delete indexed chunks for a particular file and thread.
    """

    if not thread_id or not thread_id.strip():
        raise ValueError("A valid thread_id is required.")

    if not filename or not filename.strip():
        raise ValueError("A filename is required.")

    safe_filename = Path(filename).name

    vectorstore.delete(
        where={
            "$and": [
                {"thread_id": thread_id},
                {"source": safe_filename},
            ]
        }
    )

    return {
        "filename": safe_filename,
        "thread_id": thread_id,
        "status": "delete_requested",
    }


# ============================================================
# List Documents in a Thread
# ============================================================

def list_rag_documents(
    thread_id: str,
) -> List[str]:
    """
    Return unique filenames indexed for a thread.
    """

    if not thread_id or not thread_id.strip():
        raise ValueError("A valid thread_id is required.")

    result = vectorstore.get(
        where={"thread_id": thread_id},
        include=["metadatas"],
    )

    filenames = set()

    for metadata in result.get("metadatas") or []:
        if metadata:
            source = metadata.get("source")

            if source:
                filenames.add(source)

    return sorted(filenames)


# ============================================================
# Simple RAG Test
# ============================================================

if __name__ == "__main__":
    print("Nexora-AI RAG module loaded.")
    print("Supported file types:", sorted(SUPPORTED_EXTENSIONS))
    print("ChromaDB directory:", CHROMA_DIR)
    print("Collection:", COLLECTION_NAME)

    # Example:
    #
    # result = add_document_to_rag(
    #     file_path="uploads/sample.pdf",
    #     thread_id="test-thread",
    # )
    # print(result)
    #
    # answer_context = retrieve_from_rag(
    #     query="What is this document about?",
    #     thread_id="test-thread",
    #     k=4,
    # )
    # print(answer_context)
