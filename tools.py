import math
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from datetime import datetime
from zoneinfo import ZoneInfo
from database import save_memory, search_memory
from rag import retrieve_from_rag



load_dotenv()


CURRENT_THREAD_ID = "default"


def set_current_thread_id(thread_id: str):
    global CURRENT_THREAD_ID
    CURRENT_THREAD_ID = thread_id


web_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        print(result,'=== Cal value ===')
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"
    


@tool
def search_uploaded_documents(query: str) -> str:
    """
    Search uploaded documents for relevant information.
    Use this when the user asks about uploaded PDFs, DOCX, TXT, notes, files, or documents.
    """

    return retrieve_from_rag(
        query=query,
        thread_id=CURRENT_THREAD_ID
    )




@tool
def remember_this(memory: str) -> str:
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remember something.
    """

    return save_memory(
        thread_id=CURRENT_THREAD_ID,
        memory=memory
    )



@tool
def recall_memory(query: str) -> str:
    """
    Recall saved long-term memories about the user or this conversation.
    """

    return search_memory(
        thread_id=CURRENT_THREAD_ID,
        query=query
    )





@tool
def get_current_datetime(timezone: str = "Asia/Kolkata") -> str:
    """
    Get the current date and time for a timezone.
    Use this for questions about today's date, current time,
    or the current day.
    Example timezone: Asia/Kolkata, UTC, America/New_York.
    """
    try:
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime("%A, %d %B %Y, %I:%M:%S %p %Z")

    except Exception:
        return (
            "Invalid or unsupported timezone. "
            "Use a valid IANA timezone such as Asia/Kolkata."
        )


def analyze_text(text: str) -> str:
    """
    Analyze text and return word count, character count,
    and paragraph count.
    Use when the user requests text statistics.
    """
    words = text.split()
    paragraphs = [
        p for p in text.split("\n") if p.strip()
    ]

    return (
        f"Words: {len(words)}\n"
        f"Characters: {len(text)}\n"
        f"Characters excluding spaces: "
        f"{len(''.join(text.split()))}\n"
        f"Paragraphs: {len(paragraphs)}"
    )


tools = [
    calculator,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search,
    get_current_datetime,
    analyze_text
]