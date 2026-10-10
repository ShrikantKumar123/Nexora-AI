# Nexora-AI 🤖

**Nexora-AI is an agentic AI assistant powered by Google Gemini, LangChain, and LangGraph.** It combines conversational AI, tool calling, Retrieval-Augmented Generation (RAG), web search, and persistent memory in a web-based application.

The goal of Nexora-AI is to go beyond basic question answering by enabling an AI assistant to retrieve relevant information, use tools, remember important details, and respond to user queries with contextual answers.

## ✨ Features

- **AI-Powered Conversations:** Interact with a Gemini-powered assistant using natural language.
- **Agentic AI:** Use LangGraph to orchestrate the model's reasoning and tool-calling workflow.
- **Retrieval-Augmented Generation (RAG):** Upload documents and ask questions based on their contents.
- **Document Uploads:** Supports PDF, DOCX, TXT, Markdown, Python, and CSV files.
- **Web Search:** Retrieve current information using Tavily Search.
- **Long-Term Memory:** Save and retrieve relevant user preferences and facts.
- **Conversation History:** Store conversations and retrieve previous messages.
- **Streaming Responses:** Display AI responses incrementally using Server-Sent Events (SSE).
- **Tool Integration:** Use tools for calculations, document retrieval, memory, and date/time queries.
- **Model Selection:** Configure supported Gemini models.
- **Web Interface:** Interact with the assistant through a browser-based interface.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| FastAPI | Backend API and web application |
| LangChain | LLM integration and tool definitions |
| LangGraph | Agent orchestration and workflow execution |
| Google Gemini | Large language model |
| ChromaDB | Vector storage for document retrieval |
| RAG | Context-aware document question answering |
| Tavily | Web search integration |
| SQLite | Conversation persistence and agent checkpointing |
| SQLAlchemy | Database operations and ORM |
| Jinja2 | Server-rendered web interface |
| Uvicorn | ASGI application server |
| LangSmith | Optional tracing and observability, when configured |

## 🏗️ Project Structure

```text
Nexora-AI/
├── agents.py           # Agent configuration and LangGraph workflow
├── app.py              # FastAPI application and API endpoints
├── database.py         # Database operations, history, and memory
├── rag.py              # Document processing and retrieval
├── tools.py            # Agent tools and integrations
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html       # Web interface
├── data/               # Application data and database files
├── uploads/             # Uploaded documents
├── chroma_db/           # Persistent vector database
├── .env                 # Local environment variables (not committed)
├── .gitignore
└── README.md
```

## ⚙️ Getting Started

### Prerequisites

Make sure you have the following:

- Python 3.10 or a compatible version supported by your dependencies
- A Google AI API key for Gemini
- A Tavily API key for web search
- Git

### 1. Clone the repository

```bash
git clone https://github.com/ShrikantKumar123/Nexora-AI.git
cd Nexora-AI
```

### 2. Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_ai_api_key
TAVILY_API_KEY=your_tavily_api_key

GEMINI_MODEL=gemini-2.5-flash

LANGSMITH_TRACING=false
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=Nexora-AI
```

Replace the placeholder values with your own API keys. Keep your `.env` file private and never commit credentials to GitHub.

Enable LangSmith tracing only if you have configured a valid API key and want to trace your application runs.

**Note:** Verify the environment variable names against your current code before running the application.

### 5. Run the application

```bash
python app.py
```

The application starts on port `8080` by default.

Open the following URL in your browser:

http://localhost:8080

## 🔄 How It Works

The application combines a web interface, an LLM-powered agent, external tools, and persistent storage.

1. **User input:** The user submits a question through the web interface.
2. **Agent execution:** FastAPI passes the message to the configured LangGraph agent.
3. **Decision and tool calling:** The agent determines whether it can answer directly or needs an available tool.
4. **Information retrieval:** Depending on the query, the agent can search uploaded documents, use web search, perform calculations, or retrieve saved memory.
5. **Response generation:** Gemini generates a response using the available conversation context and retrieved information.
6. **Streaming:** The backend streams the response to the browser as text becomes available.
7. **Persistence:** Conversation messages and related application state are stored for later retrieval.

## 🧠 RAG Pipeline

Nexora-AI uses Retrieval-Augmented Generation to answer questions about uploaded documents.

The general workflow is:

```text
Upload Document
      |
      v
Extract Document Text
      |
      v
Split Text into Chunks
      |
      v
Generate Embeddings
      |
      v
Store in Vector Database
      |
      v
User Asks a Question
      |
      v
Retrieve Relevant Chunks
      |
      v
Generate Context-Aware Answer
```

The RAG pipeline allows users to ask questions about their own documents instead of relying only on the model's general knowledge.

## 🔧 Available Agent Tools

| Tool | Description |
|---|---|
| `calculator` | Evaluates mathematical expressions |
| `search_uploaded_documents` | Retrieves relevant information from uploaded documents |
| `remember_this` | Saves important information to persistent memory |
| `recall_memory` | Searches saved memory |
| Tavily Search | Searches the web for relevant information |
| `get_current_datetime` | Returns the current date and time for a specified timezone |
| `analyze_text` | Calculates text statistics, including words and characters |

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the web interface |
| GET | `/conversations` | Retrieves the conversation list |
| GET | `/history/{thread_id}` | Retrieves conversation history |
| POST | `/upload` | Uploads a supported document for RAG |
| POST | `/chat/stream` | Processes a chat message and streams the response |

### Example: Chat Request

**Endpoint:** `POST /chat/stream`

```json
{
  "message": "Explain Retrieval-Augmented Generation",
  "thread_id": "demo-session",
  "model": "gemini-2.5-flash"
}
```

The endpoint returns a Server-Sent Events stream containing response tokens and a completion event.

## 🔐 Security and Configuration

- Store API keys in environment variables.
- Never commit `.env` files or private credentials.
- Validate uploaded files and restrict access to sensitive documents.
- Configure appropriate request limits and authentication before exposing the application publicly.
- Avoid logging sensitive user information.
- Review tool execution and input validation before deploying to production.

## 🚀 Future Improvements

Potential enhancements include:

- Automated evaluation of agent responses and RAG retrieval quality.
- Improved retrieval using metadata filtering and reranking.
- Stronger tool error handling and retry mechanisms.
- Authentication and user-specific conversation isolation.
- Automated tests for agent workflows and API endpoints.
- Docker-based deployment and cloud hosting.
- Improved observability, latency monitoring, and token usage tracking.

## 🎯 Learning Objectives

This project demonstrates practical implementation concepts in:

- Agentic AI and LLM application development
- LangChain and LangGraph
- Tool calling and workflow orchestration
- Retrieval-Augmented Generation
- Vector databases and document processing
- FastAPI and API development
- Conversation persistence and memory
- Streaming LLM responses
- AI application observability

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Shrikant Kumar**

- GitHub: [ShrikantKumar123](https://github.com/ShrikantKumar123)
- Project: [Nexora-AI](https://github.com/ShrikantKumar123/Nexora-AI)
