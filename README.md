# Jane — Local LLM Setup

A self-hosted AI assistant running entirely on a Mac Mini, powered by open-weight models via Ollama and served through OpenWebUI with RAG capabilities.

## Architecture

```mermaid
graph TD
    User["👤 External User<br/>(Browser)"]
    Ngrok["ngrok<br/>Secure Tunnel"]
    OpenWebUI["OpenWebUI<br/>Chat Interface"]
    Ollama["Ollama<br/>Inference Server"]
    Model["🧠 Open-Weight Model<br/>e.g. Qwen3-27B"]
    Chroma["ChromaDB<br/>Vector Database"]
    Docs["📄 Internal Documents<br/>(PDF, DOCX, TXT, ...)"]

    subgraph Mac Mini ["🖥️ Mac Mini (Local)"]
        OpenWebUI
        Ollama
        Model
        Chroma
    end

    User -- "HTTPS" --> Ngrok
    Ngrok -- "HTTP (localhost)" --> OpenWebUI
    OpenWebUI -- "LLM API" --> Ollama
    Ollama -- "runs" --> Model
    OpenWebUI -- "vector search" --> Chroma
    Docs -- "ingested into RAG" --> Chroma
```

## Components

| Component | Role |
|---|---|
| **Ollama** | Runs open-weight LLMs locally with a simple HTTP API |
| **Qwen3-27B** (or similar) | The language model used for inference |
| **OpenWebUI** | Browser-based chat UI, manages conversations and RAG pipelines |
| **ChromaDB** | Embedded vector database storing document embeddings for RAG |
| **ngrok** | Exposes the local OpenWebUI port to the public internet over HTTPS |
| **Internal Documents** | Company/project documents ingested into ChromaDB for retrieval-augmented generation |

## Data Flow

1. Internal documents are chunked, embedded, and stored in **ChromaDB**.
2. A user accesses **OpenWebUI** via the public ngrok URL.
3. OpenWebUI retrieves relevant document chunks from **ChromaDB** based on the user's query.
4. The query + retrieved context are sent to **Ollama**.
5. Ollama runs inference on the local model and streams the response back to the user.
