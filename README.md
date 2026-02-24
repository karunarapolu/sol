# Sol - GSoC RAG Chatbot

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-%3E%3D1.0-orange.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

## About

> Sol is a Retrieval-Augmented Generation (RAG) chatbot web app that combines vector search, a document store, and a modern LLM API to provide accurate, context-aware answers from project-specific data.

## Table of Contents
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How the app works](#how-the-app-works)
- [Screenshots](#screenshots)
- [Security & privacy notes](#security--privacy-notes)
- [Future Improvements](#future-improvements)
- [License](#license)

## Tech Stack

- Language: Python 3.10+
- UI: Streamlit
- Vector DB: Qdrant (`qdrant-client`)
- Embeddings: SentenceTransformers (`all-MiniLM-L6-v2`) via a small LangChain `Embeddings` wrapper
- Sparse retrieval: BM25 (`rank_bm25`)
- Orchestration: LangChain-style runnables & prompts
- LLM: Google Gemini via `langchain_google_genai.ChatGoogleGenerativeAI`
- Persistence: SQLite for chat history

## Project structure (key files)

- `bot.py` — main Streamlit app and entire RAG flow (retriever, hybrid search, prompt, LLM call, persistence)
- `GSoC_Data/` — dataset + ingestion helpers (embeddings, CSVs)
- `requirements.txt` — runtime dependencies
- `chat_history.db` — SQLite DB created at runtime (schema below)
- `README.md`, `LICENSE`

## Setup & Installation

1) Create and activate venv

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Windows (cmd)
.\.venv\Scripts\Activate
# macOS / Linux
source .venv/bin/activate
```

2) Install

```bash
pip install -r requirements.txt
```

3) Provide environment variables (export or set) and populate the vector store:

    Environment variables (required)

```
QDRANT_API_KEY=your_qdrant_key
GEMINI_API_KEY=your_gemini_key
QDRANT_URL=https://<your-qdrant-host>
```
- `QDRANT_API_KEY` — API key for Qdrant (used by `QdrantClient`)
- `GEMINI_API_KEY` — API key for Google Gemini
- `QDRANT_URL` — Qdrant endpoint 

```bash
python GSoC_Data/embeddings.py  
```

4) Run the app

```bash
streamlit run bot.py
```

## How the app works

1. Loads embeddings model (`SentenceTransformerEmbeddings`) to compute vectors.
2. Uses `QdrantVectorStore` as dense retriever (`k=20`).
3. Uses `BM25Okapi` to compute sparse relevance scores across the Qdrant-scrolled corpus.
4. Combines dense results with BM25 scores (hybrid rank) and formats top documents for context.
5. Builds a strict prompt including conversation history from SQLite.
6. Calls Google Gemini through `ChatGoogleGenerativeAI` to produce a grounded answer.
7. Persists conversation rows in `chat_history.db`.

## Screenshots
![alt text](<Images\Screenshot1.jpeg>)
---
![alt text](<Images\Screenshot2.jpeg>)

## Security & privacy notes

- Chat transcripts stored in `chat_history.db` — purge or encrypt if storing sensitive data.

## Future improvements

- Add Docker + `docker-compose` for zero-dependency deployment
- Add GitHub Actions CI: lint, unit tests, and a small e2e smoke test
- Modularize retriever and LLM adapter interfaces for easier experimentation
- Add authenticated UI, user/session isolation, and access controls

## License

MIT — see `LICENSE`.