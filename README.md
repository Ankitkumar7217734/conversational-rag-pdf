# Chat With Your PDFs (Conversational RAG)

Upload one or more PDFs and have a multi-turn conversation with their content. Answers are grounded in your documents and show their sources. Embeddings run locally and free with FastEmbed, and the LLM is Groq, so the only thing you need is a free Groq key.

> **Resume summary:** Built a conversational RAG app for multi-turn Q&A over uploaded PDFs with source citations, history-aware retrieval, and free local FastEmbed embeddings, deployed on Streamlit Community Cloud with a single free API key.

## Demo

_Add a screenshot or short GIF here._

## How it works

```
Upload PDFs  ->  split into chunks  ->  FastEmbed (local, once)  ->  Chroma vector store
                                                                          |
Ask question  ->  Groq rewrites it using chat history  ->  Chroma retrieves  ->  Groq answers + sources
```

- **Embeddings are local and free** (FastEmbed / ONNX). No OpenAI key, no per-call cost.
- **Embeddings run once** on upload; follow-up questions never re-embed the documents.
- **Chat history is per session id**, so follow-ups resolve correctly.
- **Sources are shown** for every answer (file name + page), so you can verify it.

## Key features

- **Single free key.** Only a Groq key is needed, entered in the UI. The public demo works for anyone.
- **Source citations.** Each answer lists the document chunks it used, with file and page number.
- **History-aware retrieval.** `create_history_aware_retriever` rewrites each question into a standalone query before retrieval.
- **Embed-once optimization.** The retriever is cached and only rebuilt when the uploaded file set changes.
- **Cached embedding model.** `@st.cache_resource` loads the model once, not on every rerun.

## Tech stack

| Component | Library |
|-----------|---------|
| UI | Streamlit |
| LLM | Groq, openai/gpt-oss-120b (langchain-groq) |
| Embeddings | FastEmbed, BAAI/bge-small-en-v1.5 (langchain-community) |
| Vector store | Chroma (langchain-chroma) |
| PDF loading | PyPDFLoader (langchain-community) |
| Splitting | RecursiveCharacterTextSplitter, 1000 / 200 overlap |
| RAG + memory | langchain-classic, langchain-core |

## Prerequisites

- Python 3.10+
- A free Groq API key (https://console.groq.com)

## Setup

```bash
cd "5-RAG: Implementation with Q&A Conversation with History."
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501, paste your Groq key, upload PDFs, and ask away. The first run downloads the small embedding model once (a few seconds).

## Notes

- Embeddings run on CPU via FastEmbed (ONNX), chosen so the app fits the Streamlit Community Cloud 1 GB memory limit (no PyTorch).
- Scanned/image-only PDFs have no extractable text; the app reports this instead of failing silently.
- Chroma is kept in memory for simplicity; restarting the app clears the index.
