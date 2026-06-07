# Deploying to Streamlit Community Cloud

Free, single-key deployment. Embeddings are local (FastEmbed), so no OpenAI key and no embedding cost.

## Before you push: rename this folder

The folder name contains a colon and a trailing period (`5-RAG: Implementation with Q&A Conversation with History.`), which break git paths and URLs. Rename it to something clean such as `conversational-pdf-rag` before creating the repo.

## Step 1: Push to GitHub

Make this folder the repository root so `requirements.txt` and `.streamlit/` sit at the top level.

```bash
cd conversational-pdf-rag
git init
git add .
git commit -m "Conversational PDF RAG (FastEmbed + Groq, Streamlit)"
git branch -M main
git remote add origin https://github.com/<you>/conversational-pdf-rag.git
git push -u origin main
```

Confirm `git status` shows no `.env` and no `secrets.toml` before pushing.

## Step 2: Deploy

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app -> Deploy from GitHub.**
3. Repository: `<you>/conversational-pdf-rag`, Branch: `main`, Main file path: `app.py`.
4. (Optional) Advanced settings -> Python 3.11.
5. Click **Deploy**. The first build installs dependencies and the first request downloads the embedding model once.

## Step 3: API key

The app reads the Groq key from the sidebar, so visitors bring their own free key, you pay nothing.

If you want a zero-typing private demo, set it once in **Settings -> Secrets**:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

## Why FastEmbed

Streamlit Community Cloud apps get 1 GB of RAM. `sentence-transformers` pulls in PyTorch, which is heavy and risks out-of-memory crashes on large PDFs. FastEmbed runs on ONNX Runtime with no PyTorch, so it fits the limit comfortably. No `packages.txt` is needed.
