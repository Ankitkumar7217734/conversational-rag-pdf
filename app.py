## RAG Q&A over uploaded PDFs, with conversation memory and source citations.
## Embeddings run locally with FastEmbed (no OpenAI key). LLM is Groq.

import os
import uuid
import tempfile

import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()


@st.cache_resource(show_spinner="Loading the local embedding model (first run only)...")
def get_embeddings():
    """Local, free embeddings via FastEmbed (ONNX). Cached so it loads once.

    Uses BAAI/bge-small-en-v1.5 by default. No API key, runs on CPU, small
    memory footprint so it fits the Streamlit Community Cloud 1 GB tier.
    """
    from langchain_community.embeddings import FastEmbedEmbeddings
    return FastEmbedEmbeddings()


def preset_groq_key() -> str:
    """Optional preset key for deployment (Streamlit secrets or env)."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


## ---------------------------------------------------------------------------
## Page setup
## ---------------------------------------------------------------------------
st.title("Chat With Your PDFs")
st.write("Upload one or more PDFs and ask questions. Answers are grounded in "
         "your documents, with sources shown. Free local embeddings, Groq LLM.")

api_key = st.text_input("Enter your Groq API Key", type="password",
                        value=preset_groq_key())

## Session state
if "store" not in st.session_state:
    st.session_state.store = {}
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "uid" not in st.session_state:
    # Unique per browser session: isolates this user's collection from others
    # sharing the same in-memory Chroma process on Streamlit Cloud.
    st.session_state.uid = uuid.uuid4().hex[:8]

if not api_key:
    st.warning("Please enter your Groq API Key to proceed. Get a free key at "
               "https://console.groq.com.")
    st.stop()

llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=api_key)
session_id = st.text_input("Session ID (separate chat histories)", value="default_session")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

## ---------------------------------------------------------------------------
## Embed only when the uploaded file set changes (embed-once optimization)
## ---------------------------------------------------------------------------
if uploaded_files:
    uploaded_names = sorted([f.name for f in uploaded_files])
    if uploaded_names != st.session_state.processed_files:
        with st.spinner(f"Embedding {len(uploaded_files)} PDF(s)... this runs once."):
            documents = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                file_docs = PyPDFLoader(tmp_path).load()
                # Replace the temp-file path with the real filename for clean citations
                for d in file_docs:
                    d.metadata["source"] = uploaded_file.name
                documents.extend(file_docs)
                os.unlink(tmp_path)

            if not documents:
                st.error("No text could be extracted. The PDFs may be scanned "
                         "images. Try text-based PDFs.")
                st.stop()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(documents)

            # Drop any previous collection for this session so a new upload does
            # not accumulate on top of the old documents.
            if st.session_state.vectorstore is not None:
                try:
                    st.session_state.vectorstore.delete_collection()
                except Exception:
                    pass

            vectorstore = Chroma.from_documents(
                split_docs,
                get_embeddings(),
                collection_name=f"rag_{st.session_state.uid}",
            )
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            st.session_state.processed_files = uploaded_names
            st.session_state.store = {}  # reset chat history for a new doc set
        st.success(f"Embedded {len(split_docs)} chunks from {len(uploaded_files)} file(s). Ask away!")

## ---------------------------------------------------------------------------
## Chat UI (only once documents are embedded)
## ---------------------------------------------------------------------------
if st.session_state.retriever:
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference "
        "context in the chat history, formulate a standalone question which can be "
        "understood without the chat history. Do not answer the question, just "
        "reformulate it if needed, otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    system_prompt = (
        "You are an assistant for question-answering over the user's documents. "
        "Use only the following retrieved context to answer. If the answer is not "
        "in the context, say you don't know. Be clear and concise, and do not make "
        "up information.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversation_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input("Ask a question about the uploaded PDFs:")
    with col2:
        if st.button("Clear chat"):
            st.session_state.store[session_id] = ChatMessageHistory()
            st.rerun()

    if user_input:
        response = conversation_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        st.markdown("**Assistant:** " + response["answer"])

        # Source citations: show which file + page each retrieved chunk came from
        sources = response.get("context", [])
        if sources:
            with st.expander(f"Sources ({len(sources)} chunks used)"):
                seen = set()
                for d in sources:
                    src = d.metadata.get("source", "unknown")
                    page = d.metadata.get("page")
                    label = f"{src}" + (f" (page {page + 1})" if isinstance(page, int) else "")
                    if label in seen:
                        continue
                    seen.add(label)
                    snippet = d.page_content.strip().replace("\n", " ")[:240]
                    st.markdown(f"- **{label}**: {snippet}...")

        with st.expander("Chat history"):
            for msg in get_session_history(session_id).messages:
                st.write(f"**{msg.type}:** {msg.content}")
else:
    st.info("Upload one or more PDF files above to get started.")
