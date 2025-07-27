import os
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

# Import your backend modules
# Assumes you run Streamlit from the project root (ClassQuery/)
from app import config
from app import ingest as ingest_mod
from app import query as query_mod

load_dotenv()  # load OPENAI_API_KEY if present

st.set_page_config(page_title="ClassQuery", page_icon="📚", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
UPLOADS_DIR = Path(config.UPLOADS_DIR)
INDEX_PATH = Path(config.INDEX_PATH)
META_PATH = Path(config.META_PATH)

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file) -> Path:
    """Save Streamlit UploadedFile to uploads dir and return its path."""
    target = UPLOADS_DIR / uploaded_file.name
    with target.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


def ensure_api_key_present() -> bool:
    key = os.getenv("OPENAI_API_KEY")
    return bool(key)


# -----------------------------
# Sidebar configuration
# -----------------------------
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "OpenAI model",
    options=["gpt-3.5-turbo", "gpt-4-turbo"],
    index=0,
)

top_k = st.sidebar.slider("Top-K chunks", min_value=1, max_value=10, value=5)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
max_tokens = st.sidebar.slider("Max tokens", min_value=64, max_value=1024, value=300, step=32)

st.sidebar.write("Index path:")
st.sidebar.code(str(INDEX_PATH))

# Option to rebuild index from all PDFs (advanced)
if st.sidebar.button("Rebuild index from uploads"):
    with st.spinner("Rebuilding vector index from all PDFs…"):
        ingest_mod.run_ingestion(UPLOADS_DIR)
    st.sidebar.success("Rebuild complete.")

# -----------------------------
# Main layout
# -----------------------------
st.title("ClassQuery")
st.write("Upload syllabi PDFs and ask questions about your courses.")

col1, col2 = st.columns(2)

# ---- Upload & ingest ----
with col1:
    st.subheader("1. Upload syllabus PDF(s)")
    files = st.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if files and st.button("Ingest uploaded files"):
        paths: List[Path] = []
        for f in files:
            p = save_uploaded_file(f)
            paths.append(p)
        # Simple approach: call run_ingestion on uploads dir. This will add any new vectors.
        with st.spinner("Ingesting PDFs → vectors…"):
            ingest_mod.run_ingestion(UPLOADS_DIR)
        st.success(f"Ingested {len(paths)} file(s).")

# ---- Ask question ----
with col2:
    st.subheader("2. Ask a question")
    question = st.text_input("Enter your question")

    if st.button("Get answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not ensure_api_key_present():
            st.error("OPENAI_API_KEY is missing. Add it to your environment or .env file.")
        else:
            # Patch query module to use the selected model & token caps
            def generate_answer_with_overrides(prompt: str) -> str:
                # Use the same function but override model and token params
                # We replicate the call inside query_mod.generate_answer with custom params
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                resp = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for answering course-related questions."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                return resp.choices[0].message.content.strip()

            with st.spinner("Retrieving relevant chunks…"):
                chunks = query_mod.query_documents(question, top_k=top_k)

            if not chunks:
                st.error("No relevant content found. Make sure you've ingested PDFs.")
            else:
                prompt = query_mod.build_prompt(chunks, question)
                with st.expander("Show constructed prompt"):
                    st.code(prompt)
                with st.spinner("Generating answer…"):
                    try:
                        answer = generate_answer_with_overrides(prompt)
                    except Exception as e:
                        st.exception(e)
                        answer = None

                if answer:
                    st.markdown("### Answer")
                    st.write(answer)

                    st.markdown("### Sources")
                    for i, c in enumerate(chunks, start=1):
                        st.write(f"[{i}] {c['source']} (chunk {c['chunk_id']})")
                        with st.expander(f"Show chunk {i}"):
                            st.write(c["text"])