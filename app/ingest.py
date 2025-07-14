"""
ingest.py
----------

PDF‑only ingestion module for **ClassQuery**.

Responsibilities
================
1. Discover PDF files in the uploads directory (or a user‑supplied path).
2. Extract raw text from each PDF.
3. Chunk the text for embedding.
4. Embed each chunk with a Sentence‑Transformers model.
5. Persist embeddings in a local FAISS index **and** maintain a JSON
   metadata store mapping vector IDs → document / chunk information.

Environment
===========
All tunable settings live in `config.py`.

Required packages
-----------------
pip install pdfplumber sentence-transformers faiss-cpu tqdm hf_xet
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config  # local module in app/

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def discover_pdfs(directory: Path) -> List[Path]:
    """Recursively find all `.pdf` files in *directory*."""
    return sorted(p for p in directory.rglob("*.pdf") if p.is_file())


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Return the concatenated text of all pages in *pdf_path*."""
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[str]:
    """
    Split *text* into overlapping chunks.

    Overlap helps preserve context between consecutive chunks.
    """
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


def load_faiss_index(dimension: int, index_path: Path) -> faiss.IndexFlatL2:
    """
    Load an existing FAISS index if it exists; otherwise create a new one
    with appropriate *dimension*.
    """
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        if index.d != dimension:
            raise ValueError(
                f"Existing index dimension ({index.d}) ≠ embedding dimension ({dimension})"
            )
    else:
        index = faiss.IndexFlatL2(dimension)
    return index


def load_metadata(meta_path: Path) -> List[dict]:
    """Load metadata list from *meta_path* (or return empty list)."""
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_metadata(meta: List[dict], meta_path: Path) -> None:
    """Persist *meta* list to *meta_path* (pretty‑printed JSON)."""
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Ingestion pipeline
# --------------------------------------------------------------------------- #
def ingest_pdf(pdf_path: Path, model: SentenceTransformer) -> Tuple[List[List[float]], List[dict]]:
    """
    Extract → chunk → embed a single PDF.

    Returns
    -------
    vectors : List[List[float]]
        Embedding vectors for FAISS.
    meta    : List[dict]
        One metadata record per vector.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        return [], []  # skip empty PDFs

    chunks = chunk_text(raw_text)
    vectors = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=config.BATCH_SIZE,
        normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
    )
    meta = [
        {
            "source": str(pdf_path),
            "chunk_id": i,
            "text": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]
    return vectors, meta


def run_ingestion(upload_dir: Path | None = None) -> None:
    """
    Main entry point.

    Parameters
    ----------
    upload_dir : optional Path
        Directory containing PDF uploads.  Defaults to config.UPLOADS_DIR.
    """
    upload_dir = Path(upload_dir or config.UPLOADS_DIR).expanduser()
    if not upload_dir.exists():
        raise FileNotFoundError(f"Uploads directory not found: {upload_dir}")

    pdf_files = discover_pdfs(upload_dir)
    if not pdf_files:
        print("No PDF files found — nothing to ingest.")
        return

    # Initialise model and index
    model = SentenceTransformer(config.MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    index = load_faiss_index(dim, Path(config.INDEX_PATH))
    metadata = load_metadata(Path(config.META_PATH))

    # Keep track of current vector IDs
    start_id = index.ntotal

    print(f"Ingesting {len(pdf_files)} PDF file(s)…")
    for pdf_path in tqdm(pdf_files, unit="pdf"):
        vectors, meta = ingest_pdf(pdf_path, model)
        if vectors is None or len(vectors) == 0:
            continue


        index.add(vectors)
        # Adjust vector IDs when appending metadata
        for i, record in enumerate(meta):
            record["vector_id"] = start_id + i
        metadata.extend(meta)
        start_id += len(vectors)

    # Persist index and metadata
    Path(config.INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))
    save_metadata(metadata, Path(config.META_PATH))
    print(f"Ingestion complete — total vectors in index: {index.ntotal}")


# --------------------------------------------------------------------------- #
# CLI usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDF files into ClassQuery vector store.")
    parser.add_argument(
        "-u",
        "--uploads",
        type=Path,
        help=f"Directory containing PDF uploads (default: {config.UPLOADS_DIR})",
    )
    args = parser.parse_args()
    run_ingestion(args.uploads)
