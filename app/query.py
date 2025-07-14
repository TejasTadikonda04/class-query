"""
query.py

Interactive query interface for ClassQuery RAG system.
Prompts user for a question via input() and returns relevant results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    import faiss_cpu as faiss

import config  # local config


def load_index_and_metadata() -> Tuple[faiss.IndexFlatL2, List[dict]]:
    index_path = Path(config.INDEX_PATH)
    meta_path = Path(config.META_PATH)

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Vector index or metadata file is missing. Please run ingest.py first.")

    index = faiss.read_index(str(index_path))
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def query_documents(
    user_query: str,
    top_k: int = 5,
    normalize: bool = config.NORMALIZE_EMBEDDINGS,
) -> List[dict]:
    index, metadata = load_index_and_metadata()
    model = SentenceTransformer(config.MODEL_NAME)

    query_vec = model.encode(
        [user_query],
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        record = metadata[idx].copy()
        record["score"] = float(score)
        results.append(record)

    return results


if __name__ == "__main__":
    print("\n🧠 ClassQuery: Ask me about your syllabi\n")
    question = input("🔎 Enter your question: ").strip()

    if not question:
        print("⚠️  No input provided.")
    else:
        results = query_documents(question, top_k=5)
        if not results:
            print("❌ No relevant results found.")
        else:
            print(f"\nTop {len(results)} results for: \"{question}\"\n")
            for i, r in enumerate(results, 1):
                print(f"[{i}] {r['source']} (chunk {r['chunk_id']}, score={r['score']:.4f})")
                print(f"→ {r['text'][:300]}...\n")
