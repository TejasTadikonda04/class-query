"""
query.py

Full RAG pipeline:
- Embed and retrieve top-k chunks
- Format a prompt with context
- Use OpenAI Chat API to generate a final answer
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai

import config


# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Please add it to a .env file.")


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


def build_prompt(chunks: List[dict], question: str) -> str:
    context = "\n\n".join(f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(chunks))
    prompt = (
        "You are a helpful assistant. Use the following course content to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt


def generate_answer(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for answering course-related questions."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return response["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    print("\nClassQuery RAG Assistant")
    question = input("Enter your question: ").strip()

    if not question:
        print("No input provided.")
        exit()

    print("\nRetrieving relevant content...")
    chunks = query_documents(question, top_k=5)

    if not chunks:
        print("No relevant information found.")
        exit()

    prompt = build_prompt(chunks, question)
    print("Generating answer using OpenAI...\n")
    answer = generate_answer(prompt)

    print("Answer:\n")
    print(answer)
    print("\nSources:")
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}] {chunk['source']} (chunk {chunk['chunk_id']})")