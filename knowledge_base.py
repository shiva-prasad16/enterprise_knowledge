"""Build the in-memory semantic and lexical indexes."""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_knowledge_base(
    chunks: list[dict[str, Any]],
    embedding_model_name: str = EMBEDDING_MODEL,
    reranker_model_name: str = RERANKER_MODEL,
) -> dict[str, Any]:
    """Create normalized FAISS IP, BM25, embedding, and reranker objects."""
    if not chunks:
        raise ValueError("Cannot build a knowledge base without chunks.")
    texts = [chunk["text"] for chunk in chunks]
    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings = embedding_model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return {
        "chunks": chunks,
        "embedding_model": embedding_model,
        "faiss_index": index,
        "bm25": BM25Okapi([_tokenize(text) for text in texts]),
        "reranker": CrossEncoder(reranker_model_name),
    }
