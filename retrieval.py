"""Hybrid FAISS/BM25 retrieval, RRF fusion, and cross-encoder reranking."""

from __future__ import annotations

from typing import Any

import numpy as np


def _result(chunk: dict[str, Any], score: float) -> dict[str, Any]:
    return {"chunk": chunk, "chunk_id": int(chunk["chunk_id"]), "score": float(score)}


def faiss_retrieve(query: str, knowledge_base: dict[str, Any], top_k: int = 12) -> list[dict[str, Any]]:
    model = knowledge_base["embedding_model"]
    vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    limit = min(top_k, len(knowledge_base["chunks"]))
    scores, indices = knowledge_base["faiss_index"].search(vector, limit)
    return [
        _result(knowledge_base["chunks"][idx], score)
        for score, idx in zip(scores[0], indices[0])
        if idx >= 0
    ]


def bm25_retrieve(query: str, knowledge_base: dict[str, Any], top_k: int = 12) -> list[dict[str, Any]]:
    scores = knowledge_base["bm25"].get_scores(query.lower().split())
    indices = np.argsort(scores)[::-1][: min(top_k, len(scores))]
    return [_result(knowledge_base["chunks"][idx], scores[idx]) for idx in indices]


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]], k: int = 60
) -> list[dict[str, Any]]:
    fused: dict[int, dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            chunk_id = item["chunk_id"]
            fused.setdefault(chunk_id, {**item, "score": 0.0})
            fused[chunk_id]["score"] += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda item: item["score"], reverse=True)


def hybrid_retrieve(
    query: str,
    knowledge_base: dict[str, Any],
    retrieve_k: int = 12,
    rerank_k: int = 5,
    rerank_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Fuse dense and lexical results, then rerank and threshold evidence."""
    fused = reciprocal_rank_fusion(
        [faiss_retrieve(query, knowledge_base, retrieve_k), bm25_retrieve(query, knowledge_base, retrieve_k)]
    )
    candidates = fused[: min(retrieve_k, len(fused))]
    if not candidates:
        return []
    pairs = [[query, item["chunk"]["text"]] for item in candidates]
    scores = knowledge_base["reranker"].predict(pairs)
    reranked = [
        {**item, "rrf_score": item["score"], "score": float(score)}
        for item, score in zip(candidates, scores)
        if float(score) >= rerank_threshold
    ]
    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:rerank_k]
