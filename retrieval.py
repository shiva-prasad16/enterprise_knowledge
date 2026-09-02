"""Hybrid FAISS/BM25 retrieval, reciprocal-rank fusion, and reranking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from document_processor import DocumentChunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    fused_score: float
    rerank_score: float


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        embedding_model: SentenceTransformer,
        reranker: CrossEncoder,
    ) -> None:
        if not chunks:
            raise ValueError("The knowledge base contains no readable text.")
        self.chunks = list(chunks)
        self.embedding_model = embedding_model
        self.reranker = reranker
        vectors = embedding_model.encode(
            [chunk.text for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors))
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def _semantic_rank(self, query: str, limit: int) -> list[int]:
        vector = self.embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        _, indices = self.index.search(np.ascontiguousarray(vector), min(limit, len(self.chunks)))
        return [int(i) for i in indices[0] if i >= 0]

    def _keyword_rank(self, query: str, limit: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return np.argsort(scores)[::-1][: min(limit, len(self.chunks))].astype(int).tolist()

    def search(
        self,
        query: str,
        candidate_k: int = 12,
        final_k: int = 4,
        rrf_k: int = 60,
        rerank_threshold: float = 0.18,
    ) -> list[RetrievedChunk]:
        semantic = self._semantic_rank(query, candidate_k)
        keyword = self._keyword_rank(query, candidate_k)
        fused: dict[int, float] = {}
        for ranking in (semantic, keyword):
            for rank, index in enumerate(ranking, start=1):
                fused[index] = fused.get(index, 0.0) + 1.0 / (rrf_k + rank)
        candidates = sorted(fused, key=fused.get, reverse=True)[:candidate_k]
        pairs = [(query, self.chunks[index].text) for index in candidates]
        rerank_scores = self.reranker.predict(pairs, apply_softmax=False, show_progress_bar=False)
        # ms-marco-MiniLM emits logits. Sigmoid makes the backend threshold intuitive.
        probabilities = 1.0 / (1.0 + np.exp(-np.asarray(rerank_scores, dtype=float)))
        results = [
            RetrievedChunk(self.chunks[index], fused[index], float(score))
            for index, score in zip(candidates, probabilities)
        ]
        results.sort(key=lambda item: item.rerank_score, reverse=True)
        return [item for item in results if item.rerank_score >= rerank_threshold][:final_k]
