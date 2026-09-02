"""Knowledge-base construction and model loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sentence_transformers import CrossEncoder, SentenceTransformer

from document_processor import DocumentChunk, process_pdfs
from retrieval import HybridRetriever


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class KnowledgeBase:
    signature: tuple[tuple[str, int], ...]
    chunks: list[DocumentChunk]
    retriever: HybridRetriever


def file_signature(files: Sequence[object]) -> tuple[tuple[str, int], ...]:
    """Stable key requested by the app: file name + size, sorted for order independence."""
    return tuple(sorted((str(file.name), int(file.size)) for file in files))


def build_knowledge_base(
    files: Sequence[tuple[str, bytes]],
    signature: tuple[tuple[str, int], ...],
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder,
) -> KnowledgeBase:
    chunks = process_pdfs(files)
    if not chunks:
        raise ValueError("No readable text was found. Scanned PDFs require OCR before upload.")
    return KnowledgeBase(
        signature=signature,
        chunks=chunks,
        retriever=HybridRetriever(chunks, embedding_model, reranker),
    )
