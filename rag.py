"""Grounded answer generation with inline source/page citations."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from retrieval import hybrid_retrieve


FALLBACK = "I couldn't find enough support for that answer in the uploaded documents."


def answer_question(
    query: str,
    knowledge_base: dict[str, Any],
    *,
    model: str | None = None,
    rerank_threshold: float = 0.0,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """Retrieve evidence and answer only from that evidence."""
    evidence = hybrid_retrieve(query, knowledge_base, rerank_threshold=rerank_threshold)
    if not evidence:
        return {"answer": FALLBACK, "sources": [], "supported": False}

    source_lines: list[str] = []
    sources: list[dict[str, Any]] = []
    for number, item in enumerate(evidence, start=1):
        chunk = item["chunk"]
        label = f"{chunk['document']}, p. {chunk['page']}, {chunk.get('section', 'Document')}"
        source_lines.append(f"[S{number}] {label}\n{chunk['text']}")
        sources.append({"id": f"S{number}", "document": chunk["document"], "page": chunk["page"], "section": chunk.get("section"), "score": item["score"]})

    instructions = (
        "You are an enterprise knowledge assistant. Answer using only the supplied sources. "
        "Cite every factual claim inline with [S1], [S2], etc. Never use outside knowledge. "
        f"If the sources do not answer the question, reply exactly: {FALLBACK}"
    )
    prompt = f"Question: {query}\n\nSources:\n" + "\n\n".join(source_lines)
    api = client or OpenAI()
    response = api.responses.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        instructions=instructions,
        input=prompt,
    )
    answer = response.output_text.strip()
    supported = answer != FALLBACK
    return {"answer": answer, "sources": sources if supported else [], "supported": supported}
