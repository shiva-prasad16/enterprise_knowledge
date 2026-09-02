"""Grounded answer generation with the OpenAI Responses API."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from openai import OpenAI

from retrieval import RetrievedChunk


FALLBACK = "I don't know based on the available documents."


@dataclass(frozen=True)
class Answer:
    text: str
    evidence: list[RetrievedChunk]


def _response_text(response: object) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                parts.append(str(value))
    return "\n".join(parts).strip()


def answer_question(
    question: str,
    evidence: Sequence[RetrievedChunk],
    client: OpenAI,
    model: str = "gpt-4.1-mini",
) -> Answer:
    selected = list(evidence)
    if not selected:
        return Answer(FALLBACK, [])
    sources = "\n\n".join(
        f"[S{i}] Source: {item.chunk.source} | Page: {item.chunk.page} | "
        f"Section: {item.chunk.section}\n{item.chunk.text}"
        for i, item in enumerate(selected, start=1)
    )
    instructions = f"""You are an enterprise knowledge assistant.
Answer only from the supplied evidence. Be concise, natural, and direct.
Cite each factual claim using one or more source labels such as [S1] or [S2].
Never invent a source, page, fact, or citation.
If the evidence does not answer the question, reply exactly: {FALLBACK}
"""
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=f"Question:\n{question}\n\nEvidence:\n{sources}",
        temperature=0,
    )
    text = _response_text(response) or FALLBACK
    if text != FALLBACK and not re.search(r"\[S\d+\]", text):
        text = FALLBACK
    return Answer(text, selected if text != FALLBACK else [])
