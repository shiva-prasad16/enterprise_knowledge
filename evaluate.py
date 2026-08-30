"""Retrieval and answer evaluation for a locally configured PDF corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from document_processor import process_uploaded_pdfs
from knowledge_base import build_knowledge_base
from rag import FALLBACK, answer_question
from retrieval import hybrid_retrieve


def evaluate_case(case: dict[str, Any], knowledge_base: dict[str, Any], generate_answers: bool) -> dict[str, Any]:
    results = hybrid_retrieve(case["question"], knowledge_base)
    expected_document = case.get("expected_document")
    expected_page = case.get("expected_page")
    retrieval_hit = any(
        (expected_document is None or item["chunk"]["document"] == expected_document)
        and (expected_page is None or item["chunk"]["page"] == expected_page)
        for item in results
    )
    output: dict[str, Any] = {"question": case["question"], "retrieval_hit_at_5": retrieval_hit}
    if generate_answers:
        answer = answer_question(case["question"], knowledge_base)
        output.update({"answer": answer["answer"], "supported": answer["supported"]})
        if case.get("should_fallback") is not None:
            output["fallback_correct"] = (answer["answer"] == FALLBACK) == bool(case["should_fallback"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a dynamic local PDF knowledge base.")
    parser.add_argument("--corpus", type=Path, required=True, help="Folder containing local sample PDFs")
    parser.add_argument("--cases", type=Path, required=True, help="JSON list of evaluation cases")
    parser.add_argument("--generate-answers", action="store_true", help="Call the OpenAI API and test fallback behavior")
    args = parser.parse_args()
    pdf_paths = sorted(args.corpus.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit("No PDFs found. Add local sample PDFs to the configured --corpus folder.")
    with args.cases.open(encoding="utf-8") as stream:
        cases = json.load(stream)
    streams = [path.open("rb") for path in pdf_paths]
    try:
        knowledge_base = build_knowledge_base(process_uploaded_pdfs(streams))
        results = [evaluate_case(case, knowledge_base, args.generate_answers) for case in cases]
    finally:
        for stream in streams:
            stream.close()
    hit_rate = sum(item["retrieval_hit_at_5"] for item in results) / len(results) if results else 0.0
    print(json.dumps({"retrieval_hit_rate_at_5": hit_rate, "cases": results}, indent=2))


if __name__ == "__main__":
    main()
