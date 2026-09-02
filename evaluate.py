"""Offline retrieval evaluation from a small JSON test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from knowledge_base import EMBEDDING_MODEL, RERANKER_MODEL, build_knowledge_base


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval recall and reciprocal rank.")
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found in --pdf-dir")
    payloads = [(path.name, path.read_bytes()) for path in pdfs]
    signature = tuple((path.name, path.stat().st_size) for path in pdfs)
    kb = build_knowledge_base(
        payloads,
        signature,
        SentenceTransformer(EMBEDDING_MODEL),
        CrossEncoder(RERANKER_MODEL),
    )
    tests = json.loads(args.tests.read_text(encoding="utf-8"))
    hits, reciprocal_ranks = 0, []
    for case in tests:
        results = kb.retriever.search(case["question"], final_k=args.top_k)
        expected = case["expected_source"].lower()
        rank = next(
            (i for i, result in enumerate(results, 1) if result.chunk.source.lower() == expected),
            None,
        )
        hits += int(rank is not None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        print(f"{'PASS' if rank else 'MISS'} | {case['question']} | rank={rank}")
    total = len(tests)
    if not total:
        raise SystemExit("Test set is empty")
    print(f"\nRecall@{args.top_k}: {hits / total:.3f}")
    print(f"MRR@{args.top_k}: {sum(reciprocal_ranks) / total:.3f}")


if __name__ == "__main__":
    main()
