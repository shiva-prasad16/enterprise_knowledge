import os

from openai import OpenAI

from retrieval import hybrid_retrieve


UNSUPPORTED_ANSWER = "The uploaded documents do not provide enough information to answer that."


def _build_evidence(results):
    evidence_blocks = []
    for number, result in enumerate(results, start=1):
        location = f'{result["source"]}, page {result["page"]}'
        if result.get("section"):
            location += f', section {result["section"]}'
        evidence_blocks.append(
            f"[S{number}] {location}\n{result['text']}"
        )
    return "\n\n".join(evidence_blocks)


def _build_sources(results):
    sources = []
    seen = set()

    for number, result in enumerate(results, start=1):
        key = (result["source"], result["page"], result.get("section"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "id": f"S{number}",
                "source": result["source"],
                "page": result["page"],
                "section": result.get("section"),
            }
        )
    return sources


def answer_question(query, knowledge_base):
    results = hybrid_retrieve(query, knowledge_base)

    if not results:
        return {"answer": UNSUPPORTED_ANSWER, "sources": [], "evidence": []}

    evidence = _build_evidence(results)
    prompt = f"""You are an enterprise document assistant. Answer the user's question using only the evidence below.

Rules:
- Give a natural, direct, concise answer. Do not introduce it with phrases such as "According to the documents" or "The source says".
- Synthesize all relevant evidence; do not repeat the same point.
- Place citations such as [S1] or [S1][S2] immediately after the claim they support.
- Cite only evidence that actually supports the claim. Never invent a citation, fact, condition, number, or implication.
- Do not add a separate sources list in the answer; the interface displays sources.
- If the evidence is conflicting, state the conflict and cite both sides.
- If the evidence does not support a reliable answer, reply exactly: {UNSUPPORTED_ANSWER}

Question:
{query}

Evidence:
{evidence}
"""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content.strip()

    if answer == UNSUPPORTED_ANSWER:
        return {"answer": answer, "sources": [], "evidence": []}

    return {
        "answer": answer,
        "sources": _build_sources(results),
        "evidence": results,
    }
