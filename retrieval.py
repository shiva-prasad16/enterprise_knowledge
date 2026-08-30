import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _result_from_chunk(chunk, **scores):
    result = {
        "text": chunk["text"],
        "source": chunk["source"],
        "page": chunk["page"],
        "section": chunk.get("section"),
        "chunk_id": chunk["chunk_id"],
    }
    result.update(scores)
    return result


def faiss_retrieve(query, knowledge_base, k=10, threshold=0.50):
    chunks = knowledge_base["chunks"]
    index = knowledge_base["faiss_index"]
    search_k = min(k, len(chunks))

    query_embedding = model.encode(
        [query], convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, search_k)

    results = []
    for index_position, score in zip(indices[0], scores[0]):
        if index_position != -1 and score >= threshold:
            results.append(
                _result_from_chunk(
                    chunks[index_position], faiss_score=float(score)
                )
            )
    return results


def bm25_retrieve(query, knowledge_base, k=10):
    chunks = knowledge_base["chunks"]
    bm25 = knowledge_base["bm25"]
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][: min(k, len(chunks))]

    return [
        _result_from_chunk(chunks[index_position], bm25_score=float(scores[index_position]))
        for index_position in top_indices
        if scores[index_position] > 0
    ]


def reciprocal_rank_fusion(result_lists, k=60):
    fused = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id] = dict(result)
                fused[chunk_id]["rrf_score"] = 0.0
            else:
                for key, value in result.items():
                    fused[chunk_id].setdefault(key, value)

            fused[chunk_id]["rrf_score"] += 1.0 / (k + rank)

    return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)


def hybrid_retrieve(
    query,
    knowledge_base,
    faiss_k=10,
    bm25_k=10,
    top_n=3,
    rerank_threshold=0.0,
):
    """Run FAISS + BM25, fuse with RRF, then rerank and filter in the backend."""
    faiss_results = faiss_retrieve(query, knowledge_base, k=faiss_k)
    bm25_results = bm25_retrieve(query, knowledge_base, k=bm25_k)
    fused_results = reciprocal_rank_fusion([faiss_results, bm25_results])

    if not fused_results:
        return []

    pairs = [[query, result["text"]] for result in fused_results]
    rerank_scores = reranker.predict(pairs)

    for result, rerank_score in zip(fused_results, rerank_scores):
        result["rerank_score"] = float(rerank_score)

    reranked_results = sorted(
        fused_results,
        key=lambda item: item["rerank_score"],
        reverse=True,
    )
    relevant_results = [
        result
        for result in reranked_results
        if result["rerank_score"] >= rerank_threshold
    ]
    return relevant_results[:top_n]

   
