import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def build_knowledge_base(chunks):
    """Build the in-memory FAISS and BM25 indexes for uploaded chunks."""
    if not chunks:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    tokenized_chunks = [text.lower().split() for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)

    return {
        "chunks": chunks,
        "faiss_index": index,
        "bm25": bm25,
    }

        "faiss_index": index,
        "bm25": BM25Okapi([_tokenize(text) for text in texts]),
        "reranker": CrossEncoder(reranker_model_name),
    }
