# Enterprise Knowledge Copilot

An advanced, dynamic multi-document RAG application that turns uploaded PDFs into a cited enterprise knowledge assistant. It combines semantic and keyword search, fuses both rankings, reranks evidence, and asks OpenAI to answer strictly from the retrieved passages.


## Why this project matters

Enterprise information is often scattered across policies, handbooks, reports, and manuals. This project demonstrates an end-to-end retrieval system designed for trustworthy internal Q&A: every response is grounded in uploaded documents, includes document/page citations, and falls back when the evidence is insufficient.

## Features

- Dynamic multi-PDF uploads with no prebuilt corpus
- Page- and section-aware extraction and chunking
- `all-MiniLM-L6-v2` embeddings with normalized vectors
- In-memory FAISS `IndexFlatIP` semantic search
- BM25 keyword retrieval
- Reciprocal Rank Fusion (RRF) of dense and lexical rankings
- `ms-marco-MiniLM-L-6-v2` CrossEncoder reranking
- Configurable rerank threshold to reject weak evidence
- Grounded OpenAI Responses API generation
- Inline source, page, and section citations
- Explicit unsupported-question fallback
- Practical retrieval and answer evaluation harness

## Architecture

```text
Uploaded PDFs
     |
     v
Text extraction -> section-aware chunks -> metadata (document, page, section)
     |                                      |
     +--> SentenceTransformer -> normalized vectors -> FAISS IndexFlatIP --+
     |                                                                    |
     +--> tokenized chunks -------------------------> BM25 ----------------+-> RRF
                                                                            |
                                                                            v
                                                              CrossEncoder reranker
                                                                            |
                                                     thresholded evidence + citations
                                                                            |
                                                                            v
                                                           OpenAI Responses API -> answer
```

The knowledge base is intentionally in memory. Every build reflects the current uploads, avoids external vector-database setup, and makes the architecture easy to demonstrate.

## Repository structure

```text
enterprise-knowledge-copilot/
├── app.py                       # Streamlit interface
├── document_processor.py        # PDF extraction, sections, chunks
├── knowledge_base.py            # Embeddings, FAISS, BM25, reranker
├── retrieval.py                 # Dense/lexical retrieval, RRF, reranking
├── rag.py                       # Grounded generation and citations
├── evaluate.py                  # Local evaluation harness
├── evaluation_cases.example.json
├── requirements.txt
├── .env.example
├── .gitignore
└── screenshots/
    ├── app-preview.png
    └── README.md
```




## Design choices and limitations

- Inner product over normalized vectors is cosine similarity.
- RRF makes FAISS and BM25 comparable without calibrating their raw scores.
- CrossEncoder logits are model scores, not probabilities; tune the threshold on a labeled validation set.
- In-memory indexes are rebuilt when the app process restarts and are best suited to demonstrations or modest corpora.
- Generated citations are constrained by the prompt; production deployments should add programmatic citation verification and observability.



## Screenshots

`screenshots/app-preview.png` is a representative mockup. Replace it with a real capture after deployment; see `screenshots/README.md` for a short capture checklist.

## Responsible use

Treat model output as assisted retrieval, not an authoritative decision. Verify cited source passages for legal, financial, safety, HR, or compliance decisions, and do not upload confidential material to services that your organization has not approved.
