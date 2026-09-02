"""Streamlit UI for Enterprise Knowledge Copilot."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

from knowledge_base import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    KnowledgeBase,
    build_knowledge_base,
    file_signature,
)
from rag import FALLBACK, answer_question


st.set_page_config(page_title="Enterprise Knowledge Copilot", page_icon="📚", layout="wide")
load_dotenv()


@st.cache_resource(show_spinner="Loading retrieval models…")
def load_models() -> tuple[SentenceTransformer, CrossEncoder]:
    return SentenceTransformer(EMBEDDING_MODEL), CrossEncoder(RERANKER_MODEL)


def api_key() -> str | None:
    try:
        secret = st.secrets.get("OPENAI_API_KEY")
    except (FileNotFoundError, KeyError):
        secret = None
    return str(secret or os.getenv("OPENAI_API_KEY") or "").strip() or None


def render_evidence(evidence: list) -> None:
    for index, item in enumerate(evidence, start=1):
        chunk = item.chunk
        st.markdown(f"**[S{index}] {chunk.source} — page {chunk.page}**")
        st.caption(f"Section: {chunk.section}")
        st.write(chunk.text)


st.title("📚 Enterprise Knowledge Copilot")
st.caption("Ask questions across multiple PDFs and receive grounded, traceable answers.")

with st.sidebar:
    st.header("Knowledge base")
    uploads = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    st.caption("The knowledge base rebuilds automatically when the uploaded file set changes.")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("kb_cache", {})
st.session_state.setdefault("active_signature", ())

signature = file_signature(uploads) if uploads else ()
if signature != st.session_state.active_signature:
    st.session_state.messages = []
    st.session_state.active_signature = signature

knowledge_base: KnowledgeBase | None = None
if uploads:
    if signature not in st.session_state.kb_cache:
        try:
            with st.spinner("Reading documents and building the hybrid search index…"):
                embedding_model, reranker = load_models()
                payloads = [(file.name, file.getvalue()) for file in uploads]
                st.session_state.kb_cache[signature] = build_knowledge_base(
                    payloads, signature, embedding_model, reranker
                )
        except Exception as exc:
            st.error(f"Could not build the knowledge base: {exc}")
            st.stop()
    knowledge_base = st.session_state.kb_cache[signature]
    st.success(f"Ready: {len(uploads)} document(s), {len(knowledge_base.chunks)} chunks")
else:
    st.info("Upload one or more PDFs to begin.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("evidence"):
            with st.expander("Technical evidence"):
                render_evidence(message["evidence"])

question = st.chat_input("Ask a question about your documents", disabled=knowledge_base is None)
if question and knowledge_base:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        key = api_key()
        if not key:
            response_text, evidence = (
                "Add OPENAI_API_KEY to your local .env file or Streamlit app secrets.",
                [],
            )
        else:
            with st.spinner("Finding and checking evidence…"):
                evidence = knowledge_base.retriever.search(question)
                answer = answer_question(question, evidence, OpenAI(api_key=key))
                response_text, evidence = answer.text, answer.evidence
        st.markdown(response_text)
        if evidence:
            with st.expander("Technical evidence"):
                render_evidence(evidence)
    st.session_state.messages.append(
        {"role": "assistant", "content": response_text, "evidence": evidence}
    )
