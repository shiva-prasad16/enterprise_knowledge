import streamlit as st

from document_processor import create_chunks, process_uploaded_pdfs
from knowledge_base import build_knowledge_base
from rag import answer_question


st.set_page_config(
    page_title="Enterprise Knowledge Copilot",
    page_icon="\U0001f9e0",
    layout="wide",
)


def initialize_state():
    defaults = {
        "knowledge_base": None,
        "upload_signature": (),
        "processed_files": [],
        "messages": [],
        "knowledge_base_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def upload_signature(uploaded_files):
    """Identify the active knowledge base by uploaded PDF name and size."""
    return tuple(sorted((uploaded_file.name, uploaded_file.size) for uploaded_file in uploaded_files))


def rebuild_if_uploads_changed(uploaded_files):
    current_signature = upload_signature(uploaded_files)

    if current_signature == st.session_state.upload_signature:
        return

    st.session_state.knowledge_base = None
    st.session_state.knowledge_base_error = None
    st.session_state.processed_files = []
    st.session_state.messages = []

    if not uploaded_files:
        st.session_state.upload_signature = ()
        return

    try:
        with st.spinner("Processing PDFs and preparing the knowledge base..."):
            documents = process_uploaded_pdfs(uploaded_files)
            chunks = create_chunks(documents)
            st.session_state.knowledge_base = build_knowledge_base(chunks)
            st.session_state.processed_files = list(current_signature)
            st.session_state.upload_signature = current_signature
    except Exception as error:
        st.session_state.knowledge_base_error = str(error)
        # Retain the failed signature so normal reruns do not repeat expensive work.
        st.session_state.upload_signature = current_signature


def display_sources(result):
    if not result.get("sources"):
        return

    st.markdown("#### Sources")
    for source in result["sources"]:
        label = f'**[{source["id"]}] {source["source"]} — Page {source["page"]}**'
        st.markdown(label)
        if source.get("section"):
            st.caption(source["section"])


def display_evidence(result):
    evidence = result.get("evidence", [])
    if not evidence:
        return

    with st.expander("Technical details"):
        for number, item in enumerate(evidence, start=1):
            st.markdown(f"##### Evidence S{number}")
            location = f'{item["source"]} | Page {item["page"]}'
            if item.get("section"):
                location += f' | Section: {item["section"]}'
            st.markdown(location)
            st.caption(
                f'Chunk {item["chunk_id"]} · '
                f'RRF {item["rrf_score"]:.6f} · '
                f'Rerank {item["rerank_score"]:.4f}'
            )
            st.write(item["text"])
            if number < len(evidence):
                st.divider()


initialize_state()

st.title("\U0001f9e0 Enterprise Knowledge Copilot")
st.write("Ask grounded questions across your uploaded PDFs.")

with st.sidebar:
    st.header("Knowledge base")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

rebuild_if_uploads_changed(uploaded_files or [])

with st.sidebar:
    if st.session_state.knowledge_base_error:
        st.error(f'Processing failed: {st.session_state.knowledge_base_error}')
    elif st.session_state.knowledge_base is not None:
        st.success("Knowledge base ready")
        col1, col2 = st.columns(2)
        col1.metric("Documents", len(st.session_state.processed_files))
        col2.metric("Chunks", len(st.session_state.knowledge_base["chunks"]))
    elif uploaded_files:
        st.info("Processing PDFs...")
    else:
        st.info("Upload one or more PDFs to get started.")

    st.divider()
    st.subheader("RAG pipeline")
    st.markdown(
        "FAISS + BM25  \n"
        "Reciprocal Rank Fusion  \n"
        "CrossEncoder reranking  \n"
        "Grounded answers with citations"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            display_sources(message)
            display_evidence(message)

query = st.chat_input(
    "Ask anything about your documents...",
    disabled=st.session_state.knowledge_base is None,
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            result = answer_question(query, st.session_state.knowledge_base)
        st.markdown(result["answer"])
        display_sources(result)
        display_evidence(result)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "evidence": result["evidence"],
        }
    )
