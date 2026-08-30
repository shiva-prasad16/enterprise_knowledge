"""Streamlit UI for a dynamic, upload-driven knowledge copilot."""

import streamlit as st
from dotenv import load_dotenv

from document_processor import process_uploaded_pdfs
from knowledge_base import build_knowledge_base
from rag import answer_question

load_dotenv()
st.set_page_config(page_title="Enterprise Knowledge Copilot", page_icon="📚", layout="wide")
st.title("Enterprise Knowledge Copilot")
st.caption("Ask grounded questions across your uploaded PDFs.")

with st.sidebar:
    uploads = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    threshold = st.number_input("Rerank threshold", value=0.0, step=0.5, help="Higher values require stronger evidence.")
    build = st.button("Build knowledge base", type="primary", use_container_width=True)

if build:
    if not uploads:
        st.warning("Upload at least one PDF first.")
    else:
        with st.spinner("Reading and indexing documents..."):
            chunks = process_uploaded_pdfs(uploads)
            st.session_state.knowledge_base = build_knowledge_base(chunks)
        st.success(f"Indexed {len(chunks)} chunks from {len(uploads)} document(s).")

query = st.chat_input("Ask a question about your documents")
if query:
    st.chat_message("user").write(query)
    if "knowledge_base" not in st.session_state:
        st.chat_message("assistant").warning("Build a knowledge base before asking a question.")
    else:
        with st.chat_message("assistant"), st.spinner("Finding grounded evidence..."):
            result = answer_question(query, st.session_state.knowledge_base, rerank_threshold=threshold)
            st.write(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for source in result["sources"]:
                        st.markdown(f"**[{source['id']}] {source['document']} - page {source['page']}**  \n{source.get('section') or 'Document'}")
