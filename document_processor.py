"""PDF extraction, section detection, and page-aware chunking."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def _document_name(uploaded_file: Any) -> str:
    return Path(getattr(uploaded_file, "name", "document.pdf")).name


def split_into_sections(text: str) -> list[dict[str, str]]:
    """Split page text at likely headings while preserving a default section."""
    lines = [line.strip() for line in text.splitlines()]
    sections: list[dict[str, str]] = []
    title = "Document"
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append({"title": title, "text": content})

    for line in lines:
        if not line:
            if body and body[-1] != "":
                body.append("")
            continue
        words = line.split()
        numbered = bool(re.match(r"^(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)])\s+\S+", line))
        short_title = len(words) <= 12 and len(line) <= 100 and (
            line.isupper() or line.istitle()
        )
        if (numbered or short_title) and body:
            flush()
            title, body = line, []
        elif (numbered or short_title) and not body:
            title = line
        else:
            body.append(line)
    flush()
    return sections or ([{"title": "Document", "text": text.strip()}] if text.strip() else [])


def create_chunks(
    sections: Iterable[dict[str, Any]],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[dict[str, Any]]:
    """Create overlapping chunks and retain all supplied section metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict[str, Any]] = []
    for section in sections:
        text = str(section.get("text", "")).strip()
        for part in splitter.split_text(text):
            record = {key: value for key, value in section.items() if key != "text"}
            record["text"] = part.strip()
            record["chunk_id"] = len(chunks)
            chunks.append(record)
    return chunks


def process_uploaded_pdfs(
    uploaded_files: Iterable[Any],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[dict[str, Any]]:
    """Extract uploaded PDFs and return document/page/section-aware chunks."""
    sections: list[dict[str, Any]] = []
    for uploaded_file in uploaded_files:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        reader = PdfReader(uploaded_file)
        document = _document_name(uploaded_file)
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            for section in split_into_sections(page_text):
                sections.append(
                    {
                        "document": document,
                        "page": page_number,
                        "section": section["title"],
                        "text": section["text"],
                    }
                )
    chunks = create_chunks(sections, chunk_size, chunk_overlap)
    for chunk_id, chunk in enumerate(chunks):
        chunk["chunk_id"] = chunk_id
    return chunks
