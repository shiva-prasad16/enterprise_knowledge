"""PDF extraction and section-aware chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


@dataclass(frozen=True)
class DocumentChunk:
    id: int
    text: str
    source: str
    page: int
    section: str


HEADING_PATTERNS = (
    re.compile(r"^\d+(?:\.\d+)*[.)]?\s+[A-Z][^.!?]{1,100}$"),
    re.compile(r"^[A-Z][A-Z0-9 &/(),:'-]{2,100}$"),
)


def _looks_like_heading(line: str) -> bool:
    line = " ".join(line.split())
    return bool(line and len(line) <= 110 and any(p.match(line) for p in HEADING_PATTERNS))


def _sections_from_page(text: str) -> list[tuple[str, str]]:
    """Split a page at likely headings while retaining a sensible default section."""
    sections: list[tuple[str, str]] = []
    current_title = "General"
    current_lines: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue
        if _looks_like_heading(line):
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_title, body))
            current_title, current_lines = line, []
        else:
            current_lines.append(line)
    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_title, body))
    return sections


def process_pdfs(
    files: Sequence[tuple[str, bytes | BinaryIO]],
    chunk_size: int = 900,
    chunk_overlap: int = 140,
) -> list[DocumentChunk]:
    """Extract and chunk PDFs. Page numbers are one-based for human-readable citations."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[DocumentChunk] = []
    for source, payload in files:
        stream = BytesIO(payload) if isinstance(payload, bytes) else payload
        if hasattr(stream, "seek"):
            stream.seek(0)
        reader = PdfReader(stream)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError(f"Cannot open encrypted PDF: {source}") from exc
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for section, section_text in _sections_from_page(text):
                prefix = "" if section == "General" else f"{section}\n"
                for part in splitter.split_text(section_text):
                    clean = " ".join(part.split())
                    if clean:
                        chunks.append(
                            DocumentChunk(
                                id=len(chunks),
                                text=prefix + clean,
                                source=source,
                                page=page_number,
                                section=section,
                            )
                        )
    return chunks
