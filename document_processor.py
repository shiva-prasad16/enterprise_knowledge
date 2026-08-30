import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


SECTION_PATTERN = re.compile(
    r"(?=\b\d+(?:\.\d+)+\s+[A-Z][A-Za-z ]{2,60}(?=\s|\u25cf|$))"
)
SECTION_HEADING_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)+\s+[A-Z][A-Za-z ]{2,60}?)(?=\s{2,}|\s*\u25cf|$)"
)


def process_uploaded_pdfs(uploaded_files):
    """Extract non-empty pages and retain source/page metadata."""
    documents = []

    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        reader = PdfReader(uploaded_file)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text and text.strip():
                documents.append(
                    {
                        "text": text.strip(),
                        "source": uploaded_file.name,
                        "page": page_number,
                    }
                )

    return documents


def split_into_sections(text):
    """Split flattened PDF text at numbered headings such as 3.1 Coverage."""
    sections = re.split(SECTION_PATTERN, text)
    return [section.strip() for section in sections if section.strip()]


def _section_name(section_text):
    match = SECTION_HEADING_PATTERN.match(section_text.strip())
    return match.group(1).strip() if match else None


def create_chunks(documents):
    """Create section-aware chunks with stable source, page, and section metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_id = 0

    for document in documents:
        for section in split_into_sections(document["text"]):
            section_name = _section_name(section)

            for text_chunk in splitter.split_text(section):
                cleaned_text = text_chunk.strip()
                if not cleaned_text:
                    continue

                chunks.append(
                    {
                        "text": cleaned_text,
                        "source": document["source"],
                        "page": document["page"],
                        "section": section_name,
                        "chunk_id": chunk_id,
                    }
                )
                chunk_id += 1

    return chunks
