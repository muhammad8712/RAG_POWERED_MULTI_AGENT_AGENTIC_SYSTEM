from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable

import fitz  # PyMuPDF
import pdfplumber
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _normalize_text(text: str) -> str:
    text = (text or "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix PDF extraction artifact where hyphens are replaced with 'n'
    # e.g. "director■level" or "directornlevel" → "director-level"
    # Common patterns seen in policy PDFs
    text = re.sub(r"\bn-level\b", "n-level", text)  # protect real words
    text = re.sub(r"director\s*[n■]\s*level", "director-level", text, flags=re.IGNORECASE)
    text = re.sub(r"three\s*[n■]\s*way", "three-way", text, flags=re.IGNORECASE)
    text = re.sub(r"high\s*[n■]\s*value", "high-value", text, flags=re.IGNORECASE)
    text = re.sub(r"high\s*[n■]\s*risk", "high-risk", text, flags=re.IGNORECASE)
    text = re.sub(r"anti\s*[n■]\s*corruption", "anti-corruption", text, flags=re.IGNORECASE)
    text = re.sub(r"cross\s*[n■]\s*border", "cross-border", text, flags=re.IGNORECASE)
    text = re.sub(r"long\s*[n■]\s*term", "long-term", text, flags=re.IGNORECASE)
    text = re.sub(r"three\s*[n■]\s*stage", "three-stage", text, flags=re.IGNORECASE)
    # Generic: word-n-word where n is a connector artifact (letter n between two words)
    # Only fix obvious hyphenation patterns: lowercase-n-lowercase where n is 1 char
    text = re.sub(r"([a-z])\■([a-z])", r"\1-\2", text)
    return text.strip()


def _table_to_text(table: list[list[Any]]) -> str:
    lines: list[str] = []
    for row in table:
        cells = [("" if c is None else str(c)).strip() for c in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines).strip()


def _extract_text_pymupdf(pdf_path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    doc = fitz.open(str(pdf_path))
    try:
        for i in range(len(doc)):
            t = _normalize_text(doc[i].get_text("text") or "")
            if t:
                out[i + 1] = t
    finally:
        doc.close()
    return out


def _guess_section_title(block: str) -> str | None:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return None

    first = lines[0]

    # Remove numbering like "1." or "1.2"
    cleaned = re.sub(r"^\d+(\.\d+)*[\)\.]?\s*", "", first).strip()

    if not cleaned:
        return None

    # Heuristic: short heading-like lines
    if len(cleaned) <= 80:
        return cleaned

    return None


def _split_page_into_sections(text: str) -> list[tuple[str | None, str]]:
    """
    Split page text into section-like blocks.
    Returns list of (section_title, section_text).
    """
    text = _normalize_text(text)
    if not text:
        return []

    # Split on blank lines first
    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    sections: list[tuple[str | None, str]] = []

    for block in raw_blocks:
        title = _guess_section_title(block)
        sections.append((title, block))

    return sections


def extract_documents_from_pdf(
    pdf_path: Path,
    *,
    ocr_fn: Callable[[Any], str] | None = None,
    ocr_min_text_chars: int = 40,
) -> list[Document]:
    source_name = pdf_path.name
    docs: list[Document] = []

    pymu_text_by_page: dict[int, str] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = _normalize_text(page.extract_text() or "")

            if not text and not pymu_text_by_page:
                pymu_text_by_page = _extract_text_pymupdf(pdf_path)

            if len(text) < ocr_min_text_chars and page_idx in pymu_text_by_page:
                text = pymu_text_by_page[page_idx]

            if len(text) < ocr_min_text_chars and ocr_fn is not None:
                try:
                    img = page.to_image(resolution=200).original
                    ocr_text = _normalize_text(ocr_fn(img) or "")
                    if ocr_text:
                        text = ocr_text
                except Exception:
                    pass

            if text:
                page_sections = _split_page_into_sections(text)

                if page_sections:
                    for sec_idx, (section_title, section_text) in enumerate(page_sections, start=1):
                        docs.append(
                            Document(
                                page_content=section_text,
                                metadata={
                                    "source": source_name,
                                    "page": page_idx,
                                    "type": "text",
                                    "section_index": sec_idx,
                                    "section_title": section_title or "",
                                },
                            )
                        )
                else:
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": source_name,
                                "page": page_idx,
                                "type": "text",
                                "section_index": 1,
                                "section_title": "",
                            },
                        )
                    )

            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []

            for t_idx, table in enumerate(tables, start=1):
                table_text = _table_to_text(table)
                if table_text:
                    docs.append(
                        Document(
                            page_content=f"Table (page {page_idx}, #{t_idx})\n{table_text}",
                            metadata={
                                "source": source_name,
                                "page": page_idx,
                                "type": "table",
                                "table_index": t_idx,
                                "section_title": f"Table {t_idx}",
                            },
                        )
                    )

    return docs


def _inject_context_headers(docs: list[Document]) -> list[Document]:
    """
    Prefix each document with lightweight context so retrieval is more precise.
    """
    enriched: list[Document] = []

    for d in docs:
        meta = d.metadata or {}
        source = meta.get("source", "")
        page = meta.get("page", "")
        content_type = meta.get("type", "text")
        section_title = (meta.get("section_title") or "").strip()

        prefix_parts = [f"Source: {source}", f"Page: {page}", f"Type: {content_type}"]
        if section_title:
            prefix_parts.append(f"Section: {section_title}")

        prefix = " | ".join(prefix_parts)
        enriched_text = f"{prefix}\n\n{d.page_content}".strip()

        enriched.append(
            Document(
                page_content=enriched_text,
                metadata=meta,
            )
        )

    return enriched


def ingest(input_dir: Path, output_dir: Path) -> None:
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in: {input_dir}")

    all_docs: list[Document] = []
    for pdf_path in pdf_files:
        all_docs.extend(extract_documents_from_pdf(pdf_path))

    all_docs = _inject_context_headers(all_docs)

    # Better for policy documents than one large generic chunk size
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )
    chunks = splitter.split_documents(all_docs)

    # Add chunk ids for easier debugging / explainability
    final_chunks: list[Document] = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = dict(chunk.metadata or {})
        meta["chunk_id"] = idx
        final_chunks.append(Document(page_content=_normalize_text(chunk.page_content), metadata=meta))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(final_chunks, embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    vector_db.save_local(str(output_dir))

    print(f"Ingested PDFs: {len(pdf_files)}")
    print(f"Base documents: {len(all_docs)}")
    print(f"Chunks: {len(final_chunks)}")
    print(f"FAISS index: {output_dir}")


if __name__ == "__main__":
    root = Path.cwd()
    in_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (root / "policies")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else (root / "storage/vector_store")
    ingest(in_dir, out_dir)