from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import fitz  # PyMuPDF
import pdfplumber
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
            t = (doc[i].get_text("text") or "").strip()
            if t:
                out[i + 1] = t
    finally:
        doc.close()
    return out


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
            text = (page.extract_text() or "").strip()

            if not text and not pymu_text_by_page:
                pymu_text_by_page = _extract_text_pymupdf(pdf_path)

            if len(text) < ocr_min_text_chars and page_idx in pymu_text_by_page:
                text = pymu_text_by_page[page_idx]

            if len(text) < ocr_min_text_chars and ocr_fn is not None:
                try:
                    img = page.to_image(resolution=200).original
                    ocr_text = (ocr_fn(img) or "").strip()
                    if ocr_text:
                        text = ocr_text
                except Exception:
                    pass

            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": source_name, "page": page_idx, "type": "text"},
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
                            metadata={"source": source_name, "page": page_idx, "type": "table"},
                        )
                    )

    return docs


def ingest(input_dir: Path, output_dir: Path) -> None:
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in: {input_dir}")

    all_docs: list[Document] = []
    for pdf_path in pdf_files:
        all_docs.extend(extract_documents_from_pdf(pdf_path))

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    vector_db.save_local(str(output_dir))

    print(f"Ingested PDFs: {len(pdf_files)}")
    print(f"Chunks: {len(chunks)}")
    print(f"FAISS index: {output_dir}")


if __name__ == "__main__":
    root = Path.cwd()
    in_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (root / "policies")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else (root / "storage/vector_store")
    ingest(in_dir, out_dir)