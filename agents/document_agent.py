# agents/document_agent.py

from __future__ import annotations

from typing import Any, Callable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


class DocumentAgent:
    def __init__(
        self,
        vector_store,
        llm,
        top_k: int = 5,
        ocr_fn: Callable[[Any], str] | None = None,
        ocr_min_text_chars: int = 40,
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        self.ocr_fn = ocr_fn
        self.ocr_min_text_chars = ocr_min_text_chars

        self.prompt = ChatPromptTemplate.from_template(
            """
You are an ERP assistant.
Answer only using the provided context. If the answer is not in the context, say:
"The information is not available in the provided documents."

Context:
{context}

Question:
{input}
""".strip()
        )

        self.document_chain = create_stuff_documents_chain(llm=self.llm, prompt=self.prompt)

    def _table_to_text(self, table: list[list[Any]]) -> str:
        lines: list[str] = []
        for row in table:
            cells = [("" if c is None else str(c)).strip() for c in row]
            if any(cells):
                lines.append(" | ".join(cells))
        return "\n".join(lines).strip()

    def parse_pdf(self, pdf_path: str, source: str | None = None) -> list[Document]:
        try:
            import pdfplumber
        except Exception as e:
            raise RuntimeError(
                "pdfplumber is required for PDF parsing. Install it with: pip install pdfplumber"
            ) from e

        out: list[Document] = []
        src = source or pdf_path

        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()

                if len(text) < self.ocr_min_text_chars and self.ocr_fn is not None:
                    try:
                        img = page.to_image(resolution=200).original
                        ocr_text = (self.ocr_fn(img) or "").strip()
                        if ocr_text:
                            text = ocr_text
                    except Exception:
                        pass

                if text:
                    out.append(
                        Document(
                            page_content=text,
                            metadata={"page": page_idx, "source": src, "type": "text"},
                        )
                    )

                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []

                for t_idx, table in enumerate(tables, start=1):
                    table_text = self._table_to_text(table)
                    if table_text:
                        out.append(
                            Document(
                                page_content=f"Table (page {page_idx}, #{t_idx})\n{table_text}",
                                metadata={"page": page_idx, "source": src, "type": "table"},
                            )
                        )

        return out

    def add_pdf(self, pdf_path: str, source: str | None = None) -> dict[str, Any]:
        docs = self.parse_pdf(pdf_path, source=source)

        add_fn = getattr(self.vector_store, "add_documents", None)
        if callable(add_fn):
            self.vector_store.add_documents(docs)

        counts = {"text": 0, "table": 0, "image": 0}
        for d in docs:
            counts[d.metadata.get("type", "text")] = counts.get(d.metadata.get("type", "text"), 0) + 1

        return {"source": source or pdf_path, "documents_added": len(docs), "counts": counts}

    def run(self, query: str) -> dict[str, Any]:
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=self.top_k)

        if not docs_with_scores:
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "similarity_scores": [],
                "retrieved_context": "",
            }

        docs = [doc for doc, _ in docs_with_scores]

        context_blocks: list[str] = []
        for i, d in enumerate(docs, start=1):
            page = d.metadata.get("page") or d.metadata.get("page_number", "Unknown")
            src = d.metadata.get("source", "Unknown")
            dtype = d.metadata.get("type", "text")
            context_blocks.append(f"[Chunk {i} | {dtype} | Page {page} | Source {src}]\n{d.page_content}")

        retrieved_context = "\n\n".join(context_blocks)

        answer = self.document_chain.invoke({"input": query, "context": docs})

        sources: list[dict[str, Any]] = []
        similarity_scores: list[float] = []
        for doc, score in docs_with_scores:
            page = doc.metadata.get("page") or doc.metadata.get("page_number", "Unknown")
            sources.append(
                {
                    "page": page,
                    "source": doc.metadata.get("source", "Unknown"),
                    "type": doc.metadata.get("type", "text"),
                }
            )
            similarity_scores.append(score)

        return {
            "answer": answer,
            "sources": sources,
            "similarity_scores": similarity_scores,
            "retrieved_context": retrieved_context,
        }