from __future__ import annotations

import re
from typing import Any


class CorrectiveValidationAgent:
    def __init__(
        self,
        doc_score_threshold: float = 1.2,
        min_doc_chunks: int = 1,
        require_evidence_for_numbers: bool = True,
    ):
        self.doc_score_threshold = doc_score_threshold
        self.min_doc_chunks = min_doc_chunks
        self.require_evidence_for_numbers = require_evidence_for_numbers

    def _contains_numbers(self, text: str) -> bool:
        return bool(re.search(r"\b\d+(\.\d+)?\b", text or ""))

    def run(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        next_actions: list[dict[str, Any]] = []

        intent = (state.get("intent") or "").strip()

        doc_out = state.get("document_output") or {}
        db_out = state.get("database_output") or {}
        reasoning_out = state.get("reasoning_output") or {}

        doc_scores = doc_out.get("similarity_scores") or []
        doc_sources = doc_out.get("sources") or []
        doc_has_chunks = len(doc_sources) >= self.min_doc_chunks
        best_doc_score = min(doc_scores) if doc_scores else None

        db_rows = db_out.get("result") or []
        db_row_count = db_out.get("row_count", 0) or len(db_rows)
        db_ok = db_row_count > 0 and not db_out.get("error")

        draft = ""
        if isinstance(reasoning_out, dict) and reasoning_out.get("final_decision"):
            draft = reasoning_out.get("final_decision") or ""
        elif intent == "DOCUMENT_QUERY" and isinstance(doc_out.get("answer"), str):
            draft = doc_out.get("answer") or ""
        else:
            draft = str(reasoning_out or doc_out.get("answer") or db_rows or "")

        if intent in ("DOCUMENT_QUERY", "COMPOSITE_QUERY"):
            if not doc_has_chunks:
                issues.append(
                    {"type": "insufficient_retrieval", "detail": "No document chunks returned."}
                )
                next_actions.append({"tool": "document", "args": {"query": query, "k": 6}})

            if best_doc_score is not None and best_doc_score > self.doc_score_threshold:
                issues.append(
                    {
                        "type": "low_relevance",
                        "detail": f"Best doc score {best_doc_score} > {self.doc_score_threshold}",
                    }
                )
                next_actions.append(
                    {"tool": "document", "args": {"query": f"{query} policy procedure rules", "k": 8}}
                )

        if intent in ("DATABASE_QUERY", "COMPOSITE_QUERY"):
            if not db_ok:
                issues.append(
                    {"type": "db_empty_or_error", "detail": db_out.get("error") or "No rows returned."}
                )
                next_actions.append({"tool": "database", "args": {"question": query}})

        if self.require_evidence_for_numbers and self._contains_numbers(draft):
            if not db_ok and not doc_has_chunks:
                issues.append(
                    {
                        "type": "unsupported_numeric_claim",
                        "detail": "Draft contains numbers but no evidence present.",
                    }
                )
                next_actions.append({"tool": "document", "args": {"query": query, "k": 8}})
                next_actions.append({"tool": "database", "args": {"question": query}})

        if not issues:
            return {"status": "PASS", "issues": [], "next_actions": [], "notes": "Evidence sufficient."}

        if next_actions:
            return {
                "status": "NEEDS_MORE_INFO",
                "issues": issues,
                "next_actions": next_actions,
                "notes": "Requesting corrective actions.",
            }

        return {
            "status": "FAIL",
            "issues": issues,
            "next_actions": [],
            "notes": "Cannot correct with available tools.",
        }