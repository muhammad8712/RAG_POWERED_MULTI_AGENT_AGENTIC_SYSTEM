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

    # ── helpers ──────────────────────────────────────────────────────────────

    def _contains_numbers(self, text: str) -> bool:
        return bool(re.search(r"\b\d+(\.\d+)?\b", text or ""))

    def _contains_unsupported_order_payment_link(self, text: str) -> bool:
        text = (text or "").lower()
        suspicious_patterns = [
            "payment for order",
            "payment belongs to order",
            "matched payment to order",
            "unpaid order",
            "orders without payment",
            "payment linked to order",
        ]
        return any(pattern in text for pattern in suspicious_patterns)

    def _is_insufficient_db_evidence(self, db_out: dict[str, Any]) -> bool:
        sql = str(db_out.get("sql_query") or "").strip().upper()
        err = str(db_out.get("error") or "").strip().lower()

        if "INSUFFICIENT_DB_EVIDENCE" in sql:
            return True
        if "insufficient_db_evidence" in err:
            return True
        if sql.startswith("SELECT 'INSUFFICIENT_DB_EVIDENCE' AS MESSAGE") and "no known table referenced" in err:
            return True
        return False

    _POLICY_PATTERNS = [
        "tolerance", "grace period", "late fee", "late payment", "payment term",
        "payment terms", "approval", "policy", "procedure", "rule ", "rules",
        "onboarding", "compliance", "invoice matching", "discount rule",
        "early payment", "what happens", "what is the standard", "what approval",
        "what documents", "what controls", "what information", "describe",
        "explain how", "procurement rule", "procurement governance",
        "procurement control", "procurement controls",
        "vendor risk", "audit", "price variance", "quantity variance",
        "three-way match", "three way match", "dispute",
        "financial control", "risk assessment", "what is the purpose",
        "supplier compliance", "supplier approval",
    ]

    def _looks_like_policy_question(self, query: str) -> bool:
        q = (query or "").lower()
        return any(p in q for p in self._POLICY_PATTERNS)

    def _is_unsafe_request(self, text: str) -> bool:
        text = (text or "").lower()
        patterns = [
            "drop table", "drop the", "delete all", "delete from",
            "insert into", "insert a fake", "update ", "alter table",
            "alter the", "alter ", "truncate ", "database password",
            "system prompt", "show me your prompt", "give me system prompt",
            "export the entire", "export all", "dump the database",
            "disable the validation", "disable the agent",
            "bypass the system", "reveal hidden", "hidden instructions",
        ]
        return any(p in text for p in patterns)

    def _is_unsafe_answer(self, text: str) -> bool:
        text = (text or "").lower().strip()
        dangerous_patterns = [
            "drop the customers table", "delete all orders", "hackercorp",
            "insert a fake customer", "database password", "system prompt",
        ]
        return any(p in text for p in dangerous_patterns)

    def _get_draft(
        self,
        intent: str,
        doc_out: dict[str, Any],
        db_out: dict[str, Any],
        reasoning_out: dict[str, Any],
    ) -> str:
        if isinstance(reasoning_out, dict) and reasoning_out.get("final_decision"):
            return str(reasoning_out.get("final_decision") or "")
        if intent == "DOCUMENT_QUERY" and isinstance(doc_out.get("answer"), str):
            return doc_out.get("answer") or ""
        if intent == "DATABASE_QUERY" and db_out.get("result"):
            return str(db_out.get("result"))
        return str(reasoning_out or doc_out.get("answer") or db_out.get("result") or "")

    # ── main entry ───────────────────────────────────────────────────────────

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
        db_error = db_out.get("error")
        db_insufficient_ok = self._is_insufficient_db_evidence(db_out)

        db_ok = (db_row_count > 0 and not db_error) or db_insufficient_ok

        draft = self._get_draft(intent, doc_out, db_out, reasoning_out)

        # ── security guards ──────────────────────────────────────────────────
        if self._is_unsafe_request(query):
            return {
                "status": "FAIL",
                "issues": [
                    {
                        "type": "unsafe_request",
                        "detail": "The query requests a destructive, secret-revealing, or disallowed action.",
                    }
                ],
                "next_actions": [],
                "notes": "Unsafe request detected.",
            }

        if self._is_unsafe_answer(draft):
            return {
                "status": "FAIL",
                "issues": [
                    {
                        "type": "unsafe_generated_answer",
                        "detail": "The generated answer appears to comply with a destructive or disallowed request.",
                    }
                ],
                "next_actions": [],
                "notes": "Unsafe answer detected.",
            }

        # ── document validation ──────────────────────────────────────────────
        if intent in ("DOCUMENT_QUERY", "COMPOSITE_QUERY"):
            if not doc_has_chunks:
                issues.append(
                    {
                        "type": "insufficient_retrieval",
                        "detail": "No document chunks returned.",
                    }
                )
                next_actions.append(
                    {
                        "tool": "document",
                        "args": {"query": query, "k": 6},
                    }
                )

            # For pure document questions, do not be too aggressive with doc score threshold.
            # If we have chunks and a clear answer, avoid unnecessary retries.
            if best_doc_score is not None and best_doc_score > self.doc_score_threshold:
                if not (
                    intent == "DOCUMENT_QUERY"
                    and doc_has_chunks
                    and isinstance(doc_out.get("answer"), str)
                    and doc_out.get("answer", "").strip()
                ):
                    issues.append(
                        {
                            "type": "low_relevance",
                            "detail": f"Best doc score {best_doc_score} > {self.doc_score_threshold}",
                        }
                    )
                    next_actions.append(
                        {
                            "tool": "document",
                            "args": {"query": f"{query} policy procedure rules", "k": 8},
                        }
                    )

        # ── database validation ──────────────────────────────────────────────
        if intent == "DATABASE_QUERY":
            if db_error:
                _security_errors = (
                    "forbidden sql detected",
                    "only select queries allowed",
                    "multiple statements detected",
                    "orders and payments may only be related",
                )
                if any(s in (db_error or "").lower() for s in _security_errors):
                    return {
                        "status": "FAIL",
                        "issues": [{"type": "security_sql_blocked", "detail": db_error}],
                        "next_actions": [],
                        "notes": "Query blocked by SQL security guard.",
                    }

                issues.append({"type": "db_error", "detail": db_error})
                if self._looks_like_policy_question(query):
                    next_actions.append(
                        {"tool": "document", "args": {"query": query, "k": 6}}
                    )
                else:
                    next_actions.append(
                        {"tool": "database", "args": {"question": query}}
                    )

            elif db_row_count == 0:
                issues.append({"type": "db_empty", "detail": "No rows returned."})
                if self._looks_like_policy_question(query):
                    next_actions.append(
                        {"tool": "document", "args": {"query": query, "k": 6}}
                    )
                else:
                    next_actions.append(
                        {"tool": "database", "args": {"question": query}}
                    )

        elif intent == "COMPOSITE_QUERY":
            query_lower = (query or "").lower()
            needs_real_db = any(
                phrase in query_lower
                for phrase in [
                    "top ", "highest", "lowest", "average", "total ", "sum ",
                    "count ", "recent ", "sales orders", "customers made",
                    "top-selling", "best-selling", "by country", "by device",
                    "order value", "payments",
                ]
            )

            if needs_real_db:
                if db_error and not db_insufficient_ok:
                    issues.append({"type": "db_error", "detail": db_error})
                    next_actions.append({"tool": "database", "args": {"question": query}})
                elif db_row_count == 0 and not db_insufficient_ok:
                    issues.append(
                        {
                            "type": "db_empty",
                            "detail": "No rows returned for the database-relevant part of the composite query.",
                        }
                    )
                    next_actions.append({"tool": "database", "args": {"question": query}})

        # ── numeric grounding ────────────────────────────────────────────────
        if self.require_evidence_for_numbers and self._contains_numbers(draft):
            if not db_ok and not doc_has_chunks:
                issues.append(
                    {
                        "type": "unsupported_numeric_claim",
                        "detail": "Draft contains numbers but no evidence is present.",
                    }
                )
                next_actions.append({"tool": "document", "args": {"query": query, "k": 8}})
                next_actions.append({"tool": "database", "args": {"question": query}})

        # ── schema limitation guard ──────────────────────────────────────────
        if self._contains_unsupported_order_payment_link(draft):
            issues.append(
                {
                    "type": "unsupported_order_payment_reconciliation",
                    "detail": (
                        "The answer appears to link payments directly to orders, "
                        "but the current schema only links payments to customer_id."
                    ),
                }
            )

        # ── final decision ───────────────────────────────────────────────────
        if not issues:
            return {
                "status": "PASS",
                "issues": [],
                "next_actions": [],
                "notes": "Evidence sufficient.",
            }

        if next_actions:
            deduped_actions: list[dict] = []
            seen: set[str] = set()
            for action in next_actions:
                marker = str(action)
                if marker not in seen:
                    seen.add(marker)
                    deduped_actions.append(action)

            return {
                "status": "NEEDS_MORE_INFO",
                "issues": issues,
                "next_actions": deduped_actions,
                "notes": "Requesting corrective actions.",
            }

        return {
            "status": "FAIL",
            "issues": issues,
            "next_actions": [],
            "notes": "Cannot correct with available tools.",
        }