from __future__ import annotations

from typing import Any


class ExplainabilityAgent:
    def run(self, orchestration_output: dict[str, Any]) -> dict[str, Any]:
        intent = orchestration_output.get("intent")

        document_output = orchestration_output.get("document_output") or {}
        database_output = orchestration_output.get("database_output") or {}
        api_output = orchestration_output.get("api_output") or {}
        reasoning_output = orchestration_output.get("reasoning_output") or {}
        validation_output = orchestration_output.get("validation_output") or {}
        evidence_history = orchestration_output.get("evidence_history") or []
        execution_trace = orchestration_output.get("execution_trace") or {}

        final_decision = None
        reasoning_text = None

        if isinstance(reasoning_output, dict):
            final_decision = reasoning_output.get("final_decision")
            reasoning_text = reasoning_output.get("reasoning")

        if final_decision:
            answer = final_decision
        elif intent == "DOCUMENT_QUERY":
            answer = document_output.get("answer") or "No document answer available."
        elif intent == "DATABASE_QUERY":
            answer = {
                "rows": database_output.get("result", []),
                "row_count": database_output.get("row_count", 0),
            }
        elif intent == "API_QUERY":
            answer = api_output.get("result") or "No API result available."
        elif intent == "COMPOSITE_QUERY":
            answer = final_decision or reasoning_output or "No reasoning output available."
        else:
            answer = "Unable to determine response."

        agents_used: list[str] = []
        if document_output:
            agents_used.append("DocumentAgent")
        if database_output:
            agents_used.append("DatabaseAgent")
        if api_output:
            agents_used.append("APIAgent")
        if reasoning_output:
            agents_used.append("ReasoningAgent")
        if validation_output:
            agents_used.append("CorrectiveValidation")

        # ── corrective loop summary for thesis evaluation ──────────────
        corrective_summary: list[dict[str, Any]] = []
        for snap in evidence_history:
            corrective_summary.append({
                "attempt": snap.get("attempt"),
                "decision": (snap.get("reasoning_output") or {}).get("final_decision", "")[:200],
                "validation_issues": snap.get("validation_issues", []),
            })

        return {
            "answer": answer,
            "reasoning": reasoning_text,
            "agents_used": agents_used,
            "documents_used": document_output.get("sources", []),
            "similarity_scores": document_output.get("similarity_scores", []),
            "sql_query": database_output.get("sql_query"),
            "database_rows": database_output.get("result", []),
            "database_row_count": database_output.get("row_count", 0),
            "database_error": database_output.get("error"),
            "validation": validation_output,
            "execution_trace": execution_trace,
            "evidence_history": evidence_history,
            "corrective_loop_summary": corrective_summary,
        }