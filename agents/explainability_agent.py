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
        execution_trace = orchestration_output.get("execution_trace") or {}

        final_decision = (
            reasoning_output.get("final_decision")
            if isinstance(reasoning_output, dict)
            else None
        )

        if final_decision:
            answer = final_decision
        elif intent == "DOCUMENT_QUERY":
            answer = document_output.get("answer")
        elif intent == "DATABASE_QUERY":
            answer = {
                "rows": database_output.get("result", []),
                "row_count": database_output.get("row_count", 0),
            }
        elif intent == "API_QUERY":
            answer = api_output.get("result")
        elif intent == "COMPOSITE_QUERY":
            answer = reasoning_output or "No reasoning output."
        else:
            answer = "Unable to determine response."

        agents_used: list[str] = []
        if document_output.get("answer") is not None:
            agents_used.append("DocumentAgent")
        if database_output.get("sql_query") is not None:
            agents_used.append("DatabaseAgent")
        if api_output:
            agents_used.append("APIAgent")
        if reasoning_output:
            agents_used.append("ReasoningAgent")
        if validation_output:
            agents_used.append("CorrectiveValidation")

        return {
            "answer": answer,
            "agents_used": agents_used,
            "documents_used": document_output.get("sources", []),
            "similarity_scores": document_output.get("similarity_scores", []),
            "sql_query": database_output.get("sql_query"),
            "validation": validation_output,
            "execution_trace": execution_trace,
        }