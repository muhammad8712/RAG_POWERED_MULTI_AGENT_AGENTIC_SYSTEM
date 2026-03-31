from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def _format_history(history: list[dict]) -> str:
    """Format last 10 turns for injection into the reasoning prompt."""
    if not history:
        return "None"
    recent = history[-10:]
    lines: list[str] = []
    for turn in recent:
        role = turn.get("role", "user").capitalize()
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        # Label structured table data explicitly so the LLM understands it is real grounding data
        if role == "Assistant" and content.startswith("[Table:"):
            lines.append(f"Assistant (previous answer — structured data table):\n{content}")
        else:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines) if lines else "None"


_REASONING_PROMPT = ChatPromptTemplate.from_template(
    """
You are an ERP reasoning agent.

Use only the evidence provided below.
Do not invent facts.
If the evidence is insufficient, say that clearly.

Prior Reasoning (from a previous attempt — the validator found issues listed below):
{prior_reasoning}

Validator Feedback:
{validator_feedback}


Conversation History (most recent turns):
{history}

Question:
{query}

Database Results:
{database_results}

Document Evidence:
{document_text}

FOLLOW-UP RULE (CRITICAL):
If the Conversation History contains a prior assistant message that includes a table or list of records,
AND the current Question is a follow-up (e.g. "filter that by...", "now show only...", "sort by...",
"how many of those...", "what about X", "same but for Y"):
  - You MUST treat the prior assistant table/list as your ONLY source of truth for that data.
  - Apply the requested filter, sort, or transformation ONLY to the rows already shown in history.
  - Do NOT fetch new data. Do NOT invent new rows. Do NOT use the Database Results section for new records.
  - If the filter produces zero matching rows from the history data, say "No records match that filter."
  - If the question is NOT a follow-up, ignore this rule and use Database Results normally.

General Rules:
- Prefer database evidence for numeric and structured facts.
- Use document evidence for policy, rules, thresholds, and procedures.
- If both are available, combine them clearly.
- Keep the answer concise and business-like.
- If prior reasoning is provided, treat it as your previous attempt. Incorporate new evidence to fix issues,
  but preserve conclusions that were already well-supported.


Return exactly in this format:

Final Decision:
<answer>

Reasoning:
<brief explanation>
""".strip()
)


class ReasoningAgent:
    def __init__(self, llm):
        self.llm = llm
        self.chain = _REASONING_PROMPT | self.llm | StrOutputParser()

    def _coerce_to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    def _extract_document_text(self, agent_outputs: dict[str, Any]) -> str:
        if "document_text" in agent_outputs:
            return self._coerce_to_text(agent_outputs.get("document_text"))

        doc_out = agent_outputs.get("document_output") or {}

        return self._coerce_to_text(
            doc_out.get("retrieved_context")
            or doc_out.get("answer")
            or ""
        )

    def _extract_database_text(self, agent_outputs: dict[str, Any]) -> str:
        db_out = agent_outputs.get("database_output") or {}

        error = db_out.get("error")
        if error:
            return f"Database error: {error}"

        rows = db_out.get("result", [])
        row_count = db_out.get("row_count", len(rows) if isinstance(rows, list) else 0)
        sql_query = db_out.get("sql_query", "")

        return (
            f"SQL Query: {sql_query}\n"
            f"Row Count: {row_count}\n"
            f"Rows: {rows}"
        )

    def run(
        self,
        query: str,
        agent_outputs: dict[str, Any],
        conversation_history: list[dict] | None = None,
    ) -> dict[str, str]:
        database_results = self._extract_database_text(agent_outputs) or "No database results provided."
        document_text = self._extract_document_text(agent_outputs) or "No document evidence provided."
        history_text = _format_history(conversation_history or [])

        # ── Build prior-reasoning context for corrective retries ──────────
        prior = agent_outputs.get("prior_reasoning")
        if prior:
            prior_text = (
                f"Attempt {prior['attempt']} concluded:\n"
                f"{prior['final_decision']}\n\n"
                f"Reasoning: {prior['reasoning']}"
            )
            feedback_text = "\n".join(
                f"- [{issue.get('type')}] {issue.get('detail')}"
                for issue in (prior.get("validation_issues") or [])
            ) or "No specific feedback."
        else:
            prior_text = "None (first attempt)"
            feedback_text = "N/A"

        response = self.chain.invoke(
            {
                "history": history_text,
                "query": query,
                "database_results": database_results,
                "document_text": document_text,
                "prior_reasoning": prior_text,
                "validator_feedback": feedback_text,
            }
        ).strip()

        final_decision, reasoning = self._parse_response(response)

        return {
            "final_decision": final_decision,
            "reasoning": reasoning,
        }

    def _parse_response(self, response: str) -> tuple[str, str]:
        if "Final Decision:" in response and "Reasoning:" in response:
            head, tail = response.split("Reasoning:", 1)
            final_decision = head.split("Final Decision:", 1)[1].strip()
            reasoning = tail.strip()
            return final_decision, reasoning

        return response, "Model did not follow the expected format strictly."