from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


_REASONING_PROMPT = ChatPromptTemplate.from_template(
    """
Use only the information provided. If the information is insufficient, say so.

Question:
{query}

Database:
{database_results}

Documents:
{document_text}

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
        return self._coerce_to_text(doc_out.get("retrieved_context") or "")

    def run(self, query: str, agent_outputs: dict[str, Any]) -> dict[str, str]:
        db_out = agent_outputs.get("database_output") or {}
        database_results = self._coerce_to_text(db_out.get("result", []))
        document_text = self._extract_document_text(agent_outputs) or "No document text provided."

        response = self.chain.invoke(
            {
                "query": query,
                "database_results": database_results,
                "document_text": document_text,
            }
        ).strip()

        final_decision, reasoning = self._parse_response(response)
        return {"final_decision": final_decision, "reasoning": reasoning}

    def _parse_response(self, response: str) -> tuple[str, str]:
        if "Final Decision:" in response and "Reasoning:" in response:
            head, tail = response.split("Reasoning:", 1)
            final_decision = head.split("Final Decision:", 1)[1].strip()
            reasoning = tail.strip()
            return final_decision, reasoning

        return response, "Model did not follow format strictly."