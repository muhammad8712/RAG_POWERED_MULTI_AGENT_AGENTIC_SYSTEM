from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


_PLANNER_PROMPT = ChatPromptTemplate.from_template(
    """
Plan which workers to run for the ERP assistant.

Tools: document, database, api, reasoning, validate, explainability
Intent hint: {intent}

Query:
{query}

Return JSON only:
{{"steps": ["document"|"database"|"api"|"reasoning"|"validate"|"explainability", ...], "max_iters": 2}}

Guidelines:
- Structured data (vendors/invoices/POs/status/amounts) -> database
- Policy/process/how-to -> document
- Both -> both
- reasoning when synthesis is needed or multiple sources are used
- validate must run before explainability
- explainability must be last
""".strip()
)


class OrchestratorAgent:
    _ALLOWED = ("document", "database", "api", "reasoning", "validate", "explainability")

    def __init__(self, llm, default_max_iters: int = 2):
        self.llm = llm
        self.default_max_iters = default_max_iters
        self.chain = _PLANNER_PROMPT | self.llm | StrOutputParser()

    def _fallback_plan(self, intent: str | None) -> dict[str, Any]:
        label = intent or "DOCUMENT_QUERY"
        if label == "DATABASE_QUERY":
            steps = ["database", "validate", "explainability"]
        elif label == "API_QUERY":
            steps = ["api", "validate", "explainability"]
        elif label == "COMPOSITE_QUERY":
            steps = ["database", "document", "reasoning", "validate", "explainability"]
        else:
            steps = ["document", "validate", "explainability"]

        return {"steps": steps, "max_iters": self.default_max_iters}

    def _normalize_steps(self, steps: Any) -> list[str]:
        if not isinstance(steps, list):
            return []

        allowed = set(self._ALLOWED)
        cleaned: list[str] = []
        seen: set[str] = set()

        for s in steps:
            if not isinstance(s, str):
                continue
            s2 = s.strip().lower()
            if s2 in allowed and s2 not in seen:
                cleaned.append(s2)
                seen.add(s2)

        return cleaned

    def run(self, query: str, intent: Optional[str] = None) -> dict[str, Any]:
        raw = self.chain.invoke({"query": query, "intent": intent or ""}).strip()

        try:
            obj = json.loads(raw)
            steps = self._normalize_steps(obj.get("steps"))
            if not steps:
                raise ValueError("Empty steps")

            max_iters_raw = obj.get("max_iters", self.default_max_iters)
            try:
                max_iters = int(max_iters_raw)
            except Exception:
                max_iters = self.default_max_iters
            if max_iters < 1:
                max_iters = self.default_max_iters

            if "validate" not in steps:
                steps.append("validate")

            steps = [s for s in steps if s != "explainability"]
            steps.append("explainability")

            if steps.index("validate") > steps.index("explainability"):
                steps.remove("validate")
                steps.insert(len(steps) - 1, "validate")

            return {"steps": steps, "max_iters": max_iters}
        except Exception:
            return self._fallback_plan(intent)