from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


_PLANNER_PROMPT = ChatPromptTemplate.from_template(
    """Plan which workers to run for the ERP assistant.

Available workers:
- document
- database
- api
- reasoning
- validate
- explainability

Intent hint:
{intent}

Query:
{query}

Return JSON only in this format:
{{"steps": ["document"|"database"|"api"|"reasoning"|"validate"|"explainability", ...], "max_iters": 2}}

Planning rules:
- Policy, rules, procedures, approvals, grace periods, fees, onboarding requirements, tolerances -> document
- Metrics, counts, totals, top/best/recent records, rankings, customer/order/product/payment analytics -> database
- Explicit external service or API request -> api
- If the question needs both structured data and policy/rules -> use both document and database, then reasoning
- If intent is API_COMPOSITE_QUERY: use api + document + reasoning
- reasoning should be included when synthesis across multiple sources is needed
- validate must run before explainability
- explainability must always be last

Examples:
- "What is the standard payment term?" -> ["document", "validate", "explainability"]
- "Who are the top 5 customers by total order value?" -> ["database", "validate", "explainability"]
- "Which customers made the highest total payments, and what is the standard payment term policy?" -> ["database", "document", "reasoning", "validate", "explainability"]
- "Show recent sales orders from Odoo and explain the approval rule for high-value purchase orders." -> ["database", "document", "reasoning", "validate", "explainability"]
- "Show top customers from the ERP API and explain the payment term policy." -> ["api", "document", "reasoning", "validate", "explainability"]
- "List top selling products from the API and explain invoice tolerance rules." -> ["api", "document", "reasoning", "validate", "explainability"]
""".strip()
)


class OrchestratorAgent:
    _ALLOWED = ("document", "database", "api", "reasoning", "validate", "explainability")

    def __init__(self, llm, default_max_iters: int = 2):
        self.llm = llm
        self.default_max_iters = default_max_iters
        self.chain = _PLANNER_PROMPT | self.llm | StrOutputParser()

    def _fallback_plan(self, intent: str | None) -> dict[str, Any]:
        label = (intent or "DOCUMENT_QUERY").strip().upper()

        if label == "FOLLOWUP_QUERY":
            # Answer purely from conversation history — no new data fetch
            steps = ["reasoning", "validate", "explainability"]
        elif label == "DATABASE_QUERY":
            steps = ["database", "validate", "explainability"]
        elif label == "API_QUERY":
            steps = ["api", "validate", "explainability"]
        elif label == "API_COMPOSITE_QUERY":
            steps = ["api", "document", "reasoning", "validate", "explainability"]
        elif label == "COMPOSITE_QUERY":
            steps = ["database", "document", "reasoning", "validate", "explainability"]
        else:
            steps = ["document", "validate", "explainability"]

        return {
            "steps": steps,
            "max_iters": self.default_max_iters,
        }

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

    def _postprocess_steps(self, steps: list[str], intent: str | None) -> list[str]:
        intent_label = (intent or "").strip().upper()

        if not steps:
            return self._fallback_plan(intent)["steps"]

        # FOLLOWUP_QUERY: force straight to reasoning, strip any data-fetch steps
        if intent_label == "FOLLOWUP_QUERY":
            # Keep only reasoning + validate + explainability
            steps = [s for s in steps if s in {"reasoning", "validate", "explainability"}]
            if "reasoning" not in steps:
                steps.insert(0, "reasoning")

        # If both document and database are present, reasoning should usually be present too
        if "document" in steps and "database" in steps and "reasoning" not in steps:
            steps.append("reasoning")

        # If intent is API_COMPOSITE and one source is missing, repair the plan
        if intent_label == "API_COMPOSITE_QUERY":
            if "api" not in steps:
                steps.insert(0, "api")
            if "document" not in steps:
                # Insert document after api
                idx = steps.index("api") + 1
                steps.insert(idx, "document")
            if "reasoning" not in steps:
                steps.append("reasoning")

        # If intent is composite and one source is missing, repair the plan
        if intent_label == "COMPOSITE_QUERY":
            if "database" not in steps:
                steps.insert(0, "database")
            if "document" not in steps:
                idx = steps.index("database") + 1
                steps.insert(idx, "document")
            if "reasoning" not in steps:
                steps.append("reasoning")

        # For pure document/database/api queries, avoid unnecessary reasoning
        if intent_label in {"DOCUMENT_QUERY", "DATABASE_QUERY", "API_QUERY"}:
            multi_source = ("document" in steps and "database" in steps) or ("api" in steps and ("document" in steps or "database" in steps))
            if not multi_source and "reasoning" in steps:
                steps = [s for s in steps if s != "reasoning"]

        # validate must exist
        if "validate" not in steps:
            steps.append("validate")

        # explainability must be last
        steps = [s for s in steps if s != "explainability"]
        steps.append("explainability")

        # validate must be immediately before explainability
        steps = [s for s in steps if s != "validate"]
        steps.insert(len(steps) - 1, "validate")

        # dedupe while preserving order
        final_steps: list[str] = []
        seen: set[str] = set()
        for s in steps:
            if s not in seen:
                final_steps.append(s)
                seen.add(s)

        return final_steps

    def run(self, query: str, intent: Optional[str] = None) -> dict[str, Any]:
        try:
            raw = self.chain.invoke(
                {
                    "query": query,
                    "intent": intent or "",
                }
            ).strip()

            obj = json.loads(raw)
            steps = self._normalize_steps(obj.get("steps"))
            steps = self._postprocess_steps(steps, intent)

            max_iters_raw = obj.get("max_iters", self.default_max_iters)
            try:
                max_iters = int(max_iters_raw)
            except Exception:
                max_iters = self.default_max_iters

            if max_iters < 1:
                max_iters = self.default_max_iters

            return {
                "steps": steps,
                "max_iters": max_iters,
            }

        except Exception:
            return self._fallback_plan(intent)