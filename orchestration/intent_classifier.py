from __future__ import annotations

import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


_ALLOWED = {
    "DOCUMENT_QUERY",
    "DATABASE_QUERY",
    "API_QUERY",
    "COMPOSITE_QUERY",
    "API_COMPOSITE_QUERY",
    "FOLLOWUP_QUERY",   # follow-up that should be answered from history, not re-fetched
}

DOCUMENT_KEYWORDS = [
    "policy", "rule", "rules", "procedure", "procedures", "approval",
    "required documents", "required document",
    "what is the standard payment term", "payment term", "payment terms",
    "grace period", "late fee", "late payment", "tolerance",
    "invoice matching", "matching tolerance", "price variance", "quantity variance",
    "three-way match", "three way match", "invoice dispute", "dispute resolution",
    "onboarding", "vendor onboarding", "supplier onboarding", "vendor approval",
    "vendor registration", "vendor master", "vendor risk", "risk assessment",
    "compliance", "compliance check", "what happens after", "what happens when",
    "what approval is required", "discount rule", "early payment discount",
    "early payment", "purchase order approval", "po approval",
    "procurement rule", "procurement rules", "procurement policy",
    "procurement governance", "supplier selection", "competitive bidding",
    "request for quotation", "rfq", "financial control", "audit control",
    "what is the purpose", "explain how", "describe how", "describe the",
    "what controls", "what documents", "what information",
    "procurement control", "procurement controls", "supplier compliance",
    "supplier approval",
]

DATABASE_KEYWORDS = [
    "top", "highest", "lowest", "average", "avg", "total", "sum", "count",
    "show", "list", "recent", "latest", "most recent", "most often",
    "most common", "most frequently", "most popular",
    "by country", "by device", "customers", "orders", "order value",
    "products", "best-selling", "best selling", "payment method",
    "payment methods", "sales amount", "sales", "generate the most",
    "generate the highest", "quantity", "row", "status", "how many",
    "per customer", "per product", "per country", "distribution of",
    "breakdown", "ranking", "revenue", "profit", "margin", "lifetime value",
]

API_KEYWORDS = [
    # Explicit API-service phrases (high-confidence)
    "from the erp api", "from the api", "via the api", "erp system via the api",
    "erp api", "from the erp system via", "from erp api",
    # General API / external service signals
    "api endpoint", "rest endpoint", "http endpoint",
    "service response", "external system", "external api",
    "endpoint", "http",
]

# Phrases that are strong enough to force API classification
# even when database keywords are also present in the query.
_STRONG_API_PHRASES = [
    "from the erp api", "from the api", "via the api",
    "erp system via the api", "erp api", "from erp api",
    "from the erp system via",
]

STRONG_DOC_KEYWORDS = [
    "tolerance", "grace period", "late fee", "late payment fee",
    "invoice matching", "matching tolerance", "price variance", "quantity variance",
    "what approval is required", "approval required", "early payment discount",
    "payment terms", "payment term", "procurement governance", "vendor onboarding",
    "vendor approval", "purchase order approval", "compliance", "audit control",
    "financial control", "dispute resolution", "risk assessment",
    "explain how", "describe how", "describe the", "what is the purpose",
    "what controls", "what documents", "what information",
    "three-way match", "three way match",
]

WEAK_DB_KEYWORDS = {
    "orders", "payments", "customers", "products", "quantity", "status", "row",
}

WHOLE_WORD_DB_KEYWORDS = {"top", "avg", "sum", "row", "count", "total", "list", "show", "status"}

# ── follow-up / anaphora signals ─────────────────────────────────────────────
# If these appear in a query AND history is non-empty, resolve using history.
# Split into two groups:
#   STRONG_FOLLOWUP_SIGNALS  → always resolve from history (no new DB call needed)
#   FOLLOWUP_SIGNALS         → weaker heuristic, inherit prior intent but still run agents
STRONG_FOLLOWUP_SIGNALS = [
    # pronouns pointing at a prior result set
    "among these", "which among these", "which of these", "of these",
    "from these", "in these", "these ones",
    "among those", "which among those", "which of those", "of those",
    "from those", "in those", "those ones",
    "among them", "which of them", "of them", "from them", "in them",
    "from the results", "from the list", "from the table",
    "of the above", "from the above",
    "the same ones", "these results", "those results",
    "which one", "which ones",
]

FOLLOWUP_SIGNALS = [
    # filter/refine — still beneficial to run a targeted DB query
    "now filter", "filter that", "filter those", "filter it",
    "show only", "just show", "show me only", "only show",
    "now show", "now sort", "sort that", "sort those", "sort them",
    "now limit", "limit to", "restrict to",
    # follow-up questions
    "what about", "and what about", "also show", "and show",
    "what is that", "what are those", "what were those",
    # transformations
    "break that down", "break it down",
    "same but", "do the same", "do the same for",
    "give me the top", "top of those", "top from those",
    # appending conditions
    "but only", "but for", "but from", "but in",
    "now only", "now for", "excluding", "except",
]


_PROMPT = ChatPromptTemplate.from_template(
    """
You are classifying ERP assistant user queries.

Choose exactly one label:

DOCUMENT_QUERY
DATABASE_QUERY
API_QUERY
COMPOSITE_QUERY

Classification rules:
- DOCUMENT_QUERY:
  Questions about policy, rules, procedures, approvals, grace periods, fees, onboarding
  requirements, tolerances, compliance, or what a document says.
- DATABASE_QUERY:
  Questions asking for metrics, lists, rankings, totals, averages, counts, recent records,
  statuses, or structured transactional data from the SQL database.
- API_QUERY:
  Questions that explicitly require an API or external endpoint/service.
- COMPOSITE_QUERY:
  Questions that require both document/policy knowledge and database facts together.

Conversation History (for resolving follow-up questions):
{history}

Important: if the current question is a follow-up that references something in the history
(e.g. "filter that by Germany", "now show only mobile"), classify it using the type of
the ORIGINAL question it follows up on.

Return only the label.

Query:
{query}
""".strip()
)


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def _contains_any_db(text: str) -> bool:
    for k in DATABASE_KEYWORDS:
        if k in WHOLE_WORD_DB_KEYWORDS:
            if re.search(r"\b" + re.escape(k) + r"\b", text):
                return True
        else:
            if k in text:
                return True
    return False


def _is_followup(query: str, history: list[dict]) -> bool:
    """Return True if query looks like an anaphoric follow-up to a prior turn."""
    if not history:
        return False
    q = _normalize(query)
    return (
        any(signal in q for signal in STRONG_FOLLOWUP_SIGNALS)
        or any(signal in q for signal in FOLLOWUP_SIGNALS)
    )


def _is_strong_followup(query: str, history: list[dict]) -> bool:
    """Return True if query should be answered PURELY from history — no new DB/API call."""
    if not history:
        return False
    q = _normalize(query)
    return any(signal in q for signal in STRONG_FOLLOWUP_SIGNALS)


def _last_user_intent(history: list[dict]) -> str | None:
    """
    Walk backwards through history to find the last assistant response
    that had a clear intent.  Fall back to looking at the last user message.
    We can only use heuristics here since we don't persist intents in history.
    """
    for turn in reversed(history):
        if turn.get("role") == "user":
            q = _normalize(turn.get("content") or "")
            if _contains_any_db(q) and not _contains_any(q, STRONG_DOC_KEYWORDS):
                return "DATABASE_QUERY"
            if _contains_any(q, DOCUMENT_KEYWORDS):
                return "DOCUMENT_QUERY"
    return None


def _rule_based_classify(query: str, history: list[dict] | None = None) -> str | None:
    q = _normalize(query)

    # ── strong follow-up: answer from history, skip re-fetching ─────────────
    # e.g. "which among these have highest orders?" → FOLLOWUP_QUERY
    if _is_strong_followup(q, history or []):
        return "FOLLOWUP_QUERY"

    # ── weaker follow-up: inherit prior intent but still run agents ──────────
    if _is_followup(q, history or []):
        prior_intent = _last_user_intent(history or [])
        if prior_intent:
            return prior_intent

    has_doc = _contains_any(q, DOCUMENT_KEYWORDS)
    has_db = _contains_any_db(q)
    has_api = _contains_any(q, API_KEYWORDS)

    # ── Strong API phrases override database/document signals ─────────────────
    # e.g. "Show top customers from the ERP API" should be API_QUERY, not DB.
    has_strong_api = any(phrase in q for phrase in _STRONG_API_PHRASES)

    if has_strong_api:
        # If a document policy keyword is also present → API_COMPOSITE
        if has_doc and _contains_any(q, STRONG_DOC_KEYWORDS):
            return "API_COMPOSITE_QUERY"
        # Composite also if the query explicitly joins two questions with "and"
        if has_doc and " and " in q:
            return "API_COMPOSITE_QUERY"
        return "API_QUERY"

    if has_api and not has_doc and not has_db:
        return "API_QUERY"

    if has_api and has_doc:
        return "API_COMPOSITE_QUERY"

    if has_doc and has_db:
        if " and " in q:
            return "COMPOSITE_QUERY"

        has_strong_doc = _contains_any(q, STRONG_DOC_KEYWORDS)
        if has_strong_doc:
            db_hits = [k for k in DATABASE_KEYWORDS if (
                re.search(r"\b" + re.escape(k) + r"\b", q)
                if k in WHOLE_WORD_DB_KEYWORDS else k in q
            )]
            non_weak_db = [k for k in db_hits if k not in WEAK_DB_KEYWORDS]
            if not non_weak_db:
                return "DOCUMENT_QUERY"
        return "COMPOSITE_QUERY"

    if has_doc:
        return "DOCUMENT_QUERY"

    if has_db:
        return "DATABASE_QUERY"

    return None


@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=api_key,
    )


@lru_cache(maxsize=1)
def _get_chain():
    llm = _get_llm()
    return _PROMPT | llm | StrOutputParser()


def _format_history_for_prompt(history: list[dict] | None) -> str:
    if not history:
        return "None"
    recent = (history or [])[-4:]
    lines = []
    for turn in recent:
        role = turn.get("role", "user").capitalize()
        content = str(turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "None"


def classify_intent(
    query: str,
    conversation_history: list[dict] | None = None,
) -> str:
    # Deterministic routing first
    heuristic_label = _rule_based_classify(query, conversation_history)
    if heuristic_label in _ALLOWED:
        return heuristic_label

    # LLM fallback — include history for context
    chain = _get_chain()
    raw = chain.invoke({
        "query": query,
        "history": _format_history_for_prompt(conversation_history),
    })

    label = str(raw).strip().upper()
    label = label.replace(".", "").replace(":", "").replace("-", "_")

    return label if label in _ALLOWED else "COMPOSITE_QUERY"