# orchestration/intent_classifier.py

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


_PROMPT = ChatPromptTemplate.from_template(
    """
Classify the query into exactly one label:

DOCUMENT_QUERY
DATABASE_QUERY
API_QUERY
COMPOSITE_QUERY

Return only the label.

Query:
{query}
""".strip()
)

_ALLOWED = {"DOCUMENT_QUERY", "DATABASE_QUERY", "API_QUERY", "COMPOSITE_QUERY"}


def _get_llm() -> ChatGroq:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    return ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)


def classify_intent(query: str) -> str:
    llm = _get_llm()
    chain = _PROMPT | llm | StrOutputParser()

    raw = chain.invoke({"query": query})
    label = str(raw).strip().upper()
    label = label.replace(".", "").replace(":", "").replace("-", "_")

    return label if label in _ALLOWED else "COMPOSITE_QUERY"