from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from agents.api_agent import APIAgent
from agents.database_agent import DatabaseAgent
from agents.document_agent import DocumentAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.reasoning_agent import ReasoningAgent
from orchestration.graph import build_graph
from orchestration.orchestrator_agent import OrchestratorAgent
from orchestration.validator_agent import CorrectiveValidationAgent

try:
    import numpy as np
except Exception:
    np = None


def to_json_safe(obj: Any) -> Any:
    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    return str(obj)


def build_system():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(
        "storage/vector_store",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    engine = create_engine("sqlite:///erp.db", echo=False)

    document_agent = DocumentAgent(vector_db, llm, top_k=5)
    database_agent = DatabaseAgent(engine, llm)
    api_agent = APIAgent(base_urls=["http://localhost:8000/"], timeout=15)
    reasoning_agent = ReasoningAgent(llm)
    explainability_agent = ExplainabilityAgent()

    orchestrator_agent = OrchestratorAgent(llm)
    validator_agent = CorrectiveValidationAgent()

    return build_graph(
        document_agent=document_agent,
        database_agent=database_agent,
        api_agent=api_agent,
        reasoning_agent=reasoning_agent,
        explainability_agent=explainability_agent,
        orchestrator_agent=orchestrator_agent,
        validator_agent=validator_agent,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    queries_path = root / "evaluation" / "eval_queries.json"
    out_dir = root / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    graph = build_system()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"eval_results_{ts}.jsonl"

    with out_file.open("w", encoding="utf-8") as f:
        for item in queries:
            qid = item.get("id")
            query = item.get("query")
            if not query:
                continue

            result = graph.invoke({"query": query})

            record = {
                "id": qid,
                "type": item.get("type"),
                "query": query,
                "final_response": result.get("final_response"),
                "raw_result": result,
            }
            f.write(json.dumps(record, ensure_ascii=False, default=to_json_safe) + "\n")

    print(f"Saved results to: {out_file}")


if __name__ == "__main__":
    main()