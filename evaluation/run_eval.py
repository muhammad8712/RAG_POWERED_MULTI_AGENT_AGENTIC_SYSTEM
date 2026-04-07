from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import json
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from utils.api_server_manager import ensure_api_server
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


ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = ROOT / "evaluation" / "eval_queries.json"
RESULTS_DIR = ROOT / "evaluation" / "results"
VECTOR_STORE_PATH = ROOT / "storage" / "vector_store"
DB_PATH = ROOT / "erp.db"

ERP_API_URL = "http://localhost:8000"


def _check_api_server(base_url: str = ERP_API_URL, timeout: int = 5) -> bool:
    """Ping the mock ERP API health endpoint.

    Returns True if reachable, False otherwise.
    Prints a warning without aborting so non-API queries still run.
    """
    import requests as _req

    health_url = base_url.rstrip("/") + "/api/v1/info"
    try:
        resp = _req.get(health_url, timeout=timeout)
        if resp.status_code == 200:
            print(f"[API] Mock ERP server is UP at {base_url}")
            return True
        else:
            print(
                f"[API] WARNING: Mock ERP server returned HTTP {resp.status_code} "
                f"from {health_url}. API queries (Q106-Q120) will FAIL."
            )
            return False
    except Exception as e:
        print(
            f"\n[API] WARNING: Mock ERP server NOT reachable at {base_url}.\n"
            f"  Error: {e}\n"
            f"  API-category queries (Q106-Q120) will record errors.\n"
            f"  To start the server:  python mock_erp_api/run_server.py\n"
        )
        return False


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

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(f"Vector store not found: {VECTOR_STORE_PATH}")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=api_key,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    engine = create_engine(
        f"sqlite:///{DB_PATH}", 
        echo=False,
        connect_args={"check_same_thread": False}
    )

    document_agent = DocumentAgent(vector_db, llm, top_k=8)
    database_agent = DatabaseAgent(engine, llm)
    api_agent = APIAgent(
        base_urls=[ERP_API_URL + "/"],
        erp_base_url=ERP_API_URL,
        timeout=15,
    )
    reasoning_agent = ReasoningAgent(llm)
    explainability_agent = ExplainabilityAgent()

    orchestrator_agent = OrchestratorAgent(llm)
    validator_agent = CorrectiveValidationAgent()

    graph = build_graph(
        document_agent=document_agent,
        database_agent=database_agent,
        api_agent=api_agent,
        reasoning_agent=reasoning_agent,
        explainability_agent=explainability_agent,
        orchestrator_agent=orchestrator_agent,
        validator_agent=validator_agent,
    )

    return graph


def main() -> None:
    if not QUERIES_PATH.exists():
        raise FileNotFoundError(f"Missing eval queries file: {QUERIES_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Ensure mock ERP API server is running (auto-starts if needed) ──────────
    try:
        ensure_api_server(verbose=True)
    except RuntimeError as e:
        print(f"\n[WARNING] {e}\n  API queries (Q106-Q120) will record errors but eval will continue.\n")

    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8-sig"))
    graph = build_system()

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"eval_results_{ts}.jsonl"

    with out_file.open("w", encoding="utf-8") as f:
        for item in queries:
            qid = item.get("id")
            query = item.get("query")

            if not query:
                continue

            try:
                result = graph.invoke({"query": query})

                record = {
                    "id": qid,
                    "category": item.get("category"),
                    "type": item.get("type"),
                    "difficulty": item.get("difficulty"),
                    "query": query,
                    "final_response": result.get("final_response"),
                    "raw_result": result,
                    "api_output": result.get("api_output"),
                    "status": "ok",
                }

            except Exception as e:
                record = {
                    "id": qid,
                    "category": item.get("category"),
                    "type": item.get("type"),
                    "difficulty": item.get("difficulty"),
                    "query": query,
                    "final_response": None,
                    "raw_result": None,
                    "status": "error",
                    "error": str(e),
                }

            f.write(json.dumps(record, ensure_ascii=False, default=to_json_safe) + "\n")

    print(f"Saved results to: {out_file}")


if __name__ == "__main__":
    main()