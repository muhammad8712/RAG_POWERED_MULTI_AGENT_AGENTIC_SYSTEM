from __future__ import annotations

import json
import os
from pathlib import Path

from utils.api_server_manager import ensure_api_server

from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from agents.document_agent import DocumentAgent
from agents.database_agent import DatabaseAgent
from agents.api_agent import APIAgent
from agents.reasoning_agent import ReasoningAgent
from agents.explainability_agent import ExplainabilityAgent

from orchestration.graph import build_graph
from orchestration.orchestrator_agent import OrchestratorAgent
from orchestration.validator_agent import CorrectiveValidationAgent



ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "erp.db"
VECTOR_STORE_PATH = ROOT / "storage" / "vector_store"

# Maximum conversation turns kept in memory (each turn = 1 user + 1 assistant)
MAX_HISTORY_TURNS = 10


def build_system():
    # Auto-start the Mock ERP REST API if it is not already running
    ensure_api_server(verbose=True)

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

    document_agent = DocumentAgent(vector_db, llm, top_k=5)
    database_agent = DatabaseAgent(engine, llm)

    api_agent = APIAgent(
        base_urls=["http://localhost:8000/"],
        timeout=15,
    )

    reasoning_agent = ReasoningAgent(llm)
    explainability_agent = ExplainabilityAgent()
    orchestrator_agent = OrchestratorAgent(llm)
    validator_agent = CorrectiveValidationAgent()   # auto-loads calibrated threshold

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


def _answer_to_str(answer) -> str:
    """Flatten any answer type to a plain string for history storage."""
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer
    if isinstance(answer, (dict, list)):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def _print_answer(fr: dict) -> None:
    """Pretty-print the final response to the terminal."""
    answer = fr.get("answer")
    agents = fr.get("agents_used") or []
    val = fr.get("validation") or {}
    trace = fr.get("execution_trace") or {}

    print("\n" + "=" * 60)

    # Answer
    if isinstance(answer, list):
        print("ANSWER (table):")
        for row in answer[:20]:
            print(" ", row)
        if len(answer) > 20:
            print(f"  ... ({len(answer) - 20} more rows)")
    elif isinstance(answer, dict):
        rows = answer.get("rows", [])
        if rows:
            print("ANSWER (rows):")
            for row in rows[:20]:
                print(" ", row)
        else:
            print(f"ANSWER: {json.dumps(answer, indent=2, ensure_ascii=False)}")
    else:
        print(f"ANSWER:\n{answer}")

    # Metadata
    print()
    print(f"Agents used   : {', '.join(agents) if agents else '—'}")
    print(f"Intent        : {trace.get('intent', '—')}")
    print(f"Validation    : {val.get('status', '—')}")

    sql = fr.get("sql_query")
    if sql:
        print(f"SQL           : {sql}")

    db_count = fr.get("database_row_count", 0)
    if db_count:
        print(f"DB rows       : {db_count}")

    print("=" * 60 + "\n")


def _print_welcome() -> None:
    print("\n" + "=" * 60)
    print("  ERP Multi-Agent Assistant  (multi-turn conversation)")
    print("=" * 60)
    print("  Type your question and press Enter.")
    print("  Follow-up questions like 'filter that by Germany' work.")
    print("  Commands:  'history'  — show conversation so far")
    print("             'clear'    — reset conversation")
    print("             'exit' / 'quit' / 'q'  — quit")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    graph = build_system()

    conversation_history: list[dict] = []

    _print_welcome()

    while True:
        try:
            raw_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw_input:
            continue

        # ── built-in commands ────────────────────────────────────────────────
        if raw_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        if raw_input.lower() == "clear":
            conversation_history = []
            print("  [Conversation cleared]\n")
            continue

        if raw_input.lower() == "history":
            if not conversation_history:
                print("  [No conversation history yet]\n")
            else:
                print()
                for turn in conversation_history:
                    role = turn["role"].capitalize()
                    content = turn["content"]
                    # Truncate long assistant answers for readability
                    if role == "Assistant" and len(content) > 200:
                        content = content[:200] + "…"
                    print(f"  {role}: {content}")
                print()
            continue

        # ── run the graph ────────────────────────────────────────────────────
        print("  [Running agents…]")

        result = graph.invoke({
            "query": raw_input,
            "conversation_history": conversation_history,
        })

        fr = result.get("final_response") or {}
        _print_answer(fr)


        val = fr.get("validation") or {}

        # ── update conversation history ──────────────────────────────────────
        answer_str = _answer_to_str(fr.get("answer"))
        conversation_history.append({"role": "user",      "content": raw_input})
        conversation_history.append({"role": "assistant", "content": answer_str})

        # Keep only the last MAX_HISTORY_TURNS turns
        max_entries = MAX_HISTORY_TURNS * 2
        if len(conversation_history) > max_entries:
            conversation_history = conversation_history[-max_entries:]
