import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import traceback

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
load_dotenv(dotenv_path=ROOT / ".env", override=False)


def get_config(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val:
        return str(val)
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


def ensure_groq_key() -> str:
    key = get_config("GROQ_API_KEY", "")
    if key:
        os.environ["GROQ_API_KEY"] = key
    return key


def json_serializer(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def normalize_final_response(fr: dict) -> dict:
    fr = fr or {}
    return {
        "answer": fr.get("answer"),
        "agents_used": fr.get("agents_used", []),
        "documents_used": fr.get("documents_used", []),
        "similarity_scores": fr.get("similarity_scores", []),
        "sql_query": fr.get("sql_query"),
        "validation": fr.get("validation", {}),
        "execution_trace": fr.get("execution_trace", {}),
    }


def render_answer(answer):
    if answer is None:
        st.warning("No answer returned.")
        return
    if isinstance(answer, str):
        st.markdown(answer)
        return
    if isinstance(answer, dict):
        st.json(answer)
        return
    if isinstance(answer, list):
        st.dataframe(pd.DataFrame(answer), use_container_width=True)
        return
    st.write(answer)


def render_sources_and_scores(fr: dict):
    sources = fr.get("documents_used", []) or []
    scores = fr.get("similarity_scores", []) or []

    n = max(len(sources), len(scores))
    sources = sources + [""] * (n - len(sources))
    scores = scores + [None] * (n - len(scores))

    if n == 0:
        st.info("No document sources returned.")
        return

    df = pd.DataFrame({"source": sources, "similarity_score": scores})
    st.dataframe(df, use_container_width=True)

    numeric_scores = [float(s) for s in scores if isinstance(s, (int, float, np.generic))]
    if numeric_scores:
        st.caption(f"Best score: {min(numeric_scores)} | Worst score: {max(numeric_scores)}")


def render_db_output(raw_result: dict):
    db_out = safe_get(raw_result, "database_output", default={}) or {}
    sql_query = db_out.get("sql_query") or safe_get(raw_result, "final_response", "sql_query")
    error = db_out.get("error")
    rows = db_out.get("result", []) or []
    row_count = db_out.get("row_count", len(rows))

    if sql_query:
        st.code(sql_query, language="sql")
    else:
        st.info("No SQL query generated.")

    if error:
        st.error(f"Database error: {error}")

    st.caption(f"Rows: {row_count}")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No rows returned.")


def render_validation(fr: dict):
    v = fr.get("validation") or {}
    if not v:
        st.info("No validation output.")
        return

    status = v.get("status", "UNKNOWN")
    if status == "PASS":
        st.success("Validation: PASS")
    elif status == "NEEDS_MORE_INFO":
        st.warning("Validation: NEEDS_MORE_INFO")
    elif status == "FAIL":
        st.error("Validation: FAIL")
    else:
        st.info(f"Validation: {status}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Issues")
        issues = v.get("issues", [])
        st.json(issues) if issues else st.write("None")
    with col2:
        st.subheader("Next actions")
        na = v.get("next_actions", [])
        st.json(na) if na else st.write("None")

    notes = v.get("notes")
    if notes:
        st.caption(notes)


def render_execution_trace(fr: dict):
    trace = fr.get("execution_trace") or {}
    if not trace:
        st.info("No execution trace.")
        return

    cols = st.columns(4)
    cols[0].metric("Attempts", trace.get("attempts", 0))
    cols[1].metric("Max iters", trace.get("max_iters", 0))
    cols[2].metric("Intent", trace.get("intent", ""))
    cols[3].metric("Plan steps", len(trace.get("plan", []) or []))

    st.json(trace)


def paths_status():
    vector_dir = ROOT / "storage" / "vector_store"
    policies_dir = ROOT / "policies"
    db_path = ROOT / "erp.db"

    has_vector = vector_dir.exists() and (vector_dir / "index.faiss").exists()
    has_db = db_path.exists()
    has_policies = policies_dir.exists() and any(policies_dir.glob("*.pdf"))

    return {
        "vector_dir": vector_dir,
        "policies_dir": policies_dir,
        "db_path": db_path,
        "has_vector": has_vector,
        "has_db": has_db,
        "has_policies": has_policies,
    }


def run_script(script_path: Path) -> tuple[bool, str]:
    if not script_path.exists():
        return False, f"Missing script: {script_path}"
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        ok = proc.returncode == 0
        return ok, out.strip()
    except Exception as e:
        return False, str(e)


def init_demo_data() -> dict:
    results = {"policies": None, "db": None, "vector": None}

    ok, out = run_script(ROOT / "policies" / "generate_policies.py")
    results["policies"] = {"ok": ok, "output": out}

    ok, out = run_script(ROOT / "data" / "generate_data.py")
    results["db"] = {"ok": ok, "output": out}

    ok, out = run_script(ROOT / "document_ingestion.py")
    results["vector"] = {"ok": ok, "output": out}

    return results


def log_event(event: dict, filename: str = "events.jsonl"):
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / filename

    payload = {"ts_utc": datetime.utcnow().isoformat(), **event}

    def to_json_safe(obj):
        try:
            import numpy as _np

            if isinstance(obj, _np.generic):
                return obj.item()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except Exception:
            pass
        return str(obj)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=to_json_safe) + "\n")


@st.cache_resource
def load_system():
    key = ensure_groq_key()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set.\n"
            "Local: create a .env file with GROQ_API_KEY=...\n"
            "Streamlit Cloud: set GROQ_API_KEY in App Settings → Secrets."
        )

    status = paths_status()

    if not status["has_db"]:
        raise FileNotFoundError("Database file erp.db not found.\nRun: python data/generate_data.py")

    if not status["has_vector"]:
        raise FileNotFoundError(
            "Vector store not found at storage/vector_store/index.faiss.\n"
            "Run: python document_ingestion.py"
        )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(
        str(status["vector_dir"]),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    engine = create_engine(f"sqlite:///{status['db_path'].as_posix()}", echo=False)

    document_agent = DocumentAgent(vector_db, llm, top_k=5)
    database_agent = DatabaseAgent(engine, llm)
    api_agent = APIAgent(base_urls=["http://localhost:8000/"], timeout=15)
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


st.set_page_config(page_title="ERP Multi-Agent Assistant", page_icon="🧠", layout="wide")
st.title("🧠 ERP Multi-Agent Assistant")
st.caption("Orchestrator + Corrective RAG + Explainability")

with st.sidebar:
    st.subheader("System status")
    s = paths_status()
    c1, c2 = st.columns(2)
    c1.metric("erp.db", "OK" if s["has_db"] else "Missing")
    c2.metric("vector_store", "OK" if s["has_vector"] else "Missing")
    st.write(f"Policies: `{s['policies_dir']}`")
    st.write(f"Vector store: `{s['vector_dir']}`")
    st.write(f"Database: `{s['db_path']}`")

    if not s["has_db"] or not s["has_vector"]:
        st.divider()
        st.subheader("Initialize demo data")

        st.caption("Creates mock PDFs, generates erp.db, and builds the FAISS vector store.")

        if st.button("Build DB + PDFs + Vector Store", use_container_width=True):
            with st.spinner("Building demo data..."):
                results = init_demo_data()
            st.json(results)

            st.cache_resource.clear()
            st.success("Done. Reloading...")
            st.rerun()


try:
    graph = load_system()
    st.success("✅ System loaded")
except Exception:
    st.error("❌ Failed to load system")
    st.code(traceback.format_exc())
    st.stop()


query = st.text_area(
    "Enter your ERP question",
    placeholder="e.g., Which vendors have overdue invoices and what late fee applies according to policy?",
    height=90,
)

colA, colB, colC = st.columns([1, 1, 2])
run_btn = colA.button("Run", type="primary", use_container_width=True)
clear_btn = colB.button("Clear", use_container_width=True)

if clear_btn:
    st.session_state.pop("last_result", None)
    st.rerun()

if run_btn:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Running agents..."):
            raw = graph.invoke({"query": query.strip()})

        log_event({"type": "query_run", "query": query.strip(), "final_response": raw.get("final_response")})

        fr_tmp = raw.get("final_response") or {}
        val = fr_tmp.get("validation") or {}
        if val.get("status") in ("NEEDS_MORE_INFO", "FAIL"):
            log_event(
                {"type": "validation_issue", "query": query.strip(), "validation": val},
                filename="validation.jsonl",
            )

        st.session_state["last_result"] = raw
        st.session_state["last_query"] = query.strip()
        st.session_state["ts"] = datetime.utcnow().isoformat()


raw = st.session_state.get("last_result")
if raw:
    fr = normalize_final_response(raw.get("final_response") or {})

    st.divider()

    left, mid, right = st.columns([2, 2, 2])
    left.subheader("Answer")
    mid.subheader("Agents used")
    right.subheader("Download")

    with mid:
        agents_used = fr.get("agents_used", [])
        st.write(", ".join(agents_used) if agents_used else "—")

    with right:
        blob = {"timestamp_utc": st.session_state.get("ts"), "query": st.session_state.get("last_query"), "raw_result": raw}
        st.download_button(
            label="Download JSON",
            data=json.dumps(blob, indent=2, default=json_serializer),
            file_name="run_result.json",
            mime="application/json",
            use_container_width=True,
        )

    render_answer(fr.get("answer"))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Documents", "🗄️ Database", "✅ Validation", "🧭 Trace", "🧾 Raw JSON"])

    with tab1:
        st.subheader("Sources & similarity scores")
        render_sources_and_scores(fr)

    with tab2:
        st.subheader("SQL + results")
        render_db_output(raw)

    with tab3:
        render_validation(fr)

    with tab4:
        render_execution_trace(fr)

    with tab5:
        st.json(raw)
else:
    st.info("Run a query to see results.")
