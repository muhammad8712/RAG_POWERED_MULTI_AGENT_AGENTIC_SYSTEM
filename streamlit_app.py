from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
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

from utils.api_server_manager import ensure_api_server, is_api_server_up


ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ROOT / ".env", override=True)
except ImportError:
    # Fallback if the user is running a global streamlit environment without python-dotenv
    env_file = ROOT / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k == "GROQ_API_KEY" and k not in os.environ:
                        os.environ[k] = v

# Maximum number of prior turns to keep in memory per session
MAX_HISTORY_TURNS = 10


# ---------------------------------------------------------------------------
# Config / utilities
# ---------------------------------------------------------------------------

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
        "documents_used": fr.get("documents_used") or fr.get("sources") or [],
        "similarity_scores": fr.get("similarity_scores", []),
        "sql_query": fr.get("sql_query"),
        "validation": fr.get("validation", {}),
        "execution_trace": fr.get("execution_trace", {}),
    }


def _answer_to_str(answer) -> str:
    """Flatten any answer type to a plain string for history storage.
    
    For structured data (rows/records), convert to a readable text table
    so the LLM reasoning agent can anchor its follow-up answers to it.
    """
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        # Prioritise known list keys and build a readable table
        for k in ["rows", "data", "results", "result", "items"]:
            records = answer.get(k)
            if isinstance(records, list) and records and isinstance(records[0], dict):
                # Build a plain-text table: header + rows
                cols = list(records[0].keys())
                lines = [" | ".join(cols)]
                lines.append("-" * len(lines[0]))
                for row in records:
                    lines.append(" | ".join(str(row.get(c, "")) for c in cols))
                row_count = answer.get("row_count", len(records))
                return f"[Table: {row_count} rows]\n" + "\n".join(lines)
        # Flat dict — just dump it
        return json.dumps(answer, default=json_serializer)
    if isinstance(answer, list):
        if answer and isinstance(answer[0], dict):
            cols = list(answer[0].keys())
            lines = [" | ".join(cols)]
            lines.append("-" * len(lines[0]))
            for row in answer:
                lines.append(" | ".join(str(row.get(c, "")) for c in cols))
            return "[Table: {} rows]\n".format(len(answer)) + "\n".join(lines)
        return json.dumps(answer, default=json_serializer)
    return str(answer)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def extract_table_data(obj):
    """Recursively search for a list of dictionaries to render as a table."""
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        return obj
    if isinstance(obj, dict):
        # Prefer known keys if they exist at the current level
        for k in ["rows", "data", "results", "result", "items"]:
            if k in obj and isinstance(obj[k], list) and len(obj[k]) > 0 and isinstance(obj[k][0], dict):
                return obj[k]
        # Otherwise search deeper
        for k, v in obj.items():
            res = extract_table_data(v)
            if res is not None:
                return res
    return None


def render_answer(answer):
    if answer is None:
        st.warning("No answer returned.")
        return
    if isinstance(answer, str):
        st.markdown(
            f'<div class="answer-card">{answer}</div>',
            unsafe_allow_html=True,
        )
        return
    if isinstance(answer, dict):
        table_data = extract_table_data(answer)
        if table_data:
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        else:
            # If no lists of records found, render the flat dictionary as a simple 1-row/key-value table
            st.dataframe(pd.DataFrame([answer]), use_container_width=True)
        return
    if isinstance(answer, list):
        if len(answer) > 0 and isinstance(answer[0], dict):
            st.dataframe(pd.DataFrame(answer), use_container_width=True)
        else:
            st.write(answer)
        return
    st.write(answer)


def render_sources_and_scores(fr: dict):
    sources = fr.get("documents_used", []) or []
    scores = fr.get("similarity_scores", []) or []

    if not sources and not scores:
        st.info("No document sources returned.")
        return

    n = max(len(sources), len(scores))
    scores = scores + [None] * (n - len(scores))

    # sources may be dicts {"page":..,"source":..,"type":..} or plain strings
    rows = []
    for i, src in enumerate(sources):
        score = scores[i] if i < len(scores) else None
        if isinstance(src, dict):
            rows.append({
                "source": src.get("source", "Unknown"),
                "page": src.get("page", ""),
                "type": src.get("type", ""),
                "similarity_score": score,
            })
        else:
            rows.append({"source": str(src), "page": "", "type": "", "similarity_score": score})

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    numeric_scores = [float(s) for s in scores if isinstance(s, (int, float, np.generic))]
    if numeric_scores:
        st.caption(f"Best score: {min(numeric_scores):.4f} | Worst score: {max(numeric_scores):.4f}")


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

    cols = st.columns(5)
    cols[0].metric("Attempts", trace.get("attempts", 0))
    cols[1].metric("Max iters", trace.get("max_iters", 0))
    cols[2].metric("Intent", trace.get("intent", ""))
    cols[3].metric("Plan steps", len(trace.get("initial_plan", []) or []))
    cols[4].metric("History turns", trace.get("history_turns", 0))

    st.json(trace)


def render_conversation_history():
    """Render the running conversation in a chat-style layout."""
    history = st.session_state.get("conversation_history", [])
    if not history:
        return

    st.subheader("💬 Conversation History")
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                with st.expander("View complete answer"):
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, list):
                            st.dataframe(pd.DataFrame(parsed), use_container_width=True)
                        elif isinstance(parsed, dict):
                            table_data = extract_table_data(parsed)
                            if table_data:
                                st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                            else:
                                st.dataframe(pd.DataFrame([parsed]), use_container_width=True)
                        else:
                            st.markdown(content)
                    except (json.JSONDecodeError, TypeError):
                        st.markdown(content)


# ---------------------------------------------------------------------------
# Paths / setup
# ---------------------------------------------------------------------------

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


def run_script(script_path: Path) -> dict:
    if not script_path.exists():
        return {
            "ok": False,
            "script": str(script_path),
            "returncode": None,
            "output": "Script not found.",
        }

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    combined = out + ("\n" + err if err else "")
    return {
        "ok": proc.returncode == 0,
        "script": str(script_path),
        "returncode": proc.returncode,
        "output": combined.strip(),
    }


def init_demo_data() -> dict:
    return {
        "db": run_script(ROOT / "data" / "generate_mock_erp.py"),
        "vector_store": run_script(ROOT / "ingestion" / "document_ingestion.py"),
    }





# ---------------------------------------------------------------------------
# System loader
# ---------------------------------------------------------------------------

@st.cache_resource
def load_system():
    key = ensure_groq_key()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set.\n"
            "Local: create a .env file with GROQ_API_KEY=...\n"
            "Streamlit Cloud: set GROQ_API_KEY in App Settings → Secrets."
        )

    # Auto-start the Mock ERP REST API if it is not already running.
    # NOTE: Do NOT call any st.* functions here — they are forbidden inside
    # @st.cache_resource. Any UI feedback must happen outside this function.
    try:
        ensure_api_server(verbose=False)
    except RuntimeError:
        pass  # API agent handles connection failures gracefully; warning shown below.

    status = paths_status()
    if not status["has_db"]:
        raise FileNotFoundError(
            "Database file erp.db not found. Run: python data/generate_mock_erp.py"
        )
    if not status["has_vector"]:
        raise FileNotFoundError(
            "Vector store not found. Run: python ingestion/document_ingestion.py"
        )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(
        str(status["vector_dir"]),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    engine = create_engine(
        f"sqlite:///{status['db_path'].as_posix()}", 
        echo=False, 
        connect_args={"check_same_thread": False}
    )

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


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ERP Multi-Agent Assistant", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.erp-header {
    background: linear-gradient(135deg, #1F4E79 0%, #2E75B6 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    color: white;
}
.erp-header h1 { margin: 0 0 4px 0; font-size: 1.9rem; font-weight: 700; color: white !important; }
.erp-header p  { margin: 0; font-size: 0.92rem; opacity: 0.85; color: white !important; }

.answer-card {
    background: #f8fafc;
    border-left: 4px solid #2E75B6;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 8px 0 16px 0;
    font-size: 0.97rem;
    line-height: 1.65;
}

.stForm > div { border: none !important; padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="erp-header">
  <h1>🧠 ERP Multi-Agent Assistant</h1>
  <p>Orchestrator &nbsp;·&nbsp; Corrective RAG &nbsp;·&nbsp; SQL Validation &nbsp;·&nbsp; Explainability</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

status = paths_status()

with st.sidebar:
    st.subheader("System status")
    c1, c2, c3 = st.columns(3)
    c1.metric("erp.db", "OK" if status["has_db"] else "Missing")
    c2.metric("vector_store", "OK" if status["has_vector"] else "Missing")
    c3.metric("API", "Running" if is_api_server_up() else "Offline")

    st.divider()

    history_len = len(st.session_state.get("conversation_history", []))
    st.subheader("💬 Conversation")
    st.metric("Turns in memory", history_len // 2)
    st.caption(
        f"Keeping last {MAX_HISTORY_TURNS} turns. "
        "Agents use history to resolve follow-up questions like 'filter that by Germany'."
    )

    st.divider()
    st.subheader("Initialize real ERP data")
    st.caption("Loads the real datasets into erp.db and builds the FAISS vector store from the policy PDFs.")

    if st.button("Build DB + PDFs + Vector Store", use_container_width=True):
        with st.spinner("Building system artifacts..."):
            build_results = init_demo_data()

        st.session_state["build_results"] = build_results
        load_system.clear()
        st.success("Build completed. Reloading…")
        st.rerun()

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["conversation_history"] = []
        st.session_state["last_result"] = None
        st.rerun()

    st.divider()
    render_conversation_history()


if "build_results" in st.session_state:
    with st.expander("Last build output", expanded=not (status["has_db"] and status["has_vector"])):
        st.json(st.session_state["build_results"])


missing = []
if not status["has_db"]:
    missing.append("erp.db")
if not status["has_vector"]:
    missing.append("vector_store")

if missing:
    st.warning(
        "Missing runtime artifacts: "
        + ", ".join(missing)
        + ".\n\nUse the sidebar button **Build DB + PDFs + Vector Store**."
    )
    st.stop()


try:
    graph = load_system()
    st.success("✅ System loaded")
except Exception:
    st.error("❌ Failed to load system")
    st.code(traceback.format_exc())
    st.stop()

# Show API server status outside @st.cache_resource so st.* calls are safe.
if is_api_server_up():
    st.sidebar.success("Mock ERP API: running")
else:
    st.sidebar.warning(
        "Mock ERP API is not reachable at http://localhost:8000. "
        "API queries will fail. Try reloading the page."
    )


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None


# ---------------------------------------------------------------------------
# Query input — always visible at the top of the interactive area
# ---------------------------------------------------------------------------

# Primary input: Streamlit chat_input (fixed footer bar)
query_chat = st.chat_input(
    "Ask an ERP question — or follow up on a previous answer (e.g. 'now filter that by Germany')"
)

# Fallback visible input bar (always on-screen, works in all Streamlit versions)
with st.form(key="query_form", clear_on_submit=True):
    st.markdown(
        """
        <style>
        .query-bar { display: flex; gap: 8px; margin-bottom: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        fallback_text = st.text_input(
            label="Your question",
            placeholder="e.g. Who are the top 5 customers by revenue?",
            label_visibility="collapsed",
        )
    with col_btn:
        send_clicked = st.form_submit_button("Send ➤", use_container_width=True, type="primary")

# Resolve which input to use (chat_input wins if both typed)
query: str | None = None
is_followup: bool = False

if query_chat and query_chat.strip():
    query = query_chat.strip()
    is_followup = True
elif send_clicked and fallback_text and fallback_text.strip():
    query = fallback_text.strip()
    is_followup = False

# ---------------------------------------------------------------------------
# Run the graph when a query is submitted
# ---------------------------------------------------------------------------

if query and query.strip():
    if not is_followup:
        # User is asking a brand new question from the main box. Clear prior context.
        st.session_state["conversation_history"] = []

    with st.spinner("Running agents..."):
        invoke_payload = {
            "query": query.strip(),
            "conversation_history": st.session_state["conversation_history"],
        }
        # When submitted via the follow-up chat input, force the intent so the
        # graph skips re-classification and answers purely from history.
        if is_followup:
            invoke_payload["intent"] = "FOLLOWUP_QUERY"

        raw = graph.invoke(invoke_payload)


    fr_tmp = raw.get("final_response") or {}
    answer_str = _answer_to_str(fr_tmp.get("answer"))
    st.session_state["conversation_history"].append(
        {"role": "user", "content": query.strip()}
    )
    st.session_state["conversation_history"].append(
        {"role": "assistant", "content": answer_str}
    )

    # Trim to rolling window
    max_entries = MAX_HISTORY_TURNS * 2
    if len(st.session_state["conversation_history"]) > max_entries:
        st.session_state["conversation_history"] = (
            st.session_state["conversation_history"][-max_entries:]
        )

    st.session_state["last_result"] = raw
    st.session_state["last_query"] = query.strip()
    st.session_state["ts"] = datetime.utcnow().isoformat()

    st.rerun()


# ---------------------------------------------------------------------------
# (Conversation history moved to sidebar)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Render the latest result
# ---------------------------------------------------------------------------

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
        blob = {
            "timestamp_utc": st.session_state.get("ts"),
            "query": st.session_state.get("last_query"),
            "conversation_history": st.session_state.get("conversation_history", []),
            "raw_result": raw,
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(blob, indent=2, default=json_serializer),
            file_name="run_result.json",
            mime="application/json",
            use_container_width=True,
        )

    render_answer(fr.get("answer"))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📄 Documents", "🗄️ Database", "✅ Validation", "🧭 Trace", "🧾 Raw JSON"]
    )

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
    st.info("Ask a question above to get started.")
