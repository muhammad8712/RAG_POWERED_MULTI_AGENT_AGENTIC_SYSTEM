"""
eval_dashboard.py
-----------------
Streamlit dashboard for visualizing ERP multi-agent system evaluation results.

Run:
    streamlit run evaluation/eval_dashboard.py

Expects either:
  - A scored JSONL file (output of score_eval.py)  ← preferred
  - A raw eval results JSONL (output of run_eval.py) ← will score on the fly
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# ── try importing plotly; fall back to matplotlib ────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY = True
except ImportError:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    PLOTLY = False

from evaluation.score_eval import (
    _load_jsonl,
    score_record,
    compute_summary,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCORE_COLORS = {
    "PASS":    "#2ecc71",
    "PARTIAL": "#f39c12",
    "FAIL":    "#e74c3c",
}

CATEGORY_ORDER   = ["DOCUMENT", "DATABASE", "COMPOSITE", "ADVERSARIAL"]
DIFFICULTY_ORDER = ["EASY", "MEDIUM", "HARD", "CRITICAL"]

GT_PATH = ROOT / "evaluation" / "ground_truth.json"
RESULTS_DIR = ROOT / "evaluation" / "results"

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_ground_truth() -> dict:
    if not GT_PATH.exists():
        return {}
    items = json.loads(GT_PATH.read_text(encoding="utf-8"))
    return {item["id"]: item for item in items}


@st.cache_data
def load_and_score(results_file: str) -> tuple[list[dict], dict]:
    """Load a results JSONL, score it, return (scored_records, summary)."""
    gt_by_id = load_ground_truth()
    records   = _load_jsonl(Path(results_file))

    scored = []
    for record in records:
        qid = record.get("id")
        if qid and qid in gt_by_id:
            scored.append(score_record(record, gt_by_id[qid], use_llm_judge=False))

    summary = compute_summary(scored)
    return scored, summary


def _find_result_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    files = sorted(
        list(RESULTS_DIR.glob("scored_*.jsonl")) +
        list(RESULTS_DIR.glob("eval_results_*.jsonl")),
        reverse=True,
    )
    return files


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _pie_chart(labels: list[str], values: list[int], title: str, colors: list[str]):
    if PLOTLY:
        fig = px.pie(
            names=labels,
            values=values,
            title=title,
            color=labels,
            color_discrete_map={l: c for l, c in zip(labels, colors)},
            hole=0.35,
        )
        fig.update_traces(textinfo="percent+value", textfont_size=14)
        fig.update_layout(
            title_font_size=16,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            margin=dict(t=50, b=30, l=0, r=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=140,
            wedgeprops=dict(width=0.6),
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)


def _bar_chart(df: pd.DataFrame, x: str, y_cols: list[str],
               title: str, color_map: dict):
    if PLOTLY:
        fig = go.Figure()
        for col in y_cols:
            fig.add_trace(go.Bar(
                name=col,
                x=df[x],
                y=df[col],
                marker_color=color_map.get(col, "#888"),
                text=df[col],
                textposition="inside",
            ))
        fig.update_layout(
            barmode="stack",
            title=title,
            title_font_size=16,
            xaxis_title=x.replace("_", " ").title(),
            yaxis_title="Number of Queries",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=60, b=40, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        bottom = [0] * len(df)
        for col in y_cols:
            vals = df[col].tolist()
            ax.bar(df[x], vals, bottom=bottom,
                   color=color_map.get(col, "#888"), label=col)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(x.replace("_", " ").title())
        ax.set_ylabel("Number of Queries")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)


def _gauge_metric(value: float, label: str):
    """Display a colored metric box based on score threshold."""
    if value >= 75:
        color = "#2ecc71"
        status = "Good"
    elif value >= 50:
        color = "#f39c12"
        status = "Fair"
    else:
        color = "#e74c3c"
        status = "Needs Work"

    st.markdown(
        f"""
        <div style="
            background: {color}22;
            border-left: 5px solid {color};
            border-radius: 8px;
            padding: 16px 20px;
            margin: 4px 0;
        ">
            <div style="font-size: 2rem; font-weight: 700; color: {color};">{value}%</div>
            <div style="font-size: 0.9rem; color: #666;">{label} — {status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def render_overview(summary: dict, scored: list[dict]):
    st.header("📊 Overall Performance")

    ov = summary.get("overall", {})
    total = summary.get("total_queries", 0)

    # Top KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries", total)
    c2.metric("✅ PASS", f"{ov.get('pass', 0)}  ({ov.get('pass_rate_pct', 0)}%)")
    c3.metric("⚠️ PARTIAL", f"{ov.get('partial', 0)}  ({ov.get('partial_rate_pct', 0)}%)")
    c4.metric("❌ FAIL", f"{ov.get('fail', 0)}  ({ov.get('fail_rate_pct', 0)}%)")

    st.divider()

    col_pie, col_gauge = st.columns([1, 1])

    with col_pie:
        _pie_chart(
            labels=["PASS", "PARTIAL", "FAIL"],
            values=[ov.get("pass", 0), ov.get("partial", 0), ov.get("fail", 0)],
            title="Overall Answer Quality Distribution",
            colors=[SCORE_COLORS["PASS"], SCORE_COLORS["PARTIAL"], SCORE_COLORS["FAIL"]],
        )

    with col_gauge:
        st.subheader("Score Summary")
        _gauge_metric(ov.get("pass_rate_pct", 0), "Pass Rate (exact correct)")
        _gauge_metric(ov.get("weighted_score_pct", 0), "Weighted Score (PASS=2, PARTIAL=1)")

        st.caption(
            "Weighted score formula: (PASS×2 + PARTIAL×1) / (Total×2) × 100"
        )


def render_by_category(summary: dict):
    st.header("📂 Performance by Category")

    cat_data = summary.get("by_category", {})
    if not cat_data:
        st.info("No category data available.")
        return

    # Build DataFrame in fixed order
    rows = []
    for cat in CATEGORY_ORDER:
        if cat in cat_data:
            d = cat_data[cat]
            rows.append({
                "Category": cat,
                "PASS": d["pass"],
                "PARTIAL": d["partial"],
                "FAIL": d["fail"],
                "Total": d["total"],
                "Pass Rate %": d["pass_rate_pct"],
                "Weighted %": d["weighted_score_pct"],
            })
    df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)

    with col1:
        _bar_chart(
            df, x="Category",
            y_cols=["PASS", "PARTIAL", "FAIL"],
            title="Queries by Category and Score",
            color_map=SCORE_COLORS,
        )

    with col2:
        # Pie chart per category using pass rate
        _pie_chart(
            labels=df["Category"].tolist(),
            values=df["Total"].tolist(),
            title="Query Distribution by Category",
            colors=["#3498db", "#9b59b6", "#1abc9c", "#e67e22"],
        )

    # Category pass rates
    st.subheader("Pass Rate by Category")
    cols = st.columns(len(rows))
    for i, row in enumerate(rows):
        with cols[i]:
            color = (
                "#2ecc71" if row["Pass Rate %"] >= 75
                else "#f39c12" if row["Pass Rate %"] >= 50
                else "#e74c3c"
            )
            st.markdown(
                f"""
                <div style="text-align:center; padding:12px; border-radius:8px;
                            background:{color}22; border:2px solid {color};">
                    <div style="font-size:1.5rem; font-weight:700; color:{color};">
                        {row['Pass Rate %']}%
                    </div>
                    <div style="font-size:0.8rem; color:#555;">{row['Category']}</div>
                    <div style="font-size:0.75rem; color:#888;">n={row['Total']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.dataframe(df.set_index("Category"), use_container_width=True)


def render_by_difficulty(summary: dict):
    st.header("🎯 Performance by Difficulty")

    diff_data = summary.get("by_difficulty", {})
    if not diff_data:
        st.info("No difficulty data available.")
        return

    rows = []
    for diff in DIFFICULTY_ORDER:
        if diff in diff_data:
            d = diff_data[diff]
            rows.append({
                "Difficulty": diff,
                "PASS": d["pass"],
                "PARTIAL": d["partial"],
                "FAIL": d["fail"],
                "Total": d["total"],
                "Pass Rate %": d["pass_rate_pct"],
            })
    df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)

    with col1:
        _bar_chart(
            df, x="Difficulty",
            y_cols=["PASS", "PARTIAL", "FAIL"],
            title="Score Distribution by Difficulty",
            color_map=SCORE_COLORS,
        )

    with col2:
        _pie_chart(
            labels=df["Difficulty"].tolist(),
            values=df["Total"].tolist(),
            title="Query Count by Difficulty",
            colors=["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"],
        )

    st.dataframe(df.set_index("Difficulty"), use_container_width=True)


def render_adversarial(scored: list[dict]):
    st.header("🔒 Security & Adversarial Analysis")

    adv = [s for s in scored if s.get("category") == "ADVERSARIAL"]
    if not adv:
        st.info("No adversarial queries found in results.")
        return

    passed = sum(1 for s in adv if s["label"] == "PASS")
    failed = sum(1 for s in adv if s["label"] == "FAIL")
    partial = sum(1 for s in adv if s["label"] == "PARTIAL")

    col1, col2 = st.columns([1, 1])

    with col1:
        _pie_chart(
            labels=["Correctly Blocked", "Partially Blocked", "DANGEROUS — Not Blocked"],
            values=[passed, partial, failed],
            title="Security Test Results",
            colors=["#2ecc71", "#f39c12", "#e74c3c"],
        )

    with col2:
        st.subheader("Security Score")
        security_pct = round(passed / len(adv) * 100, 1)
        _gauge_metric(security_pct, "Adversarial Block Rate")

        if failed > 0:
            st.error(
                f"⚠️ {failed} dangerous request(s) were NOT properly blocked! "
                "Review these immediately."
            )
        else:
            st.success("✅ All adversarial/security tests passed!")

    st.subheader("Adversarial Test Details")
    df = pd.DataFrame(adv)[["id", "query", "label", "reason", "validation_status"]]
    st.dataframe(df, use_container_width=True)


def render_detail_table(scored: list[dict]):
    st.header("📋 Full Results Table")

    df = pd.DataFrame(scored)
    if df.empty:
        st.info("No scored results.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        cat_filter = st.multiselect(
            "Filter by Category",
            options=df["category"].dropna().unique().tolist(),
            default=[],
        )
    with col2:
        score_filter = st.multiselect(
            "Filter by Score",
            options=["PASS", "PARTIAL", "FAIL"],
            default=[],
        )
    with col3:
        diff_filter = st.multiselect(
            "Filter by Difficulty",
            options=df["difficulty"].dropna().unique().tolist(),
            default=[],
        )

    filtered = df.copy()
    if cat_filter:
        filtered = filtered[filtered["category"].isin(cat_filter)]
    if score_filter:
        filtered = filtered[filtered["label"].isin(score_filter)]
    if diff_filter:
        filtered = filtered[filtered["difficulty"].isin(diff_filter)]

    st.caption(f"Showing {len(filtered)} of {len(df)} results")

    # Colour-code the label column
    def highlight_label(val):
        color_map = {
            "PASS":    "background-color: #d5f5e3; color: #1e8449",
            "PARTIAL": "background-color: #fdebd0; color: #935116",
            "FAIL":    "background-color: #fadbd8; color: #922b21",
        }
        return color_map.get(val, "")

    display_cols = [
        "id", "category", "difficulty", "query",
        "label", "score", "reason", "sql_generated",
        "db_row_count", "db_error", "validation_status",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    styled = (
        filtered[display_cols]
        .style
        .map(highlight_label, subset=["label"])
    )
    st.dataframe(styled, use_container_width=True, height=500)

    # Download
    csv = filtered[display_cols].to_csv(index=False)
    st.download_button(
        "📥 Download Filtered Results as CSV",
        data=csv,
        file_name="eval_results_filtered.csv",
        mime="text/csv",
    )


def render_failure_analysis(scored: list[dict]):
    st.header("🔍 Failure Analysis")

    fails = [s for s in scored if s["label"] == "FAIL"]
    partials = [s for s in scored if s["label"] == "PARTIAL"]

    if not fails and not partials:
        st.success("No failures or partial answers — perfect score!")
        return

    tab_fail, tab_partial = st.tabs([
        f"❌ FAILs ({len(fails)})",
        f"⚠️ PARTIALs ({len(partials)})",
    ])

    with tab_fail:
        if fails:
            # Group by category
            by_cat: dict[str, list] = {}
            for f in fails:
                c = f.get("category", "UNKNOWN")
                by_cat.setdefault(c, []).append(f)

            for cat, items in sorted(by_cat.items()):
                with st.expander(f"{cat} — {len(items)} failure(s)"):
                    for item in items:
                        st.markdown(f"**{item['id']}** | *{item.get('difficulty', '')}*")
                        st.markdown(f"**Query:** {item['query']}")
                        st.markdown(f"**Reason:** {item['reason']}")
                        if item.get("db_error"):
                            st.error(f"DB Error: {item['db_error']}")
                        if item.get("answer_preview"):
                            st.caption(f"Answer preview: {item['answer_preview']}")
                        st.divider()
        else:
            st.success("No failures!")

    with tab_partial:
        if partials:
            for item in partials:
                with st.expander(f"{item['id']} — {item.get('category')} / {item.get('difficulty')}"):
                    st.markdown(f"**Query:** {item['query']}")
                    st.markdown(f"**Reason:** {item['reason']}")
                    if item.get("answer_preview"):
                        st.caption(f"Answer preview: {item['answer_preview']}")
        else:
            st.success("No partial answers!")


def render_scoring_method_breakdown(scored: list[dict]):
    st.header("⚙️ Scoring Method Breakdown")

    method_stats: dict[str, dict] = {}
    for s in scored:
        method = s.get("scoring_method", "unknown")
        if method not in method_stats:
            method_stats[method] = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "total": 0}
        method_stats[method][s["label"]] += 1
        method_stats[method]["total"] += 1

    rows = []
    for method, stats in method_stats.items():
        t = stats["total"]
        rows.append({
            "Method": method,
            "Total": t,
            "PASS": stats["PASS"],
            "PARTIAL": stats["PARTIAL"],
            "FAIL": stats["FAIL"],
            "Pass Rate %": round(stats["PASS"] / t * 100, 1) if t else 0,
        })
    df = pd.DataFrame(rows)

    if not df.empty:
        _bar_chart(
            df, x="Method",
            y_cols=["PASS", "PARTIAL", "FAIL"],
            title="Results by Scoring Method",
            color_map=SCORE_COLORS,
        )
        st.dataframe(df.set_index("Method"), use_container_width=True)
        st.caption(
            "**keyword** = fast substring match | "
            "**structural** = DB rows/SQL check | "
            "**adversarial** = refusal check | "
            "**composite** = DB + doc combined | "
            "**llm_judge** = keyword fallback (LLM judge not enabled)"
        )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ERP Eval Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 ERP Multi-Agent System — Evaluation Dashboard")
st.caption("Scoring evaluation results against ground truth across 100 queries")

# ── Sidebar: file selector ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    result_files = _find_result_files()

    if not result_files:
        st.warning(
            "No result files found in:\n"
            f"`{RESULTS_DIR}`\n\n"
            "Run `python evaluation/run_eval.py` first, then "
            "`python evaluation/score_eval.py --results <file>` to score."
        )
        uploaded = st.file_uploader("Or upload a results JSONL", type=["jsonl", "json"])
        if uploaded:
            tmp = Path("/tmp/uploaded_results.jsonl")
            tmp.write_bytes(uploaded.read())
            selected_path = str(tmp)
        else:
            st.stop()
    else:
        options = [str(p) for p in result_files]
        selected_path = st.selectbox(
            "Select results file",
            options=options,
            format_func=lambda p: Path(p).name,
        )
        st.caption(f"Found {len(result_files)} result file(s)")

    st.divider()
    st.subheader("Ground Truth")
    gt_status = "✅ Loaded" if GT_PATH.exists() else "❌ Missing"
    st.markdown(f"Status: **{gt_status}**")
    st.caption(f"`{GT_PATH.relative_to(ROOT)}`")

    if not GT_PATH.exists():
        st.error("Ground truth file not found. Cannot score results.")
        st.stop()

# ── Load and score ───────────────────────────────────────────────────────────
with st.spinner("Loading and scoring results..."):
    try:
        scored, summary = load_and_score(selected_path)
    except Exception as e:
        st.error(f"Failed to load/score results: {e}")
        st.exception(e)
        st.stop()

if not scored:
    st.warning("No scoreable records found. Check that your results file contains query IDs matching ground_truth.json.")
    st.stop()

st.success(f"✅ Scored {len(scored)} queries from `{Path(selected_path).name}`")

# ── Navigation tabs ──────────────────────────────────────────────────────────
tab_overview, tab_category, tab_difficulty, tab_security, tab_methods, tab_failures, tab_detail = st.tabs([
    "📊 Overview",
    "📂 By Category",
    "🎯 By Difficulty",
    "🔒 Security",
    "⚙️ Methods",
    "🔍 Failures",
    "📋 Full Table",
])

with tab_overview:
    render_overview(summary, scored)

with tab_category:
    render_by_category(summary)

with tab_difficulty:
    render_by_difficulty(summary)

with tab_security:
    render_adversarial(scored)

with tab_methods:
    render_scoring_method_breakdown(scored)

with tab_failures:
    render_failure_analysis(scored)

with tab_detail:
    render_detail_table(scored)
