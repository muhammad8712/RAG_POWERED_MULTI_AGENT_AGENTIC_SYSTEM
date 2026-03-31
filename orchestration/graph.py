from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents.api_agent import APIAgent
from agents.database_agent import DatabaseAgent
from agents.document_agent import DocumentAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.reasoning_agent import ReasoningAgent
from orchestration.intent_classifier import classify_intent
from orchestration.orchestrator_agent import OrchestratorAgent
from orchestration.validator_agent import CorrectiveValidationAgent


# A single turn in the conversation history
# role: "user" | "assistant"
ConversationTurn = dict  # {"role": str, "content": str}


class GraphState(TypedDict, total=False):
    query: str
    intent: str

    # ── NEW: conversation history ──────────────────────────────────────────
    # Ordered list of prior turns passed in from the caller.
    # Each entry: {"role": "user"|"assistant", "content": "<text>"}
    # The graph never mutates this list — it is read-only context for agents.
    conversation_history: list[ConversationTurn]

    plan: list[str]
    initial_plan: list[str]
    attempts: int
    max_iters: int

    document_output: dict[str, Any]
    database_output: dict[str, Any]
    api_output: dict[str, Any]
    reasoning_output: dict[str, Any]
    validation_output: dict[str, Any]
    evidence_history: list[dict[str, Any]]  # accumulated evidence snapshots


    trace_steps: list[str]
    final_response: dict[str, Any]


def build_graph(
    document_agent: DocumentAgent,
    database_agent: DatabaseAgent,
    api_agent: APIAgent,
    reasoning_agent: ReasoningAgent,
    explainability_agent: ExplainabilityAgent,
    orchestrator_agent: OrchestratorAgent,
    validator_agent: CorrectiveValidationAgent,
):
    graph = StateGraph(GraphState)

    def _next_step(state: GraphState) -> str:
        plan = state.get("plan") or []
        return plan[0] if plan else "explainability"

    def _pop_plan(state: GraphState) -> GraphState:
        plan = state.get("plan") or []
        return {"plan": plan[1:]} if plan else {"plan": []}

    def _trace(state: GraphState, step: str) -> GraphState:
        steps = list(state.get("trace_steps") or [])
        steps.append(step)
        return {"trace_steps": steps}

    def _history(state: GraphState) -> list[ConversationTurn]:
        """Safe accessor — returns [] if history is absent."""
        return list(state.get("conversation_history") or [])

    # ── graph nodes ──────────────────────────────────────────────────────────

    def intent_node(state: GraphState) -> GraphState:
        # If intent was pre-set by the caller (e.g. FOLLOWUP_QUERY from UI), respect it
        if state.get("intent"):
            return {}
        return {"intent": classify_intent(state["query"], _history(state))}

    def orchestrator_node(state: GraphState) -> GraphState:
        plan_obj = orchestrator_agent.run(state["query"], state.get("intent"))
        steps = list(plan_obj["steps"])

        return {
            "plan": steps.copy(),
            "initial_plan": steps.copy(),
            "attempts": 0,
            "max_iters": int(plan_obj.get("max_iters", 2)),
            "trace_steps": [],
        }

    def database_node(state: GraphState) -> GraphState:
        return {
            **_trace(state, "database"),
            "database_output": database_agent.run(
                state["query"],
                conversation_history=_history(state),
            ),
        }

    def document_node(state: GraphState) -> GraphState:
        return {
            **_trace(state, "document"),
            "document_output": document_agent.run(
                state["query"],
                conversation_history=_history(state),
            ),
        }

    def api_node(state: GraphState) -> GraphState:
        return {
            **_trace(state, "api"),
            "api_output": api_agent.run(state["query"]),
        }

    def reasoning_node(state: GraphState) -> GraphState:
        doc_out = state.get("document_output") or {}
        db_out  = state.get("database_output") or {}

        doc_text = (
            doc_out.get("retrieved_context")
            or doc_out.get("answer")
            or "No document context provided."
        )

        # Build prior reasoning context from evidence history
        evidence_history = list(state.get("evidence_history") or [])
        prior_reasoning = None
        if evidence_history:
            prev = evidence_history[-1]
            prior_reasoning = {
                "attempt":        prev.get("attempt", 0),
                "final_decision": prev.get("reasoning_output", {}).get("final_decision", ""),
                "reasoning":      prev.get("reasoning_output", {}).get("reasoning", ""),
                "validation_issues": prev.get("validation_issues", []),
            }

        agent_outputs = {
            "database_output": db_out,
            "document_output": doc_out,
            "document_text":   doc_text,
            "prior_reasoning": prior_reasoning,
        }

        reasoning_result = reasoning_agent.run(
            state["query"],
            agent_outputs,
            conversation_history=_history(state),
        )

        # Snapshot current evidence for future attempts
        snapshot = {
            "attempt":           len(evidence_history) + 1,
            "document_output":   doc_out,
            "database_output":   db_out,
            "reasoning_output":  reasoning_result,
            "validation_issues": (state.get("validation_output") or {}).get("issues", []),
        }
        evidence_history.append(snapshot)

        return {
            **_trace(state, "reasoning"),
            "reasoning_output": reasoning_result,
            "evidence_history": evidence_history,
        }


    def validate_node(state: GraphState) -> GraphState:
        validation = validator_agent.run(state["query"], state)

        attempts = int(state.get("attempts", 0))
        max_iters = int(state.get("max_iters", 2))

        update: GraphState = {
            **_trace(state, "validate"),
            "validation_output": validation,
        }

        status = validation.get("status")

        if status == "NEEDS_MORE_INFO" and attempts < max_iters:
            tool_map = {
                "document": "document",
                "database": "database",
                "api": "api",
            }

            corrective_steps: list[str] = []
            for action in (validation.get("next_actions") or []):
                tool = tool_map.get(action.get("tool"))
                if tool and tool not in corrective_steps:
                    corrective_steps.append(tool)

            if corrective_steps:
                update["plan"] = corrective_steps + ["reasoning", "validate", "explainability"]
                update["attempts"] = attempts + 1
                return update

        plan = state.get("plan") or []
        if plan and plan[0] == "validate":
            update["plan"] = plan[1:]

        return update

    def explainability_node(state: GraphState) -> GraphState:
        payload = {
            "intent": state.get("intent"),
            "document_output": state.get("document_output"),
            "database_output": state.get("database_output"),
            "api_output": state.get("api_output"),
            "reasoning_output": state.get("reasoning_output"),
            "validation_output": state.get("validation_output"),
            "evidence_history": state.get("evidence_history") or [],
            "execution_trace": {
                "intent": state.get("intent"),
                "initial_plan": state.get("initial_plan") or [],
                "final_plan_state": state.get("plan") or [],
                "attempts": state.get("attempts"),
                "max_iters": state.get("max_iters"),
                "steps_executed": state.get("trace_steps") or [],
                "db_used": state.get("database_output") is not None,
                "doc_used": state.get("document_output") is not None,
                "api_used": state.get("api_output") is not None,
                "reasoning_used": state.get("reasoning_output") is not None,
                "validated": state.get("validation_output") is not None,
                # expose history depth for explainability
                "history_turns": len(state.get("conversation_history") or []),
            },
        }

        return {
            **_trace(state, "explainability"),
            "final_response": explainability_agent.run(payload),
        }

    # ── register nodes ───────────────────────────────────────────────────────
    graph.add_node("intent", intent_node)
    graph.add_node("orchestrator", orchestrator_node)

    graph.add_node("database", database_node)
    graph.add_node("document", document_node)
    graph.add_node("api", api_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("validate", validate_node)
    graph.add_node("explainability", explainability_node)

    graph.set_entry_point("intent")
    graph.add_edge("intent", "orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        _next_step,
        {
            "database": "database",
            "document": "document",
            "api": "api",
            "reasoning": "reasoning",
            "validate": "validate",
            "explainability": "explainability",
        },
    )

    for node_name in ("database", "document", "api", "reasoning"):
        pop_name = f"{node_name}_pop"
        graph.add_node(pop_name, _pop_plan)
        graph.add_edge(node_name, pop_name)
        graph.add_conditional_edges(
            pop_name,
            _next_step,
            {
                "database": "database",
                "document": "document",
                "api": "api",
                "reasoning": "reasoning",
                "validate": "validate",
                "explainability": "explainability",
            },
        )

    graph.add_conditional_edges(
        "validate",
        _next_step,
        {
            "database": "database",
            "document": "document",
            "api": "api",
            "reasoning": "reasoning",
            "validate": "validate",
            "explainability": "explainability",
        },
    )

    graph.add_edge("explainability", END)
    return graph.compile()