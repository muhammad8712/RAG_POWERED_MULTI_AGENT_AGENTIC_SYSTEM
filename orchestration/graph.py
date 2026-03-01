from __future__ import annotations

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


class GraphState(TypedDict, total=False):
    query: str
    intent: str

    plan: list[str]
    attempts: int
    max_iters: int

    document_output: dict[str, Any]
    database_output: dict[str, Any]
    api_output: dict[str, Any]
    reasoning_output: dict[str, Any]
    validation_output: dict[str, Any]

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

    def intent_node(state: GraphState) -> GraphState:
        return {"intent": classify_intent(state["query"])}

    def orchestrator_node(state: GraphState) -> GraphState:
        plan_obj = orchestrator_agent.run(state["query"], state.get("intent"))
        return {
            "plan": plan_obj["steps"],
            "attempts": 0,
            "max_iters": int(plan_obj.get("max_iters", 2)),
            "trace_steps": [],
        }

    def database_node(state: GraphState) -> GraphState:
        return {
            **_trace(state, "database"),
            "database_output": database_agent.run(state["query"]),
        }

    def document_node(state: GraphState) -> GraphState:
        return {
            **_trace(state, "document"),
            "document_output": document_agent.run(state["query"]),
        }

    def api_node(state: GraphState) -> GraphState:
        return {
            **_trace(state, "api"),
            "api_output": api_agent.run(state["query"]),
        }

    def reasoning_node(state: GraphState) -> GraphState:
        doc_out = state.get("document_output") or {}
        doc_text = (
            doc_out.get("retrieved_context")
            or doc_out.get("answer")
            or "No document context provided."
        )

        db_out = state.get("database_output") or {}
        db_result = db_out.get("result") or []

        agent_outputs = {
            "database_output": {"result": db_result},
            "document_text": doc_text,
        }

        return {
            **_trace(state, "reasoning"),
            "reasoning_output": reasoning_agent.run(state["query"], agent_outputs),
        }

    def validate_node(state: GraphState) -> GraphState:
        validation = validator_agent.run(state["query"], state)

        attempts = int(state.get("attempts", 0))
        max_iters = int(state.get("max_iters", 2))

        update: GraphState = {**_trace(state, "validate"), "validation_output": validation}

        if validation.get("status") == "NEEDS_MORE_INFO" and attempts < max_iters:
            tool_map = {"document": "document", "database": "database", "api": "api"}

            corrective_steps: list[str] = []
            for action in (validation.get("next_actions") or []):
                tool = tool_map.get(action.get("tool"))
                if tool and tool not in corrective_steps:
                    corrective_steps.append(tool)

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
            "execution_trace": {
                "intent": state.get("intent"),
                "initial_plan": state.get("plan"),
                "attempts": state.get("attempts"),
                "max_iters": state.get("max_iters"),
                "steps_executed": state.get("trace_steps") or [],
                "db_used": state.get("database_output") is not None,
                "doc_used": state.get("document_output") is not None,
                "api_used": state.get("api_output") is not None,
                "reasoning_used": state.get("reasoning_output") is not None,
                "validated": state.get("validation_output") is not None,
            },
        }

        return {
            **_trace(state, "explainability"),
            "final_response": explainability_agent.run(payload),
        }

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