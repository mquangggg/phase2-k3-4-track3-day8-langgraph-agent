from langgraph_agent_lab.nodes import (
    answer_node,
    approval_node,
    ask_clarification_node,
    classify_node,
    dead_letter_node,
    evaluate_node,
    finalize_node,
    intake_node,
    retry_or_fallback_node,
    risky_action_node,
    tool_node,
)
from langgraph_agent_lab.state import AgentState, Route, Scenario, initial_state


def test_intake_node() -> None:
    scenario = Scenario(id="s1", query="  Need help with order 123  ", expected_route=Route.TOOL)
    state = initial_state(scenario)
    update = intake_node(state)
    assert update["query"] == "Need help with order 123"
    assert update["attempt"] == 0
    assert update["status"] == "running"
    assert len(update["events"]) == 1


def test_intake_node_preserves_custom_max_attempts() -> None:
    scenario = Scenario(
        id="s_custom",
        query="Custom retry query",
        expected_route=Route.TOOL,
        max_attempts=5,
    )
    state = initial_state(scenario)
    assert state["max_attempts"] == 5
    update = intake_node(state)
    assert update["max_attempts"] == 5


def test_classify_node_priority() -> None:
    # Case A: Risky priority over lookup (Refund $100 for customer 123)
    s_risky: AgentState = {"query": "Refund $100 for customer 123"}
    update_risky = classify_node(s_risky)
    assert update_risky["route"] == "risky"
    assert update_risky["risk_level"] == "high"

    # Case B: Risky priority (Delete account user-123)
    s_delete: AgentState = {"query": "Delete account user-123"}
    update_delete = classify_node(s_delete)
    assert update_delete["route"] == "risky"

    # Case C: Tool priority (Check order status ORD-123)
    s_tool: AgentState = {"query": "Check order status ORD-123"}
    update_tool = classify_node(s_tool)
    assert update_tool["route"] == "tool"

    # Case D: Missing info (Vague query without ID)
    s_missing: AgentState = {"query": "Can you fix it?"}
    update_missing = classify_node(s_missing)
    assert update_missing["route"] == "missing_info"

    # Case E: Simple FAQ
    s_simple: AgentState = {"query": "How do I reset my password?"}
    update_simple = classify_node(s_simple)
    assert update_simple["route"] == "simple"


def test_tool_node_increments_attempt_and_simulates_failure() -> None:
    # Initial attempt = 0 -> tool call makes attempt = 1
    state: AgentState = {
        "route": "error",
        "attempt": 0,
        "query": "Timeout failure",
        "tool_results": [],
        "events": [],
    }
    update = tool_node(state)
    assert update["attempt"] == 1
    assert "error" in update["tool_results"][0]["status"]

    # Subsequent attempt = 1 -> tool call makes attempt = 2 and succeeds
    state["attempt"] = 1
    update2 = tool_node(state)
    assert update2["attempt"] == 2
    assert update2["tool_results"][0]["status"] == "success"


def test_evaluate_node() -> None:
    # Deterministic failure check
    state_err: AgentState = {"tool_results": [{"status": "error", "message": "ERROR: timeout"}]}
    assert evaluate_node(state_err)["evaluation_result"] == "needs_retry"

    # Deterministic success check
    state_ok: AgentState = {"tool_results": [{"status": "success", "data": "Order delivered"}]}
    assert evaluate_node(state_ok)["evaluation_result"] == "success"


def test_answer_node() -> None:
    state: AgentState = {
        "query": "Order 123",
        "route": "tool",
        "tool_results": [{"status": "success", "data": "Delivered"}],
    }
    update = answer_node(state)
    assert update["final_answer"] is not None
    assert update["status"] == "completed"


def test_ask_clarification_node() -> None:
    state: AgentState = {"query": "Fix it"}
    update = ask_clarification_node(state)
    assert update["status"] == "clarification_required"
    assert update["pending_question"] is not None
    assert update["final_answer"] == update["pending_question"]


def test_risky_action_and_approval_nodes() -> None:
    state: AgentState = {"query": "Refund 100$"}
    risky_update = risky_action_node(state)
    assert "Refund 100$" in risky_update["proposed_action"]

    # Fail-closed default when no approval decision is present
    approval_update = approval_node(state)
    assert approval_update["approval"]["approved"] is False

    # Explicit approval decision
    state["approval"] = {"approved": True, "reviewer": "admin"}
    approval_update_explicit = approval_node(state)
    assert approval_update_explicit["approval"]["approved"] is True



def test_retry_or_fallback_node_does_not_increment_attempt() -> None:
    state: AgentState = {"attempt": 2, "errors": [], "events": []}
    update = retry_or_fallback_node(state)
    assert "attempt" not in update  # Attempt is managed in tool_node
    assert len(update["errors"]) == 1


def test_dead_letter_and_finalize_nodes() -> None:
    state: AgentState = {"query": "Broken system", "attempt": 3, "status": "running"}
    dl_update = dead_letter_node(state)
    assert dl_update["status"] == "dead_letter"
    assert "3 attempts" in dl_update["final_answer"]

    # Finalize should preserve dead_letter status
    state["status"] = "dead_letter"
    fin_update = finalize_node(state)
    assert fin_update["status"] == "dead_letter"


