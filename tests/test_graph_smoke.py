from typing import Any
from unittest.mock import MagicMock, patch

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.state import AgentState, Route, Scenario, initial_state


def test_graph_node_registration() -> None:
    """Verify that graph registers all 11 business nodes."""
    graph = build_graph()
    expected_nodes = {
        "intake",
        "classify",
        "answer",
        "tool",
        "evaluate",
        "clarify",
        "risky_action",
        "approval",
        "retry",
        "dead_letter",
        "finalize",
    }
    graph_nodes = set(graph.get_graph().nodes.keys())
    for node in expected_nodes:
        assert node in graph_nodes, f"Missing node {node} in graph"


def test_graph_path_simple() -> None:
    scenario = Scenario(
        id="test_simple",
        query="How do I reset password?",
        expected_route=Route.SIMPLE,
    )
    state = initial_state(scenario)

    with patch("langgraph_agent_lab.graph.classify_node") as mock_classify:
        mock_classify.side_effect = lambda s: {
            "route": "simple",
            "events": [{"node": "classify", "event_type": "done", "message": "simple"}],
        }
        graph = build_graph()
        result = graph.invoke(state, config={"configurable": {"thread_id": "t1"}})
        mock_classify.assert_called()

    events = [e.get("node") for e in result.get("events", [])]
    assert "intake" in events
    assert "classify" in events
    assert "answer" in events
    assert "finalize" in events
    assert "tool" not in events
    assert result["status"] == "completed"
    assert result["attempt"] == 0


def test_graph_path_missing_info() -> None:
    scenario = Scenario(id="test_missing", query="Help", expected_route=Route.MISSING_INFO)
    state = initial_state(scenario)

    with patch("langgraph_agent_lab.graph.classify_node") as mock_classify:
        mock_classify.side_effect = lambda s: {
            "route": "missing_info",
            "events": [{"node": "classify", "event_type": "done", "message": "missing_info"}],
        }
        graph = build_graph()
        result = graph.invoke(state, config={"configurable": {"thread_id": "t2"}})
        mock_classify.assert_called()

    events = [e.get("node") for e in result.get("events", [])]
    assert "clarify" in events
    assert "finalize" in events
    assert "tool" not in events
    assert result["status"] == "clarification_required"


def test_graph_path_tool_success() -> None:
    scenario = Scenario(id="test_tool", query="Lookup order 123", expected_route=Route.TOOL)
    state = initial_state(scenario)

    with patch("langgraph_agent_lab.graph.classify_node") as mock_classify:
        mock_classify.side_effect = lambda s: {
            "route": "tool",
            "events": [{"node": "classify", "event_type": "done", "message": "tool"}],
        }
        graph = build_graph()
        result = graph.invoke(state, config={"configurable": {"thread_id": "t3"}})
        mock_classify.assert_called()

    events = [e.get("node") for e in result.get("events", [])]
    assert "tool" in events
    assert "evaluate" in events
    assert "answer" in events
    assert "finalize" in events
    assert result["attempt"] == 1


def test_graph_path_transient_failure_and_retry() -> None:
    scenario = Scenario(id="test_transient", query="Timeout failure", expected_route=Route.ERROR)
    state = initial_state(scenario)

    with patch("langgraph_agent_lab.graph.classify_node") as mock_classify:
        mock_classify.side_effect = lambda s: {
            "route": "error",
            "events": [{"node": "classify", "event_type": "done", "message": "error"}],
        }
        graph = build_graph()
        result = graph.invoke(state, config={"configurable": {"thread_id": "t4"}})
        mock_classify.assert_called()

    events = [e.get("node") for e in result.get("events", [])]
    assert "retry" in events
    assert result["attempt"] == 2
    assert result["status"] == "completed"


def test_graph_bounded_retry_exhaustion_dead_letter() -> None:
    """Verify retry loop is bounded and terminates at dead_letter without 4th attempt."""
    scenario = Scenario(
        id="test_dl",
        query="Permanent error",
        expected_route=Route.ERROR,
        max_attempts=3,
    )
    state = initial_state(scenario)

    call_count = 0

    def failing_tool(s: AgentState) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        curr = s.get("attempt", 0) + 1
        return {
            "attempt": curr,
            "tool_results": [{"status": "error", "message": "Database permanently down"}],
            "events": [{"node": "tool", "event_type": "fail", "message": f"attempt {curr}"}],
        }

    with (
        patch("langgraph_agent_lab.graph.classify_node") as mock_classify,
        patch("langgraph_agent_lab.graph.tool_node", side_effect=failing_tool),
    ):
        mock_classify.side_effect = lambda s: {
            "route": "error",
            "events": [{"node": "classify", "event_type": "done", "message": "error"}],
        }
        graph = build_graph()
        result = graph.invoke(state, config={"configurable": {"thread_id": "t5"}})
        mock_classify.assert_called()

    events = [e.get("node") for e in result.get("events", [])]
    assert "dead_letter" in events
    assert "finalize" in events
    assert result["attempt"] == 3
    assert result["status"] == "dead_letter"
    assert call_count == 3


def test_graph_path_risky_approved() -> None:
    scenario = Scenario(id="test_risky", query="Refund $100", expected_route=Route.RISKY)
    state = initial_state(scenario)

    mock_approval = MagicMock()
    mock_approval.side_effect = lambda s: {
        "approval": {"approved": True, "comment": "Action approved"},
        "events": [{"node": "approval", "event_type": "approve", "message": "approved"}],
    }

    with (
        patch("langgraph_agent_lab.graph.classify_node") as mock_classify,
        patch("langgraph_agent_lab.graph.approval_node", mock_approval),
    ):
        mock_classify.side_effect = lambda s: {
            "route": "risky",
            "risk_level": "high",
            "events": [{"node": "classify", "event_type": "done", "message": "risky"}],
        }
        graph = build_graph()
        result = graph.invoke(state, config={"configurable": {"thread_id": "t6"}})
        mock_classify.assert_called()
        mock_approval.assert_called()

    events = [e.get("node") for e in result.get("events", [])]
    assert "risky_action" in events
    assert "approval" in events
    assert "tool" in events
    assert result["attempt"] == 1



def test_graph_path_risky_rejected() -> None:
    scenario = Scenario(id="test_rejected", query="Delete database", expected_route=Route.RISKY)
    state = initial_state(scenario)

    mock_approval = MagicMock()
    mock_approval.side_effect = lambda s: {
        "approval": {"approved": False, "comment": "Action rejected"},
        "events": [{"node": "approval", "event_type": "reject", "message": "rejected"}],
    }

    with (
        patch("langgraph_agent_lab.graph.classify_node") as mock_classify,
        patch("langgraph_agent_lab.graph.approval_node", mock_approval),
    ):
        mock_classify.side_effect = lambda s: {
            "route": "risky",
            "events": [{"node": "classify", "event_type": "done", "message": "risky"}],
        }
        graph = build_graph()
        result = graph.invoke(state, config={"configurable": {"thread_id": "t7"}})
        mock_classify.assert_called()
        mock_approval.assert_called()

    events = [e.get("node") for e in result.get("events", [])]
    assert "approval" in events
    assert "clarify" in events
    assert "finalize" in events
    assert "tool" not in events
    assert result["attempt"] == 0



