import importlib.util
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import AgentState, Route, Scenario, initial_state


@pytest.fixture(autouse=True)
def enable_interrupt() -> None:
    """Ensure real interrupt is active for all tests in this module."""
    with patch.dict(os.environ, {"LANGGRAPH_INTERRUPT": "true"}):
        yield


def test_risky_flow_interrupts_before_tool() -> None:
    """Verify that a risky action pauses at approval node before tool execution."""
    checkpointer = MemorySaver()
    scenario = Scenario(id="hitl_01", query="Refund customer $500", expected_route=Route.RISKY)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": "thread-hitl-01"}}

    tool_call_count = 0

    def counting_tool(s: AgentState) -> dict[str, Any]:
        nonlocal tool_call_count
        tool_call_count += 1
        return {"attempt": s.get("attempt", 0) + 1, "tool_results": [{"status": "ok"}]}

    with (
        patch("langgraph_agent_lab.graph.classify_node") as mock_classify,
        patch("langgraph_agent_lab.graph.tool_node", side_effect=counting_tool),
    ):
        mock_classify.side_effect = lambda s: {
            "route": "risky",
            "risk_level": "high",
            "events": [{"node": "classify", "event_type": "done", "message": "risky"}],
        }
        graph = build_graph(checkpointer=checkpointer)
        result = graph.invoke(state, config=config)

    # 1. Verify result contains interrupt payload
    assert "__interrupt__" in result
    interrupts = result["__interrupt__"]
    assert len(interrupts) > 0
    payload = interrupts[0].value
    assert payload["type"] == "approval_required"
    assert "Refund customer $500" in payload["proposed_action"]

    # 2. Verify state snapshot before resume
    snapshot = graph.get_state(config)
    assert snapshot is not None
    events = [e.get("node") for e in snapshot.values.get("events", [])]
    assert "intake" in events
    assert "classify" in events
    assert "risky_action" in events
    assert "tool" not in events
    assert "finalize" not in events
    assert snapshot.values["attempt"] == 0
    assert tool_call_count == 0


def test_hitl_resume_approved_executes_tool() -> None:
    """Verify resuming with approval proceeds to tool and executes exactly once."""
    checkpointer = MemorySaver()
    scenario = Scenario(id="hitl_02", query="Delete account user-99", expected_route=Route.RISKY)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": "thread-hitl-02"}}

    tool_call_count = 0

    def counting_tool(s: AgentState) -> dict[str, Any]:
        nonlocal tool_call_count
        tool_call_count += 1
        return {
            "attempt": s.get("attempt", 0) + 1,
            "tool_results": [{"status": "success", "data": "deleted"}],
            "events": [{"node": "tool", "event_type": "done", "message": "deleted"}],
        }

    with (
        patch("langgraph_agent_lab.graph.classify_node") as mock_classify,
        patch("langgraph_agent_lab.graph.tool_node", side_effect=counting_tool),
    ):
        mock_classify.side_effect = lambda s: {
            "route": "risky",
            "risk_level": "high",
            "events": [{"node": "classify", "event_type": "done", "message": "risky"}],
        }
        graph = build_graph(checkpointer=checkpointer)
        # Initial call pauses
        graph.invoke(state, config=config)
        assert tool_call_count == 0

        # Resume with approval
        resume_cmd = Command(
            resume={
                "approved": True,
                "reviewer": "admin-alice",
                "comment": "Confirmed identity",
            }
        )
        final_result = graph.invoke(resume_cmd, config=config)

    events = [e.get("node") for e in final_result.get("events", [])]
    assert "approval" in events
    assert "tool" in events
    assert "evaluate" in events
    assert "answer" in events
    assert "finalize" in events

    assert tool_call_count == 1
    assert final_result["status"] == "completed"
    assert final_result["attempt"] == 1
    assert final_result["approval"]["approved"] is True
    assert final_result["approval"]["reviewer"] == "admin-alice"


def test_hitl_resume_rejected_routes_to_clarify() -> None:
    """Verify resuming with rejection routes to clarify and terminates without tool."""
    checkpointer = MemorySaver()
    scenario = Scenario(id="hitl_03", query="Wipe server logs", expected_route=Route.RISKY)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": "thread-hitl-03"}}

    tool_call_count = 0

    def counting_tool(s: AgentState) -> dict[str, Any]:
        nonlocal tool_call_count
        tool_call_count += 1
        return {"attempt": s.get("attempt", 0) + 1, "tool_results": []}

    with (
        patch("langgraph_agent_lab.graph.classify_node") as mock_classify,
        patch("langgraph_agent_lab.graph.tool_node", side_effect=counting_tool),
    ):
        mock_classify.side_effect = lambda s: {
            "route": "risky",
            "risk_level": "high",
            "events": [{"node": "classify", "event_type": "done", "message": "risky"}],
        }
        graph = build_graph(checkpointer=checkpointer)
        graph.invoke(state, config=config)
        assert tool_call_count == 0

        # Resume with rejection
        resume_cmd = Command(
            resume={
                "approved": False,
                "reviewer": "security-officer",
                "comment": "Action prohibited",
            }
        )
        final_result = graph.invoke(resume_cmd, config=config)

    events = [e.get("node") for e in final_result.get("events", [])]
    assert "approval" in events
    assert "clarify" in events
    assert "finalize" in events
    assert "tool" not in events

    assert tool_call_count == 0
    assert final_result["status"] == "clarification_required"
    assert final_result["attempt"] == 0
    assert final_result["approval"]["approved"] is False
    assert final_result["approval"]["reviewer"] == "security-officer"


def test_hitl_history_continuity_across_interrupt() -> None:
    """Verify state history checkpoint chain grows properly across interrupt and resume."""
    checkpointer = MemorySaver()
    scenario = Scenario(id="hitl_04", query="Transfer $10k", expected_route=Route.RISKY)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": "thread-hitl-04"}}

    with patch("langgraph_agent_lab.graph.classify_node") as mock_classify:
        mock_classify.side_effect = lambda s: {
            "route": "risky",
            "events": [{"node": "classify", "event_type": "done", "message": "risky"}],
        }
        graph = build_graph(checkpointer=checkpointer)
        graph.invoke(state, config=config)

    history_before = list(graph.get_state_history(config))
    assert len(history_before) >= 2

    # Resume
    graph.invoke(Command(resume={"approved": True}), config=config)
    history_after = list(graph.get_state_history(config))
    assert len(history_after) > len(history_before)


def test_hitl_wrong_thread_does_not_resume_original() -> None:
    """Verify resume to a different thread does not advance original paused state."""
    checkpointer = MemorySaver()
    scenario = Scenario(id="hitl_orig", query="Original risky task", expected_route=Route.RISKY)
    orig_config = {"configurable": {"thread_id": "hitl-original-thread"}}
    wrong_config = {"configurable": {"thread_id": "hitl-wrong-thread"}}

    tool_call_count = 0

    def counting_tool(s: AgentState) -> dict[str, Any]:
        nonlocal tool_call_count
        tool_call_count += 1
        return {"attempt": s.get("attempt", 0) + 1, "tool_results": []}

    with (
        patch("langgraph_agent_lab.graph.classify_node") as mock_classify,
        patch("langgraph_agent_lab.graph.tool_node", side_effect=counting_tool),
    ):
        mock_classify.side_effect = lambda s: {
            "route": "risky",
            "events": [{"node": "classify", "event_type": "done", "message": "risky"}],
        }
        graph = build_graph(checkpointer=checkpointer)
        # 1. Start original risky workflow -> interrupts
        res = graph.invoke(initial_state(scenario), config=orig_config)
        assert "__interrupt__" in res
        assert tool_call_count == 0

        # Snapshot before wrong resume
        orig_before = graph.get_state(orig_config)
        assert orig_before.values["attempt"] == 0

        # 2. Send resume command to wrong thread
        try:
            graph.invoke(Command(resume={"approved": True}), config=wrong_config)
        except Exception:
            pass  # Some versions raise on non-existent thread resume

        # 3. Verify original thread remains intact and was NOT advanced
        orig_after = graph.get_state(orig_config)
        orig_events = [e.get("node") for e in orig_after.values.get("events", [])]
        assert "tool" not in orig_events
        assert "finalize" not in orig_events
        assert orig_after.values["attempt"] == 0
        assert tool_call_count == 0



@pytest.mark.skipif(
    importlib.util.find_spec("langgraph.checkpoint.sqlite") is None,
    reason="langgraph-checkpoint-sqlite not installed",
)
def test_hitl_sqlite_durable_interrupt(tmp_path: Path) -> None:
    """Verify that interrupt state survives durable SQLite storage across instances."""
    db_path = tmp_path / "hitl.sqlite"
    checkpointer = build_checkpointer(kind="sqlite", database_url=str(db_path))

    scenario = Scenario(id="hitl_sql", query="Reset production db", expected_route=Route.RISKY)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": "thread-hitl-sql"}}

    with patch("langgraph_agent_lab.graph.classify_node") as mock_classify:
        mock_classify.side_effect = lambda s: {
            "route": "risky",
            "events": [{"node": "classify", "event_type": "done", "message": "risky"}],
        }
        graph = build_graph(checkpointer=checkpointer)
        res = graph.invoke(state, config=config)
        assert "__interrupt__" in res

    # Resume on a new graph instance with same checkpointer
    graph_resumed = build_graph(checkpointer=checkpointer)
    final_res = graph_resumed.invoke(Command(resume={"approved": True}), config=config)
    assert final_res["status"] == "completed"
    assert final_res["attempt"] == 1

