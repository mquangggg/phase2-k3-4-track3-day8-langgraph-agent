import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


def test_build_checkpointer_none() -> None:
    assert build_checkpointer(kind="none") is None


def test_build_checkpointer_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown checkpointer kind"):
        build_checkpointer(kind="redis_unknown")


def test_build_checkpointer_postgres_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        build_checkpointer(kind="postgres")


def test_memory_checkpointer_state_and_history() -> None:
    checkpointer = build_checkpointer(kind="memory")
    assert checkpointer is not None
    graph = build_graph(checkpointer=checkpointer)

    scenario = Scenario(id="mem_01", query="Help with password", expected_route=Route.SIMPLE)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": "thread-memory-01"}}

    with patch("langgraph_agent_lab.graph.classify_node") as mock_classify:
        mock_classify.side_effect = lambda s: {
            "route": "simple",
            "events": [{"node": "classify", "event_type": "done", "message": "simple"}],
        }
        graph.invoke(state, config=config)

    # 1. State retrieval
    snapshot = graph.get_state(config)
    assert snapshot is not None
    assert snapshot.values["route"] == "simple"
    assert snapshot.values["status"] == "completed"

    # 2. State history
    history = list(graph.get_state_history(config))
    assert len(history) > 1


@pytest.mark.skipif(
    importlib.util.find_spec("langgraph.checkpoint.sqlite") is None,
    reason="langgraph-checkpoint-sqlite not installed",
)
def test_sqlite_checkpointer_real_execution(tmp_path: Path) -> None:
    db_file = tmp_path / "test_checkpoints.sqlite"
    checkpointer = build_checkpointer(kind="sqlite", database_url=str(db_file))
    assert checkpointer is not None
    assert db_file.exists()

    graph = build_graph(checkpointer=checkpointer)
    scenario = Scenario(id="sql_01", query="Lookup order 555", expected_route=Route.TOOL)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": "thread-sqlite-01"}}

    with patch("langgraph_agent_lab.graph.classify_node") as mock_classify:
        mock_classify.side_effect = lambda s: {
            "route": "tool",
            "events": [{"node": "classify", "event_type": "done", "message": "tool"}],
        }
        result = graph.invoke(state, config=config)

    assert result["attempt"] == 1
    assert result["status"] == "completed"

    # Verify recovery via get_state
    snapshot = graph.get_state(config)
    assert snapshot.values["ticket_id"] == "sql_01"
    assert snapshot.values["attempt"] == 1

    # Verify checkpoint history exists
    history = list(graph.get_state_history(config))
    assert len(history) > 1


@pytest.mark.skipif(
    importlib.util.find_spec("langgraph.checkpoint.sqlite") is None,
    reason="langgraph-checkpoint-sqlite not installed",
)
def test_sqlite_thread_isolation(tmp_path: Path) -> None:
    db_file = tmp_path / "isolated.sqlite"
    checkpointer = build_checkpointer(kind="sqlite", database_url=str(db_file))

    config_a = {"configurable": {"thread_id": "thread-A"}}
    config_b = {"configurable": {"thread_id": "thread-B"}}

    scenario_a = Scenario(id="A", query="User A query", expected_route=Route.SIMPLE)
    scenario_b = Scenario(id="B", query="User B query", expected_route=Route.MISSING_INFO)

    with patch("langgraph_agent_lab.graph.classify_node") as mock_classify:
        mock_classify.side_effect = lambda s: {
            "route": "simple" if s.get("ticket_id") == "A" else "missing_info",
            "events": [{"node": "classify", "event_type": "done", "message": "classified"}],
        }
        graph = build_graph(checkpointer=checkpointer)
        graph.invoke(initial_state(scenario_a), config=config_a)
        graph.invoke(initial_state(scenario_b), config=config_b)

    state_a = graph.get_state(config_a).values
    state_b = graph.get_state(config_b).values

    assert state_a["ticket_id"] == "A"
    assert state_a["status"] == "completed"

    assert state_b["ticket_id"] == "B"
    assert state_b["status"] == "clarification_required"



def test_sqlite_parent_dir_creation(tmp_path: Path) -> None:
    deep_path = tmp_path / "nested" / "folder" / "custom.sqlite"
    assert not deep_path.parent.exists()
    try:
        build_checkpointer(kind="sqlite", database_url=str(deep_path))
        assert deep_path.parent.exists()
    except RuntimeError:
        pass  # If package not installed in environment, ignore
