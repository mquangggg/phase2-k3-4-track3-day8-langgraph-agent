import json
from operator import add

from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import AgentState, Route, Scenario, initial_state, make_event


def test_scenario_validation():
    scenario = Scenario(id="x", query="hello", expected_route=Route.SIMPLE)
    state = initial_state(scenario)
    assert state["thread_id"] == "thread-x"
    assert state["attempt"] == 0
    assert state["events"] == []


def test_initial_state_has_required_fields():
    """Verify initial_state includes all fields needed by the graph."""
    scenario = Scenario(id="test", query="test query", expected_route=Route.SIMPLE)
    state = initial_state(scenario)
    required_keys = [
        "ticket_id",
        "user_input",
        "query",
        "route",
        "attempt",
        "max_attempts",
        "status",
        "messages",
        "tool_results",
        "errors",
        "events",
    ]
    for key in required_keys:
        assert key in state, f"Missing key '{key}' in initial_state"


def test_state_serialization():
    """Verify that state is serializable to JSON (no un-serializable objects)."""
    scenario = Scenario(id="ser_test", query="testing serialization", expected_route=Route.TOOL)
    state = initial_state(scenario)
    state["events"].append(make_event("intake", "completed", "done"))
    state["tool_results"].append({"status": "ok", "data": 123})
    
    dumped = json.dumps(state)
    loaded = json.loads(dumped)
    assert loaded["ticket_id"] == "ser_test"
    assert len(loaded["events"]) == 1
    assert len(loaded["tool_results"]) == 1


def test_state_reducers_append_behavior():
    """Verify that list fields correctly behave with operator.add."""
    base_events = [make_event("intake", "start", "init")]
    new_events = [make_event("classify", "done", "simple")]
    combined_events = add(base_events, new_events)
    assert len(combined_events) == 2
    assert combined_events[0]["node"] == "intake"
    assert combined_events[1]["node"] == "classify"


def test_load_scenarios():
    scenarios = load_scenarios("data/sample/scenarios.jsonl")
    assert len(scenarios) >= 6
    assert {item.expected_route for item in scenarios} >= {Route.SIMPLE, Route.TOOL, Route.RISKY}

