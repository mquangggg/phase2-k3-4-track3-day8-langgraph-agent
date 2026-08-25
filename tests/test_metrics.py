from pathlib import Path

import pytest

from langgraph_agent_lab.metrics import (
    MetricsReport,
    ScenarioMetric,
    metric_from_state,
    summarize_metrics,
    write_metrics,
)
from langgraph_agent_lab.state import make_event


def test_metric_from_state_success() -> None:
    state = {
        "scenario_id": "S01",
        "route": "simple",
        "final_answer": "Your password has been reset.",
        "status": "completed",
        "events": [
            make_event("intake", "completed", "ok"),
            make_event("classify", "completed", "simple"),
            make_event("answer", "completed", "ok"),
            make_event("finalize", "completed", "done"),
        ],
        "errors": [],
        "approval": None,
    }
    m = metric_from_state(state, "simple", False)
    assert m.success is True
    assert m.expected_route == "simple"
    assert m.actual_route == "simple"
    assert m.nodes_visited == 4
    assert m.retry_count == 0
    assert m.interrupt_count == 0


def test_metric_from_state_route_mismatch() -> None:
    state = {
        "scenario_id": "S02",
        "route": "error",
        "final_answer": "Failed",
        "status": "dead_letter",
        "events": [make_event("intake", "completed", "ok")],
        "errors": ["Timeout error"],
        "approval": None,
    }
    m = metric_from_state(state, "simple", False)
    assert m.success is False
    assert m.expected_route == "simple"
    assert m.actual_route == "error"


def test_metric_from_state_approval_requirement() -> None:
    # 1. Approval required but missing -> success is False
    state_no_appr = {
        "scenario_id": "S04",
        "route": "risky",
        "final_answer": "Refund processed",
        "status": "completed",
        "events": [make_event("risky_action", "prepared", "refund")],
        "approval": None,
    }
    m_no_appr = metric_from_state(state_no_appr, "risky", True)
    assert m_no_appr.success is False
    assert m_no_appr.approval_observed is False

    # 2. Approval required and present -> success is True
    state_appr = {
        "scenario_id": "S04",
        "route": "risky",
        "final_answer": "Refund processed",
        "status": "completed",
        "events": [make_event("approval", "approved", "ok")],
        "approval": {"approved": True, "reviewer": "admin"},
    }
    m_appr = metric_from_state(state_appr, "risky", True)
    assert m_appr.success is True
    assert m_appr.approval_observed is True


def test_summarize_metrics_calculation() -> None:
    m1 = ScenarioMetric(
        scenario_id="1",
        success=True,
        expected_route="simple",
        actual_route="simple",
        nodes_visited=4,
        retry_count=0,
        interrupt_count=0,
    )
    m2 = ScenarioMetric(
        scenario_id="2",
        success=False,
        expected_route="tool",
        actual_route="error",
        nodes_visited=6,
        retry_count=2,
        interrupt_count=0,
    )
    m3 = ScenarioMetric(
        scenario_id="3",
        success=True,
        expected_route="risky",
        actual_route="risky",
        nodes_visited=5,
        retry_count=0,
        interrupt_count=1,
    )

    report = summarize_metrics([m1, m2, m3])
    assert report.total_scenarios == 3
    assert pytest.approx(report.success_rate, 0.01) == 2 / 3
    assert pytest.approx(report.avg_nodes_visited, 0.01) == 5.0
    assert report.total_retries == 2
    assert report.total_interrupts == 1


def test_summarize_metrics_empty() -> None:
    with pytest.raises(ValueError, match="No scenario metrics"):
        summarize_metrics([])


def test_write_and_validate_metrics(tmp_path: Path) -> None:
    out_file = tmp_path / "metrics" / "test_report.json"
    m1 = ScenarioMetric(
        scenario_id="1",
        success=True,
        expected_route="simple",
        actual_route="simple",
        nodes_visited=3,
    )
    report = summarize_metrics([m1])
    write_metrics(report, out_file)
    assert out_file.exists()

    loaded = MetricsReport.model_validate_json(out_file.read_text(encoding="utf-8"))
    assert loaded.total_scenarios == 1
    assert loaded.success_rate == 1.0
