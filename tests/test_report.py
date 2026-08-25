from pathlib import Path

from langgraph_agent_lab.metrics import MetricsReport, ScenarioMetric
from langgraph_agent_lab.report import render_report, write_report


def test_render_and_write_report_pass(tmp_path: Path) -> None:
    items = [
        ScenarioMetric(
            scenario_id=f"S0{i}",
            success=True,
            expected_route="simple",
            actual_route="simple",
            nodes_visited=4,
            retry_count=0,
            interrupt_count=0,
        )
        for i in range(1, 7)
    ]
    report = MetricsReport(
        total_scenarios=6,
        success_rate=1.0,
        avg_nodes_visited=4.0,
        total_retries=0,
        total_interrupts=0,
        resume_success=True,
        scenario_metrics=items,
    )

    content = render_report(report)
    assert "Vũ Minh Quang" in content
    assert "2A202601515" in content
    assert "```mermaid" in content
    assert "| **Total Scenarios** | 6 | >= 6 | PASS |" in content
    assert "| **Success Rate** | 100.00% | >= 80.0% | PASS |" in content
    assert "| **Average Nodes Visited** | 4.00 | > 0 | PASS |" in content
    assert "| **Total Retries** | 0 | Observed | INFO |" in content
    assert "| **Total Interrupts** | 0 | Observed | INFO |" in content

    out_file = tmp_path / "reports" / "lab_report.md"
    write_report(report, out_file)
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == content


def test_render_report_dynamic_fail() -> None:
    # 1. Total scenarios < 6 and success_rate < 0.8
    items = [
        ScenarioMetric(
            scenario_id="S01",
            success=False,
            expected_route="simple",
            actual_route="error",
            nodes_visited=2,
            retry_count=1,
            interrupt_count=0,
        )
    ]
    report = MetricsReport(
        total_scenarios=1,
        success_rate=0.0,
        avg_nodes_visited=2.0,
        total_retries=1,
        total_interrupts=0,
        resume_success=False,
        scenario_metrics=items,
    )

    content = render_report(report)
    assert "| **Total Scenarios** | 1 | >= 6 | FAIL |" in content
    assert "| **Success Rate** | 0.00% | >= 80.0% | FAIL |" in content
    assert "| **Average Nodes Visited** | 2.00 | > 0 | PASS |" in content
