from pathlib import Path

from typer.testing import CliRunner

from langgraph_agent_lab.cli import app
from langgraph_agent_lab.metrics import MetricsReport, ScenarioMetric, write_metrics

runner = CliRunner()



def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run-scenarios" in result.output
    assert "validate-metrics" in result.output


def test_cli_validate_metrics_valid(tmp_path: Path) -> None:
    metrics_file = tmp_path / "valid_metrics.json"
    items = [
        ScenarioMetric(
            scenario_id=f"S0{i}",
            success=True,
            expected_route="simple",
            actual_route="simple",
            nodes_visited=4,
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
    write_metrics(report, metrics_file)

    result = runner.invoke(app, ["validate-metrics", "--metrics", str(metrics_file)])
    assert result.exit_code == 0
    assert "Metrics valid" in result.output


def test_cli_validate_metrics_too_few_scenarios(tmp_path: Path) -> None:
    metrics_file = tmp_path / "invalid_metrics.json"
    items = [
        ScenarioMetric(
            scenario_id="S01",
            success=True,
            expected_route="simple",
            actual_route="simple",
            nodes_visited=4,
        )
    ]
    report = MetricsReport(
        total_scenarios=1,
        success_rate=1.0,
        avg_nodes_visited=4.0,
        total_retries=0,
        total_interrupts=0,
        resume_success=True,
        scenario_metrics=items,
    )
    write_metrics(report, metrics_file)

    result = runner.invoke(app, ["validate-metrics", "--metrics", str(metrics_file)])
    assert result.exit_code != 0
    assert "Expected at least 6 scenarios" in result.output


def test_cli_validate_metrics_low_success_rate(tmp_path: Path) -> None:
    metrics_file = tmp_path / "low_metrics.json"
    items = [
        ScenarioMetric(
            scenario_id=f"S0{i}",
            success=(i <= 3),  # 3/6 = 50% < 80%
            expected_route="simple",
            actual_route="simple",
            nodes_visited=4,
        )
        for i in range(1, 7)
    ]
    report = MetricsReport(
        total_scenarios=6,
        success_rate=0.5,
        avg_nodes_visited=4.0,
        total_retries=0,
        total_interrupts=0,
        resume_success=True,
        scenario_metrics=items,
    )
    write_metrics(report, metrics_file)

    result = runner.invoke(app, ["validate-metrics", "--metrics", str(metrics_file)])
    assert result.exit_code != 0
    assert "below threshold" in result.output


def test_cli_run_scenarios(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    output_file = tmp_path / "output_metrics.json"
    config_file.write_text(
        "scenarios_path: data/sample/scenarios.jsonl\ncheckpointer: memory\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["run-scenarios", "--config", str(config_file), "--output", str(output_file)],
    )
    assert result.exit_code == 0
    assert "Wrote metrics" in result.output
    assert output_file.exists()

    loaded = MetricsReport.model_validate_json(output_file.read_text(encoding="utf-8"))
    assert loaded.total_scenarios >= 6
    assert loaded.success_rate >= 0.0

