"""CLI for the lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml  # type: ignore[import-untyped]
from langchain_core.runnables import RunnableConfig

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)



@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    for scenario in scenarios:
        state = initial_state(scenario)
        if scenario.requires_approval and not state.get("approval"):
            if "rejected" in scenario.tags:
                state["approval"] = {
                    "approved": False,
                    "reviewer": "scenario-reviewer",
                    "comment": "Rejected by policy",
                }
            else:
                state["approval"] = {
                    "approved": True,
                    "reviewer": "scenario-reviewer",
                    "comment": "Approved by reviewer",
                }
        run_config: RunnableConfig = {"configurable": {"thread_id": state["thread_id"]}}
        final_state = graph.invoke(state, config=run_config)
        metric_item = metric_from_state(
            final_state,
            scenario.expected_route.value,
            scenario.requires_approval,
        )
        metrics.append(metric_item)
    report = summarize_metrics(metrics)
    write_metrics(report, output)
    if cfg.get("report_path"):
        try:
            from .report import write_report

            write_report(report, cfg["report_path"])
        except NotImplementedError:
            pass
    typer.echo(f"Wrote metrics to {output}")


@app.command("validate-metrics")
def validate_metrics(
    metrics: Annotated[Path, typer.Option("--metrics")],
    min_success_rate: float = 0.8,
) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    if report.success_rate < min_success_rate:
        raise typer.BadParameter(
            f"Success rate {report.success_rate:.2%} below threshold {min_success_rate:.2%}"
        )
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")



if __name__ == "__main__":
    app()
