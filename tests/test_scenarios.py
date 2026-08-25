from pathlib import Path

import pytest

from langgraph_agent_lab.scenarios import load_scenarios


def test_load_sample_scenarios() -> None:

    path = Path("data/sample/scenarios.jsonl")
    if path.exists():
        scenarios = load_scenarios(path)
        assert len(scenarios) >= 6
        assert scenarios[0].id == "S01_simple"
        assert scenarios[0].expected_route.value == "simple"


def test_load_scenarios_too_few(tmp_path: Path) -> None:
    file_path = tmp_path / "few.jsonl"
    file_path.write_text(
        '{"id":"S01","query":"hello","expected_route":"simple"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="At least 6 scenarios"):
        load_scenarios(file_path)


def test_load_scenarios_invalid_json(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid.jsonl"
    file_path.write_text("invalid json content\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid scenario at line 1"):
        load_scenarios(file_path)
