"""Integration tests for Hydra, CLI serialization, and the sample notebook."""

import json
from pathlib import Path
import subprocess

from hydra import compose, initialize_config_module
from nbclient import NotebookClient
import nbformat
import pytest

import quantum_mutant_planner.cli as cli
from quantum_mutant_planner.cli import report_payload, settings_from_config
from quantum_mutant_planner.examples import build_scenario
from quantum_mutant_planner.planner import plan_mutant_executions


@pytest.mark.parametrize("enabled", [False, True])
def test_hydra_composition_and_report_payload(enabled: bool) -> None:
    """Packaged Hydra configuration should create typed settings and JSON records."""

    with initialize_config_module(config_module="quantum_mutant_planner.conf", version_base=None):
        config = compose(config_name="config", overrides=[
            "planner.enable_equivalence=false",
            f"planner.enable_discrimination_budget={str(enabled).lower()}",
        ])
    settings = settings_from_config(config)
    records = plan_mutant_executions(build_scenario(config.scenario.name), settings)
    payload = report_payload("ghz", settings, records)
    assert payload["schema_version"] == 2
    assert payload["qiskit_version"] == "2.5.1"
    assert len(payload["records"]) == 8
    assert settings.enable_discrimination_budget is enabled
    assert payload["discrimination_budget_enabled"] is enabled


def test_older_config_defaults_to_ordinary_mutation_testing() -> None:
    """Omitting the new option remains compatible and disables discrimination."""

    with initialize_config_module(config_module="quantum_mutant_planner.conf", version_base=None):
        config = compose(config_name="config")
    del config.planner.enable_discrimination_budget
    assert settings_from_config(config).enable_discrimination_budget is False


def test_unknown_scenario_is_rejected() -> None:
    """The CLI scenario registry should reject undeclared names."""

    try:
        build_scenario("missing")
    except ValueError as error:
        assert "unknown scenario" in str(error)
    else:
        raise AssertionError("unknown scenario was accepted")


def test_console_cli_emits_json_and_writes_requested_output(tmp_path: Path) -> None:
    """The installed console command should emit and optionally persist the same report."""

    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            "quantum-mutant-plan",
            "scenario=ghz",
            "planner.enable_equivalence=false",
            f"output.path={output}",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload == json.loads(output.read_text(encoding="utf-8"))
    assert "running scenario ghz" in completed.stderr


def test_hydra_entry_body_and_main_wrapper(tmp_path: Path, monkeypatch, capsys) -> None:
    """The decorated entry body and console wrapper should both remain callable."""

    with initialize_config_module(config_module="quantum_mutant_planner.conf", version_base=None):
        config = compose(
            config_name="config",
            overrides=["planner.enable_equivalence=false", f"output.path={tmp_path / 'direct.json'}"],
        )
    cli._hydra_main.__wrapped__(config)
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario"] == "ghz"
    called: list[bool] = []
    monkeypatch.setattr(cli, "_hydra_main", lambda: called.append(True))
    cli.main()
    assert called == [True]


def test_sample_notebook_executes_headlessly() -> None:
    """The delivered notebook should execute without errors in the project environment."""

    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "notebook" / "algorithms_1_2_demo.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    executed = NotebookClient(notebook, timeout=120, kernel_name="python3").execute(cwd=root)
    assert any(cell.get("execution_count") for cell in executed.cells if cell.cell_type == "code")
