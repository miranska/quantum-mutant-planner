"""Hydra command-line entry point for registered planning scenarios."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
import qiskit

from quantum_mutant_planner.examples import build_scenario
from quantum_mutant_planner.models import (
    PilotSettings,
    PlannerSettings,
    RankingObjective,
    TranspilerSettings,
)
from quantum_mutant_planner.planner import plan_mutant_executions

LOGGER = logging.getLogger(__name__)


def settings_from_config(config: DictConfig) -> PlannerSettings:
    """Convert Hydra configuration into typed planner settings.

    :param config: Composed Hydra configuration.
    :return: Validated typed planner settings.
    """

    transpiler = TranspilerSettings(
        optimization_level=int(config.transpiler.optimization_level),
        seed=int(config.transpiler.seed),
        approximation_degree=float(config.transpiler.approximation_degree),
        basis_gates=tuple(str(name) for name in config.transpiler.basis_gates),
    )
    pilot = PilotSettings(
        target_error=float(config.pilot.target_error),
        failure_probability=float(config.pilot.failure_probability),
        max_shots=int(config.pilot.max_shots),
        seed=int(config.pilot.seed),
    )
    return PlannerSettings(
        numerical_tolerance=float(config.planner.numerical_tolerance),
        exact_state_max_qubits=int(config.planner.exact_state_max_qubits),
        exact_simulation_max_qubits=int(config.planner.exact_simulation_max_qubits),
        spectral_max_qubits=int(config.planner.spectral_max_qubits),
        max_pauli_terms=int(config.planner.max_pauli_terms),
        ranking_objective=RankingObjective(str(config.planner.ranking_objective)),
        enable_equivalence=bool(config.planner.enable_equivalence),
        enable_pilot=bool(config.planner.enable_pilot),
        transpiler=transpiler,
        pilot=pilot,
        symbolic_max_qubits=int(config.planner.symbolic_max_qubits),
        enable_discrimination_budget=bool(config.planner.get("enable_discrimination_budget", False)),
    )


def report_payload(scenario: str, settings: PlannerSettings, records: list[Any]) -> dict[str, Any]:
    """Create the versioned JSON result envelope.

    :param scenario: Registered scenario name.
    :param settings: Settings used by the planner.
    :param records: Planning records exposing ``to_dict``.
    :return: JSON-compatible report payload.
    """

    return {
        "schema_version": 2,
        "algorithm": "test-conditional-mutant-execution-planning",
        "scenario": scenario,
        "qiskit_version": qiskit.__version__,
        "numerical_tolerance": settings.numerical_tolerance,
        "pilot_run_failure_probability": settings.pilot.failure_probability,
        "discrimination_budget_enabled": settings.enable_discrimination_budget,
        "records": [record.to_dict() for record in records],
    }


@hydra.main(version_base=None, config_path="pkg://quantum_mutant_planner.conf", config_name="config")
def _hydra_main(config: DictConfig) -> None:
    """Compose configuration, run a scenario, and emit JSON.

    :param config: Hydra-composed application configuration.
    :return: None.
    """

    scenario = str(config.scenario.name)
    settings = settings_from_config(config)
    problem = build_scenario(scenario)
    LOGGER.info("running scenario %s with Qiskit %s", scenario, qiskit.__version__)
    records = plan_mutant_executions(problem, settings)
    rendered = json.dumps(report_payload(scenario, settings, records), indent=2, sort_keys=True)
    if config.output.path is not None:
        output_path = Path(str(config.output.path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        LOGGER.info("wrote planning report to %s", output_path)
    print(rendered)


def main() -> None:
    """Run the Hydra-decorated console entry point.

    :return: None.
    """

    _hydra_main()


if __name__ == "__main__":
    main()
