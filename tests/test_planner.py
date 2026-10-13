"""Tests for Algorithm 1 and Algorithm 2 branch behavior."""

from dataclasses import replace
import logging

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, UnitaryGate, XGate
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp, Statevector

from quantum_mutant_planner.analysis import is_clifford_relative, propagate_observable
from quantum_mutant_planner.examples import build_ghz_problem
from quantum_mutant_planner.models import (
    DecisionStatus,
    EffectEvidence,
    EvidenceType,
    ExecutionBudget,
    PilotSettings,
    PlannerSettings,
    PlanningProblem,
    PlanningRecord,
    QuantumTest,
    RankingObjective,
    ZeroProvenance,
)
from quantum_mutant_planner.mutations import (
    hermitian_perturbation_mutation,
    insertion_mutation,
    pauli_rotation_mutation,
    replacement_mutation,
)
from quantum_mutant_planner.pilot import PilotResult
from quantum_mutant_planner.planner import (
    _algebraic_pauli_zero,
    _rank_records,
    _status_from_evidence,
    evaluate_candidate_effect,
    plan_mutant_executions,
)


class ConstantPilot:
    """Deterministic pilot provider used to isolate planner decisions."""

    def __init__(self, value: float, error: float = 0.01) -> None:
        """Store a common estimate and declared error.

        :param value: Estimate returned for every target.
        :param error: Simultaneous per-target error.
        """

        self.value = value
        self.error = error

    def estimate(self, state, targets, settings) -> PilotResult:
        """Return deterministic estimates for all requested targets.

        :param state: Ignored mutation-site state.
        :param targets: Pauli targets to label.
        :param settings: Ignored configured pilot settings.
        :return: Deterministic pilot result.
        """

        return PilotResult(
            {target.to_label(): self.value for target in targets},
            self.error,
            shots=10,
            batches=1,
            batch_size=10,
        )


def one_qubit_problem(candidate, *, threshold: float = 0.1) -> PlanningProblem:
    """Build a one-qubit planning problem for one candidate."""

    circuit = QuantumCircuit(1)
    circuit.h(0)
    return PlanningProblem(
        circuit,
        QuantumTest(Statevector.from_label("0"), SparsePauliOp("Z"), threshold, 2.0),
        (candidate,),
        ExecutionBudget(10_000, 0.05),
    )


def test_ghz_walkthrough_matches_decisive_rules_and_ranking() -> None:
    """The eight manuscript candidates should follow the stated first rules."""

    records = plan_mutant_executions(build_ghz_problem(), PlannerSettings(enable_equivalence=False))
    statuses = {record.candidate_id: record.status for record in records}
    assert statuses == {
        "m1": DecisionStatus.SKIP_CERTIFIED_ZERO,
        "m2": DecisionStatus.SKIP_CERTIFIED_ZERO,
        "m3": DecisionStatus.SKIP_CERTIFIED_ZERO,
        "m4": DecisionStatus.SKIP_CERTIFIED_ZERO,
        "m5": DecisionStatus.SKIP_CERTIFIED_ZERO,
        "m6": DecisionStatus.SKIP_BELOW_TEST_THRESHOLD,
        "m7": DecisionStatus.EXECUTE_PREDICTED_ABOVE_THRESHOLD,
        "m8": DecisionStatus.EXECUTE_PREDICTED_ABOVE_THRESHOLD,
    }
    by_id = {record.candidate_id: record for record in records}
    assert by_id["m3"].evidence.provenance == "algebraic zero certificate"
    assert by_id["m4"].evidence.zero_provenance.value == "algebraic zero certificate"
    assert by_id["m5"].evidence.provenance.startswith("state-dependent cancellation certificate")
    assert by_id["m7"].evidence.gap == pytest.approx(1.0)
    assert by_id["m8"].evidence.gap == pytest.approx(1.0)
    assert [by_id["m7"].rank, by_id["m8"].rank] == [1, 2]


def test_pinned_transpiler_equivalence_precedes_analysis(caplog: pytest.LogCaptureFixture) -> None:
    """An identical replacement should receive semantic-equivalence provenance."""

    circuit = QuantumCircuit(1)
    circuit.x(0)
    candidate = replacement_mutation(circuit, "same", 0, XGate())
    problem = PlanningProblem(
        circuit,
        QuantumTest(Statevector.from_label("0"), SparsePauliOp("Z"), 0.1, 2.0),
        (candidate,),
        ExecutionBudget(100, 0.05),
    )
    with caplog.at_level(logging.INFO):
        record = plan_mutant_executions(problem)[0]
    assert record.status is DecisionStatus.SKIP_SEMANTICALLY_EQUIVALENT
    assert record.evidence.provenance == "pinned transpiler equality"
    assert "planning candidate same" in caplog.text


def test_spectral_branch_agrees_with_direct_simulation() -> None:
    """A declared tractable non-Clifford replacement should use an exact structured gap."""

    circuit = QuantumCircuit(1)
    circuit.h(0)
    candidate = replacement_mutation(circuit, "spectral", 0, RZGate(0.3), allow_spectral=True)
    problem = one_qubit_problem(candidate)
    propagation = propagate_observable(circuit, problem.test.observable, 1, 8, 1.0e-12)
    evidence = evaluate_candidate_effect(problem, candidate, propagation, PlannerSettings(enable_equivalence=False))
    assert evidence.evidence_type is EvidenceType.EXACT
    assert evidence.provenance == "structured gap"


def test_general_generator_returns_perturbative_and_operator_evidence() -> None:
    """State availability should separate perturbative sensitivity from its operator bound."""

    circuit = QuantumCircuit(1)
    circuit.h(0)
    candidate = hermitian_perturbation_mutation(circuit, "general", 0, Operator(Pauli("X")), 0.1)
    problem = one_qubit_problem(candidate)
    propagation = propagate_observable(circuit, problem.test.observable, 1, 8, 1.0e-12)
    state_evidence = evaluate_candidate_effect(
        problem,
        candidate,
        propagation,
        PlannerSettings(enable_equivalence=False),
    )
    assert state_evidence.evidence_type is EvidenceType.STATE_DEPENDENT_PERTURBATIVE
    assert state_evidence.score is not None
    assert state_evidence.remainder_bound is not None

    operator_evidence = evaluate_candidate_effect(
        problem,
        candidate,
        propagation,
        PlannerSettings(exact_state_max_qubits=0, enable_equivalence=False),
    )
    assert operator_evidence.evidence_type is EvidenceType.OPERATOR_PROXY
    assert operator_evidence.score == pytest.approx(2.0)

    sensitive = hermitian_perturbation_mutation(circuit, "sensitive", 0, Operator(Pauli("Y")), 0.01)
    sensitive_problem = one_qubit_problem(sensitive)
    sensitive_evidence = evaluate_candidate_effect(
        sensitive_problem,
        sensitive,
        propagation,
        PlannerSettings(enable_equivalence=False),
    )
    assert sensitive_evidence.evidence_type is EvidenceType.STATE_DEPENDENT_PERTURBATIVE
    assert not sensitive_evidence.warnings


def test_direct_simulation_and_unavailable_fallbacks() -> None:
    """Generic nonstructured mutations should simulate when tractable and otherwise remain unavailable."""

    circuit = QuantumCircuit(1)
    circuit.h(0)
    phase = UnitaryGate(np.diag([1.0, np.exp(0.3j)]))
    candidate = insertion_mutation(circuit, "generic", 1, phase, (0,), allow_spectral=False)
    problem = one_qubit_problem(candidate)
    propagation = propagate_observable(circuit, problem.test.observable, 1, 8, 1.0e-12)
    exact = evaluate_candidate_effect(
        problem,
        candidate,
        propagation,
        PlannerSettings(enable_equivalence=False),
    )
    assert exact.evidence_type is EvidenceType.EXACT
    assert exact.provenance == "simulated gap"
    unavailable = evaluate_candidate_effect(
        problem,
        candidate,
        propagation,
        PlannerSettings(exact_simulation_max_qubits=0, enable_equivalence=False),
    )
    assert unavailable.evidence_type is EvidenceType.UNAVAILABLE


def test_pilot_and_observable_side_branches() -> None:
    """A structured effect without exact state access should pilot or remain observable-side only."""

    circuit = QuantumCircuit(1)
    circuit.h(0)
    candidate = pauli_rotation_mutation(
        circuit,
        "pilot",
        1,
        Pauli("X"),
        0.0,
        np.pi / 2.0,
        qubits=(0,),
        insert=True,
    )
    problem = one_qubit_problem(candidate)
    propagation = propagate_observable(circuit, problem.test.observable, 1, 8, 1.0e-12)
    pilot_settings = PlannerSettings(
        exact_state_max_qubits=0,
        exact_simulation_max_qubits=1,
        max_pauli_terms=1,
        enable_equivalence=False,
        pilot=PilotSettings(target_error=0.1),
    )
    pilot = evaluate_candidate_effect(problem, candidate, propagation, pilot_settings, ConstantPilot(1.0))
    assert pilot.evidence_type is EvidenceType.PILOT_ESTIMATED
    assert pilot.interval is not None
    assert pilot.sampling_error > 0.0
    assert pilot.truncation_error > 0.0
    assert pilot.warnings == ("pilot used 10 shared prefix shots",)

    observable_only = evaluate_candidate_effect(
        problem,
        candidate,
        propagation,
        PlannerSettings(
            exact_state_max_qubits=0,
            exact_simulation_max_qubits=0,
            enable_equivalence=False,
            enable_pilot=False,
        ),
    )
    assert observable_only.evidence_type is EvidenceType.EXACT_OBSERVABLE_SIDE
    assert observable_only.provenance == "structured observable-difference"


def test_clifford_pilot_uses_generic_difference_and_truncation_rule() -> None:
    """A relative Clifford pilot should use the general two-epsilon allowance."""

    circuit = QuantumCircuit(1)
    circuit.h(0)
    candidate = insertion_mutation(circuit, "clifford", 1, XGate(), (0,))
    problem = one_qubit_problem(candidate)
    propagation = propagate_observable(circuit, problem.test.observable, 1, 8, 1.0e-12)
    propagation = replace(propagation, truncation_error=0.25)
    settings = PlannerSettings(
        exact_state_max_qubits=0,
        exact_simulation_max_qubits=1,
        symbolic_max_qubits=0,
        enable_equivalence=False,
    )
    evidence = evaluate_candidate_effect(problem, candidate, propagation, settings, ConstantPilot(1.0))
    assert evidence.evidence_type is EvidenceType.PILOT_ESTIMATED
    assert evidence.truncation_error == pytest.approx(0.5)
    assert not _algebraic_pauli_zero(candidate, propagation, 1)


def test_pilot_failure_falls_back_to_direct_simulation(caplog: pytest.LogCaptureFixture) -> None:
    """An unavailable pilot should be logged before the tractable simulation fallback."""

    circuit = QuantumCircuit(1)
    circuit.h(0)
    candidate = pauli_rotation_mutation(
        circuit,
        "pilot-fail",
        1,
        Pauli("X"),
        0.0,
        np.pi / 2.0,
        qubits=(0,),
        insert=True,
    )
    problem = one_qubit_problem(candidate)
    propagation = propagate_observable(circuit, problem.test.observable, 1, 8, 1.0e-12)

    class FailingPilot:
        """Pilot provider that represents a shot-budget failure."""

        def estimate(self, state, targets, settings):
            """Raise the provider's declared failure.

            :param state: Ignored state.
            :param targets: Ignored targets.
            :param settings: Ignored settings.
            :raises ValueError: Always raised to exercise fallback handling.
            """

            raise ValueError("shot ceiling")

    settings = PlannerSettings(exact_state_max_qubits=0, enable_equivalence=False)
    with caplog.at_level(logging.WARNING):
        evidence = evaluate_candidate_effect(problem, candidate, propagation, settings, FailingPilot())
    assert evidence.provenance == "simulated gap"
    assert "pilot unavailable" in caplog.text


def test_threshold_rule_covers_pilot_confidence_outcomes() -> None:
    """Pilot intervals should produce below, above, and unresolved decisions."""

    below = EffectEvidence(EvidenceType.PILOT_ESTIMATED, interval=(0.0, 0.09))
    above = EffectEvidence(EvidenceType.PILOT_ESTIMATED, interval=(0.1, 0.2))
    unresolved = EffectEvidence(EvidenceType.PILOT_ESTIMATED, interval=(0.05, 0.15))
    assert _status_from_evidence(below, 0.1) is DecisionStatus.SKIP_BELOW_TEST_THRESHOLD_CONFIDENCE
    assert _status_from_evidence(above, 0.1) is DecisionStatus.EXECUTE_PREDICTED_ABOVE_THRESHOLD
    assert _status_from_evidence(unresolved, 0.1) is DecisionStatus.EXECUTE_THRESHOLD_UNRESOLVED
    assert _status_from_evidence(EffectEvidence(EvidenceType.UNAVAILABLE), 0.1) is DecisionStatus.EXECUTE_THRESHOLD_UNRESOLVED


def test_ranking_is_within_evidence_and_supports_boundary_objective() -> None:
    """Ranking should not compare exact and pilot evidence directly."""

    records = [
        PlanningRecord(
            "a",
            DecisionStatus.EXECUTE_PREDICTED_ABOVE_THRESHOLD,
            EffectEvidence(EvidenceType.EXACT, gap=0.9, score=0.9),
        ),
        PlanningRecord(
            "b",
            DecisionStatus.EXECUTE_THRESHOLD_UNRESOLVED,
            EffectEvidence(EvidenceType.EXACT, gap=0.51, score=0.4),
        ),
        PlanningRecord(
            "c",
            DecisionStatus.EXECUTE_THRESHOLD_UNRESOLVED,
            EffectEvidence(EvidenceType.PILOT_ESTIMATED, gap=0.55, score=0.1),
        ),
        PlanningRecord("d", DecisionStatus.EXECUTE_THRESHOLD_UNRESOLVED, EffectEvidence(EvidenceType.UNAVAILABLE)),
    ]
    high = _rank_records(records, RankingObjective.HIGH_GAP, 0.5)
    assert [record.rank for record in high] == [1, 2, 1, None]
    boundary = _rank_records(records, RankingObjective.THRESHOLD_BOUNDARY, 0.5)
    assert [record.rank for record in boundary] == [2, 1, 1, None]


def test_planner_rejects_invalid_settings() -> None:
    """The planner should reject a nonpositive numerical tolerance."""

    with pytest.raises(ValueError, match="tolerance"):
        plan_mutant_executions(build_ghz_problem(), PlannerSettings(numerical_tolerance=0.0))


@pytest.mark.parametrize("eta", [-1e-9, 1e-9])
def test_near_clifford_recognition_keeps_the_true_relative_unitary(eta):
    """Numerical Clifford recognition must neither round the action nor certify zero."""
    circuit = QuantumCircuit(1)
    candidate = insertion_mutation(circuit, "near-clifford", 0, RZGate(eta), (0,), allow_spectral=False)
    problem = PlanningProblem(
        circuit, QuantumTest(Statevector.from_label("r"), SparsePauliOp("X"), .1, 2),
        (candidate,), ExecutionBudget(10000, .05),
    )
    assert is_clifford_relative(candidate)
    propagation = propagate_observable(circuit, problem.test.observable, 0, 8, 1e-12)
    # Invoke Algorithm 2 directly so the earlier scalar gates do not hide this branch.
    evidence = evaluate_candidate_effect(problem, candidate, propagation, PlannerSettings())
    assert evidence.evidence_type is EvidenceType.EXACT
    assert evidence.signed_shift == pytest.approx(np.sin(eta), rel=1e-7, abs=1e-18)
    assert evidence.gap == pytest.approx(abs(np.sin(eta)), rel=1e-7, abs=1e-18)
    assert evidence.zero_provenance is ZeroProvenance.NONE


@pytest.mark.parametrize("objective", list(RankingObjective))
def test_ranking_numerical_ties_preserve_population_order(objective):
    """Close scores or boundary distances tie without changing the evidence."""
    values = [.4, .4 + 1e-14, .8] if objective is RankingObjective.HIGH_GAP else [.6, .6 - 1e-14, .51]
    records = [
        PlanningRecord(name, DecisionStatus.EXECUTE_PREDICTED_ABOVE_THRESHOLD,
                       EffectEvidence(EvidenceType.EXACT, gap=value, score=value))
        for name, value in zip(("z", "a", "best"), values, strict=True)
    ]
    ranked = _rank_records(records, objective, .5)
    assert [record.rank for record in ranked] == [2, 3, 1]
    assert [record.evidence for record in ranked] == [record.evidence for record in records]


def test_ranking_ties_do_not_chain_across_distinct_scores():
    """Compare each tie to its group's best value, not its previous neighbor."""
    records = [
        PlanningRecord(str(index), DecisionStatus.EXECUTE_PREDICTED_ABOVE_THRESHOLD,
                       EffectEvidence(EvidenceType.EXACT, gap=value, score=value))
        for index, value in enumerate([.4, .4 + .75e-12, .4 + 1.5e-12])
    ]
    assert [record.rank for record in _rank_records(records, RankingObjective.HIGH_GAP, .1)] == [3, 1, 2]


def test_ranking_uses_relative_tolerance_and_handles_missing_gaps():
    """Large comparable proxy scores can tie; absent gaps sort last at the boundary."""
    records = [
        PlanningRecord(str(index), DecisionStatus.EXECUTE_THRESHOLD_UNRESOLVED,
                       EffectEvidence(EvidenceType.OPERATOR_PROXY, gap=gap, score=score))
        for index, (gap, score) in enumerate([(None, 1000.), (None, 1000. + 5e-10), (.6, 2000.)])
    ]
    for objective in RankingObjective:
        assert [record.rank for record in _rank_records(records, objective, .5)] == [2, 3, 1]


def test_planner_passes_configured_tolerance_to_ranking():
    """A tolerance below the GHZ roundoff difference keeps the distinct raw order."""
    records = plan_mutant_executions(
        build_ghz_problem(), PlannerSettings(enable_equivalence=False, numerical_tolerance=1e-18),
    )
    assert [record.rank for record in records[-2:]] == [2, 1]
