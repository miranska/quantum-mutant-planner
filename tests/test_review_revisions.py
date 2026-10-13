"""Regression checks for bounded decisions and symbolic certificate safety."""

from dataclasses import replace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import DensityMatrix, Operator, Pauli, SparsePauliOp, Statevector, random_density_matrix

from quantum_mutant_planner.analysis import direct_signed_gap, propagate_observable
from quantum_mutant_planner.models import (
    DecisionStatus, EffectEvidence, EvidenceType, ExecutionBudget, PilotSettings,
    PlannerSettings, PlanningProblem, QuantumTest, ZeroProvenance,
)
from quantum_mutant_planner.mutations import hermitian_perturbation_mutation, insertion_mutation, pauli_rotation_mutation
from quantum_mutant_planner.pilot import PilotResult
from quantum_mutant_planner.planner import _status_from_evidence, evaluate_candidate_effect, plan_mutant_executions


def problem_for(circuit, candidate, *, observable=None, state=None, threshold=0.1):
    """Supply a simple test without changing the declared candidate circuit."""
    width = circuit.num_qubits
    observable = SparsePauliOp('I' * (width - 1) + 'Z') if observable is None else observable
    state = Statevector.from_int(0, 2**width) if state is None else state
    eigenvalues = np.linalg.eigvalsh(observable.to_matrix())
    return PlanningProblem(
        circuit, QuantumTest(state, observable, threshold, float(eigenvalues[-1] - eigenvalues[0])),
        (candidate,), ExecutionBudget(10_000, 0.05),
    )


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('eta', [0.01, -0.01, 2 * np.pi - 0.01, -2 * np.pi + 0.01])
def test_periodic_threshold_gate_preserves_signed_offset(eta, enabled):
    """All near-identity channel representatives have the same threshold ceiling."""
    circuit = QuantumCircuit(1)
    candidate = pauli_rotation_mutation(circuit, 'small', 0, Pauli('Y'), 0, eta, qubits=(0,), insert=True)
    problem = problem_for(circuit, candidate)
    record = plan_mutant_executions(problem, PlannerSettings(enable_equivalence=False, enable_discrimination_budget=enabled))[0]
    assert candidate.eta == eta
    assert record.status is DecisionStatus.SKIP_BELOW_TEST_THRESHOLD
    assert record.evidence.gap is None
    assert record.evidence.interval is None
    assert record.evidence.deterministic_interval[1] == pytest.approx(2 * abs(np.sin(eta / 2)))
    assert direct_signed_gap(problem, candidate, 1e-12)[1] <= record.evidence.deterministic_interval[1]


@pytest.mark.parametrize('eta', [0.01, -0.01, 2 * np.pi - 0.01, -2 * np.pi + 0.01])
def test_budget_floor_remains_a_separate_periodic_decision(eta):
    """A low test threshold can leave a small mutation blocked only by the budget."""
    circuit = QuantumCircuit(1)
    candidate = pauli_rotation_mutation(circuit, 'budget', 0, Pauli('Y'), 0, eta, qubits=(0,), insert=True)
    record = plan_mutant_executions(
        problem_for(circuit, candidate, threshold=1e-8),
        PlannerSettings(enable_equivalence=False, enable_discrimination_budget=True),
    )[0]
    assert record.status is DecisionStatus.SKIP_BELOW_BUDGET_FLOOR
    assert record.evidence.gap is None

    ordinary = plan_mutant_executions(
        problem_for(circuit, candidate, threshold=1e-8), PlannerSettings(enable_equivalence=False),
    )[0]
    assert ordinary.status is DecisionStatus.EXECUTE_PREDICTED_ABOVE_THRESHOLD
    assert ordinary.evidence.gap == pytest.approx(1 - np.cos(eta), abs=1e-12)


def test_non_pauli_axis_scalar_gate_is_global_phase_invariant():
    """An arbitrary one-qubit replacement supports the same local strength ceiling."""
    circuit = QuantumCircuit(1)
    axis = (Pauli('X').to_matrix() + Pauli('Z').to_matrix()) / np.sqrt(2)
    rotation = np.cos(.005) * np.eye(2) - 1j * np.sin(.005) * axis
    for phase in [0, .7, 2.8]:
        candidate = insertion_mutation(circuit, 'axis', 0, UnitaryGate(np.exp(1j * phase) * rotation), (0,))
        record = plan_mutant_executions(problem_for(circuit, candidate), PlannerSettings(enable_equivalence=False))[0]
        assert record.status is DecisionStatus.SKIP_BELOW_TEST_THRESHOLD
        assert record.evidence.deterministic_interval[1] == pytest.approx(2 * np.sin(.005))


@pytest.mark.parametrize('eta', [-.2, -.01, .01, .2])
def test_general_generator_interval_contains_direct_gap(eta):
    """A nonunitary Hermitian generator embeds correctly and bounds the exact effect."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    generator = SparsePauliOp.from_list([('X', .3), ('Y', .5), ('Z', .2)])
    candidate = hermitian_perturbation_mutation(circuit, 'general', 0, generator, eta)
    problem = problem_for(circuit, candidate)
    propagation = propagate_observable(circuit, problem.test.observable, 1, 16, 1e-12)
    evidence = evaluate_candidate_effect(problem, candidate, propagation, PlannerSettings())
    exact = direct_signed_gap(problem, candidate, 1e-12)[1]
    lower, upper = evidence.deterministic_interval
    assert lower <= exact <= upper
    assert evidence.evidence_type is EvidenceType.STATE_DEPENDENT_PERTURBATIVE
    assert evidence.gap is None and evidence.interval is None
    assert evidence.remainder_bound > 0


def test_zero_first_order_still_has_nonzero_second_order_effect():
    """Vanishing linear response is a bounded effect, never an algebraic zero."""
    circuit = QuantumCircuit(1)
    circuit.id(0)
    candidate = hermitian_perturbation_mutation(circuit, 'quadratic', 0, SparsePauliOp('X', coeffs=[.5]), .2)
    problem = problem_for(circuit, candidate, threshold=.01)
    propagation = propagate_observable(circuit, problem.test.observable, 1, 8, 1e-12)
    evidence = evaluate_candidate_effect(problem, candidate, propagation, PlannerSettings())
    gap = direct_signed_gap(problem, candidate, 1e-12)[1]
    assert gap == pytest.approx(1 - np.cos(.2))
    assert evidence.deterministic_interval == pytest.approx((0, .02))
    assert evidence.zero_provenance is ZeroProvenance.NONE
    assert _status_from_evidence(evidence, .01) is DecisionStatus.EXECUTE_THRESHOLD_UNRESOLVED
    assert _status_from_evidence(evidence, .03) is DecisionStatus.SKIP_BELOW_TEST_THRESHOLD


def test_explicit_generator_commutation_needs_no_state():
    """A complete Pauli generator supplies an all-strength symbolic certificate."""
    circuit = QuantumCircuit(1)
    circuit.id(0)
    candidate = hermitian_perturbation_mutation(
        circuit, 'commuting', 0, SparsePauliOp.from_list([('Z', .5), ('I', .17)]), .3,
    )
    problem = problem_for(circuit, candidate)
    propagation = propagate_observable(circuit, problem.test.observable, 1, 8, 1e-12)
    evidence = evaluate_candidate_effect(problem, candidate, propagation, PlannerSettings(exact_state_max_qubits=0))
    assert evidence.zero_provenance is ZeroProvenance.ALGEBRAIC
    assert _status_from_evidence(evidence, .1) is DecisionStatus.SKIP_CERTIFIED_ZERO


@pytest.mark.parametrize('suffix', ['none', 'clifford', 'tiny_rotation'])
def test_tiny_noncommuting_terms_never_become_exact_zeros(suffix):
    """Neither numerical cutoffs nor near-Clifford gates can certify invisibility."""
    circuit = QuantumCircuit(1)
    observable = SparsePauliOp.from_list([('Z', 1), ('X', 1e-14)])
    state = Statevector.from_label('+')
    if suffix == 'clifford':
        circuit.h(0)
        axis = Pauli('X')
        # H maps the observable to X + 1e-14 Z; an X rotation preserves |+>.
        # Use |0> so the tiny Z term contributes a nonzero expectation shift.
        state = Statevector.from_label('0')
    elif suffix == 'tiny_rotation':
        circuit.ry(1e-14, 0)
        observable = SparsePauliOp('Z')
        axis = Pauli('Z')
    else:
        axis = Pauli('Z')
    candidate = pauli_rotation_mutation(circuit, 'tiny', 0, axis, 0, np.pi, qubits=(0,), insert=True)
    problem = problem_for(circuit, candidate, observable=observable, state=state)
    propagation = propagate_observable(circuit, observable, 0, 1, 1e-12)
    assert not propagation.algebraically_complete
    assert propagation.truncation_error > 0
    evidence = evaluate_candidate_effect(problem, candidate, propagation, PlannerSettings())
    assert evidence.zero_provenance is not ZeroProvenance.ALGEBRAIC
    # Allow roundoff from the unit-size term, but reject a zero or roundoff-only gap.
    assert evidence.gap == pytest.approx(2e-14, rel=0, abs=np.finfo(float).eps)
    residual = np.linalg.norm(propagation.exact_operator.data - propagation.retained_paulis.to_matrix(), ord=2)
    assert residual <= propagation.truncation_error + 1e-16


def test_discarded_generator_term_prevents_a_commutation_certificate():
    """Explicit generator metadata must also retain arbitrarily small nonzero terms."""
    circuit = QuantumCircuit(1)
    circuit.id(0)
    candidate = hermitian_perturbation_mutation(
        circuit, 'tiny-generator', 0, SparsePauliOp.from_list([('Z', .5), ('X', 1e-14)]), .1,
    )
    problem = problem_for(circuit, candidate)
    propagation = propagate_observable(circuit, problem.test.observable, 1, 8, 1e-12)
    evidence = evaluate_candidate_effect(problem, candidate, propagation, PlannerSettings())
    assert evidence.zero_provenance is not ZeroProvenance.ALGEBRAIC
    assert evidence.evidence_type is EvidenceType.STATE_DEPENDENT_PERTURBATIVE


@pytest.mark.parametrize(
    'bounds,expected', [
        ((0, .09), DecisionStatus.SKIP_BELOW_TEST_THRESHOLD),
        ((.1, .2), DecisionStatus.EXECUTE_PREDICTED_ABOVE_THRESHOLD),
        ((0, .1), DecisionStatus.EXECUTE_THRESHOLD_UNRESOLVED),
        ((.09, .11), DecisionStatus.EXECUTE_THRESHOLD_UNRESOLVED),
    ],
)
def test_deterministic_threshold_boundaries(bounds, expected):
    """The strict upper and inclusive lower boundary rules remain distinct."""
    evidence = EffectEvidence(EvidenceType.STATE_DEPENDENT_PERTURBATIVE, deterministic_interval=bounds)
    assert _status_from_evidence(evidence, .1) is expected
    assert evidence.interval is None


class AllocationPilot:
    """Record the confidence allocation passed to each pilot invocation."""
    def __init__(self):
        self.allocations = []

    def estimate(self, state, targets, settings):
        self.allocations.append(settings.failure_probability)
        return PilotResult({target.to_label(): 0.0 for target in targets}, .1, 10, 1, 10)


def test_pilot_budget_is_run_wide_but_direct_calls_are_per_call():
    """Allocate against all declared candidates, even those filtered before piloting."""
    circuit = QuantumCircuit(1)
    candidates = tuple(
        pauli_rotation_mutation(circuit, str(i), 0, Pauli('X'), 0, angle, qubits=(0,), insert=True)
        for i, angle in enumerate([.01, .5, .7])
    )
    problem = replace(problem_for(circuit, candidates[0]), candidates=candidates)
    settings = PlannerSettings(exact_state_max_qubits=0, enable_equivalence=False, pilot=PilotSettings(failure_probability=.06))
    provider = AllocationPilot()
    records = plan_mutant_executions(problem, settings, provider)
    assert provider.allocations == [.02, .02]
    assert [record.evidence.failure_probability for record in records] == [None, .02, .02]
    propagation = propagate_observable(circuit, problem.test.observable, 0, 8, 1e-12)
    direct = evaluate_candidate_effect(problem, candidates[1], propagation, settings, provider)
    assert provider.allocations[-1] == .06
    assert direct.failure_probability == .06
    assert plan_mutant_executions(replace(problem, candidates=()), settings) == []


def test_reference_entangled_inputs_obey_single_copy_ceiling_and_fidelity_floor():
    """An untouched reference does not evade the per-execution rotation bounds."""
    for seed in range(8):
        rho = random_density_matrix(4, seed=seed).data
        eta = .1 + seed / 10
        axis = (Pauli('X').to_matrix() + Pauli('Y').to_matrix()) / np.sqrt(2)
        rotation = np.cos(eta / 2) * np.eye(2) - 1j * np.sin(eta / 2) * axis
        joint = np.kron(np.eye(2), rotation)
        mutated = joint @ rho @ joint.conj().T
        distance = np.linalg.norm(rho - mutated, ord='nuc') / 2
        assert distance <= abs(np.sin(eta / 2)) + 1e-12
        assert abs(np.trace(rho @ joint))**2 >= np.cos(eta / 2)**2 - 1e-12
    bell = DensityMatrix(Statevector([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])).data
    eta = .4
    joint = np.kron(np.eye(2), np.cos(eta / 2) * np.eye(2) - 1j * np.sin(eta / 2) * Pauli('Z').to_matrix())
    assert np.linalg.norm(bell - joint @ bell @ joint.conj().T, ord='nuc') / 2 == pytest.approx(np.sin(eta / 2))


def test_removed_candidates_are_never_evaluated_or_overwritten(monkeypatch):
    """Each early exit preserves its record and the original candidate population."""
    import quantum_mutant_planner.planner as planner
    circuit = QuantumCircuit(1)
    candidates = tuple(
        pauli_rotation_mutation(circuit, name, 0, Pauli('Y'), 0., angle, qubits=(0,), insert=True)
        for name, angle in [('equivalent', 0.), ('threshold', .0001), ('budget', .01), ('active', .7)]
    )
    problem = replace(problem_for(circuit, candidates[0], threshold=.001), candidates=candidates)
    calls = []
    original_evaluator = planner.evaluate_candidate_effect

    def track(*args, **kwargs):
        calls.append(args[1].identifier)
        return original_evaluator(*args, **kwargs)

    monkeypatch.setattr(planner, 'evaluate_candidate_effect', track)
    records = planner.plan_mutant_executions(problem, PlannerSettings(enable_discrimination_budget=True))
    assert calls == ['active']
    assert problem.candidates is candidates
    assert [record.candidate_id for record in records] == [c.identifier for c in candidates]
    assert [record.status for record in records] == [
        DecisionStatus.SKIP_SEMANTICALLY_EQUIVALENT,
        DecisionStatus.SKIP_BELOW_TEST_THRESHOLD,
        DecisionStatus.SKIP_BELOW_BUDGET_FLOOR,
        DecisionStatus.EXECUTE_PREDICTED_ABOVE_THRESHOLD,
    ]


def test_spectral_replacement_does_not_use_an_unsupported_pilot():
    """Only mutation classes with the manuscript's pilot analysis enter that branch."""
    from quantum_mutant_planner.mutations import insertion_mutation
    circuit = QuantumCircuit(1)
    candidate = insertion_mutation(circuit, 'spectral', 0, UnitaryGate(np.diag([1., np.exp(.37j)])), (0,))
    problem = problem_for(circuit, candidate)
    propagation = propagate_observable(circuit, problem.test.observable, 0, 8, 1e-12)
    provider = AllocationPilot()
    evidence = evaluate_candidate_effect(problem, candidate, propagation, PlannerSettings(exact_state_max_qubits=0), provider)
    assert provider.allocations == []
    assert evidence.provenance == 'simulated gap'
