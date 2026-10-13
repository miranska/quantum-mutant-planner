"""Resource boundaries and independent dense oracles for sparse planning."""

from dataclasses import replace

import numpy as np
import pytest
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, IGate, RYGate, XGate
from qiskit.quantum_info import DensityMatrix, Operator, Pauli, SparsePauliOp, Statevector, random_unitary

from quantum_mutant_planner import analysis, certificates
from quantum_mutant_planner.analysis import direct_signed_gap, propagate_observable, validate_problem
from quantum_mutant_planner.certificates import certify_local_channel_equivalence, exact_pauli_rotation
from quantum_mutant_planner.models import (
    DecisionStatus, EvidenceType, ExecutionBudget, PlannerSettings, PlanningProblem, QuantumTest, ZeroProvenance,
)
from quantum_mutant_planner.mutations import (
    hermitian_perturbation_mutation, insertion_mutation, pauli_rotation_mutation,
)
from quantum_mutant_planner.pauli import ResourceLimitError, local_conjugation, observable_range, rotation_difference
from quantum_mutant_planner.planner import plan_mutant_executions


def problem_for(circuit, candidates, observable=None, state=None):
    width = circuit.num_qubits
    return PlanningProblem(
        circuit,
        QuantumTest(
            Statevector.from_int(0, 2**width) if state is None else state,
            SparsePauliOp('I' * (width - 1) + 'Z') if observable is None else observable,
            .01, 2.,
        ),
        tuple(candidates), ExecutionBudget(1000, .05),
    )


@pytest.fixture
def forbid_register_matrices(monkeypatch):
    """Permit small local matrices but fail any full twelve-qubit allocation path."""
    to_matrix = SparsePauliOp.to_matrix
    compose = certificates._compose_circuit
    embed = analysis.embed_local_operator

    def guarded_matrix(self, *args, **kwargs):
        assert self.num_qubits <= 4, 'full observable matrix requested'
        return to_matrix(self, *args, **kwargs)

    def guarded_compose(width, operations):
        assert width <= 4, 'full symbolic matrix requested'
        return compose(width, operations)

    def guarded_embed(operator, support, width):
        assert width <= 4, 'full embedded operator requested'
        return embed(operator, support, width)

    monkeypatch.setattr(SparsePauliOp, 'to_matrix', guarded_matrix)
    monkeypatch.setattr(certificates, '_compose_circuit', guarded_compose)
    monkeypatch.setattr(analysis, 'embed_local_operator', guarded_embed)
    monkeypatch.setattr('quantum_mutant_planner.planner.embed_local_operator', guarded_embed)


def test_wide_structural_local_and_pauli_checks_need_no_register_matrix(forbid_register_matrices):
    circuit = QuantumCircuit(12)
    circuit.h(1)
    candidates = [
        insertion_mutation(circuit, 'semantic', 1, IGate(), (0,)),
        insertion_mutation(circuit, 'outside', 1, XGate(), (9,)),
        pauli_rotation_mutation(circuit, 'commuting', 0, Pauli('Z'), 0, .4, qubits=(0,), insert=True),
        insertion_mutation(circuit, 'clifford', 1, XGate(), (0,)),
        insertion_mutation(circuit, 'spectral', 1, RYGate(.4), (0,)),
    ]
    records = plan_mutant_executions(problem_for(circuit, candidates), PlannerSettings(dense_max_qubits=0))
    assert [r.status for r in records] == [
        DecisionStatus.SKIP_SEMANTICALLY_EQUIVALENT,
        DecisionStatus.SKIP_CERTIFIED_ZERO,
        DecisionStatus.SKIP_CERTIFIED_ZERO,
        DecisionStatus.EXECUTE_THRESHOLD_UNRESOLVED,
        DecisionStatus.EXECUTE_THRESHOLD_UNRESOLVED,
    ]
    assert records[1].evidence.zero_provenance is ZeroProvenance.STRUCTURAL
    assert records[2].evidence.zero_provenance is ZeroProvenance.ALGEBRAIC
    assert records[3].evidence.evidence_type is EvidenceType.EXACT_OBSERVABLE_SIDE
    assert records[4].evidence.evidence_type is EvidenceType.EXACT_OBSERVABLE_SIDE


def test_wide_statevector_expectations_and_direct_simulation_stay_sparse(forbid_register_matrices):
    circuit = QuantumCircuit(12)
    rotation = pauli_rotation_mutation(circuit, 'rotation', 0, Pauli('Y'), 0, .4, qubits=(0,), insert=True)
    generic = insertion_mutation(circuit, 'generic', 0, RYGate(.4), (0,), allow_spectral=False)
    records = plan_mutant_executions(
        problem_for(circuit, [rotation, generic], SparsePauliOp('Z' * 12)),
        PlannerSettings(dense_max_qubits=0, exact_state_max_qubits=12, exact_simulation_max_qubits=12),
    )
    assert [r.evidence.provenance for r in records] == ['structured gap', 'simulated gap']
    assert all(r.evidence.gap == pytest.approx(1 - np.cos(.4)) for r in records)


def test_dense_state_limit_does_not_block_observable_side_work():
    circuit = QuantumCircuit(3)
    candidate = insertion_mutation(circuit, 'x', 0, XGate(), (0,))
    problem = problem_for(circuit, [candidate], state=DensityMatrix.from_int(0, 8))
    record = plan_mutant_executions(problem, PlannerSettings(dense_max_qubits=0))[0]
    assert record.evidence.evidence_type is EvidenceType.EXACT_OBSERVABLE_SIDE


@pytest.mark.parametrize('gate', [HGate(), RYGate(.3)])
def test_local_conjugation_matches_dense_oracle_on_reordered_support(gate):
    local = random_unitary(4, seed=5) if isinstance(gate, HGate) else Operator(gate).tensor(Operator(gate))
    observable = SparsePauliOp.from_list([('XYZI', .7), ('ZIYX', -.3), ('IIII', .2)])
    result = local_conjugation(observable, local, (3, 1), 2)
    full = analysis.embed_local_operator(local, (3, 1), 4).data
    expected = full.conj().T @ observable.to_matrix() @ full
    assert np.allclose(result.to_matrix(), expected)
    with pytest.raises(ResourceLimitError):
        local_conjugation(observable, local, (3, 1), 1)


@pytest.mark.parametrize('axis', ['XY', '-YZ', 'ZZ'])
def test_sparse_rotation_difference_preserves_signed_multiqubit_axes(axis):
    observable = SparsePauliOp.from_list([('XZ', .2), ('YX', -.3), ('IZ', .6)])
    pauli = Pauli(axis)
    unitary = np.cos(.35) * np.eye(4) - 1j * np.sin(.35) * pauli.to_matrix()
    expected = observable.to_matrix() - unitary.conj().T @ observable.to_matrix() @ unitary
    assert np.allclose(rotation_difference(observable, pauli, .7).to_matrix(), expected)
    assert np.allclose(rotation_difference(SparsePauliOp(pauli), pauli, .7).to_matrix(), 0)


@pytest.mark.parametrize('dense_limit,simulation_limit,expected', [
    (0, 0, EvidenceType.UNAVAILABLE),
    (1, 0, EvidenceType.EXACT),
    (0, 1, EvidenceType.EXACT),
])
def test_truncated_propagation_uses_only_authorized_fallback(dense_limit, simulation_limit, expected):
    circuit = QuantumCircuit(1)
    circuit.rz(.3, 0)
    candidate = pauli_rotation_mutation(circuit, 'y', 0, Pauli('Y'), 0, .4, qubits=(0,), insert=True)
    problem = problem_for(circuit, [candidate], SparsePauliOp('X'))
    record = plan_mutant_executions(problem, PlannerSettings(
        dense_max_qubits=dense_limit, max_pauli_terms=1, exact_state_max_qubits=1,
        exact_simulation_max_qubits=simulation_limit, enable_pilot=False,
    ))[0]
    assert record.evidence.evidence_type is expected
    if expected is EvidenceType.EXACT:
        assert record.evidence.gap == pytest.approx(direct_signed_gap(problem, candidate, 1e-12)[1])
    else:
        assert record.status is DecisionStatus.EXECUTE_THRESHOLD_UNRESOLVED
    propagation = propagate_observable(circuit, problem.test.observable, 0, 1, 1e-12, dense_max_qubits=0)
    assert propagation.complete_paulis is None
    assert propagation.exact_operator is None
    assert propagation.truncation_error == pytest.approx(abs(np.sin(.3)))


def test_general_generator_commutation_precedes_bounded_perturbative_fallback(forbid_register_matrices):
    circuit = QuantumCircuit(12)
    circuit.h(0)
    zero = hermitian_perturbation_mutation(circuit, 'zero', 0, SparsePauliOp('Z'), .3)
    general = hermitian_perturbation_mutation(circuit, 'general', 0, Operator(Pauli('Y')), .3)
    records = plan_mutant_executions(problem_for(circuit, [zero, general]), PlannerSettings(dense_max_qubits=0))
    assert records[0].evidence.zero_provenance is ZeroProvenance.ALGEBRAIC
    assert records[1].evidence.evidence_type is EvidenceType.UNAVAILABLE
    assert 'dense_max_qubits' in records[1].evidence.warnings[0]


def test_local_equivalence_checks_context_operands_and_exact_semantics():
    original = QuantumCircuit(12)
    original.h(11)
    original.s(0)
    mutant = QuantumCircuit(12)
    mutant.h(11)
    mutant.append(exact_pauli_rotation(Pauli('Z'), sp.pi / 4), [0])
    mutant.append(exact_pauli_rotation(Pauli('Z'), sp.pi / 4), [0])
    assert certify_local_channel_equivalence(original, mutant, 2)
    assert not certify_local_channel_equivalence(original, mutant, 0)
    changed_context = mutant.copy()
    changed_context.x(8)
    assert not certify_local_channel_equivalence(original, changed_context, 2)
    missing = mutant.copy()
    del missing.data[-1]
    assert not certify_local_channel_equivalence(original, missing, 2)
    wrong_operand = QuantumCircuit(12)
    wrong_operand.h(11)
    wrong_operand.s(1)
    assert not certify_local_channel_equivalence(original, wrong_operand, 2)
    original.rz(.1, 2)
    assert not certify_local_channel_equivalence(original, original.copy(), 2)
    assert not certify_local_channel_equivalence(QuantumCircuit(2, 1), QuantumCircuit(2, 1), 2)
    assert not certify_local_channel_equivalence(QuantumCircuit(2), QuantumCircuit(3), 2)
    wide_gate = QuantumCircuit(3)
    wide_gate.ccx(0, 1, 2)
    assert not certify_local_channel_equivalence(wide_gate, wide_gate.copy(), 2)


@pytest.mark.parametrize('terms', [
    [('IIII', 3.), ('XYZI', .3), ('XZZI', -.4)],
    [('IZII', .3), ('ZIIZ', -.7), ('IZIZ', .2)],
    [('IIIZ', .3), ('IIZI', .7), ('IIZZ', -.2)],
])
def test_bounded_range_validation_matches_dense_spectrum(terms):
    observable = SparsePauliOp.from_list(terms)
    eigenvalues = np.linalg.eigvalsh(observable.to_matrix())
    assert observable_range(observable, 4) == pytest.approx(eigenvalues[-1] - eigenvalues[0])


def test_range_and_gate_validation_fail_before_unbounded_allocation():
    dependent = SparsePauliOp.from_list([('ZII', 1), ('IZI', 1), ('ZZI', 1)])
    with pytest.raises(ResourceLimitError, match='spectral range'):
        observable_range(dependent, 1)
    assert observable_range(SparsePauliOp('III', coeffs=[0]), 0) == 0
    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)
    with pytest.raises(ResourceLimitError, match='gate validation'):
        validate_problem(problem_for(circuit, []), local_max_qubits=2)
    with pytest.raises(ResourceLimitError, match='suffix gate'):
        propagate_observable(circuit, SparsePauliOp('ZII'), 0, 8, 1e-12, local_max_qubits=2)
    with pytest.raises(ValueError, match='limits'):
        plan_mutant_executions(problem_for(QuantumCircuit(1), []), PlannerSettings(dense_max_qubits=-1))


def test_initial_term_cap_does_not_create_algebraic_zero():
    circuit = QuantumCircuit(2)
    observable = SparsePauliOp.from_list([('IZ', 1.), ('IX', 1e-14)])
    candidate = pauli_rotation_mutation(circuit, 'z', 0, Pauli('Z'), 0, .4, qubits=(0,), insert=True)
    problem = problem_for(circuit, [candidate], observable)
    record = plan_mutant_executions(problem, PlannerSettings(
        max_pauli_terms=1, dense_max_qubits=0, exact_state_max_qubits=0, exact_simulation_max_qubits=0,
    ))[0]
    assert record.status is DecisionStatus.EXECUTE_THRESHOLD_UNRESOLVED
    assert record.evidence.zero_provenance is not ZeroProvenance.ALGEBRAIC


def test_wide_pilot_difference_and_allocation_need_no_register_matrix(forbid_register_matrices):
    from quantum_mutant_planner.pilot import PilotResult

    calls = []

    class ExactPilot:
        def estimate(self, state, targets, settings):
            calls.append(settings.failure_probability)
            return PilotResult(
                {p.to_label(): float(state.expectation_value(p).real) for p in targets},
                .1, 100, 5, 20,
            )

    circuit = QuantumCircuit(12)
    candidates = [insertion_mutation(circuit, str(index), 0, XGate(), (index,)) for index in [0, 9]]
    record = plan_mutant_executions(
        problem_for(circuit, candidates),
        PlannerSettings(dense_max_qubits=0, exact_state_max_qubits=0, exact_simulation_max_qubits=12),
        ExactPilot(),
    )[0]
    assert calls == [.025]
    assert record.evidence.evidence_type is EvidenceType.PILOT_ESTIMATED
    assert record.evidence.gap == pytest.approx(2.)
    assert record.evidence.interval == pytest.approx((1.8, 2.2))


def test_mutant_gate_width_is_checked_before_fallback():
    circuit = QuantumCircuit(3)
    candidate = insertion_mutation(circuit, 'wide', 0, random_unitary(8, seed=7).to_instruction(), (0, 1, 2))
    with pytest.raises(ResourceLimitError, match='candidate wide'):
        plan_mutant_executions(problem_for(circuit, [candidate]), PlannerSettings(local_max_qubits=2))


def test_lazy_dense_fallback_uses_a_snapshot_of_its_sources():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    observable = SparsePauliOp('Z')
    propagation = propagate_observable(circuit, observable, 0, 8, 1e-12)
    circuit.x(0)
    observable.coeffs[0] = 2.
    assert np.allclose(propagation.exact_operator.data, Pauli('X').to_matrix())


def test_pilot_simulation_with_many_measurement_settings_is_reproducible():
    from quantum_mutant_planner.models import PilotSettings
    from quantum_mutant_planner.pilot import MedianOfMeansPilot

    state = Statevector.from_int(0, 16)
    settings = PilotSettings(target_error=1., failure_probability=.1)
    provider = MedianOfMeansPilot()
    first = provider.estimate(state, (Pauli('IIIZ'),), settings)
    second = provider.estimate(state, (Pauli('IIIZ'),), settings)
    assert first == second
    assert abs(first.estimates['IIIZ'] - 1.) <= first.target_error
