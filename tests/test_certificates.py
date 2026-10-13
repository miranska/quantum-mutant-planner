"""Independent full-circuit checks and false-certificate regressions."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, XGate
from qiskit.quantum_info import DensityMatrix, Operator, Pauli, SparsePauliOp, Statevector
import sympy as sp

import quantum_mutant_planner.certificates as certificates
from quantum_mutant_planner.certificates import (
    ExactDensityMatrix, ExactStatevector, ExactUnitaryGate, _GATES, _circuit_matrix,
    _gate_matrix, _input_density, _numeric, _pauli_matrix, certify_circuit_zero, exact_pauli_rotation,
)
from quantum_mutant_planner.examples import build_ghz_problem
from quantum_mutant_planner.models import ExecutionBudget, PlannerSettings, PlanningProblem, QuantumTest, ZeroProvenance
from quantum_mutant_planner.mutations import insertion_mutation, pauli_rotation_mutation
from quantum_mutant_planner.planner import plan_mutant_executions


def cancellation_problem(angle=sp.pi / 2):
    """Make a nontrivial cancellation with a=b*(-1)=1 and different output states."""
    circuit = QuantumCircuit(1)
    candidate = pauli_rotation_mutation(circuit, "cancel", 0, Pauli("Y"), 0, angle, qubits=(0,), insert=True)
    test = QuantumTest(Statevector.from_label("0"), SparsePauliOp.from_list([("X", 1), ("Z", 1)]), .1, 2 * np.sqrt(2))
    return PlanningProblem(circuit, test, (candidate,), ExecutionBudget(10000, .05))


def test_nontrivial_cancellation_and_nearby_nonzero():
    """A zero of the observable does not imply identical output states or operators."""
    problem = cancellation_problem()
    record = plan_mutant_executions(problem, PlannerSettings(enable_equivalence=False))[0]
    assert record.status.value == "skip--certified zero"
    assert record.evidence.provenance.startswith("state-dependent cancellation certificate")
    original = problem.test.state
    mutant = original.evolve(problem.candidates[0].mutant_circuit)
    assert not np.allclose(DensityMatrix(original).data, DensityMatrix(mutant).data)
    assert original.expectation_value(problem.test.observable) == pytest.approx(mutant.expectation_value(problem.test.observable))
    neighbor = cancellation_problem(sp.pi / 2 + sp.Rational(1, 10**8))
    assert certify_circuit_zero(neighbor, neighbor.candidates[0], 4) is None
    near_record = plan_mutant_executions(neighbor, PlannerSettings(enable_equivalence=False))[0]
    assert near_record.evidence.zero_provenance is not ZeroProvenance.ALGEBRAIC
    assert near_record.evidence.gap > 0


def test_float_pi_does_not_acquire_an_exact_certificate():
    """Rounding a number near a cancellation angle is never an algebraic proof."""
    problem = cancellation_problem(float(np.pi / 2))
    assert certify_circuit_zero(problem, problem.candidates[0], 4) is None
    assert plan_mutant_executions(problem, PlannerSettings(enable_equivalence=False))[0].evidence.zero_provenance is not ZeroProvenance.ALGEBRAIC


def test_ghz_against_separately_built_numerical_circuits():
    """The paper's cancellations survive a separate construction of the circuits."""
    problem = build_ghz_problem()
    original = QuantumCircuit(3)
    original.ry(np.pi / 2, 0)
    original.cx(0, 1)
    original.cx(0, 2)
    m4 = original.copy()
    m4.x(1)
    m5 = QuantumCircuit(3)
    m5.ry(-np.pi / 2, 0)
    m5.cx(0, 1)
    m5.cx(0, 2)
    observable = SparsePauliOp("IZI")
    mean = Statevector.from_instruction(original).expectation_value(observable)
    for circuit, candidate in zip((m4, m5), problem.candidates[3:5], strict=True):
        assert abs(mean - Statevector.from_instruction(circuit).expectation_value(observable)) < 1e-14
        assert certify_circuit_zero(problem, candidate, 4).startswith("state-dependent")
        assert certify_circuit_zero(problem, candidate, 2) is None
    numerical = plan_mutant_executions(problem, PlannerSettings(enable_equivalence=False, symbolic_max_qubits=0))
    assert numerical[3].evidence.zero_provenance is ZeroProvenance.NUMERICAL


def test_explicit_pure_mixed_and_all_strength_orthogonality():
    """Exact inputs permit proofs without recognizing approximate statevectors."""
    circuit = QuantumCircuit(2)
    state = ExactStatevector([sp.Rational(1, 2)] * 4)
    for angle in (sp.pi / 3, sp.pi / 2, sp.pi):
        candidate = pauli_rotation_mutation(circuit, "orthogonal", 0, Pauli("Y"), 0, angle, qubits=(0,), insert=True)
        problem = PlanningProblem(circuit, QuantumTest(state, SparsePauliOp("ZZ"), .1, 2), (candidate,), ExecutionBudget(10000, .05))
        assert certify_circuit_zero(problem, candidate, 4).startswith("state-dependent")
    mixed = ExactDensityMatrix(sp.eye(2) / 2)
    problem = cancellation_problem()
    problem = replace(problem, test=replace(problem.test, state=mixed))
    assert certify_circuit_zero(problem, problem.candidates[0], 4).startswith("state-dependent")
    opaque = replace(problem, test=replace(problem.test, state=DensityMatrix(mixed.data)))
    assert certify_circuit_zero(opaque, opaque.candidates[0], 4) is None


def test_observable_invariance_does_not_require_exact_state():
    """An operator-level identity is distinguishable from state cancellation."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    candidate = insertion_mutation(circuit, "commuting", 1, XGate(), (0,))
    problem = PlanningProblem(circuit, QuantumTest(Statevector.from_label("+"), SparsePauliOp("X"), .1, 2), (candidate,), ExecutionBudget(10000, .05))
    assert certify_circuit_zero(problem, candidate, 4).startswith("algebraic observable-invariance")


@pytest.mark.parametrize("gate_type", list(_GATES))
def test_exact_gate_definitions_match_qiskit(gate_type):
    """Compare every hand-specified discrete matrix with an independent library."""
    gate = gate_type()
    circuit = QuantumCircuit(gate.num_qubits + 1)
    circuit.append(gate, list(reversed(range(gate.num_qubits + 1)))[:gate.num_qubits])
    np.testing.assert_allclose(_numeric(_circuit_matrix(circuit)), Operator(circuit).data, atol=1e-15)


@pytest.mark.parametrize("label", ["X", "Y", "IZ", "-YX", "iXY", "-iZZ"])
def test_exact_pauli_phases_and_order(label):
    np.testing.assert_array_equal(_numeric(_pauli_matrix(Pauli(label))), Pauli(label).to_matrix())


def test_tiny_observable_terms_are_not_erased():
    problem = cancellation_problem()
    problem = replace(problem, test=replace(problem.test, observable=SparsePauliOp.from_list([("X", 1), ("Z", 1 + 1e-14)])))
    assert certify_circuit_zero(problem, problem.candidates[0], 4) is None
    for coefficient in (complex(1, 1e-14), np.inf):
        observable = SparsePauliOp("X")
        observable.coeffs[0] = coefficient
        invalid = replace(problem, test=replace(problem.test, observable=observable))
        assert certify_circuit_zero(invalid, invalid.candidates[0], 4) is None


@pytest.mark.parametrize("matrix", [
    [[1.0, 0], [0, 1.0]], [[sp.Symbol("x"), 0], [0, 1]], [[sp.oo, 0], [0, 1]],
    sp.ones(2, 3), sp.eye(3), [[1, 1], [0, 1]],
])
def test_invalid_exact_gates_are_rejected(matrix):
    with pytest.raises(ValueError):
        ExactUnitaryGate(matrix)


@pytest.mark.parametrize("amplitudes", [[1, 1], [[1, 0]], [1.0, 0]])
def test_invalid_exact_pure_states_are_rejected(amplitudes):
    with pytest.raises(ValueError):
        ExactStatevector(amplitudes)


@pytest.mark.parametrize("matrix", [sp.ones(2, 3), [[1, 1], [0, 0]], sp.eye(2), sp.diag(2, -1)])
def test_invalid_exact_mixed_states_are_rejected(matrix):
    with pytest.raises(ValueError):
        ExactDensityMatrix(matrix)


def test_modified_numerical_projections_lose_certificates():
    gate = exact_pauli_rotation(Pauli("Y"), sp.pi / 2)
    gate.to_matrix()[0, 0] += 1e-14
    assert _gate_matrix(gate) is None
    pure = ExactStatevector([1, 0])
    pure.data[0] += 1e-14
    assert _input_density(pure) is None
    mixed = ExactDensityMatrix(sp.eye(2) / 2)
    mixed.data[0, 0] += 1e-14
    assert _input_density(mixed) is None
    assert _input_density(DensityMatrix.from_label("1")) == sp.diag(0, 1)
    assert _input_density(Statevector.from_label("+")) is None


@pytest.mark.parametrize("state", [
    ExactStatevector([sp.sqrt(2) / 2, sp.sqrt(2) / 2]),
    ExactDensityMatrix([[sp.Rational(1, 2), sp.Rational(1, 2)], [sp.Rational(1, 2), sp.Rational(1, 2)]]),
])
def test_numerical_state_transforms_drop_exact_metadata_even_when_they_round_away(state):
    """Tiny numerical evolutions must not inherit a certificate for their input."""
    circuit = QuantumCircuit(1)
    circuit.ry(1e-20, 0)
    evolved = state.evolve(circuit)
    np.testing.assert_array_equal(evolved.data, state.data)
    assert type(evolved) in (Statevector, DensityMatrix)
    assert _input_density(evolved) is None
    copied = state.copy()
    assert type(copied) in (Statevector, DensityMatrix)
    assert _input_density(copied) is None
    assert not np.shares_memory(state.data, copied.data)


def test_unsupported_source_and_invalid_rotation_metadata():
    problem = cancellation_problem()
    problem.circuit.ry(.1, 0)
    assert certify_circuit_zero(problem, problem.candidates[0], 4) is None
    circuit = QuantumCircuit(1)
    circuit.ry(np.pi / 2, 0)
    with pytest.raises(ValueError, match="matching exact"):
        pauli_rotation_mutation(circuit, "bad", 0, Pauli("Y"), sp.pi / 2, -sp.pi / 2)
    circuit = QuantumCircuit(1)
    circuit.append(exact_pauli_rotation(Pauli("Y"), sp.pi / 2), [0])
    with pytest.raises(ValueError, match="matching exact"):
        pauli_rotation_mutation(circuit, "bad", 0, Pauli("Y"), 0., sp.pi)
    with pytest.raises(ValueError, match="original_angle"):
        pauli_rotation_mutation(circuit, "bad", 0, Pauli("Y"), .5, 1., qubits=(0,), insert=True)
    with pytest.raises(ValueError, match="Hermitian"):
        pauli_rotation_mutation(circuit, "bad", 0, Pauli("iY"), 0, 1., qubits=(0,), insert=True)
    with pytest.raises(ValueError, match="real angle"):
        exact_pauli_rotation(Pauli("Y"), sp.I)
    # Ordinary floats keep their exact binary value if mixed with exact input.
    candidate = pauli_rotation_mutation(circuit, "mixed", 0, Pauli("Y"), sp.pi / 2, .25)
    assert candidate.eta == pytest.approx(.25 - np.pi / 2)


def test_open_controls_are_not_misread_as_closed_controls():
    """The exact discrete whitelist must honor the control state."""
    assert _gate_matrix(CXGate(ctrl_state=0)) is None


def test_exact_gate_copy_and_transpilation_preserve_action():
    """Qiskit must retain exact annotations when copying mutation prefixes."""
    from qiskit import transpile
    circuit = QuantumCircuit(1)
    circuit.append(exact_pauli_rotation(Pauli("Y"), sp.pi / 2), [0])
    copied = circuit.copy()
    assert isinstance(copied.data[0].operation, ExactUnitaryGate)
    compiled = transpile(copied, basis_gates=['rz', 'sx', 'x'], seed_transpiler=1)
    np.testing.assert_allclose(Operator(compiled).data, Operator(circuit).data, atol=1e-14)


def test_repeated_exact_circuit_copies_reuse_symbolic_work():
    """Equal exact contents reuse composition and simplification across objects."""
    problem = cancellation_problem()
    with patch.object(certificates, "_embed", wraps=certificates._embed) as embed, patch.object(
        sp, "simplify", wraps=sp.simplify,
    ) as simplify:
        first = certify_circuit_zero(problem, problem.candidates[0], 4)
        counts = (embed.call_count, simplify.call_count)
        copied = replace(problem, circuit=problem.circuit.copy())
        candidate = replace(problem.candidates[0], mutant_circuit=problem.candidates[0].mutant_circuit.copy())
        assert certify_circuit_zero(copied, candidate, 4) == first
        assert first.startswith("state-dependent")
        assert (embed.call_count, simplify.call_count) == counts


def test_symbolic_cache_distinguishes_operands_and_register_width():
    """Identical local gates on different registers cannot share a composition."""
    for width, operands in [(2, (0, 1)), (2, (1, 0)), (3, (0, 1))]:
        circuit = QuantumCircuit(width)
        circuit.cx(*operands)
        np.testing.assert_array_equal(_numeric(_circuit_matrix(circuit)), Operator(circuit).data)


def test_symbolic_cache_reuses_the_original_across_distinct_mutants():
    """Candidate-specific work must not recompute the shared original unitary."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    candidates = tuple(
        pauli_rotation_mutation(circuit, label, 1, Pauli(label), 0, sp.pi / 3, qubits=(0,), insert=True)
        for label in ("X", "Y")
    )
    problem = PlanningProblem(
        circuit, QuantumTest(Statevector.from_label("0"), SparsePauliOp("Z"), .1, 2),
        candidates, ExecutionBudget(10000, .05),
    )
    certificates._compose_circuit.cache_clear()
    certificates._conjugated_observable.cache_clear()
    for candidate in candidates:
        certify_circuit_zero(problem, candidate, 4)
    assert certificates._compose_circuit.cache_info().misses == 3
    assert certificates._compose_circuit.cache_info().hits == 1
    assert certificates._conjugated_observable.cache_info().misses == 3
    assert certificates._conjugated_observable.cache_info().hits == 1


@pytest.mark.parametrize("changed", ["circuit", "gate_projection", "state_projection", "observable"])
def test_warm_symbolic_cache_never_hides_changed_inputs(changed):
    """Mutable Qiskit data must be rechecked even after a successful proof."""
    problem = cancellation_problem()
    problem = replace(problem, test=replace(problem.test, state=ExactStatevector([1, 0])))
    candidate = problem.candidates[0]
    assert certify_circuit_zero(problem, candidate, 4).startswith("state-dependent")
    if changed == "circuit":
        problem.circuit.x(0)
    elif changed == "gate_projection":
        candidate.mutant_circuit.data[0].operation.to_matrix()[0, 0] += 1e-8
    elif changed == "state_projection":
        problem.test.state.data[:] = Statevector.from_label("+").data
    else:
        problem.test.observable.coeffs[0] += 1e-8
    assert certify_circuit_zero(problem, candidate, 4) is None


def test_symbolic_certificate_survives_large_floating_point_residual():
    """A numerical nonzero must not suppress an exact cancellation certificate."""
    problem = cancellation_problem()
    problem = replace(problem, test=replace(
        problem.test, observable=1e10 * problem.test.observable,
        outcome_range=1e10 * problem.test.outcome_range,
    ))
    settings = PlannerSettings(enable_equivalence=False)
    numerical = plan_mutant_executions(problem, replace(settings, symbolic_max_qubits=0))[0]
    assert numerical.evidence.gap > settings.numerical_tolerance
    certified = plan_mutant_executions(problem, settings)[0]
    assert certified.evidence.gap == 0
    assert certified.evidence.zero_provenance is ZeroProvenance.ALGEBRAIC
    assert certified.evidence.provenance.startswith("state-dependent")
