"""Tests for structural, propagation, equivalence, and validation utilities."""

from dataclasses import replace
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate
from qiskit.quantum_info import DensityMatrix, Operator, Pauli, SparsePauliOp, Statevector

from quantum_mutant_planner.analysis import (
    direct_signed_gap,
    embed_local_operator,
    embed_local_pauli,
    information_theoretic_strength_floor,
    is_clifford_relative,
    observable_support,
    propagate_observable,
    real_expectation,
    state_at_mutation,
    structural_backward_cones,
    transpiler_equivalent,
    validate_problem,
)
from quantum_mutant_planner.examples import build_ghz_problem
from quantum_mutant_planner.models import ExecutionBudget, PlanningProblem, QuantumTest, TranspilerSettings
from quantum_mutant_planner.mutations import insertion_mutation


def simple_problem() -> PlanningProblem:
    """Build a one-qubit validated test fixture."""

    circuit = QuantumCircuit(1)
    circuit.h(0)
    candidate = insertion_mutation(circuit, "x", 1, XGate(), (0,), allow_spectral=False)
    return PlanningProblem(
        circuit,
        QuantumTest(Statevector.from_int(0, 2), SparsePauliOp("Z"), 0.1, 2.0),
        (candidate,),
        ExecutionBudget(100, 0.05),
    )


def test_observable_support_and_structural_cones() -> None:
    """Structural traversal should expand through interacting gates only."""

    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.x(2)
    observable = SparsePauliOp.from_list([("IZI", 1.0), ("III", 1.0e-13)])
    assert observable_support(observable, tolerance=1.0e-12) == {1}
    cones = structural_backward_cones(circuit, observable)
    assert cones[2] == frozenset({1})
    assert cones[1] == frozenset({1})
    assert cones[0] == frozenset({0, 1})


def test_embedding_respects_qiskit_qubit_order() -> None:
    """Local X on qubit one should match the full IXI convention."""

    full_pauli = embed_local_pauli(Pauli("X"), (1,), 3)
    assert full_pauli.to_label() == "IXI"
    embedded = embed_local_operator(Operator(Pauli("X")), (1,), 3)
    assert np.allclose(embedded.data, Operator(full_pauli).data)
    with pytest.raises(ValueError):
        embed_local_pauli(Pauli("XX"), (0,), 2)
    with pytest.raises(ValueError):
        embed_local_operator(Operator(Pauli("XX")), (0,), 2)


def test_propagation_matches_known_clifford_and_tracks_truncation() -> None:
    """Backward propagation should preserve exact matrices and bound truncation."""

    clifford = QuantumCircuit(2)
    clifford.cx(0, 1)
    result = propagate_observable(clifford, SparsePauliOp("ZI"), 0, 8, 1.0e-12)
    assert result.algebraically_complete
    assert np.allclose(result.exact_operator.data, SparsePauliOp("ZZ").to_matrix())

    nonclifford = QuantumCircuit(1)
    nonclifford.rz(0.3, 0)
    truncated = propagate_observable(nonclifford, SparsePauliOp("X"), 0, 1, 1.0e-14)
    assert not truncated.algebraically_complete
    assert truncated.truncation_error > 0.0
    with pytest.raises(ValueError, match="positive"):
        propagate_observable(nonclifford, SparsePauliOp("X"), 0, 0, 1.0e-12)


def test_state_direct_gap_expectation_and_equivalence_helpers() -> None:
    """State and direct-simulation helpers should agree on a simple test."""

    problem = simple_problem()
    candidate = problem.candidates[0]
    site_state = state_at_mutation(problem, candidate)
    assert np.allclose(site_state.data, Statevector.from_label("+").data)
    signed, gap = direct_signed_gap(problem, candidate, 1.0e-12)
    assert signed == pytest.approx(0.0, abs=1.0e-12)
    assert gap == pytest.approx(0.0, abs=1.0e-12)
    assert real_expectation(Statevector.from_label("0"), Operator(Pauli("Z")), 1.0e-12) == 1.0
    with pytest.raises(ValueError, match="imaginary"):
        real_expectation(Statevector.from_label("0"), Operator(np.array([[1j, 0], [0, 0]])), 1.0e-12)
    assert transpiler_equivalent(problem.circuit, problem.circuit.copy(), TranspilerSettings())
    assert is_clifford_relative(candidate)


def test_strength_floor_matches_manuscript_value() -> None:
    """The default example budget should reproduce about 0.0258 radians."""

    assert information_theoretic_strength_floor(10_000, 0.05) == pytest.approx(0.0258, abs=5.0e-5)


def test_validate_problem_accepts_statevector_and_density_matrix() -> None:
    """Both supported Qiskit state representations should validate."""

    problem = simple_problem()
    validate_problem(problem)
    density_test = replace(problem.test, state=DensityMatrix(problem.test.state))
    validate_problem(replace(problem, test=density_test))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: replace(p, test=replace(p.test, threshold=0.0)), "threshold"),
        (lambda p: replace(p, test=replace(p.test, outcome_range=0.0)), "range"),
        (lambda p: replace(p, test=replace(p.test, outcome_range=1.0)), "spectrum"),
        (lambda p: replace(p, budget=ExecutionBudget(0, 0.05)), "copies"),
        (lambda p: replace(p, budget=ExecutionBudget(1, 0.5)), "probability"),
        (lambda p: replace(p, candidates=(p.candidates[0], p.candidates[0])), "identifiers"),
        (lambda p: replace(p, candidates=(replace(p.candidates[0], cost=0.0),)), "cost"),
        (lambda p: replace(p, candidates=(replace(p.candidates[0], support=()),)), "support"),
        (lambda p: replace(p, candidates=(replace(p.candidates[0], suffix_start=9),)), "suffix"),
    ],
)
def test_validate_problem_rejects_invalid_scalar_and_candidate_fields(mutation, message: str) -> None:
    """Validation should reject every malformed scalar and candidate field."""

    with pytest.raises(ValueError, match=message):
        validate_problem(mutation(simple_problem()))


def test_validate_problem_rejects_dimension_and_observable_errors() -> None:
    """State, observable, mutant width, and Hermiticity must match the model."""

    problem = simple_problem()
    with pytest.raises(ValueError, match="input-state"):
        validate_problem(replace(problem, test=replace(problem.test, state=Statevector.from_int(0, 4))))
    with pytest.raises(ValueError, match="observable qubit"):
        validate_problem(replace(problem, test=replace(problem.test, observable=SparsePauliOp("ZZ"))))
    nonhermitian = SparsePauliOp.from_list([("X", 1j)])
    with pytest.raises(ValueError, match="Hermitian"):
        validate_problem(replace(problem, test=replace(problem.test, observable=nonhermitian)))
    wider = QuantumCircuit(2)
    with pytest.raises(ValueError, match="width"):
        validate_problem(replace(problem, candidates=(replace(problem.candidates[0], mutant_circuit=wider),)))


def test_validate_problem_rejects_classical_and_nonunitary_circuits() -> None:
    """Measurements and nonunitary instructions are outside the formal scope."""

    problem = simple_problem()
    measured = QuantumCircuit(1, 1)
    measured.measure(0, 0)
    with pytest.raises(ValueError, match="classical"):
        validate_problem(replace(problem, circuit=measured))

    reset = QuantumCircuit(1)
    reset.reset(0)
    reset_problem = PlanningProblem(reset, problem.test, (), problem.budget)
    with pytest.raises(ValueError, match="nonunitary"):
        validate_problem(reset_problem)

    class NonunitaryGate(Gate):
        """Test-only operation with a square but nonunitary matrix."""

        def __init__(self) -> None:
            """Initialize the invalid one-qubit operation."""

            super().__init__("nonunitary", 1, [])

        def __array__(self, dtype=None, copy=None):
            """Return a nonunitary matrix for Qiskit Operator conversion."""

            return np.asarray([[1.0, 0.0], [0.0, 0.5]], dtype=dtype)

    invalid = QuantumCircuit(1)
    invalid.append(NonunitaryGate(), [0])
    with pytest.raises(ValueError, match="not unitary"):
        validate_problem(PlanningProblem(invalid, problem.test, (), problem.budget))
