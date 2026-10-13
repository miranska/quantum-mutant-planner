"""Check manuscript identities against independently constructed full circuits."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp, random_density_matrix
from scipy.linalg import expm

from quantum_mutant_planner.analysis import embed_local_pauli, propagate_observable
from quantum_mutant_planner.models import ExecutionBudget, PlannerSettings, PlanningProblem, QuantumTest
from quantum_mutant_planner.mutations import pauli_rotation_mutation
from quantum_mutant_planner.planner import evaluate_candidate_effect


@pytest.mark.parametrize("seed", range(24))
def test_local_rotation_formula_against_independent_full_circuit(seed):
    """Check signs, ordering, negative axes, mixed inputs, and periodic strengths."""
    rng = np.random.default_rng(seed)
    width = 1 + seed % 3
    circuit = QuantumCircuit(width)
    for qubit in range(width):
        circuit.ry(rng.uniform(-2, 2), qubit)
    if width > 1:
        circuit.cx(0, width - 1)
    boundary = len(circuit.data)
    for qubit in range(width):
        circuit.rz(rng.uniform(-2, 2), qubit)
    target = seed % width
    axis = Pauli(('-' if seed % 2 else '') + ['X', 'Y', 'Z'][seed % 3])
    angle = [-2 * np.pi + .013, -.73, .37, np.pi, 2 * np.pi + .17][seed % 5]
    candidate = pauli_rotation_mutation(circuit, 'independent', boundary, axis, 0., angle, qubits=(target,), insert=True)
    # Matrix exponential and explicit circuit rebuilding do not reuse the
    # planner's relative-unitary factory or its local-response implementation.
    independent = QuantumCircuit(width)
    for index in range(len(circuit.data) + 1):
        if index == boundary:
            independent.append(UnitaryGate(expm(-.5j * angle * axis.to_matrix())), [target])
        if index < len(circuit.data):
            item = circuit.data[index]
            independent.append(item.operation, [circuit.find_bit(q).index for q in item.qubits])
    state = random_density_matrix(2**width, seed=seed)
    labels = [''.join(rng.choice(list('IXYZ'), size=width)) for _ in range(5)]
    observable = SparsePauliOp.from_list(list(zip(labels, rng.normal(size=5), strict=True)))
    values = np.linalg.eigvalsh(observable.to_matrix())
    problem = PlanningProblem(circuit, QuantumTest(state, observable, .1, float(values[-1]-values[0])), (candidate,), ExecutionBudget(10000, .05))
    propagation = propagate_observable(circuit, observable, boundary, 100, 0)
    evidence = evaluate_candidate_effect(problem, candidate, propagation, PlannerSettings(symbolic_max_qubits=0))
    original_out = state.evolve(circuit)
    mutant_out = state.evolve(independent)
    signed = float(np.real(original_out.expectation_value(observable) - mutant_out.expectation_value(observable)))
    assert evidence.signed_shift == pytest.approx(signed, abs=2e-12)
    np.testing.assert_allclose(Operator(candidate.mutant_circuit).data, Operator(independent).data, atol=2e-14)
    # The dual state formula must also agree with the independently evolved
    # state, including the sign fixed by left multiplication.
    prefix = QuantumCircuit(width)
    for item in circuit.data[:boundary]:
        prefix.append(item.operation, [circuit.find_bit(q).index for q in item.qubits])
    tau = state.evolve(prefix).data
    axis_full = embed_local_pauli(axis, (target,), width).to_matrix()
    minus = (tau - axis_full @ tau @ axis_full) / 2
    dual_difference = (1 - np.cos(angle)) * minus - 1j * np.sin(angle) * minus @ axis_full
    direct_local = expm(-.5j * angle * axis_full)
    np.testing.assert_allclose(dual_difference, tau-direct_local @ tau @ direct_local.conj().T, atol=2e-14)
    assert abs(signed) <= problem.test.outcome_range * abs(np.sin(angle / 2)) + 2e-12
