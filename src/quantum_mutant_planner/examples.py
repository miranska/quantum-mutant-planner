"""Registered, hand-checkable example planning problems."""

from __future__ import annotations

import sympy as sp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector

from quantum_mutant_planner.models import ExecutionBudget, PlanningProblem, QuantumTest
from quantum_mutant_planner.mutations import deletion_mutation, pauli_rotation_mutation
from quantum_mutant_planner.certificates import exact_pauli_rotation


def build_ghz_problem() -> PlanningProblem:
    """Build the manuscript's eight-candidate three-qubit GHZ walkthrough.

    :return: Fixed-test planning problem with threshold and budget from the manuscript.
    """

    circuit = QuantumCircuit(3, name="ghz-planning")
    circuit.append(exact_pauli_rotation(Pauli("Y"), sp.pi / 2), [0])
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    end = len(circuit.data)
    candidates = (
        deletion_mutation(circuit, "m1", 2),
        pauli_rotation_mutation(circuit, "m2", end, Pauli("X"), 0, sp.pi, qubits=(2,), insert=True),
        pauli_rotation_mutation(circuit, "m3", end, Pauli("Z"), 0, sp.pi, qubits=(1,), insert=True),
        pauli_rotation_mutation(circuit, "m4", end, Pauli("X"), 0, sp.pi, qubits=(1,), insert=True),
        pauli_rotation_mutation(circuit, "m5", 0, Pauli("Y"), sp.pi / 2, -sp.pi / 2),
        pauli_rotation_mutation(circuit, "m6", 0, Pauli("Y"), sp.pi / 2, sp.pi / 2 + sp.Rational(1, 100)),
        pauli_rotation_mutation(circuit, "m7", 0, Pauli("Y"), sp.pi / 2, sp.pi),
        deletion_mutation(circuit, "m8", 1),
    )
    test = QuantumTest(
        state=Statevector.from_int(0, 2**3),
        observable=SparsePauliOp("IZI"),
        threshold=0.1,
        outcome_range=2.0,
    )
    return PlanningProblem(
        circuit=circuit,
        test=test,
        candidates=candidates,
        budget=ExecutionBudget(copies=10_000, error_probability=0.05),
    )


def build_scenario(name: str) -> PlanningProblem:
    """Resolve a registered command-line scenario by name.

    :param name: Registered scenario name.
    :raises ValueError: If the scenario name is unknown.
    :return: Requested planning problem.
    """

    if name == "ghz":
        return build_ghz_problem()
    raise ValueError(f"unknown scenario: {name}")
