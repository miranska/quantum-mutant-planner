"""Tests for mutation factories and left multiplication."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate, ZGate
from qiskit.quantum_info import Operator, Pauli

from quantum_mutant_planner.models import MutationKind
from quantum_mutant_planner.mutations import (
    deletion_mutation,
    hermitian_perturbation_mutation,
    insertion_mutation,
    pauli_rotation_mutation,
    replacement_mutation,
)


def test_replacement_deletion_and_insertion_factories() -> None:
    """Basic factories should preserve sites, supports, and relative actions."""

    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.cx(0, 1)
    replacement = replacement_mutation(circuit, "replace", 0, ZGate())
    assert replacement.kind is MutationKind.REPLACEMENT
    assert replacement.support == (0,)
    expected = Operator(ZGate()).data @ Operator(XGate()).data.conj().T
    assert np.allclose(replacement.relative_operator.data, expected)

    deletion = deletion_mutation(circuit, "delete", 1)
    assert deletion.kind is MutationKind.DELETION
    assert deletion.suffix_start == 2
    assert len(deletion.mutant_circuit.data) == 1

    insertion = insertion_mutation(circuit, "insert", 1, ZGate(), (1,))
    assert insertion.kind is MutationKind.INSERTION
    assert insertion.suffix_start == 1
    assert insertion.mutant_circuit.data[1].operation.name == "z"


@pytest.mark.parametrize("site", [-1, 2])
def test_replacement_and_deletion_reject_invalid_sites(site: int) -> None:
    """Replacement-style factories should reject out-of-range sites."""

    circuit = QuantumCircuit(1)
    circuit.x(0)
    with pytest.raises(IndexError):
        replacement_mutation(circuit, "replace", site, ZGate())
    with pytest.raises(IndexError):
        deletion_mutation(circuit, "delete", site)


def test_replacement_rejects_mismatched_gate_dimension() -> None:
    """A replacement must act on the same number of qubits."""

    circuit = QuantumCircuit(1)
    circuit.x(0)
    with pytest.raises(ValueError, match="same dimensions"):
        replacement_mutation(circuit, "bad", 0, Operator(np.eye(4)).to_instruction())


def test_insertion_rejects_invalid_inputs() -> None:
    """Insertion boundaries, operands, and widths should be checked."""

    circuit = QuantumCircuit(1)
    with pytest.raises(IndexError, match="boundary"):
        insertion_mutation(circuit, "bad", 1, XGate(), (0,))
    with pytest.raises(IndexError, match="qubit"):
        insertion_mutation(circuit, "bad", 0, XGate(), (1,))
    with pytest.raises(ValueError, match="operand"):
        insertion_mutation(circuit, "bad", 0, XGate(), ())


def test_pauli_rotation_replacement_and_insertion() -> None:
    """Pauli factories should preserve signed coordinate offsets."""

    circuit = QuantumCircuit(1)
    circuit.ry(0.4, 0)
    replacement = pauli_rotation_mutation(circuit, "r", 0, Pauli("Y"), 0.4, 0.7)
    assert replacement.kind is MutationKind.PAULI_ROTATION
    assert replacement.eta == pytest.approx(0.3)
    insertion = pauli_rotation_mutation(
        circuit,
        "i",
        1,
        Pauli("X"),
        0.0,
        np.pi,
        qubits=(0,),
        insert=True,
    )
    assert insertion.suffix_start == 1
    with pytest.raises(ValueError, match="qubits"):
        pauli_rotation_mutation(circuit, "bad", 1, Pauli("X"), 0.0, 0.1, insert=True)
    with pytest.raises(ValueError, match="axis size"):
        pauli_rotation_mutation(circuit, "bad", 0, Pauli("XX"), 0.4, 0.5)


def test_hermitian_perturbation_preserves_left_multiplication() -> None:
    """The constructed replacement should equal exp(-i eta H) times the gate."""

    circuit = QuantumCircuit(1)
    circuit.h(0)
    generator = Operator(Pauli("Z"))
    candidate = hermitian_perturbation_mutation(circuit, "h", 0, generator, 0.2)
    relative = np.diag([np.exp(-0.2j), np.exp(0.2j)])
    assert candidate.kind is MutationKind.GENERAL_HERMITIAN
    assert np.allclose(candidate.relative_operator.data, relative)
    assert np.allclose(
        Operator(candidate.mutant_circuit.data[0].operation).data,
        relative @ Operator(circuit.data[0].operation).data,
    )


def test_hermitian_perturbation_rejects_invalid_generator() -> None:
    """General perturbations require a same-width Hermitian generator."""

    circuit = QuantumCircuit(1)
    circuit.x(0)
    with pytest.raises(IndexError):
        hermitian_perturbation_mutation(circuit, "bad", 2, Operator(Pauli("Z")), 0.1)
    with pytest.raises(ValueError, match="Hermitian"):
        hermitian_perturbation_mutation(circuit, "bad", 0, Operator([[0, 1], [0, 0]]), 0.1)
    with pytest.raises(ValueError, match="dimension"):
        hermitian_perturbation_mutation(circuit, "bad", 0, Operator(np.eye(4)), 0.1)
