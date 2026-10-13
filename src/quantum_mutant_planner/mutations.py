"""Mutation factories that preserve the manuscript's left-multiplication convention."""

from __future__ import annotations

import numpy as np
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Gate, Instruction
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp

from quantum_mutant_planner.models import MutationCandidate, MutationKind
from quantum_mutant_planner.certificates import _gate_matrix, exact_pauli_rotation


def _instruction_support(circuit: QuantumCircuit, instruction: CircuitInstruction) -> tuple[int, ...]:
    """Return original-circuit indices for an instruction's quantum operands.

    :param circuit: Circuit owning the instruction.
    :param instruction: Instruction whose operands are resolved.
    :return: Ordered tuple of qubit indices.
    """

    return tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)


def _relative_operator(original: Instruction, replacement: Instruction) -> Operator:
    """Compute the local relative unitary ``replacement * original^dagger``.

    :param original: Original gate at the mutation site.
    :param replacement: Mutated gate at the mutation site.
    :raises ValueError: If the gate dimensions differ.
    :return: Local relative unitary implementing left multiplication.
    """

    original_matrix = Operator(original).data
    replacement_matrix = Operator(replacement).data
    if original_matrix.shape != replacement_matrix.shape:
        raise ValueError("original and replacement gates must have the same dimensions")
    return Operator(replacement_matrix @ original_matrix.conj().T)


def replacement_mutation(
    circuit: QuantumCircuit,
    identifier: str,
    site: int,
    replacement: Instruction,
    *,
    cost: float = 1.0,
    allow_spectral: bool = True,
) -> MutationCandidate:
    """Replace one circuit instruction and derive its relative unitary.

    :param circuit: Original circuit.
    :param identifier: Stable mutation identifier.
    :param site: Zero-based instruction index to replace.
    :param replacement: Replacement unitary instruction.
    :param cost: Positive paired-comparison cost.
    :param allow_spectral: Whether the spectral branch is declared tractable.
    :raises IndexError: If ``site`` is outside the circuit.
    :return: Replacement candidate using the left-multiplication convention.
    """

    if site < 0 or site >= len(circuit.data):
        raise IndexError("replacement site is outside the original circuit")
    original = circuit.data[site]
    if replacement.num_qubits != len(original.qubits):
        raise ValueError("original and replacement gates must have the same dimensions")
    mutant = circuit.copy()
    mutant.data[site] = original.replace(operation=replacement)
    return MutationCandidate(
        identifier=identifier,
        mutant_circuit=mutant,
        site=site,
        suffix_start=site + 1,
        support=_instruction_support(circuit, original),
        cost=cost,
        kind=MutationKind.REPLACEMENT,
        relative_operator=_relative_operator(original.operation, replacement),
        allow_spectral=allow_spectral,
    )


def deletion_mutation(
    circuit: QuantumCircuit,
    identifier: str,
    site: int,
    *,
    cost: float = 1.0,
) -> MutationCandidate:
    """Delete one unitary instruction.

    :param circuit: Original circuit.
    :param identifier: Stable mutation identifier.
    :param site: Zero-based instruction index to delete.
    :param cost: Positive paired-comparison cost.
    :raises IndexError: If ``site`` is outside the circuit.
    :return: Deletion candidate with relative unitary ``G_j^dagger``.
    """

    if site < 0 or site >= len(circuit.data):
        raise IndexError("deletion site is outside the original circuit")
    original = circuit.data[site]
    mutant = circuit.copy()
    del mutant.data[site]
    local_dimension = 2 ** len(original.qubits)
    return MutationCandidate(
        identifier=identifier,
        mutant_circuit=mutant,
        site=site,
        suffix_start=site + 1,
        support=_instruction_support(circuit, original),
        cost=cost,
        kind=MutationKind.DELETION,
        relative_operator=Operator(np.eye(local_dimension, dtype=complex) @ Operator(original.operation).data.conj().T),
    )


def insertion_mutation(
    circuit: QuantumCircuit,
    identifier: str,
    boundary: int,
    instruction: Instruction,
    qubits: tuple[int, ...],
    *,
    cost: float = 1.0,
    allow_spectral: bool = True,
) -> MutationCandidate:
    """Insert a unitary instruction at an original-circuit boundary.

    :param circuit: Original circuit.
    :param identifier: Stable mutation identifier.
    :param boundary: Insertion position from zero through the circuit length.
    :param instruction: Inserted unitary instruction.
    :param qubits: Original-circuit qubits changed by the insertion.
    :param cost: Positive paired-comparison cost.
    :param allow_spectral: Whether the spectral branch is declared tractable.
    :raises IndexError: If the boundary or a qubit index is invalid.
    :raises ValueError: If the operand count does not match the instruction.
    :return: Insertion candidate whose relative unitary is the inserted gate.
    """

    if boundary < 0 or boundary > len(circuit.data):
        raise IndexError("insertion boundary is outside the original circuit")
    if any(qubit < 0 or qubit >= circuit.num_qubits for qubit in qubits):
        raise IndexError("insertion qubit is outside the original circuit")
    if len(qubits) != instruction.num_qubits:
        raise ValueError("instruction operand count does not match qubits")
    mutant = circuit.copy()
    qargs = tuple(mutant.qubits[index] for index in qubits)
    mutant.data.insert(boundary, CircuitInstruction(instruction, qargs, ()))
    return MutationCandidate(
        identifier=identifier,
        mutant_circuit=mutant,
        site=boundary,
        suffix_start=boundary,
        support=qubits,
        cost=cost,
        kind=MutationKind.INSERTION,
        relative_operator=Operator(instruction),
        allow_spectral=allow_spectral,
    )


def _rotation_gate(axis: Pauli, angle: float | sp.Expr) -> Gate:
    """Construct ``exp(-i angle A / 2)`` for an involutory Pauli axis.

    :param axis: Local Pauli rotation axis.
    :param angle: Signed numeric angle or explicitly exact SymPy expression.
    :return: Unitary instruction for the Pauli rotation.
    """

    if isinstance(angle, sp.Expr):
        return exact_pauli_rotation(axis, angle)
    axis_matrix = axis.to_matrix()
    dimension = axis_matrix.shape[0]
    matrix = np.cos(angle / 2.0) * np.eye(dimension) - 1j * np.sin(angle / 2.0) * axis_matrix
    return UnitaryGate(matrix, label=f"R_{axis.to_label()}({angle:.6g})")


def pauli_rotation_mutation(
    circuit: QuantumCircuit,
    identifier: str,
    site: int,
    axis: Pauli,
    original_angle: float | sp.Expr,
    mutated_angle: float | sp.Expr,
    *,
    qubits: tuple[int, ...] | None = None,
    insert: bool = False,
    cost: float = 1.0,
) -> MutationCandidate:
    """Create a same-axis Pauli rotation replacement or insertion.

    :param circuit: Original circuit.
    :param identifier: Stable mutation identifier.
    :param site: Original instruction index or insertion boundary.
    :param axis: Local involutory Pauli axis.
    :param original_angle: Original coordinate; exact expressions require a matching exact source gate.
    :param mutated_angle: Mutated coordinate; exact expressions preserve algebraic gate semantics.
    :param qubits: Operand qubits, required for insertion.
    :param insert: Whether ``site`` is an insertion boundary.
    :param cost: Positive paired-comparison cost.
    :raises ValueError: If insertion operands are missing or inconsistent.
    :return: Structured Pauli-rotation candidate.
    """

    if axis.phase % 2:
        raise ValueError("rotation axis must be a Hermitian Pauli")
    if insert and original_angle != 0:
        raise ValueError("a rotation insertion requires original_angle=0")
    eta = mutated_angle - original_angle
    if isinstance(original_angle, sp.Expr) or isinstance(mutated_angle, sp.Expr):
        original_angle = sp.Rational(original_angle) if isinstance(original_angle, float) else sp.sympify(original_angle)
        mutated_angle = sp.Rational(mutated_angle) if isinstance(mutated_angle, float) else sp.sympify(mutated_angle)
        eta = sp.simplify(mutated_angle - original_angle)
        if not insert and 0 <= site < len(circuit.data):
            source = _gate_matrix(circuit.data[site].operation)
            declared = exact_pauli_rotation(axis, original_angle).exact_matrix
            if source is None or (source - declared).applyfunc(sp.simplify) != sp.zeros(declared.rows):
                raise ValueError("exact rotation replacement requires a matching exact original gate")
    replacement = _rotation_gate(axis, mutated_angle)
    if insert:
        if qubits is None:
            raise ValueError("qubits are required for a rotation insertion")
        base = insertion_mutation(
            circuit,
            identifier,
            site,
            replacement,
            qubits,
            cost=cost,
            allow_spectral=False,
        )
    else:
        if 0 <= site < len(circuit.data) and len(circuit.data[site].qubits) != axis.num_qubits:
            raise ValueError("Pauli axis size does not match the mutated gate")
        base = replacement_mutation(
            circuit,
            identifier,
            site,
            replacement,
            cost=cost,
            allow_spectral=False,
        )
    relative = _rotation_gate(axis, eta)
    return MutationCandidate(
        identifier=base.identifier,
        mutant_circuit=base.mutant_circuit,
        site=base.site,
        suffix_start=base.suffix_start,
        support=base.support,
        cost=base.cost,
        kind=MutationKind.PAULI_ROTATION,
        relative_operator=Operator(relative),
        eta=float(eta),
        pauli_axis=axis,
        generator=Operator(axis.to_matrix() / 2.0),
        allow_spectral=False,
        generator_paulis=SparsePauliOp(axis, coeffs=[0.5]),
    )


def hermitian_perturbation_mutation(
    circuit: QuantumCircuit,
    identifier: str,
    site: int,
    generator: Operator | SparsePauliOp,
    eta: float,
    *,
    cost: float = 1.0,
    allow_spectral: bool = False,
) -> MutationCandidate:
    """Left-multiply one original gate by a Hermitian perturbation.

    :param circuit: Original circuit.
    :param identifier: Stable mutation identifier.
    :param site: Zero-based original instruction index.
    :param generator: Local Hermitian generator ``H_m``; explicit Pauli input preserves certificate metadata.
    :param eta: Signed perturbation strength.
    :param cost: Positive paired-comparison cost.
    :param allow_spectral: Whether to prefer exact spectral conjugation.
    :raises IndexError: If ``site`` is outside the circuit.
    :raises ValueError: If the generator is not Hermitian or has the wrong dimension.
    :return: General Hermitian left-multiplication candidate.
    """

    if site < 0 or site >= len(circuit.data):
        raise IndexError("perturbation site is outside the original circuit")
    generator_paulis = generator.copy() if isinstance(generator, SparsePauliOp) else None
    generator = Operator(generator)
    generator_matrix = generator.data
    if not np.allclose(generator_matrix, generator_matrix.conj().T):
        raise ValueError("generator must be Hermitian")
    values, vectors = np.linalg.eigh(generator_matrix)
    relative_matrix = (vectors * np.exp(-1j * eta * values)) @ vectors.conj().T
    original = circuit.data[site]
    original_matrix = Operator(original.operation).data
    if relative_matrix.shape != original_matrix.shape:
        raise ValueError("generator dimension does not match the mutated gate")
    replacement = UnitaryGate(relative_matrix @ original_matrix, label=f"perturb({eta:.6g})")
    base = replacement_mutation(
        circuit,
        identifier,
        site,
        replacement,
        cost=cost,
        allow_spectral=allow_spectral,
    )
    return MutationCandidate(
        identifier=base.identifier,
        mutant_circuit=base.mutant_circuit,
        site=base.site,
        suffix_start=base.suffix_start,
        support=base.support,
        cost=base.cost,
        kind=MutationKind.GENERAL_HERMITIAN,
        relative_operator=Operator(relative_matrix),
        eta=eta,
        generator=generator,
        allow_spectral=allow_spectral,
        generator_paulis=generator_paulis,
    )
