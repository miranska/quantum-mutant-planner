"""Small-circuit algebraic certificates from explicitly exact input data.

This path compares the two complete circuits independently of the planner's
local-response formulas. Floating-point matrices are never rationalized into
purported intended exact gates or states.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction
from qiskit.circuit.library import (
    CXGate, CYGate, CZGate, HGate, IGate, SGate, SdgGate,
    SwapGate, SXGate, SXdgGate, UnitaryGate, XGate, YGate, ZGate,
)
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector
import sympy as sp

from quantum_mutant_planner.models import MutationCandidate, PlanningProblem


@lru_cache(maxsize=4096)
def _simplify_scalar(expression: sp.Expr) -> sp.Expr:
    """Reuse exact simplifications of immutable scalar expressions.

    :param expression: Symbolic scalar, never a numerical equality test.
    :return: The result of SymPy's exact simplifier.
    """

    return sp.simplify(expression)


def _simplify(matrix: sp.MatrixBase) -> sp.ImmutableMatrix:
    """Simplify every entry without numerical equality heuristics.

    :param matrix: Exact matrix expression.
    :return: Immutable matrix with simplified entries.
    """

    return sp.ImmutableMatrix(matrix.applyfunc(_simplify_scalar))


def _exact_matrix(data: object) -> sp.ImmutableMatrix:
    """Validate concrete exact scalar entries.

    :param data: Matrix-like exact values.
    :raises ValueError: If entries contain floats, free symbols, or nonfinite values.
    :return: Immutable exact matrix.
    """

    matrix = sp.ImmutableMatrix(data)
    if matrix.has(sp.Float) or matrix.free_symbols or any(v.is_finite is not True for v in matrix):
        raise ValueError("exact data must be finite, bound expressions without floating-point entries")
    return matrix


def _numeric(matrix: sp.MatrixBase) -> np.ndarray:
    """Project exact expressions to the numerical simulation representation.

    :param matrix: Concrete exact matrix.
    :return: Complex floating-point matrix.
    """

    return np.array(matrix.evalf(), dtype=complex)


class ExactUnitaryGate(Gate):
    """A unitary with an authoritative exact matrix and a numerical projection.

    :param matrix: Concrete exact unitary matrix in Qiskit local qubit order.
    :param label: Optional display label.
    """

    def __init__(self, matrix: object, label: str | None = None) -> None:
        """Check exact unitarity before constructing the numerical gate.

        :param matrix: Concrete exact unitary matrix.
        :param label: Optional display label.
        :raises ValueError: If dimensions or exact unitarity cannot be established.
        :return: None.
        """

        exact = _exact_matrix(matrix)
        dimension = exact.rows
        if dimension < 2 or dimension & (dimension - 1) or exact.cols != dimension:
            raise ValueError("exact gate dimension must be a square power of two")
        if _simplify(exact.H * exact) != sp.eye(dimension):
            raise ValueError("exact gate unitarity could not be established")
        super().__init__("exact_unitary", dimension.bit_length() - 1, [], label=label)
        self.exact_matrix = exact
        self._numeric_matrix = _numeric(exact)

    def to_matrix(self) -> np.ndarray:
        """Expose the numerical projection for ordinary Qiskit simulation.

        :return: Floating-point gate matrix.
        """

        return self._numeric_matrix

    def _define(self) -> None:
        """Provide a standard unitary decomposition for transpilation.

        :return: None.
        """

        circuit = QuantumCircuit(self.num_qubits)
        circuit.append(UnitaryGate(self.to_matrix()), range(self.num_qubits))
        self.definition = circuit


class ExactStatevector(Statevector):
    """An explicitly exact normalized pure input with a numerical projection.

    Qiskit copies and numerical transformations discard the exact declaration.

    :param amplitudes: Concrete exact amplitudes in Qiskit basis order.
    """

    def __init__(self, amplitudes: object) -> None:
        """Validate an exact pure input state.

        :param amplitudes: Concrete exact column-vector amplitudes.
        :raises ValueError: If the vector is not normalized exactly.
        :return: None.
        """

        exact = _exact_matrix(amplitudes)
        if exact.cols != 1 or sp.simplify((exact.H * exact)[0]) != 1:
            raise ValueError("exact statevector must be a normalized column vector")
        super().__init__(_numeric(exact).ravel())
        self.exact_amplitudes = exact

    def __copy__(self) -> Statevector:
        """Drop exact metadata before Qiskit numerically transforms a copy.

        :return: Ordinary numerical statevector with independent storage.
        """

        return Statevector(self.data.copy(), dims=self.dims())

    def __deepcopy__(self, memo: dict[int, object]) -> Statevector:
        """Keep deep copies subject to the same numerical-only semantics.

        :param memo: Python copy protocol's object memo.
        :return: Ordinary numerical statevector with independent storage.
        """

        return self.__copy__()


class ExactDensityMatrix(DensityMatrix):
    """An explicitly exact mixed input with a numerical projection.

    Qiskit copies and numerical transformations discard the exact declaration.

    :param matrix: Exact, normalized, positive-semidefinite density matrix.
    """

    def __init__(self, matrix: object) -> None:
        """Validate an exact mixed input state.

        :param matrix: Concrete exact density matrix.
        :raises ValueError: If density-matrix validity cannot be established exactly.
        :return: None.
        """

        exact = _exact_matrix(matrix)
        if (exact.rows != exact.cols or _simplify(exact - exact.H) != sp.zeros(exact.rows)
                or sp.simplify(sp.trace(exact)) != 1 or exact.is_positive_semidefinite is not True):
            raise ValueError("exact density matrix must be Hermitian, positive semidefinite, and trace one")
        super().__init__(_numeric(exact))
        self.exact_density = exact

    def __copy__(self) -> DensityMatrix:
        """Drop exact metadata before Qiskit numerically transforms a copy.

        :return: Ordinary numerical density matrix with independent storage.
        """

        return DensityMatrix(self.data.copy(), dims=self.dims())

    def __deepcopy__(self, memo: dict[int, object]) -> DensityMatrix:
        """Keep deep copies subject to the same numerical-only semantics.

        :param memo: Python copy protocol's object memo.
        :return: Ordinary numerical density matrix with independent storage.
        """

        return self.__copy__()


def exact_pauli_rotation(axis: Pauli, angle: sp.Expr) -> ExactUnitaryGate:
    """Construct an exact Pauli rotation without recognizing rounded angles.

    :param axis: Hermitian involutory Pauli, including a possible minus sign.
    :param angle: Concrete exact angle, such as ``sympy.pi / 2``.
    :raises ValueError: If the angle or axis is not admissible exact input.
    :return: Gate with exact ``exp(-i angle axis / 2)`` semantics.
    """

    theta = _exact_matrix([[angle]])[0]
    if theta.is_real is not True or axis.phase % 2:
        raise ValueError("a real angle and Hermitian Pauli axis are required")
    matrix = _pauli_matrix(axis)
    return ExactUnitaryGate(
        sp.cos(theta / 2) * sp.eye(matrix.rows) - sp.I * sp.sin(theta / 2) * matrix,
        label=f"R_{axis.to_label()}({theta})",
    )


_I = sp.eye(2)
_X = sp.Matrix([[0, 1], [1, 0]])
_Y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
_Z = sp.diag(1, -1)
_GATES = {
    IGate: _I, XGate: _X, YGate: _Y, ZGate: _Z,
    HGate: sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2),
    SGate: sp.diag(1, sp.I), SdgGate: sp.diag(1, -sp.I),
    SXGate: ((_I + _X) + sp.I * (_I - _X)) / 2,
    SXdgGate: ((_I + _X) - sp.I * (_I - _X)) / 2,
    CXGate: sp.kronecker_product(_I, sp.diag(1, 0)) + sp.kronecker_product(_X, sp.diag(0, 1)),
    CYGate: sp.kronecker_product(_I, sp.diag(1, 0)) + sp.kronecker_product(_Y, sp.diag(0, 1)),
    CZGate: sp.diag(1, 1, 1, -1),
    SwapGate: sp.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
}


def _pauli_matrix(pauli: Pauli) -> sp.ImmutableMatrix:
    """Construct a Pauli exactly, including its phase.

    :param pauli: Qiskit Pauli with at least one qubit.
    :return: Exact matrix in Qiskit basis order.
    """

    unsigned = Pauli((pauli.z, pauli.x)).to_label()
    matrices = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
    return sp.ImmutableMatrix((-sp.I) ** int(pauli.phase) * sp.kronecker_product(*(matrices[c] for c in unsigned)))


def _gate_matrix(gate: Instruction) -> sp.ImmutableMatrix | None:
    """Read exact gate semantics without guessing from numerical matrices.

    :param gate: Original or mutated circuit instruction.
    :return: Exact matrix, or None for unsupported or modified annotations.
    """

    if isinstance(gate, ExactUnitaryGate):
        if np.array_equal(gate.to_matrix(), _numeric(gate.exact_matrix)):
            return gate.exact_matrix
        return None
    if hasattr(gate, "ctrl_state") and gate.ctrl_state != (1 << gate.num_ctrl_qubits) - 1:
        return None
    matrix = _GATES.get(gate.base_class)
    return None if matrix is None else sp.ImmutableMatrix(matrix)


def _embed(matrix: sp.MatrixBase, support: tuple[int, ...], width: int) -> sp.ImmutableMatrix:
    """Embed a local matrix by basis indices, independently of Qiskit compose.

    :param matrix: Exact local operator.
    :param support: Full-register operands in local little-endian order.
    :param width: Number of circuit qubits.
    :return: Exact full-register matrix.
    """

    result = sp.zeros(2**width)
    mask = sum(1 << q for q in support)
    for column in range(2**width):
        local_column = sum(((column >> q) & 1) << k for k, q in enumerate(support))
        for local_row in range(matrix.rows):
            row = (column & ~mask) | sum(((local_row >> k) & 1) << q for k, q in enumerate(support))
            result[row, column] = matrix[local_row, local_column]
    return sp.ImmutableMatrix(result)


def _circuit_matrix(circuit: QuantumCircuit) -> sp.ImmutableMatrix | None:
    """Compose supported exact gates in circuit order.

    Revalidate every gate before looking up the immutable contents in the cache.
    Circuit identity alone cannot detect edits or stale numerical projections.

    :param circuit: Circuit with explicit exact or recognized discrete gates.
    :return: Exact unitary up to irrelevant global phase, or None.
    """

    operations = []
    for item in circuit.data:
        matrix = _gate_matrix(item.operation)
        if matrix is None:
            return None
        support = tuple(circuit.find_bit(q).index for q in item.qubits)
        operations.append((sp.ImmutableMatrix(matrix), support))
    return _compose_circuit(circuit.num_qubits, tuple(operations))


@lru_cache(maxsize=64)
def _compose_circuit(
    width: int, operations: tuple[tuple[sp.ImmutableMatrix, tuple[int, ...]], ...],
) -> sp.ImmutableMatrix:
    """Cache full-circuit compositions by exact gates, operands, and width.

    :param width: Full register width.
    :param operations: Validated immutable gate matrices and ordered supports.
    :return: Exact unitary, ignoring only the circuit's global phase.
    """

    result = sp.eye(2**width)
    for matrix, support in operations:
        result = _embed(matrix, support, width) * result
    return _simplify(result)


@lru_cache(maxsize=128)
def _conjugated_observable(
    unitary: sp.ImmutableMatrix, observable: sp.ImmutableMatrix,
) -> sp.ImmutableMatrix:
    """Reuse full-circuit Heisenberg products without changing their algebra.

    :param unitary: Validated exact circuit unitary.
    :param observable: Complete observable with exact binary-rational coefficients.
    :return: Exact product ``U.H * O * U``, before difference simplification.
    """

    return sp.ImmutableMatrix(unitary.H * observable * unitary)


def _input_density(state: Statevector | DensityMatrix) -> sp.ImmutableMatrix | None:
    """Recover only explicitly exact or exactly represented basis inputs.

    :param state: Input state, before circuit evolution.
    :return: Exact density matrix, or None when exact semantics are unavailable.
    """

    if isinstance(state, ExactStatevector):
        if np.array_equal(state.data, _numeric(state.exact_amplitudes).ravel()):
            return sp.ImmutableMatrix(state.exact_amplitudes * state.exact_amplitudes.H)
        return None
    if isinstance(state, ExactDensityMatrix):
        if np.array_equal(state.data, _numeric(state.exact_density)):
            return state.exact_density
        return None
    density = DensityMatrix(state).data
    diagonal = np.diag(density)
    if np.count_nonzero(diagonal == 1) == 1 and np.count_nonzero(density) == 1:
        return sp.ImmutableMatrix(np.real(density).astype(int))
    return None


def certify_channel_equivalence(
    original: QuantumCircuit, mutant: QuantumCircuit, maximum_qubits: int,
) -> bool:
    """Prove equality of the original channels independently of transpilation.

    Only supported exact gate semantics are used, without rounding numerical
    parameters or comparing a selected input state or observable.
    An unavailable proof returns False, which does not prove non-equivalence.

    :param original: Original unitary circuit.
    :param mutant: Mutated unitary circuit on the same ordered qubits.
    :param maximum_qubits: Dense symbolic width limit; zero disables this path.
    :return: True only if the relative unitary is exactly a scalar phase.
    """

    width = original.num_qubits
    if maximum_qubits <= 0 or width > maximum_qubits or mutant.num_qubits != width:
        return False
    if original.num_clbits or mutant.num_clbits:
        return False
    original_matrix = _circuit_matrix(original)
    mutant_matrix = _circuit_matrix(mutant)
    if original_matrix is None or mutant_matrix is None:
        return False
    relative = _simplify(mutant_matrix * original_matrix.H)
    phase = relative[0, 0]
    return relative == phase * sp.eye(2**width) and _simplify_scalar(phase * sp.conjugate(phase)) == 1


def certify_circuit_zero(
    problem: PlanningProblem, candidate: MutationCandidate, maximum_qubits: int,
) -> str | None:
    """Prove a zero from full-circuit algebra, separately from local formulas.

    :param problem: Declared original circuit and fixed test.
    :param candidate: Mutated circuit to compare against the original.
    :param maximum_qubits: Dense symbolic width limit; zero disables this path.
    :return: Certificate provenance only when symbolic reduction proves zero.
    """

    width = problem.circuit.num_qubits
    if width > maximum_qubits:
        return None
    original = _circuit_matrix(problem.circuit)
    mutant = _circuit_matrix(candidate.mutant_circuit)
    if original is None or mutant is None:
        return None
    observable = sp.zeros(2**width)
    for pauli, weight in zip(problem.test.observable.paulis, problem.test.observable.coeffs, strict=True):
        # These are the supplied coefficient values, interpreted as exact binary
        # rationals, never rounded to a simpler intended value or dropped.
        value = complex(weight)
        if not np.isfinite(value):
            return None
        observable += (sp.Rational(value.real) + sp.I * sp.Rational(value.imag)) * _pauli_matrix(pauli)
    if _simplify(observable - observable.H) != sp.zeros(2**width):
        return None
    exact_observable = sp.ImmutableMatrix(observable)
    difference = _simplify(
        _conjugated_observable(original, exact_observable)
        - _conjugated_observable(mutant, exact_observable)
    )
    if difference == sp.zeros(2**width):
        return "algebraic observable-invariance certificate (full-circuit exact arithmetic)"
    density = _input_density(problem.test.state)
    if density is not None and _simplify_scalar(sp.trace(difference * density)) == 0:
        return "state-dependent cancellation certificate (full-circuit exact arithmetic)"
    return None
