"""Circuit, observable, and state analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (
    CXGate, CYGate, CZGate, HGate, IGate, SGate, SdgGate,
    SwapGate, SXGate, SXdgGate, UnitaryGate, XGate, YGate, ZGate,
)
from qiskit.quantum_info import Clifford, Operator, Pauli, SparsePauliOp

from quantum_mutant_planner.certificates import certify_channel_equivalence, certify_local_channel_equivalence
from quantum_mutant_planner.pauli import ResourceLimitError, local_conjugation, observable_range
from quantum_mutant_planner.models import (
    MutationCandidate,
    PlanningProblem,
    QuantumState,
    TranspilerSettings,
)


@dataclass(frozen=True, slots=True)
class PropagationResult:
    """Sparse propagation with an optional, lazily evaluated dense fallback.

    :param retained_paulis: Term-capped Pauli approximation for pilots.
    :param truncation_error: Accumulated discarded coefficient-mass bound.
    :param algebraically_complete: Whether complete discrete Clifford maps were used without truncation.
    :param symbolic_paulis: Complete discrete Clifford image, including tiny terms.
    :param complete_paulis: Complete numerical expansion when it fits the term cap.
    :param circuit: Source circuit for bounded dense fallback.
    :param observable: Source observable for bounded dense fallback.
    :param suffix_start: First instruction in the propagated suffix.
    :param dense_max_qubits: Maximum register width for the dense fallback.
    """

    retained_paulis: SparsePauliOp
    truncation_error: float
    algebraically_complete: bool
    symbolic_paulis: SparsePauliOp | None
    complete_paulis: SparsePauliOp | None
    circuit: QuantumCircuit
    observable: SparsePauliOp
    suffix_start: int
    dense_max_qubits: int

    @property
    def exact_operator(self) -> Operator | None:
        """Materialize the complete dense observable only within the width limit.

        :return: Dense numerical observable, or None above the configured limit.
        """

        if self.circuit.num_qubits > self.dense_max_qubits:
            return None
        matrix = self.observable.to_matrix()
        for instruction in reversed(self.circuit.data[self.suffix_start:]):
            support = _qubit_indices(self.circuit, instruction.qubits)
            full = embed_local_operator(Operator(instruction.operation), support, self.circuit.num_qubits).data
            matrix = full.conj().T @ matrix @ full
        return Operator(matrix)


def _qubit_indices(circuit: QuantumCircuit, qubits: tuple[object, ...]) -> tuple[int, ...]:
    """Resolve circuit qubit objects to integer indices.

    :param circuit: Circuit owning the qubits.
    :param qubits: Qubit objects to resolve.
    :return: Ordered integer indices.
    """

    return tuple(circuit.find_bit(qubit).index for qubit in qubits)


def validate_problem(problem: PlanningProblem, dense_max_qubits: int = 10, local_max_qubits: int = 4) -> None:
    """Validate the ideal-unitary Algorithm 1 input contract.

    :param problem: Complete planning problem.
    :param dense_max_qubits: Largest support used for dense observable validation.
    :param local_max_qubits: Largest gate matrix used for unitary validation.
    :raises ValueError: If any model, dimension, threshold, budget, or candidate invariant fails.
    :return: None.
    """

    circuit = problem.circuit
    test = problem.test
    if circuit.num_clbits or any(instruction.clbits for instruction in circuit.data):
        raise ValueError("classical bits and measurement operations are outside the unitary model")
    for instruction in circuit.data:
        if instruction.operation.num_qubits > local_max_qubits:
            raise ResourceLimitError("gate validation exceeds local_max_qubits")
        try:
            matrix = Operator(instruction.operation).data
        except Exception as exc:
            raise ValueError(f"nonunitary or unsupported instruction: {instruction.operation.name}") from exc
        if matrix.shape[0] != matrix.shape[1] or not np.allclose(matrix.conj().T @ matrix, np.eye(matrix.shape[0])):
            raise ValueError(f"instruction is not unitary: {instruction.operation.name}")
    if test.state.num_qubits != circuit.num_qubits:
        raise ValueError("input-state qubit count does not match the circuit")
    if test.observable.num_qubits != circuit.num_qubits:
        raise ValueError("observable qubit count does not match the circuit")
    coefficients = test.observable.simplify(atol=0.0, rtol=0.0).coeffs
    if not np.all(np.isfinite(coefficients)) or not np.allclose(coefficients.imag, 0.0):
        raise ValueError("observable must be Hermitian")
    if test.threshold <= 0.0:
        raise ValueError("test threshold must be positive")
    if test.outcome_range <= 0.0:
        raise ValueError("observable outcome range must be positive")
    spectral_range = observable_range(test.observable, dense_max_qubits)
    if not np.isclose(test.outcome_range, spectral_range, rtol=1e-12, atol=1e-12):
        raise ValueError("declared outcome range does not match the observable spectrum")
    if problem.budget.copies <= 0:
        raise ValueError("execution copies must be positive")
    if not 0.0 < problem.budget.error_probability < 0.5:
        raise ValueError("error probability must lie strictly between zero and one half")
    identifiers = [candidate.identifier for candidate in problem.candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate identifiers must be unique")
    for candidate in problem.candidates:
        if candidate.cost <= 0.0:
            raise ValueError(f"candidate {candidate.identifier} has nonpositive cost")
        if not candidate.support or any(index < 0 or index >= circuit.num_qubits for index in candidate.support):
            raise ValueError(f"candidate {candidate.identifier} has invalid support")
        if candidate.suffix_start < 0 or candidate.suffix_start > len(circuit.data):
            raise ValueError(f"candidate {candidate.identifier} has invalid suffix boundary")
        if candidate.mutant_circuit.num_qubits != circuit.num_qubits:
            raise ValueError(f"candidate {candidate.identifier} changes the circuit width")
        if any(item.operation.num_qubits > local_max_qubits for item in candidate.mutant_circuit.data):
            raise ResourceLimitError(f"candidate {candidate.identifier} gate validation exceeds local_max_qubits")


def observable_support(observable: SparsePauliOp, tolerance: float = 0.0) -> set[int]:
    """Return all qubits with nonidentity support in a Pauli observable.

    :param observable: Observable in a complete Pauli representation.
    :param tolerance: Coefficient magnitude below which a term is ignored.
    :return: Set of supported Qiskit qubit indices.
    """

    support: set[int] = set()
    for pauli, coefficient in zip(observable.paulis, observable.coeffs, strict=True):
        if abs(coefficient) <= tolerance:
            continue
        label = pauli.to_label()
        for index, symbol in enumerate(reversed(label)):
            if symbol != "I":
                support.add(index)
    return support


def structural_backward_cones(circuit: QuantumCircuit, observable: SparsePauliOp) -> dict[int, frozenset[int]]:
    """Compute exact structural backward-light-cone support at every boundary.

    :param circuit: Original unitary circuit.
    :param observable: Complete output observable.
    :return: Map from suffix-start boundary to active qubit support.
    """

    active = observable_support(observable)
    cones: dict[int, frozenset[int]] = {len(circuit.data): frozenset(active)}
    for index in range(len(circuit.data) - 1, -1, -1):
        instruction = circuit.data[index]
        operands = set(_qubit_indices(circuit, instruction.qubits))
        if active.intersection(operands):
            active.update(operands)
        cones[index] = frozenset(active)
    return cones


def embed_local_operator(operator: Operator, support: tuple[int, ...], num_qubits: int) -> Operator:
    """Embed a local operator into the full Qiskit register order.

    :param operator: Local operator acting on ``support`` in its listed order.
    :param support: Full-register qubit indices.
    :param num_qubits: Full register width.
    :raises ValueError: If the operator width and support length differ.
    :return: Full-register operator.
    """

    if operator.num_qubits != len(support):
        raise ValueError("local operator width does not match its support")
    return Operator(np.eye(2**num_qubits)).compose(operator, qargs=list(support))


def embed_local_pauli(pauli: Pauli, support: tuple[int, ...], num_qubits: int) -> Pauli:
    """Embed a local Pauli into the full Qiskit register order.

    :param pauli: Local Pauli whose qubit zero is the label's rightmost symbol.
    :param support: Full-register qubit indices in local operand order.
    :param num_qubits: Full register width.
    :raises ValueError: If the Pauli width and support length differ.
    :return: Full-register Pauli.
    """

    if pauli.num_qubits != len(support):
        raise ValueError("local Pauli width does not match its support")
    full_symbols = ["I"] * num_qubits
    for local_index, full_index in enumerate(support):
        has_x = bool(pauli.x[local_index])
        has_z = bool(pauli.z[local_index])
        full_symbols[full_index] = "Y" if has_x and has_z else "X" if has_x else "Z" if has_z else "I"
    embedded = Pauli("".join(reversed(full_symbols)))
    embedded.phase = pauli.phase
    return embedded


def _is_clifford_instruction(operation: object) -> bool:
    """Return whether Qiskit recognizes an operation as Clifford numerically.

    :param operation: Candidate Qiskit operation.
    :return: True when Qiskit accepts the operation under its recognition tolerance.
    """

    try:
        Clifford(operation)
    except Exception:
        return False
    return True


def _truncate_paulis(
    operator: SparsePauliOp, maximum_terms: int, tolerance: float = 0.0,
) -> tuple[SparsePauliOp, float]:
    """Retain the largest Pauli coefficients and bound discarded norm.

    :param operator: Pauli operator to truncate.
    :param maximum_terms: Positive retained-term cap.
    :param tolerance: Coefficient cutoff whose discarded mass is included in the bound.
    :return: Retained operator and discarded absolute coefficient mass.
    """

    order = np.argsort(np.abs(operator.coeffs))[::-1]
    kept = order[np.abs(operator.coeffs[order]) > tolerance][:maximum_terms]
    discarded = np.setdiff1d(order, kept)
    retained = (
        SparsePauliOp(operator.paulis[kept], operator.coeffs[kept])
        if len(kept)
        else SparsePauliOp("I" * operator.num_qubits, coeffs=[0.0])
    )
    return retained, float(np.sum(np.abs(operator.coeffs[discarded])))


def propagate_observable(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    suffix_start: int,
    maximum_terms: int,
    tolerance: float,
    *,
    dense_max_qubits: int = 10,
    local_max_qubits: int = 4,
) -> PropagationResult:
    """Propagate an observable backward through one original-circuit suffix.

    :param circuit: Original circuit.
    :param observable: Complete output observable.
    :param suffix_start: First suffix instruction after the mutation action.
    :param maximum_terms: Retained Pauli-term cap for the pilot representation.
    :param tolerance: Numerical simplification tolerance.
    :param dense_max_qubits: Register limit for optional dense fallback.
    :param local_max_qubits: Support limit for numerical local conjugation.
    :raises ValueError: If the term cap is not positive.
    :return: Sparse propagated observables and a lazy bounded dense fallback.
    """

    if maximum_terms <= 0:
        raise ValueError("maximum Pauli terms must be positive")
    complete: SparsePauliOp | None = observable.copy() if len(observable) <= maximum_terms else None
    symbolic: SparsePauliOp | None = observable.copy()
    retained, truncation_error = _truncate_paulis(observable, maximum_terms, tolerance)
    symbolic_gate_types = (
        IGate, XGate, YGate, ZGate, HGate, SGate, SdgGate, SXGate, SXdgGate,
        CXGate, CYGate, CZGate, SwapGate,
    )
    for instruction in reversed(circuit.data[suffix_start:]):
        support = _qubit_indices(circuit, instruction.qubits)
        # Only discrete gate types provide symbolic certificates. Matrix-based
        # Clifford recognition can round a small non-Clifford rotation to zero.
        if instruction.operation.base_class in symbolic_gate_types:
            clifford = Clifford(instruction.operation)
            retained = SparsePauliOp(retained.paulis.evolve(clifford, qargs=list(support)), retained.coeffs)
            if symbolic is not None:
                symbolic = SparsePauliOp(symbolic.paulis.evolve(clifford, qargs=list(support)), symbolic.coeffs)
            if complete is not None:
                complete = SparsePauliOp(complete.paulis.evolve(clifford, qargs=list(support)), complete.coeffs)
        else:
            symbolic = None
            if len(support) > local_max_qubits:
                raise ResourceLimitError("suffix gate exceeds local_max_qubits")
            local = Operator(instruction.operation)
            retained = local_conjugation(retained, local, support, local_max_qubits)
            if complete is not None:
                complete = local_conjugation(complete, local, support, local_max_qubits)
                if len(complete) > maximum_terms:
                    complete = None
        retained, discarded = _truncate_paulis(retained, maximum_terms, tolerance)
        truncation_error += discarded
    return PropagationResult(
        retained_paulis=retained,
        truncation_error=truncation_error,
        algebraically_complete=symbolic is not None and truncation_error == 0.0,
        symbolic_paulis=symbolic,
        complete_paulis=complete,
        circuit=circuit.copy(),
        observable=observable.copy(),
        suffix_start=suffix_start,
        dense_max_qubits=dense_max_qubits,
    )


def state_at_mutation(problem: PlanningProblem, candidate: MutationCandidate) -> QuantumState:
    """Evolve the original input through the prefix ending at the mutation action.

    :param problem: Complete planning problem.
    :param candidate: Candidate determining the prefix boundary.
    :return: Original post-gate state ``tau_j`` or insertion-boundary state.
    """

    state: QuantumState = problem.test.state.copy()
    for instruction in problem.circuit.data[: candidate.suffix_start]:
        qargs = list(_qubit_indices(problem.circuit, instruction.qubits))
        state = state.evolve(instruction.operation, qargs=qargs)
    return state


def real_expectation(state: QuantumState, operator: Operator | SparsePauliOp, tolerance: float) -> float:
    """Evaluate a Hermitian expectation and reject a material imaginary residual.

    :param state: State on which to evaluate the operator.
    :param operator: Hermitian operator.
    :param tolerance: Maximum accepted imaginary residual.
    :raises ValueError: If the expectation has a material imaginary part.
    :return: Real expectation value.
    """

    value = complex(state.expectation_value(operator))
    if abs(value.imag) > tolerance:
        raise ValueError("Hermitian expectation has a material imaginary component")
    return float(value.real)


def direct_signed_gap(problem: PlanningProblem, candidate: MutationCandidate, tolerance: float) -> tuple[float, float]:
    """Compute the ideal signed shift and gap by direct full-circuit simulation.

    :param problem: Complete planning problem.
    :param candidate: Candidate mutant circuit.
    :param tolerance: Imaginary-residual tolerance.
    :return: Signed original-minus-mutant shift and its absolute value.
    """

    original = problem.test.state.evolve(problem.circuit)
    mutant = problem.test.state.evolve(candidate.mutant_circuit)
    observable = problem.test.observable
    original_mean = real_expectation(original, observable, tolerance)
    mutant_mean = real_expectation(mutant, observable, tolerance)
    signed = original_mean - mutant_mean
    return signed, abs(signed)


def transpiler_equivalent(
    original: QuantumCircuit,
    mutant: QuantumCircuit,
    settings: TranspilerSettings,
    *,
    maximum_qubits: int = 4,
    local_maximum_qubits: int | None = None,
) -> bool:
    """Verify a pinned transpiler match with independent exact channel algebra.

    Qiskit's synthesis, optimization, and circuit equality can use numerical
    tolerances even at approximation_degree=1.0. A compiled match alone is
    therefore never an exact semantic certificate. Unsupported or over-limit
    source circuits conservatively fail this sufficient, incomplete check.

    :param original: Original circuit.
    :param mutant: Mutant circuit.
    :param settings: Pinned transpiler settings.
    :param maximum_qubits: Register limit for the dense exact comparison.
    :param local_maximum_qubits: Changed-support limit, defaulting to the register limit.
    :return: True only for a compiled match backed by exact channel equivalence.
    """

    local_limit = maximum_qubits if local_maximum_qubits is None else local_maximum_qubits
    local_proof = certify_local_channel_equivalence(original, mutant, local_limit)
    if not local_proof and (original.num_qubits > maximum_qubits or maximum_qubits <= 0):
        return False
    arguments = {
        "basis_gates": list(settings.basis_gates),
        "optimization_level": settings.optimization_level,
        "seed_transpiler": settings.seed,
        "approximation_degree": settings.approximation_degree,
    }
    compiled_original = transpile(original, **arguments)
    compiled_mutant = transpile(mutant, **arguments)
    return bool(compiled_original == compiled_mutant) and (local_proof or certify_channel_equivalence(
        original, mutant, maximum_qubits,
    ))


def information_theoretic_strength_floor(copies: int, error_probability: float) -> float:
    """Evaluate the manuscript's independent-copy rotation strength floor.

    :param copies: Available independent output copies.
    :param error_probability: Target equal-prior error probability.
    :return: Minimum strength ``eta_IT`` in radians.
    """

    base = 4.0 * error_probability * (1.0 - error_probability)
    return 2.0 * math.asin(math.sqrt(1.0 - base ** (1.0 / copies)))


def is_clifford_relative(candidate: MutationCandidate) -> bool:
    """Recognize a candidate's relative unitary using Qiskit's numerical tolerance.

    Near-Clifford rotations can pass this check; it is not an algebraic proof.
    The structured branch must retain the actual relative matrix for conjugation.

    :param candidate: Candidate with a local relative operator.
    :return: True when Qiskit accepts the matrix as approximately Clifford.
    """

    return _is_clifford_instruction(UnitaryGate(candidate.relative_operator.data))
