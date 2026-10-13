"""Pauli arithmetic with dense work confined to explicitly bounded local supports."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp


class ResourceLimitError(ValueError):
    """A requested calculation exceeds its declared representation limit."""


def local_conjugation(
    observable: SparsePauliOp, unitary: Operator, support: tuple[int, ...], maximum_qubits: int,
) -> SparsePauliOp:
    """Compute ``U.H O U`` by expanding only each term's local factors.

    :param observable: Full-width Pauli strings, without a full-width matrix.
    :param unitary: Local unitary in the ordered support convention.
    :param support: Register indices corresponding to local qubits.
    :param maximum_qubits: Maximum width of a dense local calculation.
    :raises ResourceLimitError: If the local support exceeds the limit.
    :return: Complete numerical Pauli expansion, before term truncation.
    """

    if len(support) > maximum_qubits:
        raise ResourceLimitError("local conjugation exceeds local_max_qubits")
    cache: dict[str, SparsePauliOp] = {}
    terms: list[tuple[str, complex]] = []
    for pauli, coefficient in zip(observable.paulis, observable.coeffs, strict=True):
        symbols = list(reversed(pauli.to_label()))
        local_label = "".join(symbols[index] for index in reversed(support))
        if local_label not in cache:
            matrix = Pauli(local_label).to_matrix()
            cache[local_label] = SparsePauliOp.from_operator(
                Operator(unitary.data.conj().T @ matrix @ unitary.data), atol=0.0, rtol=0.0,
            )
        for local_pauli, weight in zip(cache[local_label].paulis, cache[local_label].coeffs, strict=True):
            output = symbols.copy()
            for index, symbol in zip(support, reversed(local_pauli.to_label()), strict=True):
                output[index] = symbol
            terms.append(("".join(reversed(output)), coefficient * weight))
    return SparsePauliOp.from_list(terms).simplify(atol=0.0, rtol=0.0)


def rotation_difference(observable: SparsePauliOp, axis: Pauli, eta: float) -> SparsePauliOp:
    """Form the finite-angle difference using only anticommuting Pauli terms.

    :param observable: Observable in Pauli form.
    :param axis: Signed full-register involutory Pauli axis.
    :param eta: Rotation angle in radians.
    :return: Complete numerical difference with at most twice as many terms.
    """

    anti = ~observable.paulis.commutes(axis)
    if not np.any(anti):
        return SparsePauliOp("I" * observable.num_qubits, coeffs=[0.0])
    selected = SparsePauliOp(observable.paulis[anti], observable.coeffs[anti])
    return ((1.0 - np.cos(eta)) * selected
            + 1j * np.sin(eta) * selected.dot(SparsePauliOp(axis))).simplify(atol=0.0, rtol=0.0)


def observable_range(observable: SparsePauliOp, maximum_qubits: int) -> float:
    """Find the spectral range analytically or on the observable's active support.

    :param observable: Hermitian Pauli observable.
    :param maximum_qubits: Largest active support admitted to dense diagonalization.
    :raises ResourceLimitError: If no supported bounded range calculation applies.
    :return: Numerically evaluated spectral range.
    """

    merged = observable.simplify(atol=0.0, rtol=0.0)
    nonidentity = np.any(merged.paulis.x | merged.paulis.z, axis=1) & (merged.coeffs != 0)
    terms = merged[nonidentity]
    if len(terms) == 0:
        return 0.0
    # Pairwise anticommuting Hermitian Paulis square to a scalar identity.
    if all(not terms.paulis[i].commutes(terms.paulis[j])
           for i in range(len(terms)) for j in range(i)):
        return float(2.0 * np.linalg.norm(terms.coeffs.real))
    if all(terms.paulis[i].commutes(terms.paulis[j])
           for i in range(len(terms)) for j in range(i)):
        # Independent commuting Paulis realize every joint sign assignment.
        rows = np.concatenate((terms.paulis.x, terms.paulis.z), axis=1).copy()
        rank = 0
        for column in range(rows.shape[1]):
            pivots = np.flatnonzero(rows[rank:, column])
            if not len(pivots):
                continue
            pivot = rank + pivots[0]
            rows[[rank, pivot]] = rows[[pivot, rank]]
            for index in range(rank + 1, len(rows)):
                if rows[index, column]:
                    rows[index] ^= rows[rank]
            rank += 1
        if rank == len(terms):
            return float(2.0 * np.sum(np.abs(terms.coeffs.real)))
    support = np.flatnonzero(np.any(terms.paulis.x | terms.paulis.z, axis=0))
    if len(support) > maximum_qubits:
        raise ResourceLimitError(
            "observable spectral range requires a dense calculation beyond dense_max_qubits; "
            "use a supported Pauli observable or explicitly raise the limit"
        )
    reduced = SparsePauliOp.from_list([
        ("".join(pauli.to_label()[-1 - index] for index in reversed(support)), coefficient)
        for pauli, coefficient in zip(terms.paulis, terms.coeffs, strict=True)
    ])
    eigenvalues = np.linalg.eigvalsh(reduced.to_matrix())
    return float(eigenvalues[-1] - eigenvalues[0])
