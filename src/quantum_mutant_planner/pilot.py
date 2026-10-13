"""Signed local-Pauli classical-shadow pilot estimation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

from quantum_mutant_planner.models import PilotSettings, QuantumState


@dataclass(frozen=True, slots=True)
class PilotResult:
    """Simultaneous signed Pauli-expectation estimates.

    :param estimates: Clipped expectation estimates keyed by Pauli label.
    :param target_error: Declared simultaneous per-target error radius.
    :param shots: Total randomized local-Pauli snapshots.
    :param batches: Number of median-of-means batches.
    :param batch_size: Snapshots per batch.
    """

    estimates: dict[str, float]
    target_error: float
    shots: int
    batches: int
    batch_size: int


class PilotProvider(Protocol):
    """Interface for obtaining simultaneous signed Pauli estimates."""

    def estimate(
        self,
        state: QuantumState,
        targets: tuple[Pauli, ...],
        settings: PilotSettings,
    ) -> PilotResult:
        """Estimate every target expectation from one site-level sample.

        :param state: Original state at the mutation site.
        :param targets: Unique Pauli targets measured from the shared sample.
        :param settings: Pilot accuracy, confidence, shot, and seed settings.
        :return: Simultaneous signed expectation estimates.
        """

        ...


def pauli_weight(pauli: Pauli) -> int:
    """Count the nonidentity factors in a Pauli string.

    :param pauli: Pauli string.
    :return: Number of nonidentity tensor factors.
    """

    return sum(symbol != "I" for symbol in pauli.to_label())


def median_of_means_shot_plan(
    maximum_weight: int,
    target_count: int,
    target_error: float,
    failure_probability: float,
) -> tuple[int, int, int]:
    """Compute a simultaneous median-of-means classical-shadow shot plan.

    A weight-w local-Pauli snapshot has variance at most ``3**w``.
    A batch of at least ``4 * 3**w / target_error**2`` independent snapshots
    therefore fails the target error bound with probability at most 1/4 by
    Chebyshev's inequality.
    For K independent batches, Hoeffding's inequality bounds the probability
    that at least half fail by ``exp(-K/8)``.
    Taking ``K >= 8 * log(target_count / failure_probability)`` and applying
    a union bound gives the declared simultaneous guarantee; rounding K upward
    to an odd integer makes the median unambiguous.
    These are sufficient constants from this derivation, not the constants
    of Huang, Kueng, and Preskill's Theorem 1.

    :param maximum_weight: Maximum Pauli weight among all targets.
    :param target_count: Number of simultaneous Pauli targets.
    :param target_error: Desired additive error per target.
    :param failure_probability: Family-wise failure probability.
    :raises ValueError: If a weight, target count, error, or probability is invalid.
    :return: Total shots, odd batch count, and batch size.
    """

    if maximum_weight < 0:
        raise ValueError("maximum Pauli weight cannot be negative")
    if target_count <= 0:
        raise ValueError("at least one pilot target is required")
    if target_error <= 0.0:
        raise ValueError("pilot target error must be positive")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("pilot failure probability must lie between zero and one")
    batch_size = math.ceil(4.0 * (3**maximum_weight) / (target_error**2))
    batches = math.ceil(8.0 * math.log(target_count / failure_probability))
    if batches % 2 == 0:
        batches += 1
    return batch_size * batches, batches, batch_size


def _measurement_probabilities(state: QuantumState, bases: tuple[int, ...]) -> np.ndarray:
    """Return computational probabilities after local basis rotations.

    :param state: State to measure.
    :param bases: Per-qubit basis codes where zero, one, and two mean X, Y, and Z.
    :return: Probability vector in Qiskit's computational-basis ordering.
    """

    circuit = QuantumCircuit(state.num_qubits)
    for qubit, basis in enumerate(bases):
        if basis == 0:
            circuit.h(qubit)
        elif basis == 1:
            circuit.sdg(qubit)
            circuit.h(qubit)
    return np.asarray(state.evolve(circuit).probabilities(), dtype=float)


def _single_snapshot_estimate(pauli: Pauli, bases: tuple[int, ...], outcome: int) -> float:
    """Evaluate one local-Pauli classical-shadow snapshot for one target.

    :param pauli: Target Pauli string.
    :param bases: Per-qubit randomized basis codes.
    :param outcome: Sampled computational-basis outcome as an integer.
    :return: Unbiased single-snapshot Pauli estimator.
    """

    required_basis = {"X": 0, "Y": 1, "Z": 2}
    value = 1.0
    weight = 0
    for qubit, symbol in enumerate(reversed(pauli.to_label())):
        if symbol == "I":
            continue
        if bases[qubit] != required_basis[symbol]:
            return 0.0
        weight += 1
        value *= 1.0 if ((outcome >> qubit) & 1) == 0 else -1.0
    return value * (3**weight)


class MedianOfMeansPilot:
    """Statevector-backed randomized local-Pauli pilot provider."""

    def estimate(
        self,
        state: QuantumState,
        targets: tuple[Pauli, ...],
        settings: PilotSettings,
    ) -> PilotResult:
        """Estimate target expectations with simultaneous median-of-means bounds.

        :param state: Original state at the mutation site.
        :param targets: Unique full-register Pauli targets.
        :param settings: Pilot accuracy, confidence, shot, and seed settings.
        :raises ValueError: If targets are inconsistent or the shot ceiling is exceeded.
        :return: Clipped simultaneous signed estimates from one shared sample.
        """

        if not targets:
            raise ValueError("at least one pilot target is required")
        if any(target.num_qubits != state.num_qubits for target in targets):
            raise ValueError("pilot target width does not match the state")
        unique = {target.to_label(): target for target in targets}
        maximum_weight = max(pauli_weight(target) for target in unique.values())
        shots, batches, batch_size = median_of_means_shot_plan(
            maximum_weight,
            len(unique),
            settings.target_error,
            settings.failure_probability,
        )
        if shots > settings.max_shots:
            raise ValueError(f"pilot requires {shots} shots, above the configured ceiling {settings.max_shots}")
        rng = np.random.default_rng(settings.seed)
        cache: dict[tuple[int, ...], np.ndarray] = {}
        batch_means: dict[str, list[float]] = {label: [] for label in unique}
        for _ in range(batches):
            sums = {label: 0.0 for label in unique}
            for _ in range(batch_size):
                bases = tuple(int(value) for value in rng.integers(0, 3, size=state.num_qubits))
                probabilities = cache.get(bases)
                if probabilities is None:
                    probabilities = _measurement_probabilities(state, bases)
                    if len(cache) >= 32:
                        del cache[next(iter(cache))]
                    cache[bases] = probabilities
                outcome = int(rng.choice(len(probabilities), p=probabilities))
                for label, target in unique.items():
                    sums[label] += _single_snapshot_estimate(target, bases, outcome)
            for label in unique:
                batch_means[label].append(sums[label] / batch_size)
        estimates = {
            label: float(np.clip(np.median(means), -1.0, 1.0)) for label, means in batch_means.items()
        }
        return PilotResult(
            estimates=estimates,
            target_error=settings.target_error,
            shots=shots,
            batches=batches,
            batch_size=batch_size,
        )
