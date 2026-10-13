"""Tests for signed classical-shadow pilot estimation."""

import pytest
from qiskit.quantum_info import Pauli, Statevector

from quantum_mutant_planner.models import PilotSettings
from quantum_mutant_planner.pilot import (
    MedianOfMeansPilot,
    _single_snapshot_estimate,
    median_of_means_shot_plan,
    pauli_weight,
)


def test_pauli_weight_and_shot_plan() -> None:
    """Shot planning should follow the declared median-of-means constants."""

    assert pauli_weight(Pauli("IXYZ")) == 3
    shots, batches, batch_size = median_of_means_shot_plan(1, 1, 1.0, 0.05)
    assert batch_size == 12
    assert batches % 2 == 1
    assert shots == batches * batch_size
    _, already_odd, _ = median_of_means_shot_plan(1, 1, 1.0, 0.1)
    assert already_odd % 2 == 1
    assert _single_snapshot_estimate(Pauli("I"), (2,), 0) == 1.0


@pytest.mark.parametrize(
    "arguments",
    [(-1, 1, 1.0, 0.1), (1, 0, 1.0, 0.1), (1, 1, 0.0, 0.1), (1, 1, 1.0, 1.0)],
)
def test_shot_plan_rejects_invalid_arguments(arguments) -> None:
    """Every shot-plan domain restriction should be enforced."""

    with pytest.raises(ValueError):
        median_of_means_shot_plan(*arguments)


def test_median_of_means_pilot_is_seeded_clipped_and_shared() -> None:
    """A one-qubit pilot should be deterministic and bounded."""

    provider = MedianOfMeansPilot()
    settings = PilotSettings(target_error=1.0, failure_probability=0.1, max_shots=1_000, seed=7)
    targets = (Pauli("Z"), Pauli("X"), Pauli("Z"))
    first = provider.estimate(Statevector.from_label("0"), targets, settings)
    second = provider.estimate(Statevector.from_label("0"), targets, settings)
    assert first == second
    assert set(first.estimates) == {"X", "Z"}
    assert all(-1.0 <= value <= 1.0 for value in first.estimates.values())
    assert first.estimates["Z"] > first.estimates["X"]


def test_pilot_rejects_missing_wrong_width_and_excessive_shots() -> None:
    """Pilot execution should fail clearly for unsupported requests."""

    provider = MedianOfMeansPilot()
    settings = PilotSettings(target_error=1.0, failure_probability=0.1, max_shots=1_000)
    with pytest.raises(ValueError, match="target"):
        provider.estimate(Statevector.from_label("0"), (), settings)
    with pytest.raises(ValueError, match="width"):
        provider.estimate(Statevector.from_label("0"), (Pauli("ZZ"),), settings)
    with pytest.raises(ValueError, match="ceiling"):
        provider.estimate(
            Statevector.from_label("0"),
            (Pauli("Z"),),
            PilotSettings(target_error=0.01, failure_probability=0.05, max_shots=10),
        )
