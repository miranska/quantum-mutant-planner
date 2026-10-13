"""Fixed-shot observations, kept separate from pre-execution predictions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Sequence

import numpy as np


class ObservationStatus(StrEnum):
    """Three-way finite-shot classifications, not planner dispositions."""

    BELOW_THRESHOLD = "below threshold"
    DETECTED = "detected"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class FixedShotResult:
    """An observed gap and its fixed-time Hoeffding confidence interval.

    :param status: Observed threshold classification.
    :param estimated_gap: Absolute difference of the two sample means.
    :param interval: Absolute-gap interval, clipped to the observable range.
    :param radius: Direct two-sample Hoeffding radius.
    :param shots_per_circuit: Number of observations of each circuit.
    :param failure_probability: Prespecified error allocation for this comparison.
    """

    status: ObservationStatus
    estimated_gap: float
    interval: tuple[float, float]
    radius: float
    shots_per_circuit: int
    failure_probability: float


def hoeffding_radius(shots_per_circuit: int, outcome_range: float, failure_probability: float) -> float:
    """Bound the error of a difference of two independent sample means.

    :param shots_per_circuit: Positive equal sample size for each circuit.
    :param outcome_range: Positive finite common outcome range.
    :param failure_probability: Prespecified per-comparison failure probability.
    :raises ValueError: If the sample size, range, or allocation is invalid.
    :return: Radius ``R sqrt(log(2 / alpha) / n)``.
    """

    if isinstance(shots_per_circuit, bool) or not isinstance(shots_per_circuit, (int, np.integer)) or shots_per_circuit <= 0:
        raise ValueError("shots_per_circuit must be a positive integer")
    if not math.isfinite(outcome_range) or outcome_range <= 0:
        raise ValueError("outcome_range must be positive and finite")
    if not 0 < failure_probability < 1:
        raise ValueError("failure_probability must lie strictly between zero and one")
    return outcome_range * math.sqrt((math.log(2) - math.log(failure_probability)) / shots_per_circuit)


def classify_fixed_shot(
    original_samples: Sequence[float],
    mutant_samples: Sequence[float],
    *,
    outcome_bounds: tuple[float, float],
    threshold: float,
    failure_probability: float,
) -> FixedShotResult:
    """Classify independent, fixed-size original and mutant observations.

    The caller prespecifies the shot count and error allocation. All shots within
    a comparison must be independent. Original samples may be reused across
    different comparisons if their allocations sum to the family-wise budget.
    Repeated inspection or pilot-driven reuse needs separate sequential control.

    :param original_samples: Observed original-circuit outcomes.
    :param mutant_samples: Equally many independent mutant-circuit outcomes.
    :param outcome_bounds: Known common minimum and maximum outcome values.
    :param threshold: Positive population gap threshold.
    :param failure_probability: Prespecified test error allocation, separate from pilot and discrimination errors.
    :raises ValueError: If samples, bounds, threshold, or allocation are invalid.
    :return: Observed gap, confidence interval, and three-way classification.
    """

    lower_bound, upper_bound = outcome_bounds
    if not math.isfinite(lower_bound) or not math.isfinite(upper_bound) or lower_bound >= upper_bound:
        raise ValueError("outcome bounds must be finite and strictly increasing")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    if np.iscomplexobj(original_samples) or np.iscomplexobj(mutant_samples):
        raise ValueError("observed outcomes must be real")
    original = np.asarray(original_samples, dtype=float)
    mutant = np.asarray(mutant_samples, dtype=float)
    if original.ndim != 1 or mutant.ndim != 1 or original.size != mutant.size or not original.size:
        raise ValueError("samples must be nonempty one-dimensional arrays of equal length")
    for sample in (original, mutant):
        if not np.all(np.isfinite(sample)) or np.any(sample < lower_bound) or np.any(sample > upper_bound):
            raise ValueError("samples must be finite and inside the declared outcome bounds")
    outcome_range = upper_bound - lower_bound
    radius = hoeffding_radius(original.size, outcome_range, failure_probability)
    gap = abs(float(np.mean(original)) - float(np.mean(mutant)))
    interval = (max(0.0, gap - radius), min(outcome_range, gap + radius))
    if interval[1] < threshold:
        status = ObservationStatus.BELOW_THRESHOLD
    elif interval[0] >= threshold:
        status = ObservationStatus.DETECTED
    else:
        status = ObservationStatus.UNRESOLVED
    return FixedShotResult(status, gap, interval, radius, int(original.size), failure_probability)
