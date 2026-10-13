"""Finite-shot decisions, boundaries, and exact binomial coverage checks."""

import math

import numpy as np
import pytest

from quantum_mutant_planner.statistics import ObservationStatus, classify_fixed_shot, hoeffding_radius


def classify(original, mutant, **overrides):
    options = dict(outcome_bounds=(-1., 1.), threshold=.1, failure_probability=.05)
    options.update(overrides)
    return classify_fixed_shot(original, mutant, **options)


def test_tight_radius_and_three_observed_outcomes():
    r = hoeffding_radius(10000, 2, .05)
    assert r == pytest.approx(2 * math.sqrt(math.log(40) / 10000))
    below = classify(np.ones(10000), np.ones(10000))
    detected = classify(np.ones(10000), -np.ones(10000))
    unresolved = classify([1], [-1])
    assert below.status is ObservationStatus.BELOW_THRESHOLD
    assert below.interval == (0, r)
    assert detected.status is ObservationStatus.DETECTED
    assert detected.interval == (2 - r, 2)
    assert unresolved.status is ObservationStatus.UNRESOLVED
    assert unresolved.interval == (0, 2)
    assert below.shots_per_circuit == 10000 and below.failure_probability == .05
    assert classify(np.ones(10000), np.ones(10000), threshold=r).status is ObservationStatus.UNRESOLVED
    assert classify(np.ones(10000), -np.ones(10000), threshold=2-r).status is ObservationStatus.DETECTED


@pytest.mark.parametrize("p,q", [(0, 0), (0, 1), (.1, .8), (.5, .5), (.2, .3)])
def test_interval_coverage_by_exact_enumeration(p, q):
    """Sum binomial probabilities instead of using a stochastic coverage assertion."""
    n, alpha = 20, .2
    covered = 0.
    true_gap = 2 * abs(p - q)
    for original_ones in range(n + 1):
        px = math.comb(n, original_ones) * p**original_ones * (1-p)**(n-original_ones)
        for mutant_ones in range(n + 1):
            py = math.comb(n, mutant_ones) * q**mutant_ones * (1-q)**(n-mutant_ones)
            result = classify([1]*original_ones + [-1]*(n-original_ones), [1]*mutant_ones + [-1]*(n-mutant_ones), failure_probability=alpha)
            if result.interval[0] <= true_gap <= result.interval[1]:
                covered += px * py
    assert covered >= 1 - alpha - 1e-12


@pytest.mark.parametrize("shots,span,alpha", [(0, 2, .1), (True, 2, .1), (1.5, 2, .1), (1, 0, .1), (1, math.inf, .1), (1, 2, 0), (1, 2, 1), (1, 2, math.nan)])
def test_invalid_radius_inputs(shots, span, alpha):
    with pytest.raises(ValueError):
        hoeffding_radius(shots, span, alpha)


@pytest.mark.parametrize("original,mutant,options", [
    ([], [], {}), ([0], [0, 0], {}), ([[0]], [[0]], {}), ([math.nan], [0], {}),
    ([0], [2], {}), ([0], [0], {"threshold": 0}), ([0], [0], {"threshold": math.inf}),
    ([0], [0], {"outcome_bounds": (1, -1)}), ([0], [0], {"outcome_bounds": (-math.inf, 1)}),
    (np.array([1j]), [0], {}), ([0], np.array([1j]), {}),
])
def test_invalid_observation_inputs(original, mutant, options):
    with pytest.raises(ValueError):
        classify(original, mutant, **options)
