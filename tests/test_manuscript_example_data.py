"""Check the manuscript's distributions and measurement using independent expectations."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import sympy as sp


def test_regenerated_tables_and_bell_measurement(tmp_path: Path) -> None:
    """Regeneration must reproduce all eight distributions and the optimal readout."""
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "scripts/regenerate_manuscript_examples.py"),
         "--output-dir", str(tmp_path)], check=True, cwd=root,
    )
    with (tmp_path / "ghz.csv").open() as source:
        rows = list(csv.DictReader(source))
    s = np.sin(.01)
    expected = [
        [.5, 0, 0, 0, 0, 0, .5, 0],
        [0, .5, 0, 0, 0, 0, .5, 0],
        [.5, 0, 0, 0, 0, 0, 0, .5],
        [0, 0, .5, 0, 0, .5, 0, 0],
        [.5, 0, 0, 0, 0, 0, 0, .5],
        [(1-s)/2, 0, 0, 0, 0, 0, 0, (1+s)/2],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [.5, 0, 0, 0, 0, .5, 0, 0],
    ]
    for row, probabilities in zip(rows, expected, strict=True):
        actual = [float(row[f"p_{i:03b}"]) for i in range(8)]
        np.testing.assert_allclose(actual, probabilities, atol=1e-14)
        assert sum(actual) == pytest.approx(1)
    np.testing.assert_allclose([float(r["tvd"]) for r in rows], [.5, 1, 0, 1, 0, s/2, .5, .5], atol=1e-14)
    np.testing.assert_allclose([float(r["z1_gap"]) for r in rows], [0, 0, 0, 0, 0, s, 1, 1], atol=1e-14)
    assert sum(r["tvd_population_outcome"] == "detected" for r in rows) == 5
    report = json.loads((tmp_path / "ghz-planner.json").read_text())
    assert report["discrimination_budget_enabled"] is False
    assert len(report["records"]) == 8

    bell = json.loads((tmp_path / "bell-measurement.json").read_text())
    assert bell["readout_diagonal"] == ["1", "0", "-1", "0"]
    assert bell["readout_off_diagonal_zero"] is True
    assert sp.sympify(bell["gap"]) == sp.sqrt(2)
    assert sp.sympify(bell["original_mean"]) == 1/sp.sqrt(2)
    assert sp.sympify(bell["mutant_mean"]) == -1/sp.sqrt(2)
    assert bell["max_pauli_gap"] == "1"

    with (tmp_path / "representative-gaps.csv").open() as source:
        gaps = {r["id"]: sp.sympify(r["gap"], locals={
            "theta": sp.Symbol("theta", real=True), "eta": sp.Symbol("eta", real=True),
        }) for r in csv.DictReader(source)}
    theta, eta = sp.symbols("theta eta", real=True)
    expected_gaps = {
        "ry-offset": sp.Abs(sp.cos(theta)-sp.cos(theta+eta)),
        "ry-deletion": sp.Abs(sp.cos(theta)-1), "x-to-i": 2, "x-to-y": 0,
        "remove-control": 1, "flip-control-local": 0, "flip-control-correlation": 2,
        "delete-cnot": 1, "swap": 2, "global-phase": 0,
        "rz-insertion-x": sp.Abs(1-sp.cos(eta)), "rz-insertion-y": sp.Abs(sp.sin(eta)),
        "subcircuit-correlation": 2, "subcircuit-local": 0,
    }
    assert gaps.keys() == expected_gaps.keys()
    for key, value in expected_gaps.items():
        assert sp.simplify(gaps[key]-value) == 0, key
