"""Regenerate illustrative manuscript data without running an empirical study.

Run from the code project root after installing the package. Tensor factors in
symbolic examples are written left to right; GHZ output reverses Qiskit's
little-endian display to the manuscript's q0 q1 q2 order.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import sympy as sp
from qiskit.quantum_info import Statevector

from quantum_mutant_planner.cli import report_payload
from quantum_mutant_planner.examples import build_ghz_problem
from quantum_mutant_planner.models import PlannerSettings
from quantum_mutant_planner.planner import plan_mutant_executions


def ghz_data() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute full-circuit probabilities and gaps independently of local formulas.

    :return: Table rows and the separately computed planner report.
    """
    problem = build_ghz_problem()
    settings = PlannerSettings(enable_equivalence=False)
    records = plan_mutant_executions(problem, settings)
    original = Statevector.from_instruction(problem.circuit)
    probabilities = original.probabilities()
    original_mean = float(original.expectation_value(problem.test.observable).real)
    rows = []
    for candidate, record in zip(problem.candidates, records, strict=True):
        output = Statevector.from_instruction(candidate.mutant_circuit)
        mutated = output.probabilities()
        gap = abs(original_mean - float(output.expectation_value(problem.test.observable).real))
        distance = float(abs(probabilities - mutated).sum() / 2)
        row: dict[str, Any] = {"id": candidate.identifier}
        for value in range(8):
            label = f"{value:03b}"
            row[f"p_{label}"] = float(mutated[int(label[::-1], 2)])
        row.update({
            "z1_gap": gap,
            "z1_threshold": problem.test.threshold,
            "z1_population_outcome": "detected" if gap >= problem.test.threshold else "survivor",
            "tvd": distance,
            "tvd_threshold": 0.1,
            "tvd_population_outcome": "detected" if distance >= 0.1 else "survivor",
            "planner_status": record.status.value,
        })
        rows.append(row)
    return rows, report_payload("ghz", settings, records)


def symbolic_data() -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Evaluate representative gaps and the Bell measurement by exact matrices.

    :return: Symbolic expectation rows and the exact Bell measurement data.
    """
    identity = sp.eye(2)
    x = sp.Matrix([[0, 1], [1, 0]])
    y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    z = sp.diag(1, -1)
    zero = sp.Matrix([1, 0])
    one = sp.Matrix([0, 1])
    plus = (zero + one) / sp.sqrt(2)
    kron = sp.kronecker_product
    theta, eta, phase = sp.symbols("theta eta phase", real=True)
    cx = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    swap = sp.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    bell = (kron(zero, zero) + kron(one, one)) / sp.sqrt(2)
    opposite = (kron(zero, one) + kron(one, zero)) / sp.sqrt(2)

    def rotation(angle: sp.Expr) -> sp.Matrix:
        """Construct a real Y rotation from its defining matrix.

        :param angle: Symbolic angle.
        :return: Exact rotation matrix.
        """
        return sp.Matrix([[sp.cos(angle / 2), -sp.sin(angle / 2)],
                          [sp.sin(angle / 2), sp.cos(angle / 2)]])

    original_rotation = rotation(theta) * zero
    rz_plus = sp.diag(sp.exp(-sp.I * eta / 2), sp.exp(sp.I * eta / 2)) * plus
    cases = [
        ("ry-offset", original_rotation, rotation(theta + eta) * zero, z),
        ("ry-deletion", original_rotation, zero, z),
        ("x-to-i", x * zero, zero, z),
        ("x-to-y", x * zero, y * zero, z),
        ("remove-control", cx * kron(plus, zero), kron(plus, one), kron(identity, z)),
        ("flip-control-local", bell, opposite, kron(identity, z)),
        ("flip-control-correlation", bell, opposite, kron(z, z)),
        ("delete-cnot", bell, kron(plus, zero), kron(z, z)),
        ("swap", kron(zero, one), swap * kron(zero, one), kron(z, identity)),
        ("global-phase", bell, sp.exp(sp.I * phase) * bell, kron(z, z)),
        ("rz-insertion-x", plus, rz_plus, x),
        ("rz-insertion-y", plus, rz_plus, y),
        ("subcircuit-correlation", bell, opposite, kron(z, z)),
        ("subcircuit-local", bell, opposite, kron(z, identity)),
    ]
    rows = []
    for name, original, mutant, observable in cases:
        before = sp.simplify((original.H * observable * original)[0])
        after = sp.simplify((mutant.H * observable * mutant)[0])
        signed = sp.simplify(sp.expand_complex(before - after))
        rows.append({"id": name, "original_mean": str(before), "mutant_mean": str(after),
                     "signed_shift": str(signed), "gap": str(sp.simplify(sp.Abs(signed)))})

    mutated_bell = kron(sp.diag(1, sp.I), identity) * bell
    difference = sp.simplify(bell * bell.H - mutated_bell * mutated_bell.H)
    sign = sp.zeros(4)
    for eigenvalue, _, vectors in difference.eigenvects():
        for vector in vectors:
            sign += sp.sign(eigenvalue) * vector * vector.H / (vector.H * vector)[0]
    sign = sp.simplify(sign)
    h = (x + z) / sp.sqrt(2)
    t = sp.diag(1, (1 + sp.I) / sp.sqrt(2))
    basis = kron(h * t, identity) * cx
    readout = sp.simplify(basis * sign * basis.H)
    pauli_gaps = [sp.simplify(sp.Abs(sp.trace(kron(a, b) * difference)))
                  for a in (identity, x, y, z) for b in (identity, x, y, z)]
    details = {
        "bit_order": "q0 q1",
        "difference": str(difference),
        "sign_observable": str(sign),
        "readout_diagonal": [str(readout[i, i]) for i in range(4)],
        "readout_off_diagonal_zero": readout.is_diagonal(),
        "original_mean": str(sp.simplify((bell.H * sign * bell)[0])),
        "mutant_mean": str(sp.simplify((mutated_bell.H * sign * mutated_bell)[0])),
        "gap": str(sp.simplify(sp.trace(sign * difference))),
        "max_pauli_gap": str(max(pauli_gaps)),
        "basis_gate_order": ["cx(0,1)", "t(0)", "h(0)"],
    }
    return rows, details


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write table rows with a deterministic header and ordering.

    :param path: Destination file.
    :param rows: Nonempty ordered records.
    :return: None.
    """
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Regenerate numerical GHZ tables and exact symbolic example data.

    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("generated-examples"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ghz, report = ghz_data()
    examples, bell = symbolic_data()
    write_csv(args.output_dir / "ghz.csv", ghz)
    write_csv(args.output_dir / "representative-gaps.csv", examples)
    for name, data in [("ghz-planner.json", report), ("bell-measurement.json", bell)]:
        (args.output_dir / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote four worked-example data files to {args.output_dir}")


if __name__ == "__main__":
    main()
