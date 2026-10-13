# Quantum mutant planner

[![Tests](https://github.com/miranska/quantum-mutant-planner/actions/workflows/tests.yml/badge.svg?branch=main&event=push)](https://github.com/miranska/quantum-mutant-planner/actions/workflows/tests.yml?query=branch%3Amain+event%3Apush)
[![Coverage](https://github.com/miranska/quantum-mutant-planner/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/miranska/quantum-mutant-planner/tree/python-coverage-comment-action-data)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)

Source repository: [miranska/quantum-mutant-planner](https://github.com/miranska/quantum-mutant-planner).

This package provides supplementary material for *Circuit-Aware Mutation Testing for Quantum Programs*, including an executable reference implementation of Algorithms 1 and 2.
It plans mutant executions for a fixed expectation-value test under the manuscript's ideal, finite-dimensional, unitary model.
The package, tests, notebook, and example-data generator run independently of the manuscript files.

## Installation and usage

Run all commands in this README from the directory containing this README and `pyproject.toml`.

Install the package and its development dependencies into your active Python environment:

```sh
python -m pip install -e '.[test,notebook]'
```

Dependencies and version constraints, including Qiskit 2.5.1, are declared in [`pyproject.toml`](pyproject.toml).

Run the manuscript's GHZ example through the Hydra CLI:

```sh
quantum-mutant-plan scenario=ghz
```

Open the sample notebook in JupyterLab:

```sh
jupyter lab notebook/algorithms_1_2_demo.ipynb
```

If prompted, select the kernel for the environment where you installed the package.

Run the tests with:

```sh
python -m pytest
```

The library accepts Qiskit circuits and quantum-information objects directly.
The CLI supports the registered example scenarios.

For an exact rotation, use `exact_pauli_rotation(Pauli("Y"), sympy.pi / 2)` as the original gate and pass exact angles to `pauli_rotation_mutation`.
Exact rotation replacements must match the declared original gate, and insertions require an original angle of zero.
See the [notebook](notebook/algorithms_1_2_demo.ipynb) for examples of exact certificates, pilot estimates, and finite-shot classification.

## Reproducing manuscript examples

Run the GHZ planner with semantic filtering disabled to expose the individual analytical branches:

```sh
quantum-mutant-plan scenario=ghz planner.enable_equivalence=false
```

Regenerate the manuscript's example data with:

```sh
python scripts/regenerate_manuscript_examples.py
```

The command writes four ignored files under `generated-examples/`:

- `ghz.csv`: output distributions, gaps, distances, and outcomes for the eight GHZ mutants.
- `ghz-planner.json`: planner decisions with semantic filtering disabled.
- `representative-gaps.csv`: exact expressions for the representative-gap table.
- `bell-measurement.json`: Bell-example observables, expectations, and readout details.

Use `--output-dir PATH` to choose another output directory.
The `0.01`-radian GHZ offset is an illustrative sensitivity probe, not an empirically justified fault model.

## Limitations

The implementation uses dense matrices and is intended for small, ideal unitary circuits.
A transpiler match receives `skip--semantically equivalent` only after an independent exact channel proof; numerical transpiler equality alone is insufficient.
The symbolic certificate path requires supported exact gate data and defaults to four qubits (`symbolic_max_qubits=4`); setting the limit to zero disables it.
Unsupported or unproved matches continue through planning, and the filter may miss equivalent circuits, including some that differ only by global phase.
Failure to prove equivalence does not prove non-equivalence, and a below-threshold decision does not justify excluding a mutant as semantically equivalent.
Dense gaps and bounds use floating-point arithmetic and are not validated numerical certificates.

## Pilot allocation

In `plan_mutant_executions`, `pilot.failure_probability` is the run-wide failure budget for one fixed test.
Before filtering, the planner divides this budget equally among all declared candidates and never recycles unused allocations.
It collects at most one separate pilot per candidate, and each pilot interval reports its allocated `failure_probability`.
Direct calls to `evaluate_candidate_effect` or the pilot provider treat the supplied failure probability as a per-call budget.
Callers running multiple tests or planning runs must allocate their total budget across those calls in advance.
If the required shot count exceeds `pilot.max_shots`, the planner falls back without claiming pilot confidence.
See [`pilot.py`](src/quantum_mutant_planner/pilot.py) for the estimator and shot-count derivation.

## Project layout

- [`src/`](src/): package and Hydra configuration.
- [`tests/`](tests/): automated tests, including notebook execution and mathematical checks.
- [`notebook/`](notebook/): executable walkthrough.
- [`scripts/`](scripts/): example-data generator.
- [`.github/workflows/`](.github/workflows/): test and coverage workflows.
