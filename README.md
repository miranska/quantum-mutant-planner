# Quantum mutant planner

[![Tests](https://github.com/miranska/quantum-mutant-planner/actions/workflows/tests.yml/badge.svg?branch=main&event=push)](https://github.com/miranska/quantum-mutant-planner/actions/workflows/tests.yml?query=branch%3Amain+event%3Apush)
[![Coverage](https://github.com/miranska/quantum-mutant-planner/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/miranska/quantum-mutant-planner/tree/python-coverage-comment-action-data)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)

Source repository: [miranska/quantum-mutant-planner](https://github.com/miranska/quantum-mutant-planner).

This package is an executable reference implementation of Algorithms 1 and 2 in *Circuit-Aware Mutation Testing for Quantum Programs*.
It plans mutant executions for a fixed expectation-value test under the manuscript's ideal, finite-dimensional, unitary model.
It is sample code, not evidence that the follow-on empirical studies have been completed.
The package, tests, notebook, and example-data generator run independently of the manuscript files.

## Installation and usage

Run all commands in this README from the directory containing this README and `pyproject.toml`.

Install the package and its development dependencies into your active Python environment:

```sh
python -m pip install -e '.[test,notebook]'
```

Run the manuscript's GHZ example through the Hydra CLI:

```sh
quantum-mutant-plan scenario=ghz
```

Open the sample notebook in JupyterLab from the code project root:

```sh
jupyter lab notebook/algorithms_1_2_demo.ipynb
```

JupyterLab opens in a browser. If prompted, select the kernel associated with the environment where you installed the package.

Run the tests with:

```sh
python -m pytest
```

The library accepts Qiskit circuits and quantum-information objects directly.
The initial CLI intentionally exposes only registered example scenarios rather than defining a separate circuit-description language.

For an exact rotation, use `exact_pauli_rotation(Pauli("Y"), sympy.pi / 2)` as the original gate and pass exact angles to `pauli_rotation_mutation`.
Exact rotation replacements must match the declared original gate, and insertions require an original angle of zero.
The notebook demonstrates certified cancellations, an independent circuit comparison, and the finite-shot classifier.
The empirical Studies A--D remain future work.

## Reproducing manuscript examples

From the code project root, run the GHZ planner with semantic filtering disabled to expose the individual analytical branches:

```sh
quantum-mutant-plan scenario=ghz planner.enable_equivalence=false
```

From the code project root, regenerate the illustrative table data with:

```sh
python scripts/regenerate_manuscript_examples.py
```

The command writes four ignored files under `generated-examples/`:

- `ghz.csv` contains the eight complete output distributions in manuscript bit order `q0 q1 q2`, independently simulated `Z1` gaps, total variation distances, population outcomes, and default planning statuses.
- `ghz-planner.json` contains the reference planner report for the same candidates, with transpiler equivalence disabled to expose the analytical branches.
- `representative-gaps.csv` contains exact matrix-derived expressions for the representative-gap table, including each stated observable variant.
- `bell-measurement.json` contains the exact two-qubit difference, optimal sign observable, readout labels, expectations, and basis-change gate order.

Use `--output-dir PATH` to choose another output directory.
These are worked examples, not results from Studies A--D.
The `0.01`-radian GHZ offset is an illustrative sensitivity probe, not an empirically justified fault model.

Python dependencies and constraints are declared in `pyproject.toml`, including Qiskit 2.5.1 and SymPy 1.14.
A publicly accessible versioned archive and persistent identifier remain to be supplied; these reproduction instructions do not establish public artifact availability.

## Project layout

- [`src/`](src/) contains the installable `quantum_mutant_planner` package and its Hydra configuration.
- [`tests/`](tests/) contains package, CLI, notebook, and mathematical regression tests.
- [`notebook/`](notebook/) contains the executable GHZ and pilot-estimation walkthrough.
- [`scripts/`](scripts/) contains the illustrative example-data generator.
- [`pyproject.toml`](pyproject.toml) declares packaging, dependencies, test options, and the 95% coverage threshold.
- [`.github/workflows/`](.github/workflows/) contains standalone test and coverage-reporting workflows.
