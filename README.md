# Quantum Mutant Planner

[![arXiv](https://img.shields.io/badge/arXiv-2610.36976-b31b1b.svg)](https://arxiv.org/abs/2610.36976)
[![Tests](https://github.com/miranska/quantum-mutant-planner/actions/workflows/tests.yml/badge.svg?branch=main&event=push)](https://github.com/miranska/quantum-mutant-planner/actions/workflows/tests.yml?query=branch%3Amain+event%3Apush)
[![Coverage](https://github.com/miranska/quantum-mutant-planner/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/miranska/quantum-mutant-planner/tree/python-coverage-comment-action-data)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Python reference implementation for planning mutant executions in quantum program testing.
This package accompanies the preprint [*Toward Circuit-Aware Mutation Testing for Quantum Programs: Exact Visibility and Execution Planning*](https://arxiv.org/abs/2610.36976), and implements Algorithms 1 and 2.

## Overview

A quantum mutant can survive because it preserves program behavior, because the chosen input or measurement hides its effect, or because the test collects insufficient statistical evidence.
The planner analyzes a fixed expectation-value test under the paper's ideal, finite-dimensional, unitary model to help distinguish these cases and decide which mutants to execute.

The repository includes:

- Structural and algebraic checks, supported exact certificates, numerical effect calculations, and simulated pilot estimates.
- Planning records that retain the decision, supporting evidence, and unresolved cases for each candidate.
- A separate finite-shot classifier for differential expectation tests.
- An executable notebook, reproducible GHZ and Bell examples, and bounded synthetic scalability probes.

The package, tests, notebook, and example-data generator run independently of the manuscript files.

## Dependencies

The package requires Python 3.10 or later; continuous integration runs on Python 3.13.
Runtime dependencies are Qiskit 2.5.1, NumPy, SymPy, Hydra Core, and Hydra Colorlog.
The `test` and `notebook` extras add pytest, coverage reporting, and Jupyter tools.
All dependencies and version constraints are declared in [`pyproject.toml`](pyproject.toml) and installed by the command below.

## Installation and usage

Clone the public repository:

```sh
git clone https://github.com/miranska/quantum-mutant-planner.git
cd quantum-mutant-planner
```

Run the remaining commands from the directory containing this README and `pyproject.toml`.

Install the package and its development dependencies into your active Python environment:

```sh
python -m pip install -e '.[test,notebook]'
```

Run the manuscript's GHZ example through the Hydra CLI:

```sh
quantum-mutant-plan scenario=ghz
```

The command prints a JSON report with per-candidate decisions and evidence.

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

## Scalability probes

Run bounded synthetic checks of structure, Pauli propagation, local proofs, and dense fallbacks:

```sh
python scripts/probe_scalability.py --no-memory-screen
```

The Unix-like runner uses process timeouts and memory monitoring; results go to `generated-scalability/results.json`.
This command attempts the full declared grid while retaining the RSS watchdog and timeout.
See the [recorded results and scope](scalability-results/README.md) for the tested circuits, resource budgets, and the distinction between helper checks and the full planner.

## Limitations

This is a research reference implementation for ideal, noiseless circuits and differential expectation tests.
The paper's native-oracle extensions and full tableau-based classification algorithm are not implemented; execution savings and performance on real mutation populations require separate evaluation.

Structural cones, Pauli commutation, and bounded local conjugation run without full-register matrices.
General Pauli propagation can still grow exponentially; `max_pauli_terms` caps its retained approximation and all discarded coefficient mass enters the pilot error bound.
Dense numerical fallbacks default to ten qubits (`dense_max_qubits`), local gate matrices to four (`local_max_qubits`), and state preparation and direct simulation to ten each.
A statevector uses `2^n` amplitudes; a dense operator or density matrix uses `4^n` entries.
The input state and mutation operators are still supplied as dense Qiskit objects, and the built-in pilot simulates a state.
Validation computes supported Pauli spectral ranges analytically or diagonalizes the observable's active support within the dense limit; unsupported over-limit gates or range calculations raise an explicit error.
A transpiler match receives `skip--semantically equivalent` only after an independent exact channel proof; numerical transpiler equality alone is insufficient.
The symbolic certificate path requires supported exact gate data and defaults to four qubits (`symbolic_max_qubits=4`); setting the limit to zero disables it.
For channel equivalence, verified common circuit context can be cancelled so that this limit applies only to the changed support, even in a wider register.
Full-circuit symbolic proofs also obey the dense width limit.
When effect evaluation exhausts its limits, the planner retains the candidate with unresolved evidence.
Unsupported or unproved matches continue through planning, and the filter may miss equivalent circuits, including some that differ only by global phase.
Failure to prove equivalence does not prove non-equivalence, and a below-threshold decision does not justify excluding a mutant as semantically equivalent.
Numerical gaps and bounds use floating-point arithmetic and are not validated numerical certificates.

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
- [`scripts/`](scripts/): example-data generator and scalability probes.
- [`scalability-results/`](scalability-results/): recorded synthetic probe results and their scope.
- [`.github/workflows/`](.github/workflows/): test and coverage workflows.

## Citation

If you use or study this code, please cite the [preprint](https://arxiv.org/abs/2610.36976).
Citation metadata is provided in [`CITATION.cff`](CITATION.cff), with the preprint set as the preferred citation.
Use GitHub's **Cite this repository** button to copy a BibTeX or APA citation.

## Contact us

For bug reports, questions, or feature suggestions, please open an [issue](https://github.com/miranska/quantum-mutant-planner/issues) or a [pull request](https://github.com/miranska/quantum-mutant-planner/pulls).

## License

This code is distributed under the [Apache License 2.0](LICENSE).
