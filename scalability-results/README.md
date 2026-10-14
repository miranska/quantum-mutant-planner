# Synthetic scalability probes

Recorded on 2026-10-13 in one 39-case batch using [`probe_scalability.py`](../scripts/probe_scalability.py).
The complete grid, environment, and measurements are in [`reference.json`](reference.json).
Every case was attempted with estimate-based screening disabled; the timeout and memory watchdog remained active.
These constructed cases have known answers; they are implementation feasibility checks, not Studies A--D, a benchmark-corpus evaluation, or evidence of execution savings.
The largest completed width is coverage of this grid, not a maximum supported width.

## Reproduction and measurement scope

From the code project root, run:

```sh
python scripts/probe_scalability.py --no-memory-screen --output generated-scalability/results.json
```

The runner requires a Unix-like environment with `ps`; on macOS it also uses `sysctl` to identify the processor.
It starts a fresh process for each case, uses three timed evaluations, clears the symbolic caches before each evaluation, and requests one linear-algebra thread.
Analysis timings include the result checks but exclude imports, circuit/input construction, and cache clearing; construction time is recorded separately.
Completed workers report peak resident memory from `getrusage`, including imports, input construction, and all repetitions; this is not incremental memory of the tested function.
Stopped workers report their largest RSS sample observed by `ps`, a lower bound on the true worker peak.
The 20-second timeout covers the whole worker, including setup and all three repetitions; a timeout does not establish that one call alone takes 20 seconds.
The watchdog samples approximately every 0.2 seconds and terminates workers observed above 768 MiB.
Polling allows overshoot, so this is a stopping threshold, not a strict allocation cap.
A memory stop establishes an observed threshold crossing, not an operating-system out-of-memory failure or the memory required to finish.
Worker elapsed time includes imports, setup, analysis, monitoring, and termination; it is distinct from analysis timing.
No manuscript build or test suite ran concurrently with this batch; ordinary host activity was not isolated.
Machine-specific timings should not be treated as portable performance guarantees or fair speedup comparisons across these different tasks.

Recorded environment: Apple M1 Max; macOS-26.7.1-arm64-arm-64bit-Mach-O; Python 3.13.15, Qiskit 2.5.1, NumPy 2.5.2, SymPy 1.14.0.

The report hashes the package Python files and probe script in `source_sha256`; it excludes tests, documentation, and the result file itself.
The JSON records `memory_screen_enabled: false`; its allocation estimates did not determine which cases ran and are not used in the result tables.

## Grids and recorded outcomes

`register_qubits` (`n`) always means total register width, including idle qubits.
`rotation_count` (`r`) counts the independent `RZ(pi/4)` gates in the non-Clifford term-growth sweep; it is null for other families, not a count of every rotation appearing in their circuits.
The structural, Clifford, local-certificate, and compiler families share `n = 64, 256, 1024, 4096, 8192`.
The non-Clifford family fixes `n = 64` and varies `r = 4, 8, 12, 24`.
Statevector and dense families retain smaller width grids for their exponentially sized representations.
The schema-version-2 JSON records the complete grid in `declared_cases` and retains both parameters in every result, including workers stopped by memory monitoring or timeout.

The following table includes every declared case from this one batch.
Times are median analysis seconds, displayed with three significant figures.
MiB means the worker peak for completed cases and the largest sampled RSS for stopped cases, including timeouts.
A dash in the Seconds column means no completed analysis timing is available; stopped workers may have performed partial work.
Completed means all repetitions and checks finished, including compiler calls that decline to certify the known-equivalent pair.
Table 10 groups these same records, displaying timing and memory for the largest completed width or rotation count in each family and sampled memory separately for every stopped case.

| Computation | n | r | Outcome | Seconds | MiB |
|---|---:|---:|---|---:|---:|
| Structural cone | 64 | — | Completed | 9.74e-05 | 109.3 |
| Structural cone | 256 | — | Completed | 4.30e-04 | 110.0 |
| Structural cone | 1,024 | — | Completed | 7.71e-03 | 142.5 |
| Structural cone | 4,096 | — | Completed | 1.26e-01 | 676.0 |
| Structural cone | 8,192 | — | Memory stop | — | 1028.8 |
| Clifford Pauli propagation | 64 | — | Completed | 2.70e-02 | 109.5 |
| Clifford Pauli propagation | 256 | — | Completed | 1.05e-01 | 110.7 |
| Clifford Pauli propagation | 1,024 | — | Completed | 4.32e-01 | 112.9 |
| Clifford Pauli propagation | 4,096 | — | Completed | 1.72e+00 | 117.6 |
| Clifford Pauli propagation | 8,192 | — | Completed | 3.55e+00 | 124.5 |
| Exact local channel proof | 64 | — | Completed | 1.33e-03 | 108.6 |
| Exact local channel proof | 256 | — | Completed | 2.34e-03 | 109.1 |
| Exact local channel proof | 1,024 | — | Completed | 4.20e-03 | 110.5 |
| Exact local channel proof | 4,096 | — | Completed | 1.41e-02 | 117.7 |
| Exact local channel proof | 8,192 | — | Completed | 3.30e-02 | 126.9 |
| Compiler filter | 64 | — | Completed | 8.69e-03 | 117.9 |
| Compiler filter | 256 | — | Completed | 1.32e-02 | 121.6 |
| Compiler filter | 1,024 | — | Completed | 2.67e-02 | 132.6 |
| Compiler filter | 4,096 | — | Completed | 7.74e-02 | 174.4 |
| Compiler filter | 8,192 | — | Completed | 1.61e-01 | 213.9 |
| Non-Clifford propagation | 64 | 4 | Completed | 3.72e-03 | 109.9 |
| Non-Clifford propagation | 64 | 8 | Completed | 1.87e-02 | 111.1 |
| Non-Clifford propagation | 64 | 12 | Completed | 6.06e-02 | 113.1 |
| Non-Clifford propagation | 64 | 24 | Completed | 1.63e-01 | 113.0 |
| Statevector planner | 12 | — | Completed | 1.55e-03 | 111.1 |
| Statevector planner | 16 | — | Completed | 4.39e-03 | 114.8 |
| Statevector planner | 20 | — | Completed | 8.92e-02 | 176.0 |
| Statevector planner | 22 | — | Completed | 5.57e-01 | 367.7 |
| Statevector planner | 24 | — | Memory stop | — | 879.0 |
| Dense numerical observable | 4 | — | Completed | 4.14e-04 | 111.6 |
| Dense numerical observable | 6 | — | Completed | 5.56e-04 | 111.5 |
| Dense numerical observable | 8 | — | Completed | 6.86e-03 | 130.4 |
| Dense numerical observable | 10 | — | Completed | 2.36e-01 | 285.3 |
| Dense numerical observable | 12 | — | Memory stop | — | 1263.5 |
| Dense symbolic channel proof | 4 | — | Completed | 2.69e-02 | 111.3 |
| Dense symbolic channel proof | 6 | — | Completed | 1.21e+00 | 112.2 |
| Dense symbolic channel proof | 8 | — | Timed out | — | 116.5 |
| Dense symbolic channel proof | 10 | — | Timed out | — | 199.3 |
| Dense symbolic channel proof | 12 | — | Memory stop | — | 1138.2 |

The batch contains 39 cases: 33 completed, 4 were stopped by the memory watchdog, and 2 timed out.
Every completed case passed its result checks.

## Inputs, results, and boundaries

- **Structural cone:** an ordered chain of `n-1` CX gates with `Z` measured on the last qubit. Its initial backward cone contains all `n` qubits, exercising the implementation's storage of every boundary support. It completes through 4,096 qubits with a worker peak of 676.0 MiB; the 8,192-qubit worker crosses the memory threshold.
- **Clifford Pauli propagation:** the same chain maps the output to one full-weight `Z` string. At 8,192 qubits it remains one term, with zero discarded mass and full-register dense fallback disabled. This is a helper-level calculation and does not allocate an input state.
- **Exact local channel proof:** after a common CX chain and Hadamard, compare a SWAP on the two endpoint qubits with its three-CX implementation. Exact context cancellation leaves a two-qubit proof even at 8,192 register qubits. This helper does not compile or simulate a state.
- **Compiler filter:** apply the actual pinned compiler-plus-proof filter to the same known-equivalent pair at all shared widths through 8,192 qubits. It completes but declines all five pairs because the compiled representations do not match. This demonstrates the sufficient filter's incompleteness; the separate local proof succeeds.
- **Non-Clifford term growth:** on a 64-qubit register, conjugate an `X` string by independent `RZ(pi/4)` gates on 4, 8, 12, or 24 active qubits. The exact expansion has `2^r` terms for `r` rotations. The first two cases fit the 256-term cap. At 12 and 24 rotations, the complete expansion becomes unavailable and the recorded discarded-mass bounds are approximately 28.97 and 38.48, despite the true observable norm being one. The generic observable-difference allowance `2*epsilon` then exceeds the full range of this Pauli test; bounded computation has not preserved useful precision.
- **End-to-end statevector planner:** prepare a GHZ state from a supplied basis-state vector, append a 0.4-radian Z-axis offset, and measure the full X string. The expected gap is `1-cos(0.4)`. The planner reproduces it through 22 qubits, including validation and mutation-site state preparation. Equivalence filtering and pilots are disabled, the dense-operator limit is zero, and the state limit is explicitly raised to the tested width. The 24-qubit worker crosses the memory threshold. This is one candidate in a structured family, not an arbitrary 22-qubit workload or a change to the ten-qubit default state limit.
- **Dense numerical fallback:** explicitly materialize a propagated observable through H, RZ, and CX gates. The result agrees with its known Pauli matrix through ten qubits; the 12-qubit worker crosses the memory threshold. The ten-qubit output alone contains 16 MiB of complex entries, while intermediates, validation, and imports explain the larger process peak.
- **Full-register symbolic proof:** compare the SWAP and three-CX circuits without local cancellation, raising the symbolic limit for the probe. Four and six qubits complete; eight and ten exceed the 20-second, three-repeat worker budget. The 12-qubit worker crosses the memory threshold. These simple Clifford-containing circuits do not bound the expression growth of arbitrary exact-angle circuits.

The 8,192-qubit results do not imply that the complete planner can accept a general 8,192-qubit state: its public test input remains a dense `Statevector` or `DensityMatrix`.
A compact state interface, tableau-based expectations, and automatic state simulation on causal cones are separate extensions.

## Pilot shots are a separate constraint

The following counts are calculated from the existing median-of-means formula, not collected hardware shots or simulated pilot runs.
They use 256 targets, per-target error 0.25, and per-call failure probability 0.05, giving 69 batches.
The planner's population allocation can require a smaller per-call failure probability and more batches.

| Maximum Pauli weight | Required shots |
|---:|---:|
| 1 | 13,248 |
| 4 | 357,696 |
| 8 | 28,973,376 |
| 12 | 2,346,843,456 |
| 16 | 190,094,319,936 |

Thus the default 250,000-shot ceiling is exceeded already at weight four for this target set and precision.
A Clifford suffix can keep a single Pauli term compact while increasing its weight; cheap classical propagation therefore does not guarantee cheap shadow estimation.

## Interpretation

The four-qubit setting limits a symbolic fallback, not the identities or all planning rules.
The most useful next implementation extensions are compact state inputs and tableau or causal-cone state evaluation; simply raising a dense limit does not address exponential representation growth.
Non-Clifford propagation must be judged by both retained-term count and the resulting error interval.
Any performance comparison across mutation populations, tests, or execution policies still requires the separate frozen Study B design.
