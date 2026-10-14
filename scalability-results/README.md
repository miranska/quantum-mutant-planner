# Synthetic scalability probes

Recorded on 2026-10-13 using [`probe_scalability.py`](../scripts/probe_scalability.py).
The complete machine-readable grid, including skipped and timed-out cases, is in [`reference.json`](reference.json).
These constructed cases have known answers; they are implementation feasibility checks, not Studies A--D, a benchmark-corpus evaluation, or evidence of execution savings.
The largest completed width is coverage of this grid, not a maximum supported width.

## Reproduction and measurement scope

From the code project root, run:

```sh
python scripts/probe_scalability.py --output generated-scalability/results.json
```

The runner requires a Unix-like environment with `ps`; on macOS it also uses `sysctl` to identify the processor.
It starts a fresh process for each case, uses three timed evaluations, clears the symbolic caches before each evaluation, and requests one linear-algebra thread.
Analysis timings include the result checks but exclude imports, circuit/input construction, and cache clearing; construction time is recorded separately.
Peak resident memory covers the entire worker, including imports, input construction, and all repetitions, so it is not the incremental memory of the tested function.
The 20-second timeout covers the whole worker, including setup and all three repetitions; a timeout does not establish that one call alone takes 20 seconds.
A conservative allocation estimate screens large cases before execution; a polled RSS watchdog also terminates workers above 768 MiB.
The watchdog samples approximately every 0.2 seconds and is not a strict allocation cap.
A screened case is not an observed memory failure.
There was no concurrent manuscript build during the recorded final run; ordinary host activity was not isolated.
Machine-specific timings should not be treated as portable performance guarantees or fair speedup comparisons across these different tasks.

Recorded environment: Apple M1 Max; macOS-26.7.1-arm64-arm-64bit-Mach-O; Python 3.13.15, Qiskit 2.5.1, NumPy 2.5.2, SymPy 1.14.0.

The report hashes the package Python files and probe script in `source_sha256`; it excludes tests, documentation, and the result file itself.

## Grids and recorded outcomes

`register_qubits` (`n`) always means total register width, including idle qubits.
`rotation_count` (`r`) counts the independent `RZ(pi/4)` gates in the non-Clifford term-growth sweep; it is null for other families, not a count of every rotation appearing in their circuits.
The structural, Clifford, local-certificate, and compiler families share `n = 64, 256, 1024, 4096, 8192`.
The non-Clifford family fixes `n = 64` and varies `r = 4, 8, 12, 24`.
Statevector and dense families retain smaller width grids for their exponentially sized representations.
The schema-version-2 JSON records the complete grid in `declared_cases` and retains both parameters in every result, including screened and timed-out cases.

The following table includes every declared case.
Times are medians in seconds; peak worker RSS and preflight screen estimates are in MiB.
The screen estimate is not a memory measurement.
A dash in the time or peak column means no completed measurement is available; timeout workers may have performed partial work.
Completed means all repetitions and checks finished, including compiler calls that decline to certify the known-equivalent pair.

| Computation | n | r | Outcome | Seconds | Peak MiB | Screen estimate MiB |
|---|---:|---:|---|---:|---:|---:|
| Structural cone | 64 | — | Completed | 0.0001025 | 109.7 | 256.127 |
| Structural cone | 256 | — | Completed | 0.0006413 | 112.6 | 258.008 |
| Structural cone | 1,024 | — | Completed | 0.008132 | 140.5 | 288.031 |
| Structural cone | 4,096 | — | Screened out | — | — | 768.125 |
| Structural cone | 8,192 | — | Screened out | — | — | 2304.25 |
| Clifford Pauli propagation | 64 | — | Completed | 0.02693 | 111.6 | 256 |
| Clifford Pauli propagation | 256 | — | Completed | 0.1068 | 112.1 | 256 |
| Clifford Pauli propagation | 1,024 | — | Completed | 0.4398 | 113.4 | 256 |
| Clifford Pauli propagation | 4,096 | — | Completed | 1.755 | 119.2 | 256 |
| Clifford Pauli propagation | 8,192 | — | Completed | 3.605 | 124.6 | 256 |
| Exact local channel proof | 64 | — | Completed | 0.001318 | 110.4 | 256 |
| Exact local channel proof | 256 | — | Completed | 0.002329 | 110.7 | 256 |
| Exact local channel proof | 1,024 | — | Completed | 0.004133 | 111.6 | 256 |
| Exact local channel proof | 4,096 | — | Completed | 0.01415 | 119.2 | 256 |
| Exact local channel proof | 8,192 | — | Completed | 0.03181 | 127.5 | 256 |
| Compiler filter | 64 | — | Completed | 0.008771 | 120.4 | 256 |
| Compiler filter | 256 | — | Completed | 0.01274 | 122.2 | 256 |
| Compiler filter | 1,024 | — | Completed | 0.03172 | 134.1 | 256 |
| Compiler filter | 4,096 | — | Completed | 0.07848 | 176.5 | 256 |
| Compiler filter | 8,192 | — | Completed | 0.1631 | 213.6 | 256 |
| Non-Clifford propagation | 64 | 4 | Completed | 0.004198 | 111.1 | 256 |
| Non-Clifford propagation | 64 | 8 | Completed | 0.01996 | 112.4 | 256 |
| Non-Clifford propagation | 64 | 12 | Completed | 0.06113 | 113.5 | 256 |
| Non-Clifford propagation | 64 | 24 | Completed | 0.1621 | 113.7 | 256 |
| Statevector planner | 12 | — | Completed | 0.001965 | 111.7 | 256.25 |
| Statevector planner | 16 | — | Completed | 0.006235 | 121.2 | 260 |
| Statevector planner | 20 | — | Completed | 0.09212 | 175.9 | 320 |
| Statevector planner | 22 | — | Completed | 0.5747 | 368.5 | 512 |
| Statevector planner | 24 | — | Screened out | — | — | 1280 |
| Dense numerical observable | 4 | — | Completed | 0.0004848 | 112.9 | 256.023 |
| Dense numerical observable | 6 | — | Completed | 0.0005692 | 113.2 | 256.375 |
| Dense numerical observable | 8 | — | Completed | 0.006839 | 133.0 | 262 |
| Dense numerical observable | 10 | — | Completed | 0.2399 | 269.9 | 352 |
| Dense numerical observable | 12 | — | Screened out | — | — | 1792 |
| Dense symbolic channel proof | 4 | — | Completed | 0.02713 | 112.4 | 256.094 |
| Dense symbolic channel proof | 6 | — | Completed | 1.218 | 113.2 | 257.5 |
| Dense symbolic channel proof | 8 | — | Timed out | — | — | 280 |
| Dense symbolic channel proof | 10 | — | Timed out | — | — | 640 |
| Dense symbolic channel proof | 12 | — | Screened out | — | — | 6400 |

The full grid contains 39 cases: 32 completed, 5 were screened out before execution, and 2 timed out. No correctness check failed in the recorded run.

## Inputs, results, and boundaries

- **Structural cone:** an ordered chain of `n-1` CX gates with `Z` measured on the last qubit. Its initial backward cone contains all `n` qubits, exercising the implementation's storage of every boundary support. The screen estimates `256 MiB + 64*n*(n+1)/2 bytes`, accounting for every stored boundary support. This gives 768.125 MiB at 4,096 qubits and 2,304.25 MiB at 8,192, so both cases were screened out before any worker started. The first exceeds the 768 MiB policy by just 128 KiB; neither is an observed memory failure, and no maximum width was measured.
- **Clifford Pauli propagation:** the same chain maps the output to one full-weight `Z` string. At 8,192 qubits it remains one term, with zero discarded mass and full-register dense fallback disabled. This is a helper-level calculation and does not allocate an input state.
- **Exact local channel proof:** after a common CX chain and Hadamard, compare a SWAP on the two endpoint qubits with its three-CX implementation. Exact context cancellation leaves a two-qubit proof even at 8,192 register qubits. This helper does not compile or simulate a state.
- **Compiler filter:** apply the actual pinned compiler-plus-proof filter to the same known-equivalent pair at all shared widths through 8,192 qubits. It completes but declines all five pairs because the compiled representations do not match. This demonstrates the sufficient filter's incompleteness; the separate local proof succeeds.
- **Non-Clifford term growth:** on a 64-qubit register, conjugate an `X` string by independent `RZ(pi/4)` gates on 4, 8, 12, or 24 active qubits. The exact expansion has `2^r` terms for `r` rotations. The first two cases fit the 256-term cap. At 12 and 24 rotations, the complete expansion becomes unavailable and the recorded discarded-mass bounds are approximately 28.97 and 38.48, despite the true observable norm being one. The generic observable-difference allowance `2*epsilon` then exceeds the full range of this Pauli test; bounded computation has not preserved useful precision.
- **End-to-end statevector planner:** prepare a GHZ state from a supplied basis-state vector, append a 0.4-radian Z-axis offset, and measure the full X string. The expected gap is `1-cos(0.4)`. The planner reproduces it through 22 qubits, including validation and mutation-site state preparation. Equivalence filtering and pilots are disabled, the dense-operator limit is zero, and the state limit is explicitly raised to the tested width. The 24-qubit case is screened out. This is one candidate in a structured family, not an arbitrary 22-qubit workload or a change to the ten-qubit default state limit.
- **Dense numerical fallback:** explicitly materialize a propagated observable through H, RZ, and CX gates. The result agrees with its known Pauli matrix through ten qubits; the 12-qubit case is screened out. The ten-qubit output alone contains 16 MiB of complex entries, while intermediates, validation, and imports explain the larger process peak.
- **Full-register symbolic proof:** compare the SWAP and three-CX circuits without local cancellation, raising the symbolic limit for the probe. Four and six qubits complete; eight and ten exceed the 20-second, three-repeat worker budget. Twelve is screened out. These simple Clifford-containing circuits do not bound the expression growth of arbitrary exact-angle circuits.

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
