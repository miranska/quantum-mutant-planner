"""Bounded synthetic feasibility probes, separate from the empirical studies.

Each case runs in a fresh process. Timings include the requested analysis but
exclude imports and case construction; peak RSS includes the whole worker.
These deliberately structured cases do not establish a maximum supported width.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
import sympy

from quantum_mutant_planner.analysis import propagate_observable, structural_backward_cones, transpiler_equivalent
from quantum_mutant_planner import certificates
from quantum_mutant_planner.models import ExecutionBudget, PlannerSettings, PlanningProblem, QuantumTest, TranspilerSettings
from quantum_mutant_planner.mutations import pauli_rotation_mutation
from quantum_mutant_planner.planner import plan_mutant_executions
from quantum_mutant_planner.pilot import median_of_means_shot_plan


REGISTER_WIDTHS = (64, 256, 1024, 4096, 8192)
ROTATION_COUNTS = (4, 8, 12, 24)


def register_grid(widths):
    """Declare cases that vary register width, without a rotation-count sweep."""
    return [{'register_qubits': width, 'rotation_count': None} for width in widths]


FAMILIES = {
    'structural': register_grid(REGISTER_WIDTHS),
    'clifford_pauli': register_grid(REGISTER_WIDTHS),
    'local_certificate': register_grid(REGISTER_WIDTHS),
    'compiler_filter': register_grid(REGISTER_WIDTHS),
    'nonclifford_terms': [{'register_qubits': 64, 'rotation_count': count} for count in ROTATION_COUNTS],
    'statevector_planner': register_grid((12, 16, 20, 22, 24)),
    'dense_numeric': register_grid((4, 6, 8, 10, 12)),
    'dense_symbolic': register_grid((4, 6, 8, 10, 12)),
}


def chain(width: int) -> QuantumCircuit:
    """Construct an ordered CX chain whose backward Z cone spans the register."""
    circuit = QuantumCircuit(width)
    for index in range(width - 1):
        circuit.cx(index, index + 1)
    return circuit


def prepare(family: str, register_qubits: int, rotation_count: int | None = None):
    """Construct one fixed, analytically checkable case and its timed operation."""
    width = register_qubits
    if width < 2:
        raise ValueError('register_qubits must be at least two')
    if family == 'nonclifford_terms':
        if rotation_count is None or not 1 <= rotation_count <= width:
            raise ValueError('rotation_count must be between one and register_qubits')
    elif rotation_count is not None:
        raise ValueError('rotation_count applies only to nonclifford_terms')
    details = {'register_qubits': width, 'rotation_count': rotation_count}
    if family == 'structural':
        circuit = chain(width)
        observable = SparsePauliOp('Z' + 'I' * (width - 1))

        def run():
            cones = structural_backward_cones(circuit, observable)
            assert len(cones[0]) == width
            return {'initial_cone_qubits': len(cones[0]), 'stored_boundaries': len(cones)}

    elif family in {'clifford_pauli', 'nonclifford_terms'}:
        circuit = chain(width) if family == 'clifford_pauli' else QuantumCircuit(width)
        observable = SparsePauliOp('Z' + 'I' * (width - 1))
        if family == 'nonclifford_terms':
            observable = SparsePauliOp('I' * (width - rotation_count) + 'X' * rotation_count)
            for index in range(rotation_count):
                circuit.rz(np.pi / 4, index)
            details['untruncated_term_count'] = 2**rotation_count
        details['term_cap'] = 256

        def run():
            result = propagate_observable(circuit, observable, 0, 256, 1e-12, dense_max_qubits=0)
            assert result.exact_operator is None
            assert len(result.retained_paulis) <= 256
            if family == 'clifford_pauli':
                assert result.symbolic_paulis.paulis.to_labels() == ['Z' * width]
                assert result.truncation_error == 0
            else:
                assert len(result.retained_paulis) == min(256, 2**rotation_count)
                assert (result.complete_paulis is not None) == (rotation_count <= 8)
            return {
                'retained_terms': len(result.retained_paulis),
                'complete_expansion': result.complete_paulis is not None,
                'discarded_mass_bound': result.truncation_error,
            }

    elif family in {'local_certificate', 'compiler_filter', 'dense_symbolic'}:
        circuit = chain(width)
        circuit.h(0)
        mutant = circuit.copy()
        circuit.swap(0, width - 1)
        mutant.cx(0, width - 1)
        mutant.cx(width - 1, 0)
        mutant.cx(0, width - 1)
        details['changed_support_qubits'] = 2
        details['symbolic_limit'] = width if family == 'dense_symbolic' else 4

        def run():
            if family == 'local_certificate':
                proved = certificates.certify_local_channel_equivalence(circuit, mutant, 4)
            elif family == 'compiler_filter':
                proved = transpiler_equivalent(circuit, mutant, TranspilerSettings(), maximum_qubits=0, local_maximum_qubits=4)
            else:
                proved = certificates.certify_channel_equivalence(circuit, mutant, width)
            if family == 'compiler_filter':
                # A canonical compiler match is a sufficient, incomplete filter.
                return {'known_equivalent': True, 'filter_accepted': bool(proved)}
            assert proved
            return {'equivalent': bool(proved)}

    elif family == 'statevector_planner':
        circuit = QuantumCircuit(width)
        circuit.h(0)
        for index in range(width - 1):
            circuit.cx(index, index + 1)
        candidate = pauli_rotation_mutation(circuit, 'z-offset', len(circuit.data), Pauli('Z'), 0, .4, qubits=(0,), insert=True)
        problem = PlanningProblem(circuit, QuantumTest(Statevector.from_int(0, 2**width), SparsePauliOp('X' * width), .01, 2.),
                                  (candidate,), ExecutionBudget(1000, .05))
        settings = PlannerSettings(dense_max_qubits=0, symbolic_max_qubits=0, exact_state_max_qubits=width,
                                   exact_simulation_max_qubits=0, enable_equivalence=False, enable_pilot=False)
        details.update({'dense_limit': 0, 'state_limit': width, 'input_state_bytes': problem.test.state.data.nbytes})

        def run():
            record = plan_mutant_executions(problem, settings)[0]
            assert record.evidence.provenance == 'structured gap'
            assert abs(record.evidence.gap - (1 - np.cos(.4))) < 1e-12
            return {'gap': record.evidence.gap, 'status': record.status.value}

    elif family == 'dense_numeric':
        circuit = QuantumCircuit(width)
        circuit.h(0)
        circuit.rz(.31, 0)
        circuit.cx(0, width - 1)
        observable = SparsePauliOp('Z' + 'I' * (width - 1))
        propagation = propagate_observable(circuit, observable, 0, 256, 1e-12, dense_max_qubits=width)
        details['dense_limit'] = width

        def run():
            result = propagation.exact_operator
            expected = SparsePauliOp('Z' + 'I' * (width - 2) + 'X').to_matrix()
            error = float(np.max(np.abs(result.data - expected)))
            assert error < 1e-12
            return {'operator_bytes': result.data.nbytes, 'maximum_entry_error': error}

    else:
        raise ValueError(f'unknown family: {family}')
    details['original_gates'] = len(circuit.data)
    return run, details


def memory_estimate(family: str, register_qubits: int) -> int:
    """Conservatively screen known large allocations; this is not measured RSS."""
    baseline = 256 * 2**20
    if family == 'structural':
        return baseline + 64 * register_qubits * (register_qubits + 1) // 2
    if family == 'statevector_planner':
        return baseline + 4 * 16 * 2**register_qubits
    if family == 'dense_numeric':
        return baseline + 6 * 16 * 4**register_qubits
    if family == 'dense_symbolic':
        return baseline + 6 * 64 * 4**register_qubits
    return baseline


def worker(family: str, register_qubits: int, repeats: int, rotation_count: int | None = None) -> dict:
    """Measure analysis separately from construction and check each result."""
    start = time.perf_counter()
    run, details = prepare(family, register_qubits, rotation_count)
    setup_seconds = time.perf_counter() - start
    times = []
    for _ in range(repeats):
        # Report uncached exact comparisons, not an already memoized answer.
        certificates._simplify_scalar.cache_clear()
        certificates._compose_circuit.cache_clear()
        certificates._conjugated_observable.cache_clear()
        start = time.perf_counter()
        result = run()
        times.append(time.perf_counter() - start)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {**details, 'result': result, 'analysis_seconds': times,
            'median_analysis_seconds': statistics.median(times), 'setup_seconds': setup_seconds,
            'process_peak_rss_bytes': int(rss if sys.platform == 'darwin' else rss * 1024)}


def run_case(family: str, case: dict, args: argparse.Namespace) -> dict:
    """Run a case with a preflight screen, worker timeout, and RSS watchdog."""
    record = {'family': family, **case, 'estimated_peak_bytes': memory_estimate(family, case['register_qubits'])}
    ceiling = args.memory_mib * 2**20
    if record['estimated_peak_bytes'] > ceiling:
        return {**record, 'status': 'not_run_memory_screen'}
    environment = dict(os.environ)
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']:
        environment[name] = '1'
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', family,
               '--register-qubits', str(case['register_qubits']), '--repeats', str(args.repeats)]
    if case['rotation_count'] is not None:
        command.extend(['--rotation-count', str(case['rotation_count'])])
    start = time.monotonic()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment) as process:
        status = None
        while process.poll() is None:
            try:
                stdout, stderr = process.communicate(timeout=.2)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - start > args.timeout_seconds:
                    status = 'timeout'
                current = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, text=True)
                if current.stdout.strip() and int(current.stdout.strip()) * 1024 > ceiling:
                    status = 'memory_watchdog'
                if status:
                    process.kill()
                    break
        stdout, stderr = process.communicate()
        if status:
            return {**record, 'status': status, 'worker_wall_seconds': time.monotonic() - start}
        if process.returncode:
            return {**record, 'status': 'failed', 'error': stderr[-4000:]}
        return {**record, 'status': 'completed', **json.loads(stdout)}


def source_digest() -> str:
    """Hash the package and probe source so results identify their implementation."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted([*root.joinpath('src').rglob('*.py'), Path(__file__).resolve()]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    """Run the declared grid and save every completion, failure, and boundary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('generated-scalability/results.json'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout-seconds', type=float, default=20.)
    parser.add_argument('--memory-mib', type=int, default=768)
    parser.add_argument('--families', nargs='+', choices=list(FAMILIES), default=list(FAMILIES))
    parser.add_argument('--worker', choices=list(FAMILIES))
    parser.add_argument('--register-qubits', type=int, help='worker register width, including idle qubits')
    parser.add_argument('--rotation-count', type=int, help='number of independent rotations for a non-Clifford worker')
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout_seconds <= 0 or args.memory_mib <= 0:
        parser.error('repeats, timeout, and memory ceiling must be positive')
    if args.worker:
        if args.register_qubits is None:
            parser.error('--worker requires --register-qubits')
        print(json.dumps(worker(args.worker, args.register_qubits, args.repeats, args.rotation_count)))
        return
    if args.register_qubits is not None or args.rotation_count is not None:
        parser.error('--register-qubits and --rotation-count require --worker')
    # Refuse to start if the RSS watchdog cannot inspect worker processes.
    subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], check=True, capture_output=True)
    cpu_model = platform.processor()
    if sys.platform == 'darwin':
        cpu_model = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    report = {
        'schema_version': 2,
        'purpose': 'synthetic implementation feasibility, not Studies A--D or a maximum-width benchmark',
        'platform': platform.platform(), 'processor': cpu_model, 'python': platform.python_version(),
        'qiskit': qiskit.__version__, 'numpy': np.__version__, 'sympy': sympy.__version__,
        'source_sha256': source_digest(), 'repeats': args.repeats, 'worker_timeout_seconds': args.timeout_seconds,
        'memory_ceiling_mib': args.memory_mib, 'linear_algebra_threads': 1,
        'declared_cases': {family: FAMILIES[family] for family in args.families},
        'timing_scope': 'analysis excludes imports and setup; peak RSS includes the worker; symbolic caches cleared per repeat',
        'shadow_shot_plans': [
            {'maximum_weight': weight, 'target_count': 256, 'target_error': .25,
             'failure_probability': .05, 'shots': median_of_means_shot_plan(weight, 256, .25, .05)[0]}
            for weight in [1, 4, 8, 12, 16]
        ],
        'cases': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for family in args.families:
        for case in FAMILIES[family]:
            record = run_case(family, case, args)
            report['cases'].append(record)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            rotation_label = '-' if case['rotation_count'] is None else str(case['rotation_count'])
            print(f"{family:22} n={case['register_qubits']:5} r={rotation_label:>2} "
                  f"{record['status']:22} {record.get('median_analysis_seconds', '')}", flush=True)
    if any(case['status'] == 'failed' for case in report['cases']):
        raise SystemExit('one or more correctness checks failed; inspect the report')


if __name__ == '__main__':
    main()
