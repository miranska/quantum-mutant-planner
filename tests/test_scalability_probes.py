"""Small correctness cases for the reproducible scaling harness and norm bound."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, random_density_matrix


def load_probe():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'probe_scalability.py'
    spec = importlib.util.spec_from_file_location('scalability_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('family', [
    'structural', 'clifford_pauli', 'local_certificate', 'compiler_filter',
    'nonclifford_terms', 'statevector_planner', 'dense_numeric', 'dense_symbolic',
])
def test_probe_worker_checks_small_cases(family):
    rotations = 4 if family == 'nonclifford_terms' else None
    width = 64 if rotations is not None else 4
    result = load_probe().worker(family, width, 1, rotations)
    assert result['median_analysis_seconds'] >= 0
    assert result['process_peak_rss_bytes'] > 0
    assert result['register_qubits'] == width
    assert result['rotation_count'] == rotations
    if family == 'compiler_filter':
        assert result['result']['known_equivalent'] is True
        assert isinstance(result['result']['filter_accepted'], bool)


def test_probe_memory_screen_avoids_large_allocations():
    probe = load_probe()
    assert probe.memory_estimate('dense_numeric', 12) > 768 * 2**20
    assert probe.memory_estimate('statevector_planner', 24) > 768 * 2**20
    assert probe.memory_estimate('clifford_pauli', 4096) <= 768 * 2**20


def test_probe_grids_share_widths_and_separate_rotation_counts():
    probe = load_probe()
    for family in ['structural', 'clifford_pauli', 'local_certificate', 'compiler_filter']:
        assert [case['register_qubits'] for case in probe.FAMILIES[family]] == [64, 256, 1024, 4096, 8192]
        assert all(case['rotation_count'] is None for case in probe.FAMILIES[family])
    cases = probe.FAMILIES['nonclifford_terms']
    assert {case['register_qubits'] for case in cases} == {64}
    assert [case['rotation_count'] for case in cases] == [4, 8, 12, 24]


def test_screened_case_keeps_parameters_without_starting_worker(monkeypatch):
    probe = load_probe()

    def unexpected_worker(*args, **kwargs):
        pytest.fail('a screened case must not start a worker')

    monkeypatch.setattr(probe.subprocess, 'Popen', unexpected_worker)
    case = {'register_qubits': 4096, 'rotation_count': None}
    result = probe.run_case('structural', case, SimpleNamespace(memory_mib=768, memory_screen=True))
    assert result['status'] == 'not_run_memory_screen'
    assert result['register_qubits'] == 4096
    assert result['rotation_count'] is None
    assert 'process_peak_rss_bytes' not in result


@pytest.mark.parametrize('sampled_rss_kib, expected_status', [
    (800 * 1024, 'memory_watchdog'),
    (100 * 1024, 'timeout'),
    (None, 'failed'),
])
def test_bypassing_screen_keeps_guards_and_records_observed_rss(monkeypatch, sampled_rss_kib, expected_status):
    probe = load_probe()

    class Worker:
        pid = 123
        returncode = None
        killed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            if not self.killed:
                raise probe.subprocess.TimeoutExpired('worker', timeout)
            return '', ''

        def kill(self):
            self.killed = True
            self.returncode = -9

    worker = Worker()
    monkeypatch.setattr(probe.subprocess, 'Popen', lambda *args, **kwargs: worker)
    monkeypatch.setattr(probe.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(stdout='' if sampled_rss_kib is None else str(sampled_rss_kib)))
    elapsed = 21. if expected_status == 'timeout' else 1.
    ticks = iter([0., elapsed, elapsed])
    monkeypatch.setattr(probe.time, 'monotonic', lambda: next(ticks))
    args = SimpleNamespace(memory_mib=768, memory_screen=False, repeats=3, timeout_seconds=20.)
    result = probe.run_case('structural', {'register_qubits': 4096, 'rotation_count': None}, args)
    assert result['estimated_peak_bytes'] > 768 * 2**20
    assert result['status'] == expected_status
    assert worker.killed
    assert result['sampled_peak_rss_bytes'] == (sampled_rss_kib or 0) * 1024
    assert result['worker_wall_seconds'] == elapsed
    assert 'process_peak_rss_bytes' not in result
    if expected_status == 'failed':
        assert 'watchdog' in result['error']


@pytest.mark.parametrize('rotation_count', [None, 0, 65])
def test_nonclifford_probe_requires_valid_rotation_count(rotation_count):
    with pytest.raises(ValueError, match='rotation_count'):
        load_probe().prepare('nonclifford_terms', 64, rotation_count)


@pytest.mark.parametrize('seed', range(5))
def test_coefficient_remainder_bound_contains_actual_gap(seed):
    rng = np.random.default_rng(seed)
    h = SparsePauliOp.from_list(list(zip(['XI', 'YZ', 'ZZ'], rng.normal(size=3))))
    o = SparsePauliOp.from_list(list(zip(['IY', 'XZ', 'ZX'], rng.normal(size=3))))
    hm, om = h.to_matrix(), o.to_matrix()
    rho = random_density_matrix(4, seed=seed).data
    eta = .1 * (-1)**seed
    eigenvalues, vectors = np.linalg.eigh(hm)
    unitary = (vectors * np.exp(-1j * eta * eigenvalues)) @ vectors.conj().T
    gap = abs(np.trace((om - unitary.conj().T @ om @ unitary) @ rho))
    commutator = hm @ om - om @ hm
    sensitivity = abs(np.trace(-1j * commutator @ rho))
    exact_remainder = eta**2 * np.linalg.norm(hm @ commutator - commutator @ hm, ord=2) / 2
    coefficient_remainder = 2 * eta**2 * np.sum(np.abs(h.coeffs))**2 * np.sum(np.abs(o.coeffs))
    assert exact_remainder <= coefficient_remainder + 1e-12
    assert max(0, abs(eta) * sensitivity - coefficient_remainder) <= gap + 1e-12
    assert gap <= abs(eta) * sensitivity + coefficient_remainder + 1e-12
