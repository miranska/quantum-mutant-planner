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
    result = probe.run_case('structural', case, SimpleNamespace(memory_mib=768))
    assert result['status'] == 'not_run_memory_screen'
    assert result['register_qubits'] == 4096
    assert result['rotation_count'] is None
    assert 'process_peak_rss_bytes' not in result


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
