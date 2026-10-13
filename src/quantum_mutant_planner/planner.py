"""Implementations of the manuscript's Algorithms 1 and 2."""

from __future__ import annotations

from dataclasses import replace
import logging
import math

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator, Pauli, SparsePauliOp

from quantum_mutant_planner.analysis import (
    PropagationResult,
    _truncate_paulis,
    direct_signed_gap,
    embed_local_operator,
    embed_local_pauli,
    information_theoretic_strength_floor,
    is_clifford_relative,
    propagate_observable,
    real_expectation,
    state_at_mutation,
    structural_backward_cones,
    transpiler_equivalent,
    validate_problem,
)
from quantum_mutant_planner.models import (
    DecisionStatus,
    EffectEvidence,
    EvidenceType,
    MutationCandidate,
    MutationKind,
    PlannerSettings,
    PlanningProblem,
    PlanningRecord,
    RankingObjective,
    ZeroProvenance,
)
from quantum_mutant_planner.pilot import MedianOfMeansPilot, PilotProvider
from quantum_mutant_planner.certificates import certify_circuit_zero
from quantum_mutant_planner.pauli import local_conjugation, rotation_difference

LOGGER = logging.getLogger(__name__)


def _efficiency(gap: float, outcome_range: float, cost: float) -> float:
    """Compute the normalized Hoeffding expectation-estimation score.

    :param gap: Absolute observable gap.
    :param outcome_range: Observable spectral range.
    :param cost: Positive paired-comparison cost.
    :return: Normalized squared-gap efficiency.
    """

    return gap**2 / (outcome_range**2 * cost)


def _operator_norm(matrix: np.ndarray) -> float:
    """Compute the spectral operator norm of a dense matrix.

    :param matrix: Dense square matrix.
    :return: Largest singular value.
    """

    return float(np.linalg.norm(matrix, ord=2))


def _generic_difference(propagated: Operator, relative: Operator) -> Operator:
    """Construct ``O_j - W_m^dagger O_j W_m``.

    :param propagated: Complete propagated observable.
    :param relative: Full-register relative unitary.
    :return: Exact observable-side difference.
    """

    observable = propagated.data
    unitary = relative.data
    return Operator(observable - unitary.conj().T @ observable @ unitary)


def _retained_difference(
    candidate: MutationCandidate,
    propagation: PropagationResult,
    num_qubits: int,
    settings: PlannerSettings,
) -> tuple[SparsePauliOp, float]:
    """Construct and cap the retained Pauli observable difference.

    :param candidate: Candidate whose relative action is applied.
    :param propagation: Complete and retained propagated observable.
    :param num_qubits: Full register width.
    :param settings: Numerical and retained-term settings.
    :return: Retained difference and added discarded coefficient mass.
    """

    paulis = _sparse_difference(candidate, propagation.retained_paulis, num_qubits, settings)
    return _truncate_paulis(paulis, settings.max_pauli_terms, settings.numerical_tolerance)


def _sparse_difference(
    candidate: MutationCandidate, observable: SparsePauliOp, num_qubits: int, settings: PlannerSettings,
) -> SparsePauliOp:
    """Apply a relative action without embedding a full-register matrix.

    :param candidate: Local relative action or Pauli rotation.
    :param observable: Complete or explicitly retained observable.
    :param num_qubits: Register width for embedding a Pauli string.
    :param settings: Local dense-operation limit.
    :return: Numerical Pauli difference before any truncation.
    """

    if candidate.kind is MutationKind.PAULI_ROTATION:
        assert candidate.pauli_axis is not None and candidate.eta is not None
        axis = embed_local_pauli(candidate.pauli_axis, candidate.support, num_qubits)
        return rotation_difference(observable, axis, candidate.eta)
    return (observable - local_conjugation(
        observable, candidate.relative_operator, candidate.support, settings.local_max_qubits,
    )).simplify(atol=0.0, rtol=0.0)


def _state_available(problem: PlanningProblem, maximum_qubits: int, settings: PlannerSettings) -> bool:
    """Check state limits, including the separate density-matrix storage limit.

    :param problem: Problem supplying the input representation and width.
    :param maximum_qubits: State-preparation or direct-simulation limit.
    :param settings: Dense fallback limit.
    :return: Whether numerical state evolution is permitted.
    """

    width = problem.circuit.num_qubits
    return width <= maximum_qubits and (
        not isinstance(problem.test.state, DensityMatrix) or width <= settings.dense_max_qubits
    )


def _algebraic_pauli_zero(
    candidate: MutationCandidate,
    propagation: PropagationResult,
    num_qubits: int,
) -> bool:
    """Check exact Pauli commutation without using a numerical zero tolerance.

    :param candidate: Pauli-rotation candidate.
    :param propagation: Propagated observable and exactness metadata.
    :param num_qubits: Full register width.
    :return: True only when every complete Pauli term commutes with the axis.
    """

    if candidate.kind is not MutationKind.PAULI_ROTATION or propagation.symbolic_paulis is None:
        return False
    assert candidate.pauli_axis is not None
    axis = embed_local_pauli(candidate.pauli_axis, candidate.support, num_qubits)
    return all(
        pauli.commutes(axis)
        for pauli, coefficient in zip(propagation.symbolic_paulis.paulis, propagation.symbolic_paulis.coeffs, strict=True)
        if coefficient != 0.0
    )


def _algebraic_generator_zero(
    candidate: MutationCandidate, propagation: PropagationResult, num_qubits: int,
) -> bool:
    """Certify generator commutation from complete explicit Pauli terms.

    :param candidate: Mutation with optional explicit generator metadata.
    :param propagation: Complete symbolic observable, if available.
    :param num_qubits: Full register width.
    :return: True when every nonzero generator term commutes with every observable term.
    """

    if candidate.generator_paulis is None or propagation.symbolic_paulis is None:
        return False
    return all(
        embed_local_pauli(axis, candidate.support, num_qubits).commutes(pauli)
        for axis, weight in zip(candidate.generator_paulis.paulis, candidate.generator_paulis.coeffs, strict=True)
        if weight != 0.0
        for pauli, coefficient in zip(propagation.symbolic_paulis.paulis, propagation.symbolic_paulis.coeffs, strict=True)
        if coefficient != 0.0
    )


def _exact_structured_evidence(
    problem: PlanningProblem,
    candidate: MutationCandidate,
    difference: Operator | SparsePauliOp,
    settings: PlannerSettings,
) -> EffectEvidence:
    """Evaluate one exact structured difference in the original mutation-site state.

    :param problem: Complete planning problem.
    :param candidate: Candidate mutation.
    :param difference: Exact structured observable difference.
    :param settings: Numerical settings.
    :return: Exact structured gap evidence with explicit zero provenance.
    """

    state = state_at_mutation(problem, candidate)
    signed = real_expectation(state, difference, settings.numerical_tolerance)
    gap = abs(signed)
    zero = ZeroProvenance.NUMERICAL if gap <= settings.numerical_tolerance else ZeroProvenance.NONE
    return EffectEvidence(
        evidence_type=EvidenceType.EXACT,
        provenance="structured gap",
        gap=gap,
        signed_shift=signed,
        score=_efficiency(gap, problem.test.outcome_range, candidate.cost),
        zero_provenance=zero,
    )


def _pilot_evidence(
    problem: PlanningProblem,
    candidate: MutationCandidate,
    propagation: PropagationResult,
    settings: PlannerSettings,
    pilot_provider: PilotProvider,
) -> EffectEvidence:
    """Evaluate a retained signed Pauli difference using one pilot sample.

    :param problem: Complete planning problem.
    :param candidate: Candidate mutation.
    :param propagation: Complete and retained propagated observable.
    :param settings: Planner and pilot settings.
    :param pilot_provider: Provider of simultaneous signed Pauli estimates.
    :raises ValueError: If no retained terms are available or the pilot cannot run.
    :return: Pilot-estimated gap interval and cancellation diagnostic.
    """

    retained, added_discard = _retained_difference(candidate, propagation, problem.circuit.num_qubits, settings)
    if len(retained) == 0:
        raise ValueError("retained observable difference has no pilot targets")
    targets = tuple(Pauli(pauli) for pauli in retained.paulis)
    pilot = pilot_provider.estimate(state_at_mutation(problem, candidate), targets, settings.pilot)
    signed_complex = sum(
        coefficient * pilot.estimates[pauli.to_label()]
        for pauli, coefficient in zip(retained.paulis, retained.coeffs, strict=True)
    )
    if abs(complex(signed_complex).imag) > settings.numerical_tolerance:
        raise ValueError("pilot signed estimate has a material imaginary component")
    signed = float(complex(signed_complex).real)
    gap = abs(signed)
    coefficient_mass = float(np.sum(np.abs(retained.coeffs)))
    sampling_error = pilot.target_error * coefficient_mass
    if candidate.kind is MutationKind.PAULI_ROTATION:
        assert candidate.eta is not None
        truncation_error = 2.0 * abs(np.sin(candidate.eta / 2.0)) * propagation.truncation_error
    else:
        truncation_error = 2.0 * propagation.truncation_error
    truncation_error += added_discard
    total_error = truncation_error + sampling_error
    interval = (max(0.0, gap - total_error), gap + total_error)
    magnitude = float(
        sum(
            abs(coefficient) * abs(pilot.estimates[pauli.to_label()])
            for pauli, coefficient in zip(retained.paulis, retained.coeffs, strict=True)
        )
    )
    cancellation = None if magnitude == 0.0 else float(np.clip(1.0 - gap / magnitude, 0.0, 1.0))
    return EffectEvidence(
        evidence_type=EvidenceType.PILOT_ESTIMATED,
        gap=gap,
        signed_shift=signed,
        interval=interval,
        score=_efficiency(gap, problem.test.outcome_range, candidate.cost),
        truncation_error=truncation_error,
        sampling_error=sampling_error,
        cancellation_diagnostic=cancellation,
        failure_probability=settings.pilot.failure_probability,
        warnings=(f"pilot used {pilot.shots} shared prefix shots",),
    )


def _perturbative_evidence(
    problem: PlanningProblem,
    candidate: MutationCandidate,
    propagation: PropagationResult,
    settings: PlannerSettings,
) -> EffectEvidence:
    """Compute state-dependent or operator-level first-order evidence.

    :param problem: Complete planning problem.
    :param candidate: General Hermitian mutation.
    :param propagation: Complete propagated observable.
    :param settings: State-availability and numerical settings.
    :return: Perturbative interval and score, or a state-independent operator proxy.
    """

    assert candidate.generator is not None
    assert candidate.eta is not None
    if problem.circuit.num_qubits > min(settings.dense_max_qubits, propagation.dense_max_qubits):
        return EffectEvidence(
            evidence_type=EvidenceType.UNAVAILABLE,
            warnings=("dense perturbative fallback exceeds dense_max_qubits",),
        )
    generator = embed_local_operator(candidate.generator, candidate.support, problem.circuit.num_qubits).data
    dense = propagation.exact_operator
    assert dense is not None
    observable = dense.data
    commutator = generator @ observable - observable @ generator
    sensitivity_operator = Operator(-1j * commutator)
    if _state_available(problem, settings.exact_state_max_qubits, settings):
        state = state_at_mutation(problem, candidate)
        sensitivity = abs(real_expectation(state, sensitivity_operator, settings.numerical_tolerance))
        nested = generator @ commutator - commutator @ generator
        remainder = candidate.eta**2 * _operator_norm(nested) / 2.0
        warning = ()
        if remainder >= abs(candidate.eta) * sensitivity:
            warning = ("second-order bound does not establish first-order dominance",)
        return EffectEvidence(
            evidence_type=EvidenceType.STATE_DEPENDENT_PERTURBATIVE,
            provenance="double-commutator remainder bound (floating-point evaluation)",
            score=_efficiency(abs(candidate.eta) * sensitivity, problem.test.outcome_range, candidate.cost),
            deterministic_interval=(
                max(0.0, abs(candidate.eta) * sensitivity - remainder),
                min(problem.test.outcome_range, abs(candidate.eta) * sensitivity + remainder),
            ),
            remainder_bound=remainder,
            warnings=warning,
        )
    return EffectEvidence(
        evidence_type=EvidenceType.OPERATOR_PROXY,
        score=_operator_norm(commutator),
        warnings=("operator proxy is a state-independent upper bound, not a realized gap",),
    )


def _direct_simulation_evidence(
    problem: PlanningProblem,
    candidate: MutationCandidate,
    settings: PlannerSettings,
) -> EffectEvidence:
    """Compute exact ideal evidence by direct original/mutant simulation.

    :param problem: Complete planning problem.
    :param candidate: Candidate mutant.
    :param settings: Numerical settings.
    :return: Exact simulated gap evidence.
    """

    signed, gap = direct_signed_gap(problem, candidate, settings.numerical_tolerance)
    zero = ZeroProvenance.NUMERICAL if gap <= settings.numerical_tolerance else ZeroProvenance.NONE
    return EffectEvidence(
        evidence_type=EvidenceType.EXACT,
        provenance="simulated gap",
        gap=gap,
        signed_shift=signed,
        score=_efficiency(gap, problem.test.outcome_range, candidate.cost),
        zero_provenance=zero,
    )


def evaluate_candidate_effect(
    problem: PlanningProblem,
    candidate: MutationCandidate,
    propagation: PropagationResult,
    settings: PlannerSettings,
    pilot_provider: PilotProvider | None = None,
) -> EffectEvidence:
    """Implement Algorithm 2 for one candidate and propagated observable.

    :param problem: Complete planning problem.
    :param candidate: Candidate mutation.
    :param propagation: Complete and retained propagated observable.
    :param settings: Computational and numerical settings.
    :param pilot_provider: Optional signed pilot provider.
    :return: Best available evidence under the prescribed branch order.
    """

    num_qubits = problem.circuit.num_qubits
    if candidate.kind is MutationKind.GENERAL_HERMITIAN and _algebraic_generator_zero(candidate, propagation, num_qubits):
        return EffectEvidence(
            evidence_type=EvidenceType.EXACT,
            provenance="algebraic generator commutation certificate",
            gap=0.0,
            signed_shift=0.0,
            score=0.0,
            zero_provenance=ZeroProvenance.ALGEBRAIC,
        )
    if _algebraic_pauli_zero(candidate, propagation, num_qubits):
        LOGGER.info("%s: algebraic Pauli commutation zero", candidate.identifier)
        return EffectEvidence(
            evidence_type=EvidenceType.EXACT,
            provenance="algebraic zero certificate",
            gap=0.0, signed_shift=0.0, score=0.0,
            zero_provenance=ZeroProvenance.ALGEBRAIC,
        )
    local_available = len(candidate.support) <= settings.local_max_qubits
    clifford = local_available and is_clifford_relative(candidate)
    structured = (candidate.kind is MutationKind.PAULI_ROTATION or clifford
                  or (local_available and candidate.allow_spectral
                      and candidate.relative_operator.num_qubits <= settings.spectral_max_qubits))
    if not structured and candidate.kind is MutationKind.GENERAL_HERMITIAN:
        return _perturbative_evidence(problem, candidate, propagation, settings)

    if structured:
        certificate = certify_circuit_zero(
            problem, candidate, min(settings.symbolic_max_qubits, settings.dense_max_qubits),
        )
        if certificate is not None:
            return EffectEvidence(
                evidence_type=EvidenceType.EXACT,
                provenance=certificate,
                gap=0.0, signed_shift=0.0, score=0.0,
                zero_provenance=ZeroProvenance.ALGEBRAIC,
            )
        difference: Operator | SparsePauliOp | None = None
        complete = propagation.complete_paulis
        if complete is not None:
            difference = _sparse_difference(candidate, complete, num_qubits, settings)
        if _state_available(problem, settings.exact_state_max_qubits, settings):
            if difference is None and num_qubits <= min(settings.dense_max_qubits, propagation.dense_max_qubits):
                dense = propagation.exact_operator
                assert dense is not None
                relative = embed_local_operator(candidate.relative_operator, candidate.support, num_qubits)
                difference = _generic_difference(dense, relative)
            if difference is not None:
                return _exact_structured_evidence(problem, candidate, difference, settings)
        if (settings.enable_pilot and _state_available(problem, settings.exact_simulation_max_qubits, settings)
                and (candidate.kind is MutationKind.PAULI_ROTATION or clifford)):
            provider = pilot_provider or MedianOfMeansPilot()
            try:
                return _pilot_evidence(problem, candidate, propagation, settings, provider)
            except ValueError as exc:
                LOGGER.warning("%s: pilot unavailable: %s", candidate.identifier, exc)
        if _state_available(problem, settings.exact_simulation_max_qubits, settings):
            return _direct_simulation_evidence(problem, candidate, settings)
        if difference is not None:
            return EffectEvidence(
                evidence_type=EvidenceType.EXACT_OBSERVABLE_SIDE,
                provenance="structured observable-difference",
                warnings=("mutation-site state and executable pilot are unavailable",),
            )
        return EffectEvidence(
            evidence_type=EvidenceType.UNAVAILABLE,
            warnings=("complete propagation exceeds the Pauli-term and dense limits; no executable pilot",),
        )

    if _state_available(problem, settings.exact_simulation_max_qubits, settings):
        return _direct_simulation_evidence(problem, candidate, settings)
    return EffectEvidence(
        evidence_type=EvidenceType.UNAVAILABLE,
        warnings=("no supported gap, interval, perturbative, or proxy quantity was obtained",),
    )


def _status_from_evidence(evidence: EffectEvidence, threshold: float) -> DecisionStatus:
    """Apply the manuscript's threshold rule to one evidence record.

    :param evidence: Algorithm 2 evidence.
    :param threshold: Declared test gap threshold.
    :return: Final execution-decision status.
    """

    if evidence.zero_provenance is ZeroProvenance.ALGEBRAIC:
        return DecisionStatus.SKIP_CERTIFIED_ZERO
    if evidence.evidence_type is EvidenceType.EXACT and evidence.gap is not None:
        if evidence.gap >= threshold:
            return DecisionStatus.EXECUTE_PREDICTED_ABOVE_THRESHOLD
        return DecisionStatus.SKIP_BELOW_TEST_THRESHOLD
    if evidence.deterministic_interval is not None:
        lower, upper = evidence.deterministic_interval
        if upper < threshold:
            return DecisionStatus.SKIP_BELOW_TEST_THRESHOLD
        if lower >= threshold:
            return DecisionStatus.EXECUTE_PREDICTED_ABOVE_THRESHOLD
    if evidence.evidence_type is EvidenceType.PILOT_ESTIMATED and evidence.interval is not None:
        lower, upper = evidence.interval
        if upper < threshold:
            return DecisionStatus.SKIP_BELOW_TEST_THRESHOLD_CONFIDENCE
        if lower >= threshold:
            return DecisionStatus.EXECUTE_PREDICTED_ABOVE_THRESHOLD
    return DecisionStatus.EXECUTE_THRESHOLD_UNRESOLVED


def _rank_records(
    records: list[PlanningRecord],
    objective: RankingObjective,
    threshold: float,
    tolerance: float = 1e-12,
) -> list[PlanningRecord]:
    """Rank executable candidates only within comparable evidence types.

    Form ties relative to each group's best value, avoiding nontransitive chains
    of pairwise-close values, and preserve input order within each tie group.
    Only ranks change; evidence and threshold classifications remain untouched.

    :param records: Unranked planning records.
    :param objective: High-gap or threshold-boundary objective.
    :param threshold: Declared test threshold.
    :param tolerance: Absolute and relative tolerance used only for ranking ties.
    :return: Records with one-based within-evidence ranks.
    """

    executable = {
        DecisionStatus.EXECUTE_PREDICTED_ABOVE_THRESHOLD,
        DecisionStatus.EXECUTE_THRESHOLD_UNRESOLVED,
    }
    grouped: dict[EvidenceType, list[int]] = {}
    for index, record in enumerate(records):
        if record.status in executable and record.evidence.score is not None:
            grouped.setdefault(record.evidence.evidence_type, []).append(index)
    ranked = list(records)
    for indices in grouped.values():
        if objective is RankingObjective.THRESHOLD_BOUNDARY:
            values = {
                index: abs(ranked[index].evidence.gap - threshold)
                if ranked[index].evidence.gap is not None
                else float("inf")
                for index in indices
            }
        else:
            values = {index: ranked[index].evidence.score or 0.0 for index in indices}
        indices.sort(key=values.__getitem__, reverse=objective is RankingObjective.HIGH_GAP)
        ordered = []
        start = 0
        while start < len(indices):
            anchor = values[indices[start]]
            end = start + 1
            while end < len(indices) and math.isclose(
                values[indices[end]], anchor, rel_tol=tolerance, abs_tol=tolerance,
            ):
                end += 1
            ordered.extend(sorted(indices[start:end]))
            start = end
        for rank, index in enumerate(ordered, start=1):
            ranked[index] = replace(ranked[index], rank=rank)
    return ranked


def _strength_sine(candidate: MutationCandidate) -> float | None:
    """Obtain the phase-invariant strength ceiling when a rotation is known.

    :param candidate: Pauli rotation or fixed one-qubit replacement.
    :return: Absolute half-angle sine, or None for unsupported replacements.
    """

    if candidate.kind is MutationKind.PAULI_ROTATION and candidate.eta is not None:
        return float(abs(np.sin(candidate.eta / 2.0)))
    if candidate.relative_operator.num_qubits == 1:
        eigenvalues = np.linalg.eigvals(candidate.relative_operator.data)
        return float(min(1.0, abs(eigenvalues[0] - eigenvalues[1]) / 2.0))
    return None


def plan_mutant_executions(
    problem: PlanningProblem,
    settings: PlannerSettings | None = None,
    pilot_provider: PilotProvider | None = None,
) -> list[PlanningRecord]:
    """Implement Algorithm 1 for a fixed test and declared candidates.

    :param problem: Original circuit, test, candidates, and execution budget.
    :param settings: Optional computational and numerical settings.
    :param pilot_provider: Optional signed pilot provider shared by Algorithm 2.
    :raises ValueError: If the planning problem or settings violate the model contract.
    :return: One status and evidence record for every declared candidate.
    """

    active_settings = settings or PlannerSettings()
    if active_settings.numerical_tolerance <= 0.0:
        raise ValueError("numerical tolerance must be positive")
    if min(active_settings.dense_max_qubits, active_settings.local_max_qubits,
           active_settings.symbolic_max_qubits, active_settings.exact_state_max_qubits,
           active_settings.exact_simulation_max_qubits) < 0 or active_settings.max_pauli_terms <= 0:
        raise ValueError("width limits must be nonnegative and maximum Pauli terms must be positive")
    validate_problem(problem, active_settings.dense_max_qubits, active_settings.local_max_qubits)
    # One call per candidate at most; unused allocations are not recycled.
    effect_settings = replace(
        active_settings,
        pilot=replace(
            active_settings.pilot,
            failure_probability=active_settings.pilot.failure_probability / max(1, len(problem.candidates)),
        ),
    )
    cones = structural_backward_cones(problem.circuit, problem.test.observable)
    floor = (
        information_theoretic_strength_floor(
            problem.budget.copies,
            problem.budget.error_probability,
        )
        if active_settings.enable_discrimination_budget
        else None
    )
    records: list[PlanningRecord] = []
    for candidate in problem.candidates:
        LOGGER.info("planning candidate %s", candidate.identifier)
        if active_settings.enable_equivalence and transpiler_equivalent(
            problem.circuit,
            candidate.mutant_circuit,
            active_settings.transpiler,
            maximum_qubits=min(active_settings.symbolic_max_qubits, active_settings.dense_max_qubits),
            local_maximum_qubits=min(active_settings.symbolic_max_qubits, active_settings.local_max_qubits),
        ):
            records.append(
                PlanningRecord(
                    candidate.identifier,
                    DecisionStatus.SKIP_SEMANTICALLY_EQUIVALENT,
                    EffectEvidence(
                        EvidenceType.EXACT,
                        provenance="pinned transpiler match verified by exact channel comparison",
                    ),
                )
            )
            continue
        strength_sine = _strength_sine(candidate)
        if strength_sine is not None and problem.test.outcome_range * strength_sine < problem.test.threshold:
            records.append(
                PlanningRecord(
                    candidate.identifier,
                    DecisionStatus.SKIP_BELOW_TEST_THRESHOLD,
                    EffectEvidence(
                        EvidenceType.EXACT,
                        provenance="rotation threshold ceiling (floating-point evaluation)",
                        deterministic_interval=(0.0, problem.test.outcome_range * strength_sine),
                    ),
                )
            )
            continue
        if floor is not None and strength_sine is not None and strength_sine < np.sin(floor / 2.0):
            records.append(
                PlanningRecord(
                    candidate.identifier,
                    DecisionStatus.SKIP_BELOW_BUDGET_FLOOR,
                    EffectEvidence(
                        EvidenceType.EXACT,
                        provenance="strength-bound certificate",
                    ),
                )
            )
            continue
        if set(candidate.support).isdisjoint(cones[candidate.suffix_start]):
            records.append(
                PlanningRecord(
                    candidate.identifier,
                    DecisionStatus.SKIP_CERTIFIED_ZERO,
                    EffectEvidence(
                        EvidenceType.EXACT,
                        provenance="structural zero certificate",
                        gap=0.0,
                        signed_shift=0.0,
                        score=0.0,
                        zero_provenance=ZeroProvenance.STRUCTURAL,
                    ),
                )
            )
            continue
        propagation = propagate_observable(
            problem.circuit,
            problem.test.observable,
            candidate.suffix_start,
            active_settings.max_pauli_terms,
            active_settings.numerical_tolerance,
            dense_max_qubits=active_settings.dense_max_qubits,
            local_max_qubits=active_settings.local_max_qubits,
        )
        evidence = evaluate_candidate_effect(
            problem,
            candidate,
            propagation,
            effect_settings,
            pilot_provider,
        )
        records.append(
            PlanningRecord(
                candidate.identifier,
                _status_from_evidence(evidence, problem.test.threshold),
                evidence,
            )
        )
    return _rank_records(
        records, active_settings.ranking_objective, problem.test.threshold,
        active_settings.numerical_tolerance,
    )
