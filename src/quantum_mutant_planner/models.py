"""Typed data contracts for mutant execution planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, Pauli, SparsePauliOp, Statevector

QuantumState = Statevector | DensityMatrix


class DecisionStatus(StrEnum):
    """Execution-decision values prescribed by Algorithm 1."""

    SKIP_SEMANTICALLY_EQUIVALENT = "skip--semantically equivalent"
    SKIP_BELOW_BUDGET_FLOOR = "skip--below budget floor"
    SKIP_CERTIFIED_ZERO = "skip--certified zero"
    SKIP_BELOW_TEST_THRESHOLD = "skip--below test threshold"
    SKIP_BELOW_TEST_THRESHOLD_CONFIDENCE = "skip--below test threshold at declared confidence"
    EXECUTE_PREDICTED_ABOVE_THRESHOLD = "execute--predicted above threshold"
    EXECUTE_THRESHOLD_UNRESOLVED = "execute--threshold unresolved"


class EvidenceType(StrEnum):
    """Evidence categories prescribed by Algorithm 2."""

    EXACT = "exact"
    EXACT_OBSERVABLE_SIDE = "exact observable-side update only"
    STATE_DEPENDENT_PERTURBATIVE = "state-dependent perturbative"
    OPERATOR_PROXY = "operator proxy"
    PILOT_ESTIMATED = "pilot-estimated"
    UNAVAILABLE = "unavailable"


class MutationKind(StrEnum):
    """Supported mutation construction categories."""

    REPLACEMENT = "replacement"
    DELETION = "deletion"
    INSERTION = "insertion"
    PAULI_ROTATION = "pauli rotation"
    GENERAL_HERMITIAN = "general Hermitian"


class ZeroProvenance(StrEnum):
    """Provenance labels that distinguish exact and numerical zeros."""

    NONE = "none"
    STRUCTURAL = "analytical light-cone exclusion"
    ALGEBRAIC = "algebraic zero certificate"
    NUMERICAL = "ordinary floating-point numerical compatibility"


class RankingObjective(StrEnum):
    """Supported within-evidence ranking objectives."""

    HIGH_GAP = "high_gap"
    THRESHOLD_BOUNDARY = "threshold_boundary"


@dataclass(frozen=True, slots=True)
class QuantumTest:
    """One fixed input-state and observable test.

    :param state: Original circuit input state.
    :param observable: Hermitian observable represented in the Pauli basis.
    :param threshold: Declared observable-gap detection threshold.
    :param outcome_range: Spectral range used in the efficiency score.
    """

    state: QuantumState
    observable: SparsePauliOp
    threshold: float
    outcome_range: float


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    """Independent-copy discrimination budget.

    :param copies: Available independent output copies per candidate.
    :param error_probability: Declared equal-prior error probability.
    """

    copies: int
    error_probability: float


@dataclass(frozen=True, slots=True)
class TranspilerSettings:
    """Pinned settings for finding transpiler matches to verify exactly.

    :param optimization_level: Qiskit preset optimization level.
    :param seed: Seed for stochastic transpiler passes.
    :param approximation_degree: Synthesis approximation degree.
    :param basis_gates: Pinned target basis-gate names.
    """

    optimization_level: int = 3
    seed: int = 12345
    approximation_degree: float = 1.0
    basis_gates: tuple[str, ...] = ("rz", "sx", "x", "cx")


@dataclass(frozen=True, slots=True)
class PilotSettings:
    """Configuration for signed classical-shadow pilot estimates.

    :param target_error: Simultaneous per-Pauli additive error target.
    :param failure_probability: Run-wide pilot failure budget in the planner, or per-call budget when estimating directly.
    :param max_shots: Maximum permitted prefix pilot shots.
    :param seed: Sampling seed.
    """

    target_error: float = 0.25
    failure_probability: float = 0.05
    max_shots: int = 250_000
    seed: int = 12345


@dataclass(frozen=True, slots=True)
class PlannerSettings:
    """Computational and numerical limits for the reference planner.

    :param dense_max_qubits: Largest full-register dense operator or density-matrix fallback.
    :param local_max_qubits: Largest dense gate support for validation and Pauli updates.
    :param numerical_tolerance: Floating-point compatibility tolerance, also used as both absolute and relative ranking tie tolerance.
    :param exact_state_max_qubits: Largest state evaluated directly at a mutation site.
    :param exact_simulation_max_qubits: Largest full circuit evaluated directly.
    :param spectral_max_qubits: Largest relative unitary admitted to the spectral branch.
    :param max_pauli_terms: Maximum retained terms for pilot propagation.
    :param ranking_objective: Within-evidence execution-ranking objective.
    :param enable_equivalence: Whether to verify pinned transpiler matches with exact channel comparison.
    :param enable_pilot: Whether to attempt the signed pilot branch.
    :param transpiler: Pinned transpiler configuration.
    :param pilot: Pilot-estimator configuration.
    :param symbolic_max_qubits: Largest circuit compared with exact symbolic arithmetic; zero disables certificates from this path.
    :param enable_discrimination_budget: Opt into the separate unknown-source discrimination policy; disabled for ordinary mutation testing.
    """

    numerical_tolerance: float = 1.0e-12
    exact_state_max_qubits: int = 10
    exact_simulation_max_qubits: int = 10
    spectral_max_qubits: int = 2
    max_pauli_terms: int = 256
    ranking_objective: RankingObjective = RankingObjective.HIGH_GAP
    enable_equivalence: bool = True
    enable_pilot: bool = True
    transpiler: TranspilerSettings = field(default_factory=TranspilerSettings)
    pilot: PilotSettings = field(default_factory=PilotSettings)
    symbolic_max_qubits: int = 4
    enable_discrimination_budget: bool = False
    dense_max_qubits: int = 10
    local_max_qubits: int = 4


@dataclass(frozen=True, slots=True)
class MutationCandidate:
    """One declared mutation and its local relative-unitary metadata.

    :param identifier: Stable candidate identifier.
    :param mutant_circuit: Complete mutated circuit.
    :param site: Original instruction index or insertion boundary.
    :param suffix_start: First original instruction in the suffix after the mutation.
    :param support: Original-circuit qubit indices changed by the mutation.
    :param cost: Positive paired-comparison cost.
    :param kind: Mutation construction category.
    :param relative_operator: Local relative unitary on ``support``.
    :param eta: Signed mutation strength when defined.
    :param pauli_axis: Local involutory Pauli axis when defined.
    :param generator: Local Hermitian generator when defined.
    :param allow_spectral: Whether the spectral branch is declared tractable.
    :param generator_paulis: Complete explicit Pauli generator used for algebraic commutation certificates.
    """

    identifier: str
    mutant_circuit: QuantumCircuit
    site: int
    suffix_start: int
    support: tuple[int, ...]
    cost: float
    kind: MutationKind
    relative_operator: Operator
    eta: float | None = None
    pauli_axis: Pauli | None = None
    generator: Operator | None = None
    allow_spectral: bool = True
    generator_paulis: SparsePauliOp | None = None


@dataclass(frozen=True, slots=True)
class PlanningProblem:
    """Complete input to Algorithm 1.

    :param circuit: Original unitary circuit.
    :param test: Fixed expectation-value test.
    :param candidates: Declared candidate mutations.
    :param budget: Independent-copy execution budget.
    """

    circuit: QuantumCircuit
    test: QuantumTest
    candidates: tuple[MutationCandidate, ...]
    budget: ExecutionBudget


@dataclass(frozen=True, slots=True)
class EffectEvidence:
    """Algorithm 2 evidence returned before threshold classification.

    :param evidence_type: Evidence category.
    :param provenance: Exact certificate or construction provenance when applicable.
    :param gap: Exact or point-estimated absolute gap.
    :param signed_shift: Signed original-minus-mutant expectation shift.
    :param interval: Simultaneous absolute-gap confidence interval.
    :param deterministic_interval: Analytic absolute-gap bound, evaluated at the reported numerical precision.
    :param failure_probability: Allocated pilot failure probability for this record's interval.
    :param score: Efficiency or proxy score.
    :param zero_provenance: Exact or numerical zero classification.
    :param truncation_error: Observable-propagation error allowance.
    :param sampling_error: Pilot-sampling error allowance.
    :param cancellation_diagnostic: Signed-versus-magnitude cancellation diagnostic.
    :param remainder_bound: Perturbative second-order remainder bound.
    :param warnings: Nonfatal limitations attached to the record.
    """

    evidence_type: EvidenceType
    provenance: str | None = None
    gap: float | None = None
    signed_shift: float | None = None
    interval: tuple[float, float] | None = None
    score: float | None = None
    zero_provenance: ZeroProvenance = ZeroProvenance.NONE
    truncation_error: float = 0.0
    sampling_error: float = 0.0
    cancellation_diagnostic: float | None = None
    remainder_bound: float | None = None
    warnings: tuple[str, ...] = ()
    deterministic_interval: tuple[float, float] | None = None
    failure_probability: float | None = None


@dataclass(frozen=True, slots=True)
class PlanningRecord:
    """Final status and evidence record for one candidate.

    :param candidate_id: Stable candidate identifier.
    :param status: Execution-decision status.
    :param evidence: Algorithm 2 evidence or a pre-analysis certificate.
    :param rank: One-based rank within a comparable evidence type.
    """

    candidate_id: str
    status: DecisionStatus
    evidence: EffectEvidence
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a JSON-compatible dictionary.

        :return: Dictionary containing enum values and numeric evidence.
        """

        data = asdict(self)
        data["status"] = self.status.value
        data["evidence"]["evidence_type"] = self.evidence.evidence_type.value
        data["evidence"]["zero_provenance"] = self.evidence.zero_provenance.value
        return data
