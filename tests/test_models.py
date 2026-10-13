"""Tests for public result contracts."""

from quantum_mutant_planner.models import (
    DecisionStatus,
    EffectEvidence,
    EvidenceType,
    PlanningRecord,
    ZeroProvenance,
)


def test_record_serializes_enum_values() -> None:
    """Planning records should produce JSON-compatible enum strings."""

    record = PlanningRecord(
        "m",
        DecisionStatus.SKIP_CERTIFIED_ZERO,
        EffectEvidence(
            EvidenceType.EXACT,
            provenance="certificate",
            gap=0.0,
            zero_provenance=ZeroProvenance.ALGEBRAIC,
        ),
    )
    data = record.to_dict()
    assert data["status"] == "skip--certified zero"
    assert data["evidence"]["evidence_type"] == "exact"
    assert data["evidence"]["zero_provenance"] == "algebraic zero certificate"


def test_status_and_evidence_vocabularies_are_complete() -> None:
    """The code should preserve all manuscript status and evidence values."""

    assert len(DecisionStatus) == 7
    assert len(EvidenceType) == 6
