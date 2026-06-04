import pytest
from pydantic import ValidationError

from statvocab.contracts import (
    MeasureRelation,
    RelationType,
    RunManifest,
    RunState,
    VocabularyCategory,
    stable_id,
)


def test_stable_id_is_deterministic() -> None:
    first = stable_id("term", "Population", "Eurostat")
    second = stable_id("term", " population ", "eurostat")

    assert first == second
    assert first.startswith("term_")


def test_stable_id_rejects_unknown_prefix() -> None:
    with pytest.raises(ValueError, match="Unknown stable ID prefix"):
        stable_id("unknown", "value")


def test_enum_contract_values() -> None:
    assert VocabularyCategory.MEASURE == "measure"
    assert VocabularyCategory.OTHER_AMBIGUOUS == "other_ambiguous"
    assert RelationType.BROADER_THAN == "broader_than"
    assert RelationType.RELATED_TO == "related_to"


def test_manifest_serializes_to_json_contract() -> None:
    manifest = RunManifest(
        run_id="run_abc",
        config_id="cfg_abc",
        state=RunState.STARTED,
        corpus="fixture",
        pipeline_stage="foundation",
    )

    payload = manifest.model_dump(mode="json")

    assert payload["state"] == "started"
    assert payload["corpus"] == "fixture"
    assert payload["started_at"].endswith(("Z", "+00:00"))


def test_relation_rejects_self_relation() -> None:
    with pytest.raises(ValidationError, match="self-relations"):
        MeasureRelation(
            relation_id="relation_abc",
            source_term_id="term_same",
            target_term_id="term_same",
            relation_type=RelationType.RELATED_TO,
            evidence_ids=("evidence_abc",),
            confidence=0.5,
        )
