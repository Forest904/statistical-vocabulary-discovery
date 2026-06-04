from statvocab.classify_rules import classify_feature_row
from statvocab.contracts import VocabularyCategory


def _row(term: str, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "term_id": "term_test",
        "canonical_term": term,
        "matching_key": term.casefold(),
        "occurrence_count": 1,
        "table_count": 1,
        "role_summary_json": '{"metadata_value": 1}',
        "has_header_evidence": False,
        "has_title_evidence": False,
        "has_metadata_value_evidence": True,
        "has_conflicting_roles": False,
    }
    row.update(updates)
    return row


def test_rule_classifies_metadata_header_as_dimension_name() -> None:
    decision = classify_feature_row(
        _row(
            "Unit of measure",
            role_summary_json='{"header_name": 1}',
            has_header_evidence=True,
            has_metadata_value_evidence=False,
        )
    )

    assert decision.category == VocabularyCategory.DIMENSION_NAME
    assert decision.protected


def test_rule_classifies_unit_expression() -> None:
    decision = classify_feature_row(_row("Million euro"))

    assert decision.category == VocabularyCategory.UNIT
    assert decision.protected


def test_rule_classifies_frequency_value() -> None:
    decision = classify_feature_row(_row("Annual"))

    assert decision.category == VocabularyCategory.DIMENSION_VALUE
    assert decision.protected


def test_rule_classifies_title_measure() -> None:
    decision = classify_feature_row(
        _row(
            "Pesticide sales by categorisation of active substances",
            role_summary_json='{"title_full": 1}',
            has_title_evidence=True,
            has_metadata_value_evidence=False,
        )
    )

    assert decision.category == VocabularyCategory.MEASURE


def test_rule_uses_ambiguous_for_conflicting_roles() -> None:
    decision = classify_feature_row(
        _row(
            "Total",
            role_summary_json='{"metadata_value": 2, "title_clause": 1}',
            has_title_evidence=True,
            has_conflicting_roles=True,
        )
    )

    assert decision.category == VocabularyCategory.OTHER_AMBIGUOUS
