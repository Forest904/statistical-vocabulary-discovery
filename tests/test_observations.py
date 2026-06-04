from statvocab.config import load_config
from statvocab.ingest import (
    EurostatStarAdapter,
    fixture_vocabulary_candidates,
    parse_observation_cell,
)


def test_numeric_observation_parses_as_number() -> None:
    parsed = parse_observation_cell(2, "2020", "98.1")

    assert parsed.numeric_value == 98.1
    assert parsed.missing is False
    assert parsed.flag is None


def test_missing_markers_parse_as_missing() -> None:
    colon = parse_observation_cell(2, "2020", ":")
    empty = parse_observation_cell(2, "2020", "")

    assert colon.missing is True
    assert empty.missing is True


def test_flagged_observation_splits_value_and_flag() -> None:
    parsed = parse_observation_cell(2, "2021", "55p")

    assert parsed.numeric_value == 55.0
    assert parsed.flag == "p"
    assert parsed.missing is False


def test_observation_values_do_not_enter_fixture_vocabulary_candidates() -> None:
    config = load_config("configs/evaluation.yaml")
    adapter = EurostatStarAdapter(config)
    reference = next(ref for ref in adapter.iter_references() if ref.table_id == "fixture_regular")
    parsed = adapter.parse_table(reference)

    candidates = fixture_vocabulary_candidates(parsed)

    assert "Persons" in candidates
    assert "Male" in candidates
    assert "10" not in candidates
    assert "22e" not in candidates
