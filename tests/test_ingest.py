from pathlib import Path

import pyarrow.parquet as pq
import pytest

from statvocab.config import load_config
from statvocab.contracts import ParseStatus
from statvocab.ingest import (
    AEI_HRI_DEFECT_REASON,
    EurostatStarAdapter,
    _diagnostics,
    detect_time_start,
    fixture_vocabulary_candidates,
    is_gzip_wrapped_tar_without_member,
    parsed_table_to_row,
    write_tables_parquet,
)


def test_fixture_corpus_inventories_four_tables() -> None:
    config = load_config("configs/evaluation.yaml")
    adapter = EurostatStarAdapter(config)

    references = list(adapter.iter_references())

    assert len(references) == 4


def test_core_corpus_inventory_expects_2000_when_raw_cache_exists() -> None:
    raw_cache = Path("data/raw/eurostat_2000_tables")
    if not raw_cache.exists():
        pytest.skip("raw core corpus is not present in this checkout")

    config = load_config("configs/core.yaml")
    adapter = EurostatStarAdapter(config)

    assert len(list(adapter.iter_references())) == 2000


def test_known_defect_detector_identifies_core_aei_hri() -> None:
    raw_cache = Path("data/raw/eurostat_2000_tables")
    if not raw_cache.exists():
        pytest.skip("raw core corpus is not present in this checkout")

    assert is_gzip_wrapped_tar_without_member(raw_cache / "aei_hri.csv", "aei_hri.csv")


def test_aei_hri_override_resolves_to_full_corpus_copy_when_present() -> None:
    raw_cache = Path("data/raw/eurostat_2000_tables")
    replacement = Path("data/raw/eurostat_7605_tables/aei_hri.csv")
    if not raw_cache.exists() or not replacement.exists():
        pytest.skip("raw core/full corpora are not present in this checkout")

    config = load_config("configs/core.yaml")
    adapter = EurostatStarAdapter(config)
    references = list(adapter.iter_references())
    reference = next(ref for ref in references if ref.table_id == "aei_hri")
    parsed = adapter.parse_table(reference)

    assert len(references) == 2000
    assert reference.filename == "aei_hri.csv"
    assert reference.source_repaired is True
    assert reference.original_local_path == str(raw_cache / "aei_hri.csv")
    assert reference.local_path == str(replacement)
    assert reference.repair_reason == AEI_HRI_DEFECT_REASON
    assert parsed.parse_status == ParseStatus.PARSED
    assert parsed.row_count == 64
    assert parsed.observation_count == 768
    assert adapter.source_repairs
    assert adapter.source_repairs[0].repair_status == "applied"


def test_repair_metadata_is_written_to_table_rows_and_diagnostics(tmp_path: Path) -> None:
    raw_cache = Path("data/raw/eurostat_2000_tables")
    replacement = Path("data/raw/eurostat_7605_tables/aei_hri.csv")
    if not raw_cache.exists() or not replacement.exists():
        pytest.skip("raw core/full corpora are not present in this checkout")

    config = load_config("configs/core.yaml")
    adapter = EurostatStarAdapter(config)
    reference = next(ref for ref in adapter.iter_references() if ref.table_id == "aei_hri")
    parsed = adapter.parse_table(reference)
    row = parsed_table_to_row(parsed)
    output_path = write_tables_parquet([row], tmp_path / "tables.parquet")
    rows = pq.read_table(output_path).to_pylist()
    diagnostics = _diagnostics(config, [parsed], source_repairs=adapter.source_repairs)

    assert rows[0]["source_repaired"] is True
    assert rows[0]["original_file_md5"] != rows[0]["parsed_file_md5"]
    assert rows[0]["repair_reason"] == AEI_HRI_DEFECT_REASON
    assert diagnostics["source_repairs"][0]["repair_status"] == "applied"
    assert diagnostics["source_repairs"][0]["defect_reason"] == AEI_HRI_DEFECT_REASON


def test_missing_known_defect_replacement_keeps_original_failure(tmp_path: Path) -> None:
    core_dir = tmp_path / "raw" / "eurostat_2000_tables"
    core_dir.mkdir(parents=True)
    source = Path("data/raw/eurostat_2000_tables/aei_hri.csv")
    if not source.exists():
        pytest.skip("raw core corpus is not present in this checkout")
    (core_dir / "aei_hri.csv").write_bytes(source.read_bytes())

    base_config = load_config("configs/core.yaml")
    config = base_config.model_copy(
        update={"paths": base_config.paths.model_copy(update={"raw_dir": tmp_path / "raw"})}
    )
    adapter = EurostatStarAdapter(config)
    reference = next(adapter.iter_references())
    parsed = adapter.parse_table(reference)

    assert reference.source_repaired is False
    assert parsed.parse_status == ParseStatus.FAILED
    assert adapter.source_repairs[0].repair_status == "missing_replacement"


def test_eurostat_backslash_time_header_keeps_dimension_metadata() -> None:
    header = ["Time frequency", "Geopolitical entity (reporting)\\Time", "2020.0", "2021.0"]

    assert detect_time_start(header) == 2


def test_quoted_fixture_values_with_commas_parse_correctly() -> None:
    config = load_config("configs/evaluation.yaml")
    adapter = EurostatStarAdapter(config)
    reference = next(ref for ref in adapter.iter_references() if ref.table_id == "fixture_quoted")
    parsed = adapter.parse_table(reference)

    candidates = fixture_vocabulary_candidates(parsed)

    assert parsed.parse_status == ParseStatus.PARSED
    assert "Food, beverages and tobacco" in candidates
    assert "Dangerous goods" in candidates


def test_no_time_fixture_records_warning_without_crashing() -> None:
    config = load_config("configs/evaluation.yaml")
    adapter = EurostatStarAdapter(config)
    reference = next(ref for ref in adapter.iter_references() if ref.table_id == "fixture_no_time")

    parsed = adapter.parse_table(reference)

    assert parsed.parse_status == ParseStatus.WARNING
    assert any("no time columns" in warning.reason for warning in parsed.warnings)


def test_malformed_rows_are_listed_with_reasons() -> None:
    config = load_config("configs/evaluation.yaml")
    adapter = EurostatStarAdapter(config)
    reference = next(ref for ref in adapter.iter_references() if ref.table_id == "fixture_no_time")

    parsed = adapter.parse_table(reference)

    assert parsed.malformed_row_count == 1
    assert any(
        warning.expected_columns == 4 and warning.actual_columns == 3
        for warning in parsed.warnings
    )
