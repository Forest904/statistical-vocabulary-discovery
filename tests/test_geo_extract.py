from pathlib import Path

from statvocab.config import ExtractionConfig
from statvocab.geo_extract import (
    build_geography_matcher,
    infer_nuts_level,
    load_eurostat_geo_entries,
)


def test_nuts_level_inference() -> None:
    assert infer_nuts_level("IT") == "country"
    assert infer_nuts_level("ITI") == "nuts1"
    assert infer_nuts_level("ITI4") == "nuts2"
    assert infer_nuts_level("ITI43") == "nuts3"


def test_nuts_exact_and_normalized_matching(tmp_path: Path) -> None:
    nuts_path = tmp_path / "NUTS_AT_2024.csv"
    nuts_path.write_text(
        "CNTR_CODE,NUTS_ID,NAME_LATN,NUTS_NAME,MOUNT_TYPE,URBN_TYPE,COAST_TYPE\n"
        "IT,IT,Italy,Italy,,,\n"
        "IT,ITI43,Roma,Roma,,,\n",
        encoding="utf-8",
    )
    matcher = build_geography_matcher(
        ExtractionConfig(nuts_2024_path=nuts_path),
        variant="nuts",
    )

    code = matcher.match("IT", table_id="t", source_area="metadata_value", location="row[2].col[1]")
    name = matcher.match(
        " italy ",
        table_id="t",
        source_area="metadata_value",
        location="row[3].col[1]",
    )

    assert code is not None
    assert code.code == "IT"
    assert code.match_method == "exact_code"
    assert name is not None
    assert name.code == "IT"
    assert name.match_method == "normalized_name"


def test_enhanced_geo_codelist_matches_aggregates(tmp_path: Path) -> None:
    geo_path = tmp_path / "geo.xml"
    geo_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<Structure>
  <Code id="EU27_2020"><Name>European Union - 27 countries (from 2020)</Name></Code>
</Structure>
""",
        encoding="utf-8",
    )

    entries = load_eurostat_geo_entries(geo_path)
    matcher = build_geography_matcher(
        ExtractionConfig(
            nuts_2024_path=tmp_path / "missing.csv",
            eurostat_geo_codelist_path=geo_path,
        ),
        variant="enhanced",
    )
    match = matcher.match(
        "EU27_2020",
        table_id="t",
        source_area="metadata_value",
        location="row[2].col[1]",
    )

    assert entries[0].code == "EU27_2020"
    assert match is not None
    assert match.level == "aggregate"
    assert match.variant == "enhanced"
