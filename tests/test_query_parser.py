from pathlib import Path

from statvocab.config import AppConfig, load_config
from statvocab.search.query_parser import parse_query


def _fixture_config(tmp_path: Path) -> AppConfig:
    base = load_config("configs/evaluation.yaml")
    paths = base.paths.model_copy(update={"external_dir": tmp_path / "external"})
    extraction = base.extraction.model_copy(
        update={
            "nuts_2024_path": tmp_path / "external" / "missing_nuts.csv",
            "eurostat_geo_codelist_path": tmp_path / "external" / "missing_geo.xml",
        }
    )
    return base.model_copy(update={"paths": paths, "extraction": extraction})


def test_query_parser_extracts_time_geography_and_remainder(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)

    parsed = parse_query(config, "How many police officers were there in Romania in 2001?")

    assert parsed.original == "How many police officers were there in Romania in 2001?"
    assert [geo.name for geo in parsed.geographies] == ["Romania"]
    assert [time.normalized_value for time in parsed.times] == ["2001"]
    assert "Romania" not in parsed.semantic_remainder
    assert "2001" not in parsed.semantic_remainder
    assert "police officers" in parsed.semantic_remainder


def test_query_parser_handles_blank_query(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)

    parsed = parse_query(config, "   ")

    assert parsed.is_blank is True
    assert parsed.semantic_remainder == ""
    assert parsed.geographies == ()
    assert parsed.times == ()
