import json
import sys
from pathlib import Path

from statvocab.config import AppConfig, load_config
from statvocab.search.explain import NOTICE, explain_candidate
from statvocab.search.query_parser import parse_query
from statvocab.search.rank import RankedCandidate


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


def test_explanation_includes_components_evidence_and_notice(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    parsed = parse_query(config, "Police officers in Romania in 2001")
    candidate = RankedCandidate(
        table_id="crim_plce",
        score=0.91,
        score_components={
            "semantic": 0.9,
            "lexical": 0.8,
            "measure": 1.0,
            "dimension": 0.0,
            "geography": 1.0,
            "time": 1.0,
        },
        document={
            "table_id": "crim_plce",
            "title_clean": "Police officers",
            "source_url": "https://ec.europa.eu/eurostat/databrowser/view/crim_plce",
            "measures_json": json.dumps(["Police officers"]),
            "dimension_names_json": json.dumps([]),
            "dimension_values_json": json.dumps([]),
            "units_json": json.dumps([]),
            "geographies_json": json.dumps([{"code": "RO", "name": "Romania"}]),
            "times_json": json.dumps([{"normalized_value": "2001"}]),
        },
    )

    result = explain_candidate(rank=1, query=parsed, candidate=candidate)

    assert result["rank"] == 1
    assert result["table_id"] == "crim_plce"
    assert result["notice"] == NOTICE
    assert result["score_components"]["semantic"] == 0.9
    assert result["matched_terms"] == [{"category": "measure", "term": "Police officers"}]
    assert result["geographies"] == [{"code": "RO", "name": "Romania"}]
    assert result["times"] == [{"normalized_value": "2001"}]
    assert result["evidence"]["warnings"] == []


def test_query_explanation_path_does_not_import_paid_llm_code(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    existing = sys.modules.pop("statvocab.classify_llm", None)

    try:
        parsed = parse_query(config, "Police officers in Romania in 2001")

        assert parsed.semantic_remainder
        assert "statvocab.classify_llm" not in sys.modules
    finally:
        if existing is not None:
            sys.modules["statvocab.classify_llm"] = existing
