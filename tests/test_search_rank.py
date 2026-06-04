import json
from pathlib import Path

from statvocab.config import AppConfig, load_config
from statvocab.search.lexical import LexicalHit
from statvocab.search.query_parser import parse_query
from statvocab.search.rank import rank_candidates
from statvocab.search.semantic import SemanticHit


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


def _document(table_id: str, *, country: str = "RO", time: str = "2001") -> dict[str, object]:
    return {
        "table_id": table_id,
        "title_clean": "Police officers",
        "source_url": "https://example.test/table",
        "measures_json": json.dumps(["Police officers"]),
        "dimension_names_json": json.dumps(["unit of measure"]),
        "dimension_values_json": json.dumps(["total"]),
        "units_json": json.dumps(["number"]),
        "geographies_json": json.dumps(
            [
                {
                    "code": country,
                    "name": "Romania" if country == "RO" else "France",
                    "normalized_value": "Romania" if country == "RO" else "France",
                    "country": country,
                }
            ]
        ),
        "times_json": json.dumps(
            [
                {
                    "normalized_value": time,
                    "start_date": f"{time}-01-01",
                    "end_date": f"{time}-12-31",
                }
            ]
        ),
    }


def test_rank_candidates_normalizes_and_fuses_scores(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    parsed = parse_query(config, "Police officers in Romania in 2001")
    documents = {
        "crim_plce": _document("crim_plce"),
        "other": _document("other", country="FR", time="1999"),
    }

    ranked = rank_candidates(
        query=parsed,
        documents=documents,
        lexical_hits=[
            LexicalHit("crim_plce", score=5.0, rank=-5.0),
            LexicalHit("other", score=1.0, rank=-1.0),
        ],
        semantic_hits=[SemanticHit("crim_plce", score=0.9), SemanticHit("unknown", score=1.0)],
        config=config.search,
        limit=10,
    )

    assert [candidate.table_id for candidate in ranked] == ["crim_plce", "other"]
    assert ranked[0].score > ranked[1].score
    assert ranked[0].score_components["semantic"] == 1.0
    assert ranked[0].score_components["geography"] == 1.0
    assert ranked[0].score_components["time"] == 1.0
    assert all(candidate.table_id != "unknown" for candidate in ranked)


def test_rank_candidates_returns_grounded_empty_for_weak_evidence(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    search = config.search.model_copy(update={"min_fused_score": 0.95})
    parsed = parse_query(config, "Police officers in Romania in 2001")
    documents = {"crim_plce": _document("crim_plce")}

    ranked = rank_candidates(
        query=parsed,
        documents=documents,
        lexical_hits=[LexicalHit("crim_plce", score=0.1, rank=-0.1)],
        semantic_hits=[],
        config=search,
        limit=10,
    )

    assert ranked == []
