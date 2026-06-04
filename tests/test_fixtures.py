import csv
from pathlib import Path

FIXTURE_DIR = Path("data/fixtures/eurostat_star_edge_cases")


def test_fixture_corpus_exists() -> None:
    csv_files = sorted(FIXTURE_DIR.glob("fixture_*.csv"))

    assert len(csv_files) == 4
    assert (FIXTURE_DIR / "titles.csv").exists()


def test_fixture_csvs_are_basic_parseable() -> None:
    for csv_path in sorted(FIXTURE_DIR.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))

        assert rows
        assert rows[0]
