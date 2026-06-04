from statvocab.config import ExtractionConfig
from statvocab.time_extract import extract_title_times, parse_time_value


def test_year_header_accepts_integer_like_form() -> None:
    parsed = parse_time_value(
        "2024.0",
        table_id="t",
        source_area="header_time",
        location="header[0]",
        config=ExtractionConfig(),
    )

    assert parsed is not None
    assert parsed.normalized_value == "2024"
    assert parsed.start_date == "2024-01-01"
    assert parsed.end_date == "2024-12-31"
    assert parsed.granularity == "year"


def test_quarter_month_day_and_until_title_rules() -> None:
    config = ExtractionConfig()
    quarter = parse_time_value(
        "Q1 2022",
        table_id="t",
        source_area="header_time",
        location="header[0]",
        config=config,
    )
    month = parse_time_value(
        "2020-M03",
        table_id="t",
        source_area="header_time",
        location="header[1]",
        config=config,
    )
    day = parse_time_value(
        "2020-03-15",
        table_id="t",
        source_area="header_time",
        location="header[2]",
        config=config,
    )
    title_times = extract_title_times("t", "Historical data until 2009 and January 2020", config)

    assert quarter is not None
    assert quarter.normalized_value == "2022-Q1"
    assert month is not None
    assert month.normalized_value == "2020-03"
    assert day is not None
    assert day.granularity == "day"
    assert {item.normalized_value for item in title_times} == {"until 2009", "2020-01"}


def test_title_range_prevents_duplicate_standalone_years() -> None:
    title_times = extract_title_times(
        "fixture_regular",
        "Employment by sex, age and NUTS 2 region, 2019-2021",
        ExtractionConfig(),
    )

    assert [item.normalized_value for item in title_times] == ["2019-2021"]
    assert title_times[0].granularity == "range"


def test_french_quarter_headers_normalize_to_standard_quarters() -> None:
    config = ExtractionConfig()

    first = parse_time_value(
        "Janvier-Mars 1995",
        table_id="t",
        source_area="header_time",
        location="header[0]",
        config=config,
    )
    second = parse_time_value(
        "Avril-Juin 2010",
        table_id="t",
        source_area="header_time",
        location="header[1]",
        config=config,
    )
    typo_third = parse_time_value(
        "Jullet-Septembre 2024",
        table_id="t",
        source_area="header_time",
        location="header[2]",
        config=config,
    )
    fourth = parse_time_value(
        "Octobre-Décembre 2025",
        table_id="t",
        source_area="header_time",
        location="header[3]",
        config=config,
    )

    assert first is not None
    assert (first.normalized_value, first.start_date, first.end_date) == (
        "1995-Q1",
        "1995-01-01",
        "1995-03-31",
    )
    assert second is not None
    assert second.normalized_value == "2010-Q2"
    assert typo_third is not None
    assert typo_third.normalized_value == "2024-Q3"
    assert fourth is not None
    assert fourth.normalized_value == "2025-Q4"
