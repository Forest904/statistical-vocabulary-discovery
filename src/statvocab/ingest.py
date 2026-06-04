"""Eurostat STAR source adapter and ingestion artifacts."""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
import tarfile
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

import pyarrow as pa
import pyarrow.parquet as pq

from statvocab.config import AppConfig
from statvocab.contracts import (
    ParsedColumn,
    ParsedObservation,
    ParsedTable,
    ParseStatus,
    ParseWarning,
    ResourceRecord,
    SourceRepair,
    TableReference,
)
from statvocab.manifests import complete_manifest, create_manifest, write_manifest
from statvocab.resources import file_md5

TIME_HEADER_RE = re.compile(
    r"^(?:\d{4}(?:\.0)?|\d{4}[- ]?Q[1-4]|\d{4}[- ]?M(?:0?[1-9]|1[0-2])|\d{4}-\d{2}(?:-\d{2})?)$",
    re.IGNORECASE,
)
NUMERIC_CELL_RE = re.compile(
    r"^\s*(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?P<flag>[A-Za-z]+)?\s*$"
)
MISSING_MARKERS = {"", ":"}
AEI_HRI_DEFECT_REASON = (
    "core file is gzip-wrapped tar content; nested tar does not contain aei_hri.csv"
)


@dataclass(frozen=True)
class KnownSourceDefect:
    """Known upstream corpus packaging defect with a documented replacement."""

    table_id: str
    reason: str
    replacement_relative_path: Path
    replacement_source_corpus: str


KNOWN_SOURCE_DEFECTS: dict[str, KnownSourceDefect] = {
    "aei_hri": KnownSourceDefect(
        table_id="aei_hri",
        reason=AEI_HRI_DEFECT_REASON,
        replacement_relative_path=Path("eurostat_7605_tables") / "aei_hri.csv",
        replacement_source_corpus="full",
    )
}


class SourceAdapter(Protocol):
    """Protocol for corpus-specific table adapters."""

    def iter_references(self) -> Iterator[TableReference]:
        """Yield table references selected for this corpus."""

    def parse_table(self, reference: TableReference) -> ParsedTable:
        """Parse one source table into a summary and parse status."""


def is_time_header(value: str) -> bool:
    """Return whether a header cell should start the time-observation block."""

    cleaned = value.strip()
    if cleaned.casefold() == "time":
        return True
    return bool(TIME_HEADER_RE.match(cleaned))


def detect_time_start(header: list[str]) -> int | None:
    """Return the first observation-column index for a parsed header row."""

    for index, value in enumerate(header):
        if "\\Time" in value:
            return index + 1
        if is_time_header(value):
            return index
    return None


def parse_observation_cell(row_number: int, column_name: str, raw_value: str) -> ParsedObservation:
    """Parse a numeric, missing, or flagged observation cell."""

    raw = raw_value.strip()
    if raw in MISSING_MARKERS:
        return ParsedObservation(
            row_number=row_number,
            column_name=column_name,
            raw_value=raw_value,
            missing=True,
        )
    if raw.startswith(":"):
        flag = raw[1:].strip() or None
        return ParsedObservation(
            row_number=row_number,
            column_name=column_name,
            raw_value=raw_value,
            missing=True,
            flag=flag,
        )

    match = NUMERIC_CELL_RE.match(raw)
    if not match:
        return ParsedObservation(
            row_number=row_number,
            column_name=column_name,
            raw_value=raw_value,
            missing=True,
            flag="non_numeric",
        )

    return ParsedObservation(
        row_number=row_number,
        column_name=column_name,
        raw_value=raw_value,
        numeric_value=float(match.group("number")),
        missing=False,
        flag=match.group("flag"),
    )


def _has_gzip_magic(path: Path) -> bool:
    with path.open("rb") as file:
        return file.read(2) == b"\x1f\x8b"


def is_gzip_wrapped_tar_without_member(path: Path, member_filename: str) -> bool:
    """Return whether a file is a gzip-wrapped tar that lacks the expected member."""

    if not path.exists() or not _has_gzip_magic(path):
        return False
    try:
        decompressed = gzip.decompress(path.read_bytes())
        if decompressed[257:262] != b"ustar":
            return False
        with tarfile.open(fileobj=io.BytesIO(decompressed), mode="r:") as tar:
            return all(Path(member.name).name != member_filename for member in tar.getmembers())
    except (OSError, tarfile.TarError, EOFError):
        return False


@contextmanager
def _open_source_csv(path: Path) -> Iterator[TextIO]:
    if _has_gzip_magic(path):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as file:
            yield file
    else:
        with path.open("r", encoding="utf-8", newline="") as file:
            yield file


def _read_csv_rows(path: Path) -> list[list[str]]:
    with _open_source_csv(path) as file:
        return [row for row in csv.reader(file) if row and any(cell.strip() for cell in row)]


def _corpus_dir(config: AppConfig) -> Path:
    if config.corpus.name == "fixture":
        if config.corpus.fixture_path is None:
            raise ValueError("fixture corpus requires corpus.fixture_path")
        return config.corpus.fixture_path

    archive_name = (
        config.corpus.archive_name
        or f"eurostat_{config.corpus.expected_table_count}_tables.tgz"
    )
    return config.paths.raw_dir / archive_name.removesuffix(".tgz")


def _iter_table_files(config: AppConfig) -> Iterator[Path]:
    corpus_dir = _corpus_dir(config)
    pattern = "fixture_*.csv" if config.corpus.name == "fixture" else "*.csv"
    yield from sorted(corpus_dir.glob(pattern))


def _evidence_url(table_id: str) -> str:
    return f"https://ec.europa.eu/eurostat/databrowser/view/{table_id}/default/table?lang=en"


def _load_fixture_titles(config: AppConfig) -> dict[str, str]:
    if config.corpus.fixture_path is None:
        return {}
    path = config.corpus.fixture_path / "titles.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as file:
        return {
            row["table_id"].strip(): row["title"].strip()
            for row in csv.DictReader(file)
            if row.get("table_id") and row.get("title")
        }


def _candidate_title_paths(config: AppConfig) -> tuple[Path, ...]:
    return (
        config.paths.external_dir / "eurostat_toc_en.txt",
        config.paths.external_dir / "titles.csv",
        config.paths.raw_dir / "titles.csv",
    )


def _load_catalog_titles(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    sample = path.read_text(encoding="utf-8", errors="replace")[:4096]
    delimiter = "\t" if "\t" in sample else ","
    titles: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        if reader.fieldnames:
            fields = {field.casefold(): field for field in reader.fieldnames}
            code_field = fields.get("code") or fields.get("table_id") or fields.get("id")
            title_field = fields.get("title") or fields.get("label")
            if code_field and title_field:
                for row in reader:
                    code = (row.get(code_field) or "").strip()
                    title = (row.get(title_field) or "").strip()
                    if code and title:
                        titles[code] = title
                return titles

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split(delimiter) if part.strip()]
        if len(parts) < 2:
            continue
        code = next((part for part in parts if re.match(r"^[A-Za-z][A-Za-z0-9_$.-]+$", part)), "")
        if not code:
            continue
        title = max((part for part in parts if part != code), key=len, default="")
        if title:
            titles[code] = title
    return titles


def load_titles(config: AppConfig) -> dict[str, str]:
    """Load table titles from fixture metadata or acquired Eurostat catalog files."""

    if config.corpus.name == "fixture":
        return _load_fixture_titles(config)

    titles: dict[str, str] = {}
    for path in _candidate_title_paths(config):
        titles.update(_load_catalog_titles(path))
    return titles


class EurostatStarAdapter:
    """Adapter for the STAR Eurostat CSV corpus."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.titles = load_titles(config)
        self.source_repairs: list[SourceRepair] = []

    def _resolve_source_repair(self, csv_path: Path, table_id: str) -> SourceRepair | None:
        defect = KNOWN_SOURCE_DEFECTS.get(table_id)
        if defect is None:
            return None

        original_md5 = file_md5(csv_path)
        original_size = csv_path.stat().st_size
        replacement_path = self.config.paths.raw_dir / defect.replacement_relative_path
        if not replacement_path.exists():
            return SourceRepair(
                table_id=table_id,
                original_path=str(csv_path),
                original_md5=original_md5,
                original_size_bytes=original_size,
                defect_reason=defect.reason,
                replacement_path=str(replacement_path),
                replacement_md5="",
                replacement_size_bytes=0,
                replacement_source_corpus=defect.replacement_source_corpus,
                repair_status="missing_replacement",
                message="known-defect replacement file is missing",
            )

        if not is_gzip_wrapped_tar_without_member(csv_path, csv_path.name):
            return None

        replacement_md5 = file_md5(replacement_path)
        replacement_size = replacement_path.stat().st_size
        parsed_probe = self._parse_reference(
            TableReference(
                table_id=table_id,
                filename=csv_path.name,
                local_path=str(replacement_path),
                original_local_path=str(csv_path),
                title=self.titles.get(table_id),
                evidence_url=_evidence_url(table_id),
                file_md5=replacement_md5,
                original_file_md5=original_md5,
                size_bytes=replacement_size,
                original_size_bytes=original_size,
                source_repaired=True,
                repair_reason=defect.reason,
            )
        )
        if parsed_probe.parse_status == ParseStatus.FAILED:
            return SourceRepair(
                table_id=table_id,
                original_path=str(csv_path),
                original_md5=original_md5,
                original_size_bytes=original_size,
                defect_reason=defect.reason,
                replacement_path=str(replacement_path),
                replacement_md5=replacement_md5,
                replacement_size_bytes=replacement_size,
                replacement_source_corpus=defect.replacement_source_corpus,
                repair_status="invalid_replacement",
                message="known-defect replacement file did not parse",
            )

        return SourceRepair(
            table_id=table_id,
            original_path=str(csv_path),
            original_md5=original_md5,
            original_size_bytes=original_size,
            defect_reason=defect.reason,
            replacement_path=str(replacement_path),
            replacement_md5=replacement_md5,
            replacement_size_bytes=replacement_size,
            replacement_source_corpus=defect.replacement_source_corpus,
            repair_status="applied",
            message="known upstream packaging defect repaired during ingestion",
        )

    def iter_references(self) -> Iterator[TableReference]:
        self.source_repairs = []
        for csv_path in _iter_table_files(self.config):
            table_id = csv_path.stem
            source_repair = self._resolve_source_repair(csv_path, table_id)
            if source_repair is not None:
                self.source_repairs.append(source_repair)
            if source_repair is not None and source_repair.repair_status == "applied":
                yield TableReference(
                    table_id=table_id,
                    filename=csv_path.name,
                    local_path=source_repair.replacement_path,
                    original_local_path=source_repair.original_path,
                    title=self.titles.get(table_id),
                    evidence_url=_evidence_url(table_id),
                    file_md5=source_repair.replacement_md5,
                    original_file_md5=source_repair.original_md5,
                    size_bytes=source_repair.replacement_size_bytes,
                    original_size_bytes=source_repair.original_size_bytes,
                    source_repaired=True,
                    repair_reason=source_repair.defect_reason,
                )
                continue

            original_md5 = file_md5(csv_path)
            yield TableReference(
                table_id=table_id,
                filename=csv_path.name,
                local_path=str(csv_path),
                original_local_path=str(csv_path),
                title=self.titles.get(table_id),
                evidence_url=_evidence_url(table_id),
                file_md5=original_md5,
                original_file_md5=original_md5,
                size_bytes=csv_path.stat().st_size,
                original_size_bytes=csv_path.stat().st_size,
            )

    def parse_table(self, reference: TableReference) -> ParsedTable:
        return self._parse_reference(reference)

    def _parse_reference(self, reference: TableReference) -> ParsedTable:
        warnings: list[ParseWarning] = []
        path = Path(reference.local_path)
        try:
            rows = _read_csv_rows(path)
        except Exception as exc:
            return ParsedTable(
                reference=reference,
                columns=(),
                row_count=0,
                observation_count=0,
                malformed_row_count=0,
                warnings=(ParseWarning(table_id=reference.table_id, reason=str(exc)),),
                parse_status=ParseStatus.FAILED,
            )

        if not rows or not rows[0]:
            return ParsedTable(
                reference=reference,
                columns=(),
                row_count=0,
                observation_count=0,
                malformed_row_count=0,
                warnings=(ParseWarning(table_id=reference.table_id, reason="empty CSV"),),
                parse_status=ParseStatus.FAILED,
            )

        header = rows[0]
        if any("\x00" in cell for cell in header):
            return ParsedTable(
                reference=reference,
                columns=(),
                row_count=0,
                observation_count=0,
                malformed_row_count=0,
                warnings=(
                    ParseWarning(
                        table_id=reference.table_id,
                        reason="decompressed content contains NUL bytes; not a valid CSV header",
                    ),
                ),
                parse_status=ParseStatus.FAILED,
            )

        first_time_index = detect_time_start(header)
        if first_time_index is None:
            warnings.append(
                ParseWarning(table_id=reference.table_id, reason="no time columns detected")
            )
            first_time_index = len(header)

        if reference.title is None:
            warnings.append(ParseWarning(table_id=reference.table_id, reason="missing title"))

        columns = tuple(
            ParsedColumn(
                name=name,
                index=index,
                role="metadata" if index < first_time_index else "time",
            )
            for index, name in enumerate(header)
        )
        time_columns = header[first_time_index:]
        malformed_rows = 0
        observation_count = 0
        non_numeric_warnings = 0

        for row_number, row in enumerate(rows[1:], start=2):
            if len(row) != len(header):
                malformed_rows += 1
                warnings.append(
                    ParseWarning(
                        table_id=reference.table_id,
                        row_number=row_number,
                        reason="row has unexpected column count",
                        expected_columns=len(header),
                        actual_columns=len(row),
                    )
                )
                continue

            for column_name, raw_value in zip(time_columns, row[first_time_index:], strict=False):
                parsed = parse_observation_cell(row_number, column_name, raw_value)
                observation_count += 1
                if parsed.flag == "non_numeric":
                    non_numeric_warnings += 1

        if non_numeric_warnings:
            warnings.append(
                ParseWarning(
                    table_id=reference.table_id,
                    reason=f"{non_numeric_warnings} non-numeric observation cells",
                )
            )

        status = ParseStatus.PARSED if not warnings else ParseStatus.WARNING
        if not time_columns:
            status = ParseStatus.WARNING

        return ParsedTable(
            reference=reference,
            columns=columns,
            row_count=len(rows) - 1,
            observation_count=observation_count,
            malformed_row_count=malformed_rows,
            warnings=tuple(warnings),
            parse_status=status,
        )


def parsed_table_to_row(table: ParsedTable) -> dict[str, object]:
    """Convert a parsed table summary to a flat Parquet/JSON row."""

    metadata_columns = [column.name for column in table.columns if column.role == "metadata"]
    time_columns = [column.name for column in table.columns if column.role == "time"]
    return {
        "table_id": table.reference.table_id,
        "filename": table.reference.filename,
        "title": table.reference.title,
        "source_url": table.reference.evidence_url,
        "file_md5": table.reference.file_md5,
        "original_file_md5": table.reference.original_file_md5,
        "parsed_file_md5": table.reference.file_md5,
        "size_bytes": table.reference.size_bytes,
        "original_size_bytes": table.reference.original_size_bytes,
        "source_repaired": table.reference.source_repaired,
        "repair_reason": table.reference.repair_reason,
        "original_local_path": table.reference.original_local_path,
        "parsed_local_path": table.reference.local_path,
        "metadata_columns": metadata_columns,
        "time_columns": time_columns,
        "row_count": table.row_count,
        "observation_count": table.observation_count,
        "malformed_row_count": table.malformed_row_count,
        "warning_count": len(table.warnings),
        "parse_status": table.parse_status.value,
        "warning_reasons": [warning.reason for warning in table.warnings],
    }


def _diagnostics(
    config: AppConfig,
    parsed_tables: Iterable[ParsedTable],
    resources: Iterable[ResourceRecord] = (),
    source_repairs: Iterable[SourceRepair] = (),
) -> dict[str, object]:
    tables = list(parsed_tables)
    status_counts = Counter(table.parse_status.value for table in tables)
    warning_tables = [
        {
            "table_id": table.reference.table_id,
            "parse_status": table.parse_status.value,
            "reasons": [warning.reason for warning in table.warnings],
        }
        for table in tables
        if table.warnings
    ]
    failed_tables = [
        {
            "table_id": table.reference.table_id,
            "reasons": [warning.reason for warning in table.warnings],
        }
        for table in tables
        if table.parse_status == ParseStatus.FAILED
    ]
    return {
        "config_name": config.config_name,
        "corpus": config.corpus.name,
        "expected_table_count": config.corpus.expected_table_count,
        "inventoried_table_count": len(tables),
        "status_counts": dict(status_counts),
        "warning_table_count": len(warning_tables),
        "failed_table_count": len(failed_tables),
        "warning_tables": warning_tables,
        "failed_tables": failed_tables,
        "resource_validation": [resource.model_dump(mode="json") for resource in resources],
        "source_repairs": [repair.model_dump(mode="json") for repair in source_repairs],
    }


def write_tables_parquet(rows: list[dict[str, object]], output_path: Path) -> Path:
    """Write table inventory rows to Parquet."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)
    return output_path


def write_ingestion_diagnostics(payload: dict[str, object], output_path: Path) -> Path:
    """Write ingestion diagnostics as stable JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def run_ingestion(
    config: AppConfig,
    *,
    resources: Iterable[ResourceRecord] = (),
) -> tuple[Path, Path, Path, dict[str, object]]:
    """Run Milestone 1 ingestion and write all core artifacts."""

    manifest = create_manifest(config, "ingest")
    adapter = EurostatStarAdapter(config)
    references = list(adapter.iter_references())
    parsed_tables = [adapter.parse_table(reference) for reference in references]
    rows = [parsed_table_to_row(table) for table in parsed_tables]

    tables_path = write_tables_parquet(rows, config.paths.processed_dir / "tables.parquet")
    diagnostics = _diagnostics(
        config,
        parsed_tables,
        resources,
        adapter.source_repairs,
    )
    diagnostics_path = write_ingestion_diagnostics(
        diagnostics,
        config.paths.processed_dir / "ingestion_diagnostics.json",
    )
    completed = complete_manifest(
        manifest,
        resources=tuple(resource.resource_id for resource in resources),
        artifacts=(str(tables_path), str(diagnostics_path)),
    )
    manifest_path = write_manifest(
        completed,
        config.paths.outputs_dir / "manifests" / f"{manifest.run_id}_ingest.json",
    )
    return tables_path, diagnostics_path, manifest_path, diagnostics


def fixture_vocabulary_candidates(parsed_table: ParsedTable) -> tuple[str, ...]:
    """Return only metadata names and values from a parsed fixture table.

    This helper exists to lock the Milestone 1 invariant that observation values
    do not flow into vocabulary candidate outputs.
    """

    header = [column.name for column in parsed_table.columns]
    metadata_indexes = [
        column.index for column in parsed_table.columns if column.role == "metadata"
    ]
    values: set[str] = {header[index] for index in metadata_indexes}
    rows = _read_csv_rows(Path(parsed_table.reference.local_path))
    for row in rows[1:]:
        if len(row) != len(header):
            continue
        for index in metadata_indexes:
            if row[index].strip():
                values.add(row[index].strip())
    return tuple(sorted(values))
