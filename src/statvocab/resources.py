"""Resource acquisition and validation for Milestone 1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from statvocab.config import AppConfig
from statvocab.contracts import ResourceRecord, ResourceValidationStatus, stable_id

CHUNK_SIZE = 1024 * 1024
USER_AGENT = "statvocab/0.1 resource-acquisition"
STAR_REPOSITORY_VERSION = "31f69971e6286f1d9dada7851c03a649b8ab150b"


@dataclass(frozen=True)
class ResourceSpec:
    """A downloadable or locally verifiable resource."""

    name: str
    url: str
    local_path: Path
    license: str
    version: str
    expected_md5: str | None = None
    expected_size_bytes: int | None = None
    required: bool = True


def file_md5(path: Path) -> str:
    """Return the MD5 checksum for a file without loading it all into memory."""

    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_files(path: Path) -> int:
    """Count CSV files directly inside a corpus folder."""

    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for _ in path.glob("*.csv"))


def core_resource_specs(config: AppConfig) -> tuple[ResourceSpec, ...]:
    """Build the Milestone 1 resource list for the active configuration."""

    raw_dir = config.paths.raw_dir
    external_dir = config.paths.external_dir
    archive_name = config.corpus.archive_name or "eurostat_2000_tables.tgz"
    archive_md5 = config.corpus.expected_md5
    zenodo_file = archive_name
    if archive_name not in {"eurostat_2000_tables.tgz", "eurostat_7605_tables.tgz"}:
        zenodo_file = "eurostat_2000_tables.tgz"

    return (
        ResourceSpec(
            name=f"zenodo_{archive_name}",
            url=f"https://zenodo.org/records/15681384/files/{zenodo_file}?download=1",
            local_path=raw_dir / archive_name,
            license="Zenodo record license; verify from record metadata",
            version="Zenodo record 15681384",
            expected_md5=archive_md5,
            expected_size_bytes=10_500_000 if archive_name == "eurostat_2000_tables.tgz" else None,
            required=False,
        ),
        ResourceSpec(
            name="eurostat_title_catalog_en",
            url="https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/txt?lang=en",
            local_path=external_dir / "eurostat_toc_en.txt",
            license="Eurostat reuse notice",
            version="retrieved-current",
        ),
        ResourceSpec(
            name="nuts_2024_attributes",
            url="https://gisco-services.ec.europa.eu/distribution/v2/nuts/csv/NUTS_AT_2024.csv",
            local_path=external_dir / "NUTS_AT_2024.csv",
            license="EuroGeographics-Eurostat GISCO reuse notice",
            version="NUTS 2024",
        ),
        ResourceSpec(
            name="eurostat_geo_codelist_sdmx3",
            url=(
                "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/"
                "structure/codelist/ESTAT/GEO/14.0?compress=false"
            ),
            local_path=external_dir / "eurostat_geo_codelist.xml",
            license="Eurostat reuse notice",
            version="GEO 14.0",
            required=False,
        ),
        ResourceSpec(
            name="star_original_questions",
            url=(
                "https://raw.githubusercontent.com/AntoineGauquier/"
                "efficient_and_scalable_search_for_statistics/"
                f"{STAR_REPOSITORY_VERSION}/S_i.csv"
            ),
            local_path=external_dir / "star" / "S_i.csv",
            license="Upstream GitHub repository license; verify before redistribution",
            version=STAR_REPOSITORY_VERSION,
        ),
        ResourceSpec(
            name="star_reformulated_questions",
            url=(
                "https://raw.githubusercontent.com/AntoineGauquier/"
                "efficient_and_scalable_search_for_statistics/"
                f"{STAR_REPOSITORY_VERSION}/S_r.csv"
            ),
            local_path=external_dir / "star" / "S_r.csv",
            license="Upstream GitHub repository license; verify before redistribution",
            version=STAR_REPOSITORY_VERSION,
        ),
        ResourceSpec(
            name="star_annotations",
            url=(
                "https://raw.githubusercontent.com/AntoineGauquier/"
                "efficient_and_scalable_search_for_statistics/"
                f"{STAR_REPOSITORY_VERSION}/annotations.csv"
            ),
            local_path=external_dir / "star" / "annotations.csv",
            license="Upstream GitHub repository license; verify before redistribution",
            version=STAR_REPOSITORY_VERSION,
        ),
    )


def _record(
    spec: ResourceSpec,
    status: ResourceValidationStatus,
    *,
    checksum_md5: str | None = None,
    size_bytes: int | None = None,
    message: str = "",
) -> ResourceRecord:
    return ResourceRecord(
        resource_id=stable_id("resource", spec.name, spec.url, spec.version),
        name=spec.name,
        url=spec.url,
        local_path=str(spec.local_path),
        license=spec.license,
        version=spec.version,
        expected_md5=spec.expected_md5,
        expected_size_bytes=spec.expected_size_bytes,
        checksum_md5=checksum_md5,
        size_bytes=size_bytes,
        retrieved_at=datetime.now(UTC),
        validation_status=status,
        message=message,
    )


def validate_local_resource(spec: ResourceSpec) -> ResourceRecord | None:
    """Return a validation record when a resource already exists locally."""

    if not spec.local_path.exists():
        return None

    checksum = file_md5(spec.local_path)
    size = spec.local_path.stat().st_size
    if spec.expected_md5 and checksum != spec.expected_md5:
        return _record(
            spec,
            ResourceValidationStatus.FAILED,
            checksum_md5=checksum,
            size_bytes=size,
            message=f"MD5 mismatch: expected {spec.expected_md5}, found {checksum}",
        )

    message = "local resource is valid"
    if spec.expected_size_bytes is not None:
        tolerance = max(1, int(spec.expected_size_bytes * 0.15))
        if abs(size - spec.expected_size_bytes) > tolerance:
            message = (
                f"MD5 valid; size {size} differs from approximate expected "
                f"{spec.expected_size_bytes}"
            )

    return _record(
        spec,
        ResourceValidationStatus.VERIFIED if spec.expected_md5 else ResourceValidationStatus.REUSED,
        checksum_md5=checksum,
        size_bytes=size,
        message=message,
    )


def download_resource(spec: ResourceSpec, *, timeout_seconds: float = 120.0) -> ResourceRecord:
    """Download one resource with simple resume support and validate it."""

    local_record = validate_local_resource(spec)
    if local_record and local_record.validation_status in {
        ResourceValidationStatus.VERIFIED,
        ResourceValidationStatus.REUSED,
    }:
        return local_record

    spec.local_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = spec.local_path.with_name(f"{spec.local_path.name}.part")
    existing = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    mode = "ab" if existing else "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"

    try:
        with httpx.stream(
            "GET",
            spec.url,
            headers=headers,
            timeout=timeout_seconds,
            follow_redirects=True,
        ) as response:
            if existing and response.status_code == httpx.codes.OK:
                mode = "wb"
            response.raise_for_status()
            with partial_path.open(mode) as file:
                for chunk in response.iter_bytes(CHUNK_SIZE):
                    file.write(chunk)
        partial_path.replace(spec.local_path)
    except Exception as exc:
        return _record(spec, ResourceValidationStatus.FAILED, message=str(exc))

    validated = validate_local_resource(spec)
    if validated is None:
        return _record(
            spec,
            ResourceValidationStatus.MISSING,
            message="download did not create file",
        )
    if validated.validation_status == ResourceValidationStatus.REUSED:
        return validated.model_copy(
            update={"validation_status": ResourceValidationStatus.DOWNLOADED}
        )
    return validated


def acquire_resources(config: AppConfig) -> tuple[ResourceRecord, ...]:
    """Acquire or reuse all Milestone 1 resources for a configuration."""

    records: list[ResourceRecord] = []
    extracted_dir = config.paths.raw_dir / (config.corpus.archive_name or "").removesuffix(".tgz")
    extracted_count = count_csv_files(extracted_dir)

    for spec in core_resource_specs(config):
        if spec.name.startswith("zenodo_"):
            record = download_resource(spec)
            if (
                record.validation_status == ResourceValidationStatus.FAILED
                and extracted_count == config.corpus.expected_table_count
            ):
                record = _record(
                    spec,
                    ResourceValidationStatus.NOT_VERIFIED_ARCHIVE_ABSENT,
                    message=(
                        f"archive unavailable or invalid; using extracted cache with "
                        f"{extracted_count} CSV files"
                    ),
                )
            records.append(record)
            continue

        records.append(download_resource(spec))

    return tuple(records)


def write_resource_manifest(records: Iterable[ResourceRecord], output_path: Path) -> Path:
    """Write resource acquisition metadata as stable JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.model_dump(mode="json") for record in records]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
