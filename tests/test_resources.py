from pathlib import Path

from statvocab.config import load_config
from statvocab.contracts import ResourceValidationStatus
from statvocab.resources import (
    ResourceSpec,
    core_resource_specs,
    download_resource,
    validate_local_resource,
)


def test_checksum_validation_accepts_expected_md5(tmp_path: Path) -> None:
    path = tmp_path / "resource.txt"
    path.write_text("abc", encoding="utf-8")
    spec = ResourceSpec(
        name="fixture_resource",
        url="https://example.test/resource.txt",
        local_path=path,
        license="test",
        version="v1",
        expected_md5="900150983cd24fb0d6963f7d28e17f72",
    )

    record = validate_local_resource(spec)

    assert record is not None
    assert record.validation_status == ResourceValidationStatus.VERIFIED


def test_valid_local_resource_is_reused_without_download(tmp_path: Path) -> None:
    path = tmp_path / "resource.txt"
    path.write_text("already here", encoding="utf-8")
    spec = ResourceSpec(
        name="fixture_resource",
        url="https://example.test/resource.txt",
        local_path=path,
        license="test",
        version="v1",
    )

    record = download_resource(spec)

    assert record.validation_status == ResourceValidationStatus.REUSED
    assert record.checksum_md5 is not None


def test_failed_partial_download_keeps_part_file(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "resource.txt"
    partial_path = tmp_path / "resource.txt.part"
    partial_path.write_bytes(b"partial")
    spec = ResourceSpec(
        name="fixture_resource",
        url="https://example.test/resource.txt",
        local_path=path,
        license="test",
        version="v1",
    )

    def raise_stream(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("statvocab.resources.httpx.stream", raise_stream)

    record = download_resource(spec)

    assert record.validation_status == ResourceValidationStatus.FAILED
    assert partial_path.exists()
    assert not path.exists()


def test_resource_specs_include_provenance_fields() -> None:
    config = load_config("configs/core.yaml")

    records = core_resource_specs(config)

    assert records
    assert all(record.url for record in records)
    assert all(record.license for record in records)
    assert all(record.version for record in records)
