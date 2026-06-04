from pathlib import Path

import pytest

from statvocab.config import load_config


@pytest.mark.parametrize(
    ("path", "expected_name", "expected_count"),
    [
        ("configs/core.yaml", "core", 2000),
        ("configs/full.yaml", "full", 7605),
        ("configs/evaluation.yaml", "fixture", 4),
    ],
)
def test_load_configs(path: str, expected_name: str, expected_count: int) -> None:
    config = load_config(path)

    assert config.corpus.name == expected_name
    assert config.corpus.expected_table_count == expected_count
    assert isinstance(config.paths.outputs_dir, Path)

