from pathlib import Path

import pytest

nbclient = pytest.importorskip("nbclient")
nbformat = pytest.importorskip("nbformat")


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
FINAL_ARTIFACTS = (
    ROOT / "data" / "processed" / "vocabulary.parquet",
    ROOT / "outputs" / "measures.csv",
    ROOT / "report" / "classification_metrics.json",
)


@pytest.mark.parametrize("notebook_path", sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.ipynb")))
def test_professor_notebooks_execute_from_clean_kernel(notebook_path: Path) -> None:
    """Execute professor-facing notebooks in filename order without hidden state."""

    missing = [path for path in FINAL_ARTIFACTS if not path.exists()]
    if missing:
        joined = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        pytest.skip(f"final generated artifacts are required for notebook execution: {joined}")

    notebook = nbformat.read(notebook_path, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )

    client.execute()
