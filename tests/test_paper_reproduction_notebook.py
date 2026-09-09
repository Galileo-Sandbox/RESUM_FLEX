"""Structural checks for the paper reproduction notebook."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "paper_2410_03873_reproduction.ipynb"


def load_notebook() -> dict:
    """Load the notebook without adding a runtime notebook dependency."""
    with NOTEBOOK.open(encoding="utf-8") as stream:
        return json.load(stream)


def test_paper_reproduction_notebook_has_an_empty_cnp_placeholder() -> None:
    """Keep the CNP cell genuinely empty until portable event data exist."""
    cells = load_notebook()["cells"]
    placeholders = [
        cell
        for cell in cells
        if "cnp-todo" in cell.get("metadata", {}).get("tags", [])
    ]

    assert len(placeholders) == 1
    assert placeholders[0]["cell_type"] == "code"
    assert placeholders[0]["source"] == []
    assert placeholders[0]["outputs"] == []
    assert placeholders[0]["execution_count"] is None


def test_paper_reproduction_notebook_uses_aggregate_csv_contract() -> None:
    """The current replay must not silently depend on unavailable HDF5 data."""
    cells = load_notebook()["cells"]
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if cell["cell_type"] == "code"
    )

    assert "paper_2410_03873" in source
    assert "MultiFidelityGP" in source
    assert "hf_validation_data_v1.2.csv" in source
    assert "h5py" not in source
    assert "/home/" not in source
    assert "RESUM_LEGACY_ROOT" not in source


def test_paper_reproduction_notebook_contains_executed_figures() -> None:
    """Commit a complete, error-free run while preserving the empty CNP cell."""
    cells = load_notebook()["cells"]
    executed = [
        cell
        for cell in cells
        if cell["cell_type"] == "code"
        and "cnp-todo" not in cell.get("metadata", {}).get("tags", [])
    ]
    outputs = [output for cell in executed for output in cell["outputs"]]

    assert all(cell["execution_count"] is not None for cell in executed)
    assert not any(output["output_type"] == "error" for output in outputs)
    assert sum("image/png" in output.get("data", {}) for output in outputs) == 7
    assert "/tmp/" not in json.dumps(outputs)


def test_no_cnp_comparison_uses_two_separate_code_cells() -> None:
    """Keep the historical-style and current-framework refits visibly separate."""
    code_sources = [
        "".join(cell.get("source", []))
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "code"
    ]

    assert sum("historical_no_cnp_model" in source for source in code_sources) == 1
    assert sum("current_no_cnp_model" in source for source in code_sources) == 1


def test_no_cnp_validation_preserves_trial_order() -> None:
    """Figure 7 must retain validation-file order rather than sort predictions."""
    cells = load_notebook()["cells"]
    figure_7_start = next(
        index
        for index, cell in enumerate(cells)
        if "## 7. Paper Figure 7" in "".join(cell.get("source", []))
    )
    figure_7_end = next(
        index
        for index, cell in enumerate(cells[figure_7_start + 1 :], figure_7_start + 1)
        if "## 8. Reproduction boundary" in "".join(cell.get("source", []))
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells[figure_7_start:figure_7_end]
    )

    assert "argsort" not in source
    assert "sorted by posterior mean" not in source
    assert "original CSV order" in source
    assert source.count("include_likelihood=True") == 1
    assert source.count("include_likelihood=False") == 1
    assert "Future-observation band" in source


def test_bundled_aggregate_data_is_unchanged() -> None:
    """Pin the exact portable aggregate inputs used by the replay."""
    expected = {
        "cnp_v1.6_output.csv": (
            "1903f03a82c3442389cf588a01227a909e679196b2ba94eb715d14bd6fe1c2a3"
        ),
        "hf_validation_data_v1.2.csv": (
            "307723b5b40b42acd68152f4a30f59b120a7b8d0b18d57602c4916eecc6151bf"
        ),
    }
    data_root = ROOT / "notebooks" / "data" / "paper_2410_03873"

    for name, checksum in expected.items():
        digest = hashlib.sha256((data_root / name).read_bytes()).hexdigest()
        assert digest == checksum


def test_bundled_aggregate_data_has_expected_row_counts() -> None:
    """Document the paper-versus-artifact sample-count boundary."""
    data_root = ROOT / "notebooks" / "data" / "paper_2410_03873"
    with (data_root / "cnp_v1.6_output.csv").open(newline="", encoding="utf-8") as stream:
        train_rows = list(csv.DictReader(stream))
    with (data_root / "hf_validation_data_v1.2.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        validation_rows = list(csv.DictReader(stream))

    assert sum(float(row["fidelity"]) == 0 for row in train_rows) == 309
    assert sum(float(row["fidelity"]) == 1 for row in train_rows) == 10
    assert len(validation_rows) == 100
