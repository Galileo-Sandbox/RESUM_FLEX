"""Packaging invariants for the complementary Pixi and uv setup."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]


def test_torch_numpy_round_trip() -> None:
    """Exercise the C-ABI boundary under both locked NumPy test groups."""
    array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    tensor = torch.from_numpy(array)
    np.testing.assert_array_equal(tensor.numpy(), array)


def test_pixi_delegates_python_resolution_to_uv() -> None:
    """Pixi must not grow a second, drifting list of Python packages."""
    with (ROOT / "pixi.toml").open("rb") as stream:
        manifest = tomllib.load(stream)

    assert set(manifest["dependencies"]) == {"uv"}
    assert "pypi-dependencies" not in manifest

    tasks = manifest["tasks"]
    assert "--frozen" in tasks["test-numpy1"]["cmd"]
    assert "--frozen" in tasks["test-numpy2"]["cmd"]
    assert (
        tasks["test-numpy1"]["env"]["UV_PROJECT_ENVIRONMENT"]
        != (tasks["test-numpy2"]["env"]["UV_PROJECT_ENVIRONMENT"])
    )
    assert tasks["test-numpy1"]["env"]["OPENBLAS_NUM_THREADS"] == "2"
    assert tasks["test-numpy2"]["env"]["OPENBLAS_NUM_THREADS"] == "1"


def test_legacy_gp_extra_is_self_contained() -> None:
    """The published legacy extra must reject an incompatible modern stack."""
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    requirements = project["project"]["optional-dependencies"]["gp-numpy1"]
    assert "numpy>=1.24,<2" in requirements
    assert "scipy>=1.10,<=1.12" in requirements
