"""Smoke tests for the dim-dispatch plotter.

We don't pixel-check the output — we verify that the dispatcher accepts
1D / 2D inputs, writes a non-empty PNG to the requested path, and refuses
≥3D inputs (caller must slice).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from viz.dispatch import plot_field


def test_1d_writes_png(tmp_path: Path) -> None:
    x = np.linspace(-1, 1, 64)
    y = np.exp(-(x**2) * 4)
    out = tmp_path / "field_1d.png"
    plot_field(y, [x], out_path=out, title="1D test", axis_labels=["x"])
    assert out.exists() and out.stat().st_size > 0


def test_1d_with_overlay(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 64)
    y = np.exp(-(x**2) * 4)
    samples_x = rng.uniform(-1, 1, size=200)
    samples_y = (rng.uniform(0, 1, size=200) < np.exp(-(samples_x**2) * 4)).astype(float)
    out = tmp_path / "field_1d_overlay.png"
    plot_field(
        y,
        [x],
        out_path=out,
        title="1D + scatter",
        axis_labels=["x"],
        overlay_xy=(samples_x, samples_y),
    )
    assert out.exists() and out.stat().st_size > 0


def test_2d_writes_png(tmp_path: Path) -> None:
    x0 = np.linspace(-1, 1, 32)
    x1 = np.linspace(-1, 1, 40)
    grid_0, grid_1 = np.meshgrid(x0, x1, indexing="ij")
    z = np.exp(-(grid_0**2 + grid_1**2) * 2)
    out = tmp_path / "field_2d.png"
    plot_field(z, [x0, x1], out_path=out, title="2D test", axis_labels=["x0", "x1"])
    assert out.exists() and out.stat().st_size > 0


def test_3d_raises(tmp_path: Path) -> None:
    grids = [np.linspace(0, 1, 4)] * 3
    values = np.zeros((4, 4, 4))
    with pytest.raises(NotImplementedError):
        plot_field(values, grids, out_path=tmp_path / "x.png", title="3D")


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    x = np.linspace(0, 1, 10)
    y = np.zeros(11)  # wrong length
    with pytest.raises(ValueError):
        plot_field(y, [x], out_path=tmp_path / "x.png", title="bad")


def test_axis_label_count_mismatch_raises(tmp_path: Path) -> None:
    x = np.linspace(0, 1, 10)
    y = np.zeros(10)
    with pytest.raises(ValueError):
        plot_field(
            y,
            [x],
            out_path=tmp_path / "x.png",
            title="bad",
            axis_labels=["a", "b"],
        )
