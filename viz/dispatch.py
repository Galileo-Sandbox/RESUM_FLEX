"""Dimension-dispatching plotting primitives.

Single entry point :func:`plot_field` takes values that have already been
evaluated on a regular axis grid and chooses the layout based on the
number of axes:

* 1 axis  → line plot (with optional scatter overlay for raw samples).
* 2 axes  → heatmap (``imshow``) with a colorbar.
* ≥3 axes → :class:`NotImplementedError`. Higher-dim fields are scenario
  specific (e.g. fix one axis, slice another); callers should reduce to
  1D / 2D themselves and call this twice if they want both views.

Modules calling this never branch on ``dim`` themselves — that's the
point of the dispatcher.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_field(
    values: np.ndarray,
    axis_grids: Sequence[np.ndarray],
    *,
    out_path: str | Path,
    title: str,
    axis_labels: Sequence[str] | None = None,
    value_label: str = "p",
    overlay_xy: tuple[np.ndarray, np.ndarray] | None = None,
    overlay_label: str = "samples",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """Plot a scalar field defined on a regular grid.

    Parameters
    ----------
    values
        Array of scalar values. Shape must match ``tuple(len(g) for g in axis_grids)``.
    axis_grids
        One 1D array of axis ticks per input dimension.
    out_path
        Where to save the figure (PNG inferred from suffix).
    title
        Figure title.
    axis_labels
        One label per axis. Default: ``["x_0", "x_1", ...]``.
    value_label
        Label for the dependent variable (line y-axis or colorbar).
    overlay_xy
        Optional ``(x, y)`` arrays to scatter on top of a 1D plot — used to
        show binary samples ``X`` against the analytical ``p(x)`` curve.
        Ignored for 2D plots.
    overlay_label
        Legend label for the overlay scatter.
    cmap, vmin, vmax
        Forwarded to matplotlib for 2D plots.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_dims = len(axis_grids)
    if axis_labels is None:
        axis_labels = [f"x_{i}" for i in range(n_dims)]
    if len(axis_labels) != n_dims:
        raise ValueError(
            f"axis_labels has {len(axis_labels)} entries but field has {n_dims} axes"
        )
    expected_shape = tuple(len(g) for g in axis_grids)
    if values.shape != expected_shape:
        raise ValueError(
            f"values.shape={values.shape} does not match grid shape {expected_shape}"
        )

    if n_dims == 1:
        _plot_1d(
            values,
            axis_grids[0],
            out_path=out_path,
            title=title,
            xlabel=axis_labels[0],
            ylabel=value_label,
            overlay_xy=overlay_xy,
            overlay_label=overlay_label,
        )
    elif n_dims == 2:
        _plot_2d(
            values,
            axis_grids,
            out_path=out_path,
            title=title,
            xlabels=axis_labels,
            cbar_label=value_label,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        raise NotImplementedError(
            f"plot_field supports 1D and 2D fields; got {n_dims}D. "
            "Slice or marginalize before calling."
        )
    return out_path


def _plot_1d(
    y: np.ndarray,
    x: np.ndarray,
    *,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    overlay_xy: tuple[np.ndarray, np.ndarray] | None,
    overlay_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, color="C0", linewidth=2, label=ylabel)
    if overlay_xy is not None:
        ox, oy = overlay_xy
        # Jitter binary samples vertically by class so 0s and 1s don't overlap.
        jitter = (np.random.default_rng(0).uniform(-0.02, 0.02, size=ox.shape))
        ax.scatter(
            ox,
            oy + jitter,
            s=8,
            alpha=0.35,
            c=oy,
            cmap="coolwarm",
            label=overlay_label,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_comparison_1d(
    x: np.ndarray,
    analytical: np.ndarray,
    predicted: np.ndarray,
    *,
    out_path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str = "p",
    overlay_xy: tuple[np.ndarray, np.ndarray] | None = None,
    overlay_label: str = "samples",
    analytical_label: str = "analytical p",
    predicted_label: str = "predicted β",
) -> Path:
    """Overlay analytical and predicted curves on the same 1-D axes.

    Implements the comparison rule for 1-D fields: both curves on a
    single set of axes, with raw samples (binary X or per-trial rates)
    optionally scattered underneath for context.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if analytical.shape != predicted.shape or analytical.shape != x.shape:
        raise ValueError(
            f"shapes must match: x={x.shape}, analytical={analytical.shape}, "
            f"predicted={predicted.shape}"
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if overlay_xy is not None:
        ox, oy = overlay_xy
        jitter = np.random.default_rng(0).uniform(-0.02, 0.02, size=ox.shape)
        ax.scatter(
            ox, oy + jitter,
            s=8, alpha=0.3, c=oy, cmap="coolwarm", label=overlay_label,
            zorder=1,
        )
    ax.plot(x, analytical, color="C0", linewidth=2.4, label=analytical_label, zorder=3)
    ax.plot(
        x, predicted, color="C3", linewidth=2.0, linestyle="--",
        label=predicted_label, zorder=2,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_comparison_2d(
    analytical: np.ndarray,
    predicted: np.ndarray,
    axis_grids: Sequence[np.ndarray],
    *,
    out_path: str | Path,
    title: str,
    axis_labels: Sequence[str],
    value_label: str = "p",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """Side-by-side heatmaps ``[analytical | predicted]`` with a shared colorbar.

    Implements the comparison rule for 2-D fields: both panels on the
    same colorbar so a darker spot on the right is darker for the right
    reason. ``vmin`` / ``vmax`` default to the joint min/max across the
    two arrays so noise differences don't induce spurious contrast.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(axis_grids) != 2:
        raise ValueError(f"plot_comparison_2d expects 2 axis grids, got {len(axis_grids)}")
    expected_shape = (len(axis_grids[0]), len(axis_grids[1]))
    for name, arr in [("analytical", analytical), ("predicted", predicted)]:
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name}.shape={arr.shape} does not match grid {expected_shape}"
            )
    if vmin is None:
        vmin = float(min(analytical.min(), predicted.min()))
    if vmax is None:
        vmax = float(max(analytical.max(), predicted.max()))

    g0, g1 = axis_grids
    extent = (g1[0], g1[-1], g0[0], g0[-1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, arr, panel_title in [
        (axes[0], analytical, "analytical p"),
        (axes[1], predicted, "predicted β"),
    ]:
        im = ax.imshow(
            arr, origin="lower", extent=extent, aspect="auto",
            cmap=cmap, vmin=vmin, vmax=vmax,
        )
        ax.set_title(panel_title)
        ax.set_xlabel(axis_labels[1])
    axes[0].set_ylabel(axis_labels[0])
    fig.suptitle(title)
    fig.colorbar(im, ax=axes, label=value_label, shrink=0.9)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_coverage_test(
    y_raw: np.ndarray,
    y_predicted: np.ndarray,
    sigma_predicted: np.ndarray,
    *,
    out_path: str | Path,
    title: str,
    xlabel: str = "Trial index",
    predicted_label: str = "y_predicted",
    raw_label: str = "y_raw = m/N",
) -> dict[str, float]:
    """Figure-5-style coverage diagnostic with a calibration sub-panel.

    Two-panel figure:

    * **Left (time series)** — ``y_predicted`` as a line, ``±1/2/3 σ``
      shaded bands, ``y_raw`` overlaid as black dots. Trials are kept in
      their natural input order (no sorting), so the bands and dots jump
      around together — and the "are dots inside the bands?" question
      stays honest, not visually choreographed by sorting.
    * **Right (calibration)** — paired bars per σ level: target Gaussian
      coverage (68.27 / 95.45 / 99.73 %) vs measured. A well-calibrated
      predictor has paired bars of equal height; under-tight bands sit
      below target, over-wide ones sit above.

    Returns the measured coverage dict.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not (y_raw.shape == y_predicted.shape == sigma_predicted.shape):
        raise ValueError(
            f"shape mismatch: y_raw={y_raw.shape}, y_predicted={y_predicted.shape}, "
            f"sigma={sigma_predicted.shape}"
        )
    if y_raw.ndim != 1:
        raise ValueError(f"inputs must be 1-D arrays of length B; got ndim={y_raw.ndim}")
    if np.any(sigma_predicted < 0):
        raise ValueError("sigma_predicted must be non-negative")

    abs_diff = np.abs(y_raw - y_predicted)
    coverage = {
        "1sigma": float((abs_diff <= 1.0 * sigma_predicted).mean()),
        "2sigma": float((abs_diff <= 2.0 * sigma_predicted).mean()),
        "3sigma": float((abs_diff <= 3.0 * sigma_predicted).mean()),
    }
    target = {"1sigma": 0.6827, "2sigma": 0.9545, "3sigma": 0.9973}

    x = np.arange(len(y_predicted))

    fig, (ax_ts, ax_cal) = plt.subplots(
        1, 2, figsize=(13, 5),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # ---- Left: time series ----
    ax_ts.fill_between(
        x, y_predicted - 3 * sigma_predicted, y_predicted + 3 * sigma_predicted,
        color="#d62728", alpha=0.18,
        label=f"±3σ ({coverage['3sigma'] * 100:.0f}%)",
    )
    ax_ts.fill_between(
        x, y_predicted - 2 * sigma_predicted, y_predicted + 2 * sigma_predicted,
        color="#ffd700", alpha=0.40,
        label=f"±2σ ({coverage['2sigma'] * 100:.0f}%)",
    )
    ax_ts.fill_between(
        x, y_predicted - 1 * sigma_predicted, y_predicted + 1 * sigma_predicted,
        color="#2ca02c", alpha=0.50,
        label=f"±1σ ({coverage['1sigma'] * 100:.0f}%)",
    )
    ax_ts.plot(x, y_predicted, color="C0", linewidth=1.5, label=predicted_label, zorder=3)
    ax_ts.scatter(x, y_raw, c="black", s=14, alpha=0.85, label=raw_label, zorder=4)
    ax_ts.set_xlabel(xlabel)
    ax_ts.set_ylabel("y")
    lo = min(-0.02, float((y_predicted - 3 * sigma_predicted).min()) - 0.02)
    hi = max(
        1.02,
        float(y_raw.max()) + 0.05,
        float((y_predicted + 3 * sigma_predicted).max()) + 0.02,
    )
    ax_ts.set_ylim(lo, hi)
    ax_ts.legend(loc="best", fontsize=9)
    ax_ts.grid(alpha=0.3)

    # ---- Right: calibration ----
    levels = ["1σ", "2σ", "3σ"]
    target_vals = [target["1sigma"], target["2sigma"], target["3sigma"]]
    actual_vals = [coverage["1sigma"], coverage["2sigma"], coverage["3sigma"]]
    pos = np.arange(len(levels))
    w = 0.38
    ax_cal.bar(
        pos - w / 2, target_vals, w,
        color="lightgray", edgecolor="black", label="target (Gaussian)",
    )
    ax_cal.bar(
        pos + w / 2, actual_vals, w,
        color="C0", edgecolor="black", label="actual",
    )
    for p, t_, a_ in zip(pos, target_vals, actual_vals, strict=True):
        ax_cal.text(p - w / 2, t_ + 0.015, f"{t_ * 100:.1f}%", ha="center", fontsize=8)
        ax_cal.text(p + w / 2, a_ + 0.015, f"{a_ * 100:.1f}%", ha="center", fontsize=8)
    ax_cal.set_xticks(pos)
    ax_cal.set_xticklabels(levels)
    ax_cal.set_ylim(0.0, 1.12)
    ax_cal.set_ylabel("coverage fraction")
    ax_cal.set_title("calibration")
    ax_cal.legend(loc="lower right", fontsize=8)
    ax_cal.grid(alpha=0.3, axis="y")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return coverage


def _plot_2d(
    values: np.ndarray,
    axis_grids: Sequence[np.ndarray],
    *,
    out_path: Path,
    title: str,
    xlabels: Sequence[str],
    cbar_label: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    # values has shape (len(axis_grids[0]), len(axis_grids[1])); first axis is rows.
    # imshow's extent is (xmin, xmax, ymin, ymax); we want axis_grids[0] on Y and
    # axis_grids[1] on X so transpose semantics align with the row/col convention.
    g0, g1 = axis_grids
    extent = (g1[0], g1[-1], g0[0], g0[-1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        values,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabels[1])
    ax.set_ylabel(xlabels[0])
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
