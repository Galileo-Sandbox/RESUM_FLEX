"""Generate Phase 1 ground-truth plots for every scenario in the validation matrix.

Produces ``viz_output/pseudo_ground_truth_S{1..8}.png``. Each plot is one
informative view of the analytical ``t(θ, φ)`` for the scenario, with
sample points overlaid where it helps:

* 1D scenarios (S5, S7) — line plot of ``t`` plus a scatter of either
  the binary ``X`` (S5) or the per-trial empirical rate (S7), so we can
  see at a glance that the noisy samples cluster around the peak.
* 2D scenarios (S1, S6, S8) — heatmap of ``t``.
* 3D / 4D scenarios (S2, S3, S4) — heatmap of ``t`` along the two axes
  in question, with the other parameter(s) sliced at the bump peak.

Run from the repo root:

    python scripts/phase1_plot_ground_truth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so `python scripts/foo.py` works without PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from data.pseudo_generator import for_scenario  # noqa: E402
from viz.dispatch import plot_field  # noqa: E402

OUT_DIR = Path("viz_output")
N_GRID = 80
N_TRIALS = 32
N_EVENTS = 256


def _grid_1d(domain=(-1.0, 1.0), n: int = N_GRID) -> np.ndarray:
    return np.linspace(domain[0], domain[1], n)


def _grid_2d(n: int = N_GRID) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (axis_0, axis_1, mesh) where mesh is the (n, n, 2) input grid."""
    a0 = _grid_1d(n=n)
    a1 = _grid_1d(n=n)
    G0, G1 = np.meshgrid(a0, a1, indexing="ij")
    mesh = np.stack([G0, G1], axis=-1)  # (n, n, 2)
    return a0, a1, mesh


# ---------------------------------------------------------------------------
# Per-scenario plotters.
# ---------------------------------------------------------------------------

def plot_S1(out: Path) -> None:
    gen = for_scenario("S1")
    a0 = _grid_1d()  # θ
    a1 = _grid_1d()  # φ
    G0, G1 = np.meshgrid(a0, a1, indexing="ij")
    theta = G0[..., None]  # (G, G, 1)
    phi = G1[..., None]
    p = gen.truth.evaluate(theta=theta, phi=phi)  # (G, G)
    plot_field(
        p, [a0, a1],
        out_path=out,
        title="S1 — FULL, dim(θ)=1, dim(φ)=1",
        axis_labels=["θ", "φ"],
        value_label="t(θ, φ)",
        vmin=0.0, vmax=gen.truth.t_max,
    )


def plot_S2(out: Path) -> None:
    """2D θ × 1D φ. Slice φ at φ_peak, heatmap over (θ₁, θ₂)."""
    gen = for_scenario("S2")
    a0, a1, mesh = _grid_2d()  # θ-grid
    theta = mesh  # (G, G, 2)
    phi_slice = np.broadcast_to(gen.truth.phi_peak, theta.shape[:-1] + (1,))
    p = gen.truth.evaluate(theta=theta, phi=phi_slice)
    plot_field(
        p, [a0, a1],
        out_path=out,
        title=f"S2 — FULL, dim(θ)=2, dim(φ)=1 (φ sliced at φ_peak={gen.truth.phi_peak[0]:.2f})",
        axis_labels=["θ₁", "θ₂"],
        value_label="t(θ; φ_peak)",
        vmin=0.0, vmax=gen.truth.t_max,
    )


def plot_S3(out: Path) -> None:
    """1D θ × 2D φ. Slice θ at θ_peak, heatmap over (φ₁, φ₂)."""
    gen = for_scenario("S3")
    a0, a1, mesh = _grid_2d()  # φ-grid
    phi = mesh
    theta_slice = np.broadcast_to(gen.truth.theta_peak, phi.shape[:-1] + (1,))
    p = gen.truth.evaluate(theta=theta_slice, phi=phi)
    plot_field(
        p, [a0, a1],
        out_path=out,
        title=f"S3 — FULL, dim(θ)=1, dim(φ)=2 (θ sliced at θ_peak={gen.truth.theta_peak[0]:.2f})",
        axis_labels=["φ₁", "φ₂"],
        value_label="t(θ_peak; φ)",
        vmin=0.0, vmax=gen.truth.t_max,
    )


def plot_S4(out: Path) -> None:
    """2D θ × 2D φ. Slice both pairs at peaks; show θ-heatmap (φ at φ_peak)."""
    gen = for_scenario("S4")
    a0, a1, mesh = _grid_2d()  # θ-grid
    theta = mesh
    phi_slice = np.broadcast_to(gen.truth.phi_peak, theta.shape[:-1] + (2,))
    p = gen.truth.evaluate(theta=theta, phi=phi_slice)
    plot_field(
        p, [a0, a1],
        out_path=out,
        title="S4 — FULL, dim(θ)=2, dim(φ)=2 (φ sliced at φ_peak)",
        axis_labels=["θ₁", "θ₂"],
        value_label="t(θ; φ_peak)",
        vmin=0.0, vmax=gen.truth.t_max,
    )


def plot_S5(out: Path) -> None:
    """1D φ. Line plot + binary X scatter."""
    gen = for_scenario("S5", seed=0)
    grid = _grid_1d()
    p = gen.truth.evaluate(theta=None, phi=grid[:, None])  # (G,)
    batch = gen.generate(n_trials=1, n_events=400)
    assert batch.phi is not None
    overlay = (batch.phi[0, :, 0], batch.labels[0].astype(float))
    plot_field(
        p, [grid],
        out_path=out,
        title="S5 — EVENT_ONLY, dim(φ)=1  (binary X overlay)",
        axis_labels=["φ"],
        value_label="t(φ)",
        overlay_xy=overlay,
        overlay_label="X (binary)",
    )


def plot_S6(out: Path) -> None:
    gen = for_scenario("S6")
    a0, a1, mesh = _grid_2d()
    p = gen.truth.evaluate(theta=None, phi=mesh)
    plot_field(
        p, [a0, a1],
        out_path=out,
        title="S6 — EVENT_ONLY, dim(φ)=2",
        axis_labels=["φ₁", "φ₂"],
        value_label="t(φ)",
        vmin=0.0, vmax=gen.truth.t_max,
    )


def plot_S7(out: Path) -> None:
    """1D θ DESIGN_ONLY. Line + per-trial empirical rate scatter."""
    gen = for_scenario("S7", seed=0)
    grid = _grid_1d()
    p = gen.truth.evaluate(theta=grid[:, None], phi=None)  # (G,)
    batch = gen.generate(n_trials=N_TRIALS, n_events=N_EVENTS)
    assert batch.theta is not None
    rate = batch.labels.mean(axis=1)  # (B,)
    overlay = (batch.theta[:, 0], rate)
    plot_field(
        p, [grid],
        out_path=out,
        title=f"S7 — DESIGN_ONLY, dim(θ)=1  (per-trial empirical rate overlay, N={N_EVENTS})",
        axis_labels=["θ"],
        value_label="t(θ)",
        overlay_xy=overlay,
        overlay_label=f"empirical rate (per trial, N={N_EVENTS})",
    )


def plot_S8(out: Path) -> None:
    gen = for_scenario("S8")
    a0, a1, mesh = _grid_2d()
    p = gen.truth.evaluate(theta=mesh, phi=None)
    plot_field(
        p, [a0, a1],
        out_path=out,
        title="S8 — DESIGN_ONLY, dim(θ)=2",
        axis_labels=["θ₁", "θ₂"],
        value_label="t(θ)",
        vmin=0.0, vmax=gen.truth.t_max,
    )


PLOTTERS = {
    "S1": plot_S1, "S2": plot_S2, "S3": plot_S3, "S4": plot_S4,
    "S5": plot_S5, "S6": plot_S6, "S7": plot_S7, "S8": plot_S8,
}


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for name, fn in PLOTTERS.items():
        out = OUT_DIR / f"pseudo_ground_truth_{name}.png"
        fn(out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
