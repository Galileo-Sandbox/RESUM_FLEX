"""Generate Phase 1 ground-truth plots for every scenario in the validation matrix.

Produces ``viz_output/phase1_ground_truth/pseudo_ground_truth_S{1..8}.png``. Each plot is one
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

OUT_DIR = Path("viz_output/phase1_ground_truth")
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
    """1D θ DESIGN_ONLY. Line + raw per-event scatter."""
    gen = for_scenario("S7", seed=0)
    grid = _grid_1d()
    p = gen.truth.evaluate(theta=grid[:, None], phi=None)  # (G,)
    batch = gen.generate(n_trials=N_TRIALS, n_events=N_EVENTS)
    assert batch.theta is not None
    # Raw event dots: (trial's θ, individual X_i) — one dot per event.
    raw_theta = np.repeat(batch.theta[:, 0], N_EVENTS)
    raw_x = batch.labels.flatten().astype(float)
    plot_field(
        p, [grid],
        out_path=out,
        title="S7 — DESIGN_ONLY, dim(θ)=1  (raw events overlay)",
        axis_labels=["θ"],
        value_label="t(θ)",
        overlay_xy=(raw_theta, raw_x),
        overlay_label="X (raw, per event)",
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


def _plot_theta_1d_marginal(
    name: str, gen, out: Path, *, n_trials: int = N_TRIALS, n_events: int = N_EVENTS
) -> None:
    """Marginal-θ plot for FULL-mode scenarios with dim(θ)=1 (S1, S3).

    Analytical curve is ``p̄(θ) = ∫ p(θ, φ) g(φ) dφ`` estimated by MC over
    a uniform φ grid. Dots are raw per-event (θ_k, X_ki) — one dot per
    event, never an aggregate.
    """
    grid = _grid_1d()
    G = grid.shape[0]
    rng = np.random.default_rng(11)
    n_phi_mc = 200
    phi_mc = rng.uniform(-1.0, 1.0, size=(G, n_phi_mc, gen.dim_phi))
    theta_per_event = grid[:, None, None]  # [G, 1, 1]
    p_at_phi = gen.truth.evaluate(theta=theta_per_event, phi=phi_mc)  # [G, n_phi_mc]
    p_curve = p_at_phi.mean(axis=1)

    batch = gen.generate(n_trials=n_trials, n_events=n_events)
    assert batch.theta is not None
    raw_theta = np.repeat(batch.theta[:, 0], n_events)
    raw_x = batch.labels.flatten().astype(float)

    plot_field(
        p_curve, [grid],
        out_path=out,
        title=(
            f"{name} — θ-projection (marginal over φ ~ U[-1,1]^{gen.dim_phi}); "
            f"raw events overlay"
        ),
        axis_labels=["θ"],
        value_label="t̄(θ)",
        overlay_xy=(raw_theta, raw_x),
        overlay_label="X (raw, per event)",
    )


PLOTTERS = {
    "S1": plot_S1, "S2": plot_S2, "S3": plot_S3, "S4": plot_S4,
    "S5": plot_S5, "S6": plot_S6, "S7": plot_S7, "S8": plot_S8,
}

# Scenarios with dim(θ)=1 whose main plot is 2D; add a θ-projection too.
THETA_1D_EXTRA_FOR = {"S1", "S3"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in PLOTTERS.items():
        out = OUT_DIR / f"pseudo_ground_truth_{name}.png"
        fn(out)
        print(f"  wrote {out}")
        if name in THETA_1D_EXTRA_FOR:
            extra = OUT_DIR / f"pseudo_ground_truth_{name}_theta.png"
            _plot_theta_1d_marginal(name, for_scenario(name, seed=0), extra)
            print(f"  wrote {extra}")


if __name__ == "__main__":
    main()
