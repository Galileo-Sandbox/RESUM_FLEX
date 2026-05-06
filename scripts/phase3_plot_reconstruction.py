"""Phase 3 visual evidence: CNP-reconstructed β vs analytical p, S1..S8.

Trains a CNP per scenario (a few seconds each on CPU) and produces
``viz_output/cnp_reconstruction_S{1..8}.png`` following the comparison
rule:

* 1-D scenarios (S5, S7) → :func:`plot_comparison_1d` overlays analytical
  ``p`` and predicted ``β`` on the same axes, with the binary ``X``
  scatter (S5) or per-trial empirical rates (S7) underneath.
* 2-D scenarios (S1, S6, S8) → :func:`plot_comparison_2d` side-by-side
  heatmaps with a shared colorbar.
* 3-D / 4-D scenarios (S2, S3, S4) → side-by-side heatmaps along the
  same slice axes used by Phase 1, so the eye can compare directly.

Each scenario gets its own training budget (matching the recovery
gates in ``tests/test_cnp_recovery.py``); reproducibility comes from
seeding torch and numpy at the start of every plotter.

Run from the repo root::

    python scripts/phase3_plot_reconstruction.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from core import build_cnp, train_cnp  # noqa: E402
from data.pseudo_generator import PseudoDataGenerator, for_scenario  # noqa: E402
from schemas.config import (  # noqa: E402
    CNPConfig,
    EncoderConfig,
    TrainingConfig,
)
from schemas.data_models import InputMode, StandardBatch  # noqa: E402
from viz import plot_comparison_1d, plot_comparison_2d  # noqa: E402

OUT_DIR = Path("viz_output")
N_GRID = 80
N_CTX = 64
BUDGET = {
    "S1": 500, "S2": 800, "S3": 800, "S4": 1000,
    "S5": 600, "S6": 600, "S7": 600, "S8": 600,
}


# ---------------------------------------------------------------------------
# Common helpers.
# ---------------------------------------------------------------------------


def _enc_cfg() -> EncoderConfig:
    return EncoderConfig(
        type="mlp", latent_dim=32, hidden_dims=[64, 64], dropout=0.0
    )


def _cnp_cfg() -> CNPConfig:
    return CNPConfig(
        n_context_min=32, n_context_max=96,
        output_activation="sigmoid", mixup_alpha=0.1,
    )


def _train_cfg(name: str) -> TrainingConfig:
    return TrainingConfig(
        n_steps=BUDGET[name],
        learning_rate=1.0e-3,
        batch_size=16,
        n_events_per_trial=128,
        n_mc_samples=4,
        eval_every=0,
        seed=0,
    )


def _train(name: str):
    """Build and train a CNP for one scenario; return (gen, cnp)."""
    torch.manual_seed(0)
    np.random.seed(0)
    gen = for_scenario(name, seed=0)
    cnp = build_cnp(_enc_cfg(), gen.dim_theta, gen.dim_phi)
    train_cnp(cnp, gen, cnp_config=_cnp_cfg(), training_config=_train_cfg(name))
    cnp.eval()
    return gen, cnp


def _grid_1d(n: int = N_GRID) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n)


def _grid_2d(n: int = N_GRID) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a0 = _grid_1d(n)
    a1 = _grid_1d(n)
    G0, G1 = np.meshgrid(a0, a1, indexing="ij")
    mesh = np.stack([G0, G1], axis=-1)  # (n, n, 2)
    return a0, a1, mesh


def _build_context_FULL(
    gen, theta: np.ndarray, n_ctx: int = N_CTX, seed: int = 7,
) -> StandardBatch:
    """Generate a context batch where each trial uses one of the supplied θ.

    ``theta`` has shape ``[B, D_θ]``; sampling drives X via the truth.
    """
    rng = np.random.default_rng(seed)
    B = theta.shape[0]
    D_phi = gen.dim_phi
    phi = rng.uniform(-1.0, 1.0, size=(B, n_ctx, D_phi))
    theta_per_event = theta[:, None, :]  # [B, 1, D_θ]
    p_ctx = gen.truth.evaluate(theta=theta_per_event, phi=phi)
    labels = (rng.uniform(size=(B, n_ctx)) < p_ctx).astype(np.int8)
    return StandardBatch(mode=InputMode.FULL, theta=theta, phi=phi, labels=labels)


def _build_context_EVENT_ONLY(
    gen, B: int, n_ctx: int = N_CTX, seed: int = 7,
) -> StandardBatch:
    rng = np.random.default_rng(seed)
    D_phi = gen.dim_phi
    phi = rng.uniform(-1.0, 1.0, size=(B, n_ctx, D_phi))
    p_ctx = gen.truth.evaluate(theta=None, phi=phi)
    labels = (rng.uniform(size=(B, n_ctx)) < p_ctx).astype(np.int8)
    return StandardBatch(mode=InputMode.EVENT_ONLY, phi=phi, labels=labels)


def _build_context_DESIGN_ONLY(
    gen, theta: np.ndarray, n_ctx: int = N_CTX, seed: int = 7,
) -> StandardBatch:
    """One trial per supplied θ; events all share that trial's p."""
    rng = np.random.default_rng(seed)
    B = theta.shape[0]
    p_per_trial = gen.truth.evaluate(theta=theta, phi=None)  # [B]
    p_broadcast = np.broadcast_to(p_per_trial[:, None], (B, n_ctx))
    labels = (rng.uniform(size=(B, n_ctx)) < p_broadcast).astype(np.int8)
    return StandardBatch(mode=InputMode.DESIGN_ONLY, theta=theta, labels=labels)


def _predict_at(cnp, ctx, target) -> np.ndarray:
    with torch.no_grad():
        beta = cnp.predict_beta(ctx, target).cpu().numpy()
    return beta


# ---------------------------------------------------------------------------
# 1-D scenarios: S5 (EVENT_ONLY 1D φ), S7 (DESIGN_ONLY 1D θ).
# ---------------------------------------------------------------------------


def plot_S5(out: Path) -> None:
    gen, cnp = _train("S5")
    grid = _grid_1d()
    p = gen.truth.evaluate(theta=None, phi=grid[:, None])

    # Single trial (B=1), context from the same generator, target along grid.
    ctx = _build_context_EVENT_ONLY(gen, B=1, n_ctx=N_CTX)
    tgt_phi = grid[None, :, None]              # [1, G, 1]
    tgt_labels = np.zeros((1, N_GRID), dtype=np.int8)
    tgt = StandardBatch(mode=InputMode.EVENT_ONLY, phi=tgt_phi, labels=tgt_labels)

    beta = _predict_at(cnp, ctx, tgt)[0]        # [G]

    overlay = (ctx.phi[0, :, 0], ctx.labels[0].astype(float))
    plot_comparison_1d(
        x=grid, analytical=p, predicted=beta, out_path=out,
        title="S5 — EVENT_ONLY, dim(φ)=1: analytical p vs predicted β",
        xlabel="φ", overlay_xy=overlay, overlay_label="X (binary, context)",
    )


def plot_S7(out: Path) -> None:
    gen, cnp = _train("S7")
    grid = _grid_1d()
    p = gen.truth.evaluate(theta=grid[:, None], phi=None)        # [G]

    # B=G trials, one per θ in grid; per-trial context with N_CTX events.
    theta = grid[:, None]
    ctx = _build_context_DESIGN_ONLY(gen, theta=theta, n_ctx=N_CTX)
    # Target: B=G trials with N=1 dummy event each (DESIGN_ONLY ⇒ same p across events).
    tgt_labels = np.zeros((N_GRID, 1), dtype=np.int8)
    tgt = StandardBatch(mode=InputMode.DESIGN_ONLY, theta=theta, labels=tgt_labels)
    beta = _predict_at(cnp, ctx, tgt)[:, 0]   # [G]

    rate = ctx.labels.mean(axis=1)
    overlay = (theta[:, 0], rate)
    plot_comparison_1d(
        x=grid, analytical=p, predicted=beta, out_path=out,
        title=f"S7 — DESIGN_ONLY, dim(θ)=1: analytical p vs predicted β (per-trial overlay, N={N_CTX})",
        xlabel="θ", overlay_xy=overlay,
        overlay_label=f"empirical rate per trial (N={N_CTX})",
    )


# ---------------------------------------------------------------------------
# 2-D scenarios: S1 (FULL 1D×1D), S6 (EVENT_ONLY 2D φ), S8 (DESIGN_ONLY 2D θ).
# ---------------------------------------------------------------------------


def plot_S1(out: Path) -> None:
    gen, cnp = _train("S1")
    a_theta = _grid_1d()
    a_phi = _grid_1d()
    G = N_GRID

    # B=G trials, one per θ; target spans the full φ grid for each trial.
    theta = a_theta[:, None]
    ctx = _build_context_FULL(gen, theta=theta, n_ctx=N_CTX)
    tgt_phi = np.tile(a_phi[None, :, None], (G, 1, 1))   # [G, G, 1]
    tgt_labels = np.zeros((G, G), dtype=np.int8)
    tgt = StandardBatch(mode=InputMode.FULL, theta=theta, phi=tgt_phi, labels=tgt_labels)
    beta = _predict_at(cnp, ctx, tgt)                    # [G, G]

    Theta, Phi = np.meshgrid(a_theta, a_phi, indexing="ij")
    p_grid = gen.truth.evaluate(
        theta=Theta[..., None], phi=Phi[..., None]
    )

    plot_comparison_2d(
        analytical=p_grid, predicted=beta,
        axis_grids=[a_theta, a_phi], out_path=out,
        title="S1 — FULL, dim(θ)=1, dim(φ)=1",
        axis_labels=["θ", "φ"], value_label="p / β",
        vmin=0.0, vmax=gen.truth.t_max,
    )


def plot_S6(out: Path) -> None:
    gen, cnp = _train("S6")
    a0, a1, mesh = _grid_2d()
    p_grid = gen.truth.evaluate(theta=None, phi=mesh)    # [G, G]

    # Single trial; target the entire 2-D grid in one go.
    G = N_GRID
    ctx = _build_context_EVENT_ONLY(gen, B=1, n_ctx=N_CTX)
    tgt_phi = mesh.reshape(1, G * G, 2)
    tgt_labels = np.zeros((1, G * G), dtype=np.int8)
    tgt = StandardBatch(mode=InputMode.EVENT_ONLY, phi=tgt_phi, labels=tgt_labels)
    beta_flat = _predict_at(cnp, ctx, tgt)[0]            # [G*G]
    beta = beta_flat.reshape(G, G)

    plot_comparison_2d(
        analytical=p_grid, predicted=beta,
        axis_grids=[a0, a1], out_path=out,
        title="S6 — EVENT_ONLY, dim(φ)=2",
        axis_labels=["φ₁", "φ₂"], value_label="p / β",
        vmin=0.0, vmax=gen.truth.t_max,
    )


def plot_S8(out: Path) -> None:
    gen, cnp = _train("S8")
    a0, a1, mesh = _grid_2d()
    p_grid = gen.truth.evaluate(theta=mesh, phi=None)    # [G, G]

    # B = G*G trials, each with its own θ from the grid.
    G = N_GRID
    theta_flat = mesh.reshape(G * G, 2)
    ctx = _build_context_DESIGN_ONLY(gen, theta=theta_flat, n_ctx=N_CTX)
    tgt_labels = np.zeros((G * G, 1), dtype=np.int8)
    tgt = StandardBatch(mode=InputMode.DESIGN_ONLY, theta=theta_flat, labels=tgt_labels)
    beta_flat = _predict_at(cnp, ctx, tgt)[:, 0]
    beta = beta_flat.reshape(G, G)

    plot_comparison_2d(
        analytical=p_grid, predicted=beta,
        axis_grids=[a0, a1], out_path=out,
        title="S8 — DESIGN_ONLY, dim(θ)=2",
        axis_labels=["θ₁", "θ₂"], value_label="p / β",
        vmin=0.0, vmax=gen.truth.t_max,
    )


# ---------------------------------------------------------------------------
# 3-D / 4-D scenarios: S2, S3, S4 — slice along the Phase-1 axes.
# ---------------------------------------------------------------------------


def plot_S2(out: Path) -> None:
    """FULL 2D θ × 1D φ. Slice φ at φ_peak; heatmap over (θ₁, θ₂)."""
    gen, cnp = _train("S2")
    a0, a1, mesh = _grid_2d()                            # θ-grid
    G = N_GRID
    phi_slice = np.broadcast_to(gen.truth.phi_peak, (G, G, 1))
    p_grid = gen.truth.evaluate(theta=mesh, phi=phi_slice)

    theta_flat = mesh.reshape(G * G, 2)
    ctx = _build_context_FULL(gen, theta=theta_flat, n_ctx=N_CTX)
    tgt_phi = np.broadcast_to(gen.truth.phi_peak, (G * G, 1, 1)).copy()
    tgt_labels = np.zeros((G * G, 1), dtype=np.int8)
    tgt = StandardBatch(mode=InputMode.FULL, theta=theta_flat, phi=tgt_phi, labels=tgt_labels)
    beta_flat = _predict_at(cnp, ctx, tgt)[:, 0]
    beta = beta_flat.reshape(G, G)

    plot_comparison_2d(
        analytical=p_grid, predicted=beta,
        axis_grids=[a0, a1], out_path=out,
        title=f"S2 — FULL dim(θ)=2, dim(φ)=1  (φ sliced at φ_peak={gen.truth.phi_peak[0]:.2f})",
        axis_labels=["θ₁", "θ₂"], value_label="p / β",
        vmin=0.0, vmax=gen.truth.t_max,
    )


def plot_S3(out: Path) -> None:
    """FULL 1D θ × 2D φ. Slice θ at θ_peak; heatmap over (φ₁, φ₂)."""
    gen, cnp = _train("S3")
    a0, a1, mesh = _grid_2d()                            # φ-grid
    G = N_GRID
    theta_slice = np.broadcast_to(gen.truth.theta_peak, (G, G, 1))
    p_grid = gen.truth.evaluate(theta=theta_slice, phi=mesh)

    # B=1 trial at θ_peak, target the φ grid.
    theta_one = gen.truth.theta_peak[None, :]            # [1, 1]
    ctx = _build_context_FULL(gen, theta=theta_one, n_ctx=N_CTX)
    tgt_phi = mesh.reshape(1, G * G, 2)
    tgt_labels = np.zeros((1, G * G), dtype=np.int8)
    tgt = StandardBatch(
        mode=InputMode.FULL, theta=theta_one, phi=tgt_phi, labels=tgt_labels,
    )
    beta_flat = _predict_at(cnp, ctx, tgt)[0]
    beta = beta_flat.reshape(G, G)

    plot_comparison_2d(
        analytical=p_grid, predicted=beta,
        axis_grids=[a0, a1], out_path=out,
        title=f"S3 — FULL dim(θ)=1, dim(φ)=2  (θ sliced at θ_peak={gen.truth.theta_peak[0]:.2f})",
        axis_labels=["φ₁", "φ₂"], value_label="p / β",
        vmin=0.0, vmax=gen.truth.t_max,
    )


def plot_S4(out: Path) -> None:
    """FULL 2D θ × 2D φ. Slice φ at φ_peak; heatmap over (θ₁, θ₂)."""
    gen, cnp = _train("S4")
    a0, a1, mesh = _grid_2d()                            # θ-grid
    G = N_GRID
    phi_slice = np.broadcast_to(gen.truth.phi_peak, (G, G, 2))
    p_grid = gen.truth.evaluate(theta=mesh, phi=phi_slice)

    theta_flat = mesh.reshape(G * G, 2)
    ctx = _build_context_FULL(gen, theta=theta_flat, n_ctx=N_CTX)
    tgt_phi = np.broadcast_to(gen.truth.phi_peak, (G * G, 1, 2)).copy()
    tgt_labels = np.zeros((G * G, 1), dtype=np.int8)
    tgt = StandardBatch(mode=InputMode.FULL, theta=theta_flat, phi=tgt_phi, labels=tgt_labels)
    beta_flat = _predict_at(cnp, ctx, tgt)[:, 0]
    beta = beta_flat.reshape(G, G)

    plot_comparison_2d(
        analytical=p_grid, predicted=beta,
        axis_grids=[a0, a1], out_path=out,
        title="S4 — FULL dim(θ)=2, dim(φ)=2  (φ sliced at φ_peak)",
        axis_labels=["θ₁", "θ₂"], value_label="p / β",
        vmin=0.0, vmax=gen.truth.t_max,
    )


PLOTTERS: dict[str, Callable[[Path], None]] = {
    "S1": plot_S1, "S2": plot_S2, "S3": plot_S3, "S4": plot_S4,
    "S5": plot_S5, "S6": plot_S6, "S7": plot_S7, "S8": plot_S8,
}


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for name, fn in PLOTTERS.items():
        out = OUT_DIR / f"cnp_reconstruction_{name}.png"
        print(f"  training & plotting {name} ...")
        fn(out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
