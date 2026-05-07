"""Phase 5 visual evidence: IVR acquisition surfaces and σ shrinkage.

For each θ-bearing scenario (S1, S2, S3, S4, S7, S8):

  1. Train a CNP on synthetic data (Phase 3 budgets).
  2. Build LF / HF datasets and fit a 3-fidelity MFGP.
  3. Run a 5-step IVR active-learning loop.
  4. Emit one figure per step:
     - 1-D θ: σ(θ) + acquisition score over a fine grid.
     - 2-D θ: σ heatmap + acquisition heatmap (shared aspect), with
       sampled θ overlaid as dots and θ_next as a red star.
  5. Emit one summary figure per scenario tracking integrated variance
     across iterations (the canonical "band shrinks" sanity check).

Outputs land in ``viz_output/phase5_optimizer/``:

  * ``optimizer_{S}_step{n}.png`` — per-step panels, n = 1 .. N_STEPS.
  * ``optimizer_{S}_iv.png``      — integrated-variance trace per scenario.

Run from the repo root:

    python scripts/phase5_plot_optimizer.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from core import (  # noqa: E402
    ActiveLearningLoop,
    ActiveLearningStep,
    BoxBounds,
    build_cnp,
    fit_mfgp_three_fidelity,
    integrated_variance,
    prepare_mfgp_datasets,
    train_cnp,
)
from data import for_scenario  # noqa: E402
from data.pseudo_generator import PseudoDataGenerator  # noqa: E402
from schemas.config import (  # noqa: E402
    CNPConfig,
    EncoderConfig,
    TrainingConfig,
)

OUT_DIR = Path("viz_output/phase5_optimizer")

OPTIMIZER_SCENARIOS = ["S1", "S2", "S3", "S4", "S7", "S8"]
CNP_STEPS = {"S1": 500, "S2": 800, "S3": 800, "S4": 1000, "S7": 600, "S8": 600}

N_LF_TRIALS, N_LF_EVENTS = 200, 64
N_HF_TRIALS, N_HF_EVENTS = 12, 128
N_AL_STEPS = 5
N_MC_SAMPLES = 800
N_CAND_PER_AXIS = 50
REFIT_N_RESTARTS = 5


# ---------------------------------------------------------------------------
# Common config helpers (mirror Phase 4).
# ---------------------------------------------------------------------------


def _enc_cfg() -> EncoderConfig:
    return EncoderConfig(type="mlp", latent_dim=32, hidden_dims=[64, 64], dropout=0.0)


def _cnp_cfg() -> CNPConfig:
    return CNPConfig(
        n_context_min=32, n_context_max=96,
        output_activation="sigmoid", mixup_alpha=0.1,
    )


def _train_cfg(name: str) -> TrainingConfig:
    return TrainingConfig(
        n_steps=CNP_STEPS[name],
        learning_rate=1.0e-3, batch_size=16,
        n_events_per_trial=128, n_mc_samples=4,
        eval_every=0, seed=0,
    )


# ---------------------------------------------------------------------------
# Pipeline up to the active-learning loop.
# ---------------------------------------------------------------------------


def _train_pipeline(name: str, n_hf_trials: int = N_HF_TRIALS):
    """CNP + initial 3-fidelity MFGP, ready for active learning."""
    torch.manual_seed(0)
    np.random.seed(0)
    gen = for_scenario(name, seed=0)
    cnp = build_cnp(_enc_cfg(), gen.dim_theta, gen.dim_phi)
    train_cnp(cnp, gen, cnp_config=_cnp_cfg(), training_config=_train_cfg(name))

    data = prepare_mfgp_datasets(
        cnp, gen,
        n_lf_trials=N_LF_TRIALS, n_lf_events=N_LF_EVENTS,
        n_hf_trials=n_hf_trials, n_hf_events=N_HF_EVENTS,
        seed=0,
    )
    mfgp = fit_mfgp_three_fidelity(data, n_restarts=REFIT_N_RESTARTS)
    return gen, cnp, mfgp, data


# ---------------------------------------------------------------------------
# 1-D and 2-D step panels.
# ---------------------------------------------------------------------------


def _truth_curve_1d(gen: PseudoDataGenerator, theta_grid: np.ndarray) -> np.ndarray:
    """Analytical t̄(θ) on a 1-D grid (FULL or DESIGN_ONLY)."""
    if gen.dim_phi is None:
        # DESIGN_ONLY — t̄(θ) ≡ t(θ).
        return gen.truth.evaluate(theta=theta_grid)
    # FULL — marginalize over a uniform φ ~ U(-1, 1)^D_φ.
    rng = np.random.default_rng(0)
    n_phi = 2000
    phi = rng.uniform(-1.0, 1.0, size=(n_phi, gen.dim_phi))
    p = gen.truth.evaluate(
        theta=theta_grid[:, None, :], phi=phi[None, :, :],
    )                                       # [G, n_phi]
    return p.mean(axis=1)


def _truth_grid_2d(
    gen: PseudoDataGenerator, ax0: np.ndarray, ax1: np.ndarray,
) -> np.ndarray:
    """Analytical t̄(θ_0, θ_1) on a 2-D θ grid (FULL or DESIGN_ONLY).

    Returned shape ``(len(ax0), len(ax1))`` — first axis is θ_0 (rows),
    matching the convention used by ``BoxBounds.grid_2d`` and
    :class:`ActiveLearningStep.acquisition`.
    """
    g0, g1 = np.meshgrid(ax0, ax1, indexing="ij")
    flat = np.stack([g0.ravel(), g1.ravel()], axis=1)         # [G, 2]
    if gen.dim_phi is None:
        # DESIGN_ONLY — t̄(θ) ≡ t(θ).
        return gen.truth.evaluate(theta=flat).reshape(g0.shape)
    rng = np.random.default_rng(0)
    n_phi = 1500
    phi = rng.uniform(-1.0, 1.0, size=(n_phi, gen.dim_phi))   # [n_phi, D_φ]
    p = gen.truth.evaluate(
        theta=flat[:, None, :], phi=phi[None, :, :],
    )                                                         # [G, n_phi]
    return p.mean(axis=1).reshape(g0.shape)


def _arg_target(values: np.ndarray, target: str) -> np.intp:
    """``np.argmax`` or ``np.argmin`` of a 1-D array, dispatched on ``target``."""
    if target == "max":
        return np.argmax(values)
    if target == "min":
        return np.argmin(values)
    raise ValueError(f"target must be 'max' or 'min', got {target!r}")


def _plot_step_1d(
    gen: PseudoDataGenerator,
    record: ActiveLearningStep,
    out_path: Path,
    title: str,
    *,
    target: str,
    mu_after: np.ndarray,
    acquisition: str = "ivr",
) -> None:
    grid = record.grid_axes[0]
    sigma = record.sigma
    acq = record.acquisition
    theta_grid_2d = grid[:, None]
    truth = _truth_curve_1d(gen, theta_grid_2d)

    # True / predicted optima of the response (max or min, dispatched).
    theta_true_opt = float(grid[_arg_target(truth, target)])
    theta_pred_opt = float(grid[_arg_target(mu_after, target)])
    target_label = "argmax" if target == "max" else "argmin"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    ax_s, ax_a = axes

    sampled = record.sampled_theta[:, 0]
    # All but the last are prior samples; the last is the just-acquired one.
    prior = sampled[:-1]
    chosen = record.theta_next[0]

    # Twin-axis layout: σ(θ) on the LEFT (primary, the thing we care
    # about for the active-learning question), analytical t̄(θ) on the
    # RIGHT (context only). Without this, σ ~0.03 vs t̄ ~0.2 makes the
    # σ curve visually flat against the larger truth signal.
    sigma_line, = ax_s.plot(
        grid, sigma, color="C2", linewidth=2.2, label="GP σ(θ)",
    )
    sigma_pad = max(0.05 * sigma.max(), 1e-6)
    ax_s.set_ylim(max(0.0, sigma.min() - sigma_pad), sigma.max() + sigma_pad)
    ax_s.set_xlabel("θ")
    ax_s.set_ylabel("posterior σ(θ)", color="C2")
    ax_s.tick_params(axis="y", labelcolor="C2")
    ax_s.set_title("posterior σ(θ)  +  analytical t̄(θ) for context")
    ax_s.grid(alpha=0.3)
    true_line = ax_s.axvline(
        theta_true_opt, color="C0", linestyle=":", linewidth=1.6,
        label=f"true {target_label}",
    )
    pred_line = ax_s.axvline(
        theta_pred_opt, color="C3", linestyle="-.", linewidth=1.4,
        label=f"predicted {target_label}",
    )

    ax_t = ax_s.twinx()
    truth_line, = ax_t.plot(
        grid, truth, color="C0", linewidth=1.4, alpha=0.55,
        linestyle="--", label="analytical t̄(θ)",
    )
    ax_t.set_ylabel("analytical t̄(θ)", color="C0")
    ax_t.tick_params(axis="y", labelcolor="C0")
    ax_t.set_ylim(0.0, max(truth.max() * 1.05, 1e-6))

    # Sampled-θ markers anchored to the σ axis at its bottom edge so
    # their position stays meaningful even after the y-limits change.
    y_marker = ax_s.get_ylim()[0]
    sampled_dots = ax_s.scatter(
        prior, np.full_like(prior, y_marker),
        color="black", s=36, zorder=4, label="sampled θ",
    )
    chosen_star = ax_s.scatter(
        [chosen], [y_marker],
        color="red", s=200, marker="*", zorder=5,
        edgecolor="black", linewidths=0.6, label="θ_next",
    )
    ax_s.legend(
        handles=[sigma_line, truth_line, true_line, pred_line, sampled_dots, chosen_star],
        loc="best", fontsize=7,
    )

    acq_label = {"ivr": "IVR acquisition", "ei": "EI acquisition"}.get(
        acquisition, "acquisition"
    )
    ax_a.plot(grid, acq, color="C3", linewidth=2.0, label=acq_label)
    ax_a.axvline(theta_true_opt, color="C0", linestyle=":", linewidth=1.6,
                 label=f"true {target_label}")
    ax_a.axvline(theta_pred_opt, color="C3", linestyle="-.", linewidth=1.4,
                 label=f"predicted {target_label}")
    ax_a.scatter(
        prior, np.zeros_like(prior),
        color="black", s=36, zorder=4, label="sampled θ",
    )
    ax_a.scatter(
        [chosen], [acq.max()],
        color="red", s=180, marker="*", zorder=5, label="θ_next",
    )
    ax_a.set_xlabel("θ"); ax_a.set_ylabel("acquisition")
    ax_a.set_title(f"{acq_label} surface")
    ax_a.legend(loc="best", fontsize=7)
    ax_a.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_step_2d(
    gen: PseudoDataGenerator,
    record: ActiveLearningStep,
    out_path: Path,
    title: str,
    *,
    target: str,
    mu_after: np.ndarray,
    acquisition: str = "ivr",
) -> None:
    ax0, ax1 = record.grid_axes
    sigma = record.sigma
    acq = record.acquisition
    extent = (ax1[0], ax1[-1], ax0[0], ax0[-1])

    sampled = record.sampled_theta
    prior = sampled[:-1]
    chosen = record.theta_next

    # True / predicted optima from t̄ and the post-step μ.
    truth_grid = _truth_grid_2d(gen, ax0, ax1)
    flat_truth = truth_grid.ravel()
    flat_mu = mu_after.ravel()
    g0_grid, g1_grid = np.meshgrid(ax0, ax1, indexing="ij")
    flat_pts = np.stack([g0_grid.ravel(), g1_grid.ravel()], axis=1)
    theta_true_opt = flat_pts[_arg_target(flat_truth, target)]
    theta_pred_opt = flat_pts[_arg_target(flat_mu, target)]
    target_label = "argmax" if target == "max" else "argmin"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, arr, panel_title, cmap in [
        (axes[0], sigma, "posterior σ(θ)", "viridis"),
        (axes[1], acq, {"ivr": "IVR acquisition", "ei": "EI acquisition"}.get(acquisition, "acquisition"), "magma"),
    ]:
        im = ax.imshow(arr, origin="lower", extent=extent, aspect="auto", cmap=cmap)
        ax.scatter(
            prior[:, 1], prior[:, 0],
            color="white", edgecolor="black", s=42, zorder=4, label="sampled θ",
        )
        ax.scatter(
            [chosen[1]], [chosen[0]],
            color="red", marker="*", s=240, edgecolor="black", zorder=5,
            label="θ_next",
        )
        # True optimum (cyan X) + predicted optimum (red diamond).
        ax.scatter(
            [theta_true_opt[1]], [theta_true_opt[0]],
            color="cyan", marker="X", s=180, edgecolor="black", linewidths=1.0,
            zorder=6, label=f"true {target_label}",
        )
        ax.scatter(
            [theta_pred_opt[1]], [theta_pred_opt[0]],
            color="red", marker="D", s=110, edgecolor="black", linewidths=0.8,
            zorder=6, label=f"predicted {target_label}",
        )
        ax.set_title(panel_title)
        ax.set_xlabel("θ_1")
        fig.colorbar(im, ax=ax, shrink=0.85)
    axes[0].set_ylabel("θ_0")
    axes[0].legend(loc="upper right", fontsize=7)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_iv_trace(
    name: str,
    iv0: float,
    records: list[ActiveLearningStep],
    out_path: Path,
) -> None:
    """Integrated-variance trace across active-learning steps."""
    steps = list(range(len(records) + 1))
    iv = [iv0] + [r.integrated_variance_after for r in records]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, iv, marker="o", linewidth=2, color="C2")
    ax.fill_between(steps, 0, iv, alpha=0.18, color="C2")
    for s, v in zip(steps, iv, strict=True):
        ax.annotate(f"{v:.2e}", (s, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_title(f"{name} — integrated posterior variance ∫σ²(θ)dθ across AL steps")
    ax.set_xlabel("active-learning step"); ax.set_ylabel("integrated σ²")
    ax.set_xticks(steps)
    ax.set_ylim(0, max(iv) * 1.15)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Final trajectory plot + per-step optimization metrics.
# ---------------------------------------------------------------------------


def _plot_trajectory_1d(
    name: str,
    gen: PseudoDataGenerator,
    bounds: BoxBounds,
    initial_hf_theta: np.ndarray,
    al_thetas: np.ndarray,
    final_mu: np.ndarray,
    final_sigma: np.ndarray,
    grid: np.ndarray,
    truth: np.ndarray,
    target: str,
    out_path: Path,
) -> None:
    """Search-history view: t̄(θ), μ(θ) ± σ, sampled trajectory, target marker."""
    target_label = "argmax" if target == "max" else "argmin"
    theta_true_opt = float(grid[_arg_target(truth, target)])
    theta_pred_opt = float(grid[_arg_target(final_mu, target)])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grid, truth, color="C0", linewidth=2.4, label="analytical t̄(θ)")
    ax.plot(grid, final_mu, color="C3", linewidth=2.0, linestyle="--",
            label=f"MFGP μ(θ) after {len(al_thetas)} AL steps")
    ax.fill_between(
        grid, final_mu - final_sigma, final_mu + final_sigma,
        color="C3", alpha=0.18, label="μ ± σ",
    )

    y_marker = ax.get_ylim()[0]
    init_dots = ax.scatter(
        initial_hf_theta[:, 0], np.full(initial_hf_theta.shape[0], y_marker),
        color="gray", s=28, zorder=4, alpha=0.8, label="initial HF θ",
    )
    # Numbered AL stars in the order they were chosen. When several
    # stars sit at (nearly) the same θ — common when IVR keeps probing
    # the boundary — cascade the labels vertically so all step numbers
    # remain visible.
    label_collision_tol = 0.04 * (bounds.high[0] - bounds.low[0])
    label_stack: dict[float, int] = {}
    for k, theta in enumerate(al_thetas, start=1):
        ax.scatter(
            [theta[0]], [y_marker],
            color="red", marker="*", s=240, edgecolor="black",
            linewidths=0.6, zorder=5,
        )
        rounded = float(np.round(theta[0] / label_collision_tol) * label_collision_tol)
        stack_idx = label_stack.get(rounded, 0)
        label_stack[rounded] = stack_idx + 1
        ax.annotate(
            str(k), (theta[0], y_marker),
            textcoords="offset points",
            xytext=(0, 10 + 14 * stack_idx),
            ha="center", fontsize=9, fontweight="bold", color="red",
        )

    true_line = ax.axvline(
        theta_true_opt, color="C0", linestyle=":", linewidth=2.0,
        label=f"true {target_label} (θ={theta_true_opt:+.3f})",
    )
    pred_line = ax.axvline(
        theta_pred_opt, color="C3", linestyle="-.", linewidth=1.6,
        label=f"predicted {target_label} (θ={theta_pred_opt:+.3f})",
    )

    ax.set_xlim(bounds.low[0], bounds.high[0])
    ax.set_xlabel("θ")
    ax.set_ylabel("response")
    gap = abs(theta_true_opt - theta_pred_opt)
    mae = float(np.abs(final_mu - truth).mean())
    ax.set_title(
        f"{name} — search history (target = {target}, "
        f"gap={gap:.3f}, MAE={mae:.3e})"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_trajectory_2d(
    name: str,
    gen: PseudoDataGenerator,
    initial_hf_theta: np.ndarray,
    al_thetas: np.ndarray,
    final_mu: np.ndarray,
    truth_grid: np.ndarray,
    grid_axes: list[np.ndarray],
    target: str,
    out_path: Path,
) -> None:
    """Side-by-side heatmaps t̄(θ) | μ(θ) with the search history overlaid."""
    target_label = "argmax" if target == "max" else "argmin"
    ax0, ax1 = grid_axes
    extent = (ax1[0], ax1[-1], ax0[0], ax0[-1])

    flat_truth = truth_grid.ravel()
    flat_mu = final_mu.ravel()
    g0_grid, g1_grid = np.meshgrid(ax0, ax1, indexing="ij")
    flat_pts = np.stack([g0_grid.ravel(), g1_grid.ravel()], axis=1)
    theta_true_opt = flat_pts[_arg_target(flat_truth, target)]
    theta_pred_opt = flat_pts[_arg_target(flat_mu, target)]
    gap = float(np.linalg.norm(theta_true_opt - theta_pred_opt))
    mae = float(np.abs(final_mu - truth_grid).mean())

    vmin = float(min(truth_grid.min(), final_mu.min()))
    vmax = float(max(truth_grid.max(), final_mu.max()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, arr, panel_title in [
        (axes[0], truth_grid, "analytical t̄(θ)"),
        (axes[1], final_mu, f"MFGP μ(θ) after {len(al_thetas)} AL steps"),
    ]:
        im = ax.imshow(
            arr, origin="lower", extent=extent, aspect="auto",
            cmap="viridis", vmin=vmin, vmax=vmax,
        )
        ax.scatter(
            initial_hf_theta[:, 1], initial_hf_theta[:, 0],
            color="lightgray", edgecolor="black", s=34, zorder=3,
            label="initial HF θ",
        )
        label_stack_2d: dict[tuple[float, float], int] = {}
        tol_0 = 0.04 * (ax0[-1] - ax0[0])
        tol_1 = 0.04 * (ax1[-1] - ax1[0])
        for k, theta in enumerate(al_thetas, start=1):
            ax.scatter(
                [theta[1]], [theta[0]],
                color="red", marker="*", s=260, edgecolor="black",
                linewidths=0.6, zorder=4,
            )
            key = (
                float(np.round(theta[0] / tol_0) * tol_0),
                float(np.round(theta[1] / tol_1) * tol_1),
            )
            stack_idx = label_stack_2d.get(key, 0)
            label_stack_2d[key] = stack_idx + 1
            ax.annotate(
                str(k), (theta[1], theta[0]),
                textcoords="offset points",
                xytext=(8, 6 + 12 * stack_idx),
                color="red", fontsize=10, fontweight="bold",
            )
        ax.scatter(
            [theta_true_opt[1]], [theta_true_opt[0]],
            color="cyan", marker="X", s=220, edgecolor="black",
            linewidths=1.0, zorder=5, label=f"true {target_label}",
        )
        ax.scatter(
            [theta_pred_opt[1]], [theta_pred_opt[0]],
            color="orange", marker="D", s=130, edgecolor="black",
            linewidths=0.8, zorder=5, label=f"predicted {target_label}",
        )
        ax.set_title(panel_title)
        ax.set_xlabel("θ_1")
        fig.colorbar(im, ax=ax, shrink=0.85)
    axes[0].set_ylabel("θ_0")
    axes[0].legend(loc="upper left", fontsize=7)

    fig.suptitle(
        f"{name} — search history (target = {target}, "
        f"gap={gap:.3f}, MAE={mae:.3e})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_metrics_trajectory(
    name: str,
    gap_history: list[float],
    mae_history: list[float],
    out_path: Path,
) -> None:
    """Per-step trajectory of optimization gap and surrogate MAE.

    Step 0 is the state *before* any AL points are added (initial fit),
    step k is the state after the k-th AL refit.
    """
    steps = list(range(len(gap_history)))

    fig, (ax_g, ax_m) = plt.subplots(1, 2, figsize=(11, 4))
    ax_g.plot(steps, gap_history, marker="o", color="C3", linewidth=2)
    ax_g.set_title(f"{name} — optimization gap |θ_true − θ_pred|")
    ax_g.set_xlabel("active-learning step (0 = initial fit)")
    ax_g.set_ylabel("gap")
    ax_g.set_xticks(steps); ax_g.grid(alpha=0.3)
    ax_g.set_ylim(0, max(gap_history) * 1.15 + 1e-6)
    for s, v in zip(steps, gap_history, strict=True):
        ax_g.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                      xytext=(0, 8), ha="center", fontsize=8)

    ax_m.plot(steps, mae_history, marker="o", color="C2", linewidth=2)
    ax_m.set_title(f"{name} — surrogate MAE(μ, t̄)")
    ax_m.set_xlabel("active-learning step (0 = initial fit)")
    ax_m.set_ylabel("MAE")
    ax_m.set_xticks(steps); ax_m.grid(alpha=0.3)
    ax_m.set_ylim(0, max(mae_history) * 1.15 + 1e-6)
    for s, v in zip(steps, mae_history, strict=True):
        ax_m.annotate(f"{v:.3e}", (s, v), textcoords="offset points",
                      xytext=(0, 8), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def _predict_on_grid_axes(
    mfgp, grid_axes: list[np.ndarray],
) -> np.ndarray:
    """Run ``mfgp.predict`` on the candidate grid implied by ``grid_axes``.

    Returns the mean reshaped to match the surface arrays the loop
    records (1-D for ``len(grid_axes)==1``, 2-D ``[G0, G1]`` for 2).
    """
    if len(grid_axes) == 1:
        pts = grid_axes[0][:, None]
        mu, _ = mfgp.predict(pts)
        return mu
    if len(grid_axes) == 2:
        ax0, ax1 = grid_axes
        g0, g1 = np.meshgrid(ax0, ax1, indexing="ij")
        flat = np.stack([g0.ravel(), g1.ravel()], axis=1)
        mu, _ = mfgp.predict(flat)
        return mu.reshape(g0.shape)
    raise ValueError(f"unsupported grid dimensionality {len(grid_axes)}")


def _eval_grid(
    bounds: BoxBounds, dim: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Fine grid for trajectory / metrics evaluation."""
    if dim == 1:
        ax0 = np.linspace(bounds.low[0], bounds.high[0], 200)
        return ax0[:, None], [ax0]
    if dim == 2:
        ax0 = np.linspace(bounds.low[0], bounds.high[0], 60)
        ax1 = np.linspace(bounds.low[1], bounds.high[1], 60)
        g0, g1 = np.meshgrid(ax0, ax1, indexing="ij")
        flat = np.stack([g0.ravel(), g1.ravel()], axis=1)
        return flat, [ax0, ax1]
    raise ValueError(f"unsupported dim {dim}")


def _gap_and_mae(
    truth: np.ndarray,
    mu: np.ndarray,
    eval_pts: np.ndarray,
    target: str,
    dim: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Optimization gap and surrogate MAE on the eval grid.

    ``truth`` and ``mu`` are flat 1-D arrays of length ``len(eval_pts)``.
    Returns ``(gap, mae, theta_true_opt, theta_pred_opt)``. Gap is
    Euclidean distance in θ-space.
    """
    theta_true = eval_pts[_arg_target(truth, target)]
    theta_pred = eval_pts[_arg_target(mu, target)]
    gap = float(np.linalg.norm(theta_true - theta_pred))
    mae = float(np.abs(mu - truth).mean())
    return gap, mae, theta_true, theta_pred


def _run_scenario(
    name: str,
    *,
    target: str,
    acquisition: str = "ivr",
    n_hf_trials: int = N_HF_TRIALS,
    out_dir: Path = OUT_DIR,
) -> dict[str, float]:
    print(f"  ── {name} (acq={acquisition}, target={target}, n_hf_init={n_hf_trials}) ──")
    gen, cnp, mfgp, data = _train_pipeline(name, n_hf_trials=n_hf_trials)
    bounds = BoxBounds(
        low=np.full(gen.dim_theta, -1.0),
        high=np.full(gen.dim_theta, 1.0),
    )
    iv0 = integrated_variance(mfgp, bounds, n_mc_samples=2000, seed=0)
    initial_hf_theta = data["X_hf"].copy()

    # Fine grid + analytical t̄ for trajectory & metrics.
    eval_pts, eval_axes = _eval_grid(bounds, gen.dim_theta)
    if gen.dim_theta == 1:
        truth_flat = _truth_curve_1d(gen, eval_pts)
    else:
        truth_grid_2d = _truth_grid_2d(gen, eval_axes[0], eval_axes[1])
        truth_flat = truth_grid_2d.ravel()

    # Step 0 metrics — initial MFGP fit, no AL yet.
    mu0_flat, _ = mfgp.predict(eval_pts)
    g0, m0, _, _ = _gap_and_mae(
        truth_flat, mu0_flat, eval_pts, target, gen.dim_theta,
    )
    gap_history = [g0]
    mae_history = [m0]

    loop = ActiveLearningLoop(
        mfgp=mfgp, generator=gen, cnp=cnp, bounds=bounds, data=data,
        n_hf_events=N_HF_EVENTS, n_mc_samples=N_MC_SAMPLES,
        n_candidates_per_axis=N_CAND_PER_AXIS,
        seed=0, refit_n_restarts=REFIT_N_RESTARTS,
        acquisition=acquisition, target=target,
    )

    # Run AL one step at a time so we can grab post-step μ snapshots
    # from the *just-refit* MFGP — both on the candidate grid (for
    # the per-step plots) and on the fine eval grid (for metrics).
    records: list[ActiveLearningStep] = []
    mu_afters: list[np.ndarray] = []
    al_thetas: list[np.ndarray] = []
    for _ in range(N_AL_STEPS):
        rec = loop.step()
        records.append(rec)
        mu_afters.append(_predict_on_grid_axes(loop.mfgp, rec.grid_axes))
        al_thetas.append(rec.theta_next.copy())
        mu_eval, _ = loop.mfgp.predict(eval_pts)
        gap_k, mae_k, _, _ = _gap_and_mae(
            truth_flat, mu_eval, eval_pts, target, gen.dim_theta,
        )
        gap_history.append(gap_k)
        mae_history.append(mae_k)

    for k, (rec, mu_after) in enumerate(
        zip(records, mu_afters, strict=True), start=1,
    ):
        out = out_dir / f"optimizer_{name}_step{k}.png"
        title = (
            f"{name} — AL step {k}/{N_AL_STEPS}  "
            f"(IV {rec.integrated_variance_before:.2e} → "
            f"{rec.integrated_variance_after:.2e})"
        )
        if gen.dim_theta == 1:
            _plot_step_1d(
                gen, rec, out, title,
                target=target, mu_after=mu_after, acquisition=acquisition,
            )
        else:
            _plot_step_2d(
                gen, rec, out, title,
                target=target, mu_after=mu_after, acquisition=acquisition,
            )
        print(f"  wrote {out}")

    # Final-state trajectory + metrics plots.
    final_mu_flat, final_var_flat = loop.mfgp.predict(eval_pts)
    final_sigma_flat = np.sqrt(final_var_flat)
    al_thetas_arr = np.stack(al_thetas, axis=0)

    traj_path = out_dir / f"optimizer_{name}_trajectory.png"
    if gen.dim_theta == 1:
        _plot_trajectory_1d(
            name, gen, bounds, initial_hf_theta, al_thetas_arr,
            final_mu_flat, final_sigma_flat,
            grid=eval_axes[0], truth=truth_flat,
            target=target, out_path=traj_path,
        )
    else:
        truth_grid_2d_local = truth_flat.reshape(
            len(eval_axes[0]), len(eval_axes[1])
        )
        final_mu_grid_2d = final_mu_flat.reshape(
            len(eval_axes[0]), len(eval_axes[1])
        )
        _plot_trajectory_2d(
            name, gen, initial_hf_theta, al_thetas_arr,
            final_mu_grid_2d, truth_grid_2d_local,
            grid_axes=eval_axes, target=target, out_path=traj_path,
        )
    print(f"  wrote {traj_path}")

    metrics_path = out_dir / f"optimizer_{name}_metrics.png"
    _plot_metrics_trajectory(name, gap_history, mae_history, metrics_path)
    print(f"  wrote {metrics_path}")

    iv_path = out_dir / f"optimizer_{name}_iv.png"
    _plot_iv_trace(name, iv0, records, iv_path)
    print(f"  wrote {iv_path}")

    iv_final = records[-1].integrated_variance_after
    iv_min = min(r.integrated_variance_after for r in records)
    final_gap = gap_history[-1]
    final_mae = mae_history[-1]
    theta_true_opt = eval_pts[_arg_target(truth_flat, target)]
    theta_pred_opt = eval_pts[_arg_target(final_mu_flat, target)]
    return {
        "iv_start": iv0,
        "iv_final": iv_final,
        "iv_min": iv_min,
        "theta_true_opt": theta_true_opt,
        "theta_pred_opt": theta_pred_opt,
        "gap_initial": gap_history[0],
        "gap_final": final_gap,
        "mae_initial": mae_history[0],
        "mae_final": final_mae,
        "gap_history": gap_history,
        "mae_history": mae_history,
    }


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios", default=",".join(OPTIMIZER_SCENARIOS),
        help="Comma-separated scenario names to run (default: all six).",
    )
    parser.add_argument(
        "--target", choices=["max", "min"], default="max",
        help=(
            "Optimization target. With ``--acquisition ivr`` (default), "
            "this only controls which point is annotated as the true / "
            "predicted optimum on the plots — IVR itself is "
            "direction-agnostic. With ``--acquisition ei``, this also "
            "controls the direction of optimization in the EI formula."
        ),
    )
    parser.add_argument(
        "--acquisition", choices=["ivr", "ei"], default="ivr",
        help=(
            "Active-learning acquisition. 'ivr' (default) is pure "
            "exploration — minimizes integrated posterior variance. "
            "'ei' is Expected Improvement, exploitation-leaning — "
            "stars cluster near the predicted optimum once σ shrinks."
        ),
    )
    parser.add_argument(
        "--n-initial-hf", type=int, default=N_HF_TRIALS,
        help=(
            f"Number of HF trials in the initial dataset before AL "
            f"begins (default: {N_HF_TRIALS}). Lower values = more "
            f"uncertain initial fit = clearer AL signal, but the GP "
            f"refits become numerically less stable."
        ),
    )
    parser.add_argument(
        "--out-dir", default=str(OUT_DIR),
        help=(
            f"Directory where plots are written. Default: "
            f"{OUT_DIR}. Use a separate subfolder when running "
            f"non-default configs (e.g. EI / target=min stress test) "
            f"so the IVR baseline output is preserved."
        ),
    )
    args = parser.parse_args(argv)
    requested = [s.strip().upper() for s in args.scenarios.split(",") if s.strip()]
    unknown = [s for s in requested if s not in OPTIMIZER_SCENARIOS]
    if unknown:
        raise SystemExit(
            f"unknown scenario(s) {unknown}; valid: {OPTIMIZER_SCENARIOS}"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for name in requested:
        summary[name] = _run_scenario(
            name,
            target=args.target,
            acquisition=args.acquisition,
            n_hf_trials=args.n_initial_hf,
            out_dir=out_dir,
        )

    print("\n  IV summary (start / final / min, lower is better):")
    print("    scenario   start       final       min        min/start")
    for name, s in summary.items():
        ratio = s["iv_min"] / s["iv_start"] if s["iv_start"] > 0 else float("nan")
        print(
            f"    {name:<10} {s['iv_start']:.3e}   "
            f"{s['iv_final']:.3e}   {s['iv_min']:.3e}   {ratio:.2f}"
        )

    print(f"\n  Optimization summary (target = {args.target}):")
    print(
        f"    scenario   θ_true_opt              θ_pred_opt              "
        f"gap_init  gap_final  MAE_init     MAE_final"
    )
    for name, s in summary.items():
        true_str = np.array2string(
            s["theta_true_opt"], precision=3, separator=",",
            suppress_small=True,
        )
        pred_str = np.array2string(
            s["theta_pred_opt"], precision=3, separator=",",
            suppress_small=True,
        )
        print(
            f"    {name:<10} {true_str:<22}  {pred_str:<22}  "
            f"{s['gap_initial']:.3f}     {s['gap_final']:.3f}      "
            f"{s['mae_initial']:.3e}    {s['mae_final']:.3e}"
        )


if __name__ == "__main__":
    main()
