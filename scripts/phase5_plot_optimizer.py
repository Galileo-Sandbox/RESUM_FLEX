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


def _train_pipeline(name: str):
    """CNP + initial 3-fidelity MFGP, ready for active learning."""
    torch.manual_seed(0)
    np.random.seed(0)
    gen = for_scenario(name, seed=0)
    cnp = build_cnp(_enc_cfg(), gen.dim_theta, gen.dim_phi)
    train_cnp(cnp, gen, cnp_config=_cnp_cfg(), training_config=_train_cfg(name))

    data = prepare_mfgp_datasets(
        cnp, gen,
        n_lf_trials=N_LF_TRIALS, n_lf_events=N_LF_EVENTS,
        n_hf_trials=N_HF_TRIALS, n_hf_events=N_HF_EVENTS,
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


def _plot_step_1d(
    gen: PseudoDataGenerator,
    record: ActiveLearningStep,
    out_path: Path,
    title: str,
) -> None:
    grid = record.grid_axes[0]
    sigma = record.sigma
    acq = record.acquisition
    theta_grid_2d = grid[:, None]
    truth = _truth_curve_1d(gen, theta_grid_2d)

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
        handles=[sigma_line, truth_line, sampled_dots, chosen_star],
        loc="best", fontsize=8,
    )

    ax_a.plot(grid, acq, color="C3", linewidth=2.0, label="IVR acquisition")
    ax_a.scatter(
        prior, np.zeros_like(prior),
        color="black", s=36, zorder=4, label="sampled θ",
    )
    ax_a.scatter(
        [chosen], [acq.max()],
        color="red", s=180, marker="*", zorder=5, label="θ_next",
    )
    ax_a.set_xlabel("θ"); ax_a.set_ylabel("acquisition")
    ax_a.set_title("IVR acquisition surface")
    ax_a.legend(loc="best", fontsize=8)
    ax_a.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_step_2d(
    record: ActiveLearningStep,
    out_path: Path,
    title: str,
) -> None:
    ax0, ax1 = record.grid_axes
    sigma = record.sigma
    acq = record.acquisition
    extent = (ax1[0], ax1[-1], ax0[0], ax0[-1])

    sampled = record.sampled_theta
    prior = sampled[:-1]
    chosen = record.theta_next

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, arr, panel_title, cmap in [
        (axes[0], sigma, "posterior σ(θ)", "viridis"),
        (axes[1], acq, "IVR acquisition", "magma"),
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
        ax.set_title(panel_title)
        ax.set_xlabel("θ_1")
        fig.colorbar(im, ax=ax, shrink=0.85)
    axes[0].set_ylabel("θ_0")
    axes[0].legend(loc="upper right", fontsize=8)

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
# Driver.
# ---------------------------------------------------------------------------


def _run_scenario(name: str) -> dict[str, float]:
    print(f"  ── {name} ──")
    gen, cnp, mfgp, data = _train_pipeline(name)
    bounds = BoxBounds(
        low=np.full(gen.dim_theta, -1.0),
        high=np.full(gen.dim_theta, 1.0),
    )
    iv0 = integrated_variance(mfgp, bounds, n_mc_samples=2000, seed=0)

    loop = ActiveLearningLoop(
        mfgp=mfgp, generator=gen, cnp=cnp, bounds=bounds, data=data,
        n_hf_events=N_HF_EVENTS, n_mc_samples=N_MC_SAMPLES,
        n_candidates_per_axis=N_CAND_PER_AXIS,
        seed=0, refit_n_restarts=REFIT_N_RESTARTS,
    )
    records = loop.run(N_AL_STEPS)

    for k, rec in enumerate(records, start=1):
        out = OUT_DIR / f"optimizer_{name}_step{k}.png"
        title = (
            f"{name} — AL step {k}/{N_AL_STEPS}  "
            f"(IV {rec.integrated_variance_before:.2e} → "
            f"{rec.integrated_variance_after:.2e})"
        )
        if gen.dim_theta == 1:
            _plot_step_1d(gen, rec, out, title)
        else:
            _plot_step_2d(rec, out, title)
        print(f"  wrote {out}")

    iv_path = OUT_DIR / f"optimizer_{name}_iv.png"
    _plot_iv_trace(name, iv0, records, iv_path)
    print(f"  wrote {iv_path}")

    iv_final = records[-1].integrated_variance_after
    iv_min = min(r.integrated_variance_after for r in records)
    return {
        "iv_start": iv0,
        "iv_final": iv_final,
        "iv_min": iv_min,
    }


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios", default=",".join(OPTIMIZER_SCENARIOS),
        help="Comma-separated scenario names to run (default: all six).",
    )
    args = parser.parse_args(argv)
    requested = [s.strip().upper() for s in args.scenarios.split(",") if s.strip()]
    unknown = [s for s in requested if s not in OPTIMIZER_SCENARIOS]
    if unknown:
        raise SystemExit(
            f"unknown scenario(s) {unknown}; valid: {OPTIMIZER_SCENARIOS}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float]] = {}
    for name in requested:
        summary[name] = _run_scenario(name)

    print("\n  IV summary (start / final / min, lower is better):")
    print("    scenario   start       final       min        min/start")
    for name, s in summary.items():
        ratio = s["iv_min"] / s["iv_start"] if s["iv_start"] > 0 else float("nan")
        print(
            f"    {name:<10} {s['iv_start']:.3e}   "
            f"{s['iv_final']:.3e}   {s['iv_min']:.3e}   {ratio:.2f}"
        )


if __name__ == "__main__":
    main()
