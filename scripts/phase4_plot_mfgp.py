"""Phase 4 visual evidence: MFGP posterior, coverage, QQ-calibration.

For each of S1, S2, S3, S4, S7, S8 (skipping the EVENT_ONLY scenarios
that have no θ to drive an MFGP):

  1. Train a CNP on synthetic data (Phase 3 budgets).
  2. Build LF / HF datasets and fit a 3-fidelity MFGP.
  3. Generate held-out HF observations and assemble three plots:
     - mfgp_posterior_<S>.png  (1-D or 2-D depending on dim_θ)
     - mfgp_coverage_<S>.png   (Figure-5 time series + calibration bars)
     - residuals contributing to a combined mfgp_qq.png

Run from the repo root:

    python scripts/phase4_plot_mfgp.py
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
from scipy.stats import norm  # noqa: E402

from core import (  # noqa: E402
    build_cnp,
    evaluate_mfgp_coverage,
    fit_mfgp_three_fidelity,
    prepare_mfgp_datasets,
    train_cnp,
)
from data.pseudo_generator import for_scenario  # noqa: E402
from schemas.config import (  # noqa: E402
    CNPConfig,
    EncoderConfig,
    TrainingConfig,
)
from schemas.data_models import InputMode  # noqa: E402
from viz import plot_comparison_1d, plot_coverage_test  # noqa: E402

OUT_DIR = Path("viz_output")

MFGP_SCENARIOS = ["S1", "S2", "S3", "S4", "S7", "S8"]
CNP_STEPS = {"S1": 500, "S2": 800, "S3": 800, "S4": 1000, "S7": 600, "S8": 600}
N_LF_TRIALS, N_LF_EVENTS = 200, 64
N_HF_TRIALS, N_HF_EVENTS = 50, 128
N_TEST_TRIALS, N_TEST_EVENTS = 100, 128
N_RESTARTS = 10


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
        n_steps=CNP_STEPS[name], learning_rate=1.0e-3, batch_size=16,
        n_events_per_trial=128, n_mc_samples=4, eval_every=0, seed=0,
    )


def _train_pipeline(name: str):
    """Train CNP + fit MFGP for one scenario; return (gen, cnp, mfgp, data)."""
    torch.manual_seed(0)
    np.random.seed(0)
    gen = for_scenario(name, seed=0)
    cnp = build_cnp(_enc_cfg(), gen.dim_theta, gen.dim_phi)
    train_cnp(cnp, gen, cnp_config=_cnp_cfg(), training_config=_train_cfg(name))
    cnp.eval()
    data = prepare_mfgp_datasets(
        cnp, gen,
        n_lf_trials=N_LF_TRIALS, n_lf_events=N_LF_EVENTS,
        n_hf_trials=N_HF_TRIALS, n_hf_events=N_HF_EVENTS,
        seed=0,
    )
    mfgp = fit_mfgp_three_fidelity(data, n_restarts=N_RESTARTS)
    return gen, cnp, mfgp, data


def _grid_1d(n: int = 80) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n)


def _grid_2d(n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a0 = _grid_1d(n)
    a1 = _grid_1d(n)
    G0, G1 = np.meshgrid(a0, a1, indexing="ij")
    mesh = np.stack([G0, G1], axis=-1)
    return a0, a1, mesh


# ---------------------------------------------------------------------------
# Per-scenario posterior plot.
# ---------------------------------------------------------------------------


def plot_posterior_1d(name: str, gen, mfgp, data, out: Path) -> None:
    """1-D θ posterior: analytical t̄(θ), MFGP μ ± σ, HF y_raw scatter."""
    grid = _grid_1d()
    theta_grid = grid[:, None]

    # Analytical t̄(θ) — marginal over φ for FULL, exact for DESIGN_ONLY.
    if gen.mode is InputMode.FULL:
        rng = np.random.default_rng(11)
        n_phi_mc = 200
        phi_mc = rng.uniform(-1.0, 1.0, size=(grid.shape[0], n_phi_mc, gen.dim_phi))
        p_at_phi = gen.truth.evaluate(theta=theta_grid[:, None, :], phi=phi_mc)
        t_curve = p_at_phi.mean(axis=1)
    else:  # DESIGN_ONLY
        t_curve = gen.truth.evaluate(theta=theta_grid, phi=None)

    mu, var = mfgp.predict(theta_grid, fidelity=mfgp.n_fidelities - 1)
    sigma = np.sqrt(var)

    # HF training points overlay (the data the MFGP actually saw).
    overlay_xy = (data["X_hf"][:, 0], data["Y_hf_raw"].flatten())

    plot_comparison_1d(
        x=grid,
        analytical=t_curve,
        predicted=mu,
        predicted_sigma=sigma,
        out_path=out,
        title=f"{name} — MFGP posterior on θ ({gen.mode.value})",
        xlabel="θ",
        ylabel="y",
        analytical_label="analytical t̄(θ)",
        predicted_label="MFGP μ(θ)",
        overlay_xy=overlay_xy,
        overlay_label=f"HF y_raw (n={N_HF_TRIALS})",
    )


def plot_posterior_2d(name: str, gen, mfgp, data, out: Path) -> None:
    """2-D θ posterior: 3-panel [analytical t̄ | MFGP μ | MFGP σ]."""
    a0, a1, mesh = _grid_2d()
    G = mesh.shape[0]
    theta_flat = mesh.reshape(G * G, 2)

    # Analytical t̄(θ).
    if gen.mode is InputMode.FULL:
        rng = np.random.default_rng(11)
        n_phi_mc = 200
        phi_mc = rng.uniform(-1.0, 1.0, size=(G * G, n_phi_mc, gen.dim_phi))
        p_at_phi = gen.truth.evaluate(theta=theta_flat[:, None, :], phi=phi_mc)
        t_grid = p_at_phi.mean(axis=1).reshape(G, G)
    else:
        t_grid = gen.truth.evaluate(theta=theta_flat, phi=None).reshape(G, G)

    mu_flat, var_flat = mfgp.predict(theta_flat, fidelity=mfgp.n_fidelities - 1)
    mu_grid = mu_flat.reshape(G, G)
    sigma_grid = np.sqrt(var_flat).reshape(G, G)

    vmin = float(min(t_grid.min(), mu_grid.min()))
    vmax = float(max(t_grid.max(), mu_grid.max()))
    extent = (a1[0], a1[-1], a0[0], a0[-1])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, arr, panel_title, cmap, vlim in [
        (axes[0], t_grid, "analytical t̄(θ)", "viridis", (vmin, vmax)),
        (axes[1], mu_grid, "MFGP μ(θ)", "viridis", (vmin, vmax)),
        (axes[2], sigma_grid, "MFGP σ(θ)", "magma", (None, None)),
    ]:
        im = ax.imshow(
            arr, origin="lower", extent=extent, aspect="auto",
            cmap=cmap, vmin=vlim[0], vmax=vlim[1],
        )
        ax.set_title(panel_title)
        ax.set_xlabel("θ₂")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("θ₁")
    # Mark the HF training θ points so the eye sees where data was.
    hf_theta = data["X_hf"]
    for ax in axes[:2]:
        ax.scatter(
            hf_theta[:, 1], hf_theta[:, 0],
            s=18, facecolors="none", edgecolors="white", linewidths=0.8,
        )
    fig.suptitle(f"{name} — MFGP 2-D posterior on θ ({gen.mode.value})")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Coverage plot per scenario (reuse Phase 3 helper).
# ---------------------------------------------------------------------------


def plot_coverage(name: str, gen, mfgp, cnp, out: Path) -> dict[str, float]:
    result = evaluate_mfgp_coverage(
        mfgp, cnp, gen,
        n_test_trials=N_TEST_TRIALS, n_test_events=N_TEST_EVENTS, seed=12345,
    )
    coverage = plot_coverage_test(
        y_raw=result["y_obs"],
        y_predicted=result["mu"],
        sigma_predicted=result["sigma"],
        out_path=out,
        title=f"{name} — MFGP coverage on {N_TEST_TRIALS} held-out HF trials",
        predicted_label="MFGP μ",
    )
    return coverage, result


# ---------------------------------------------------------------------------
# Combined QQ plot across all valid scenarios.
# ---------------------------------------------------------------------------


def plot_qq(residuals_by_scenario: dict[str, np.ndarray], out: Path) -> None:
    """QQ-plot of standardised residuals z = (y_obs - μ) / σ.

    A well-calibrated MFGP produces z ~ N(0, 1); the empirical and
    theoretical quantiles should lie on the y=x diagonal.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    diag = np.linspace(-3.5, 3.5, 200)
    ax.plot(diag, diag, color="black", linewidth=1, alpha=0.5, label="y = x (ideal)")

    for i, (name, z) in enumerate(residuals_by_scenario.items()):
        z_sorted = np.sort(z)
        n = len(z_sorted)
        theoretical = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        ax.plot(theoretical, z_sorted, marker=".", linestyle="none",
                markersize=5, alpha=0.7, label=name, color=f"C{i}")

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel("theoretical quantile (N(0, 1))")
    ax.set_ylabel("empirical residual quantile  z = (y_obs − μ) / σ")
    ax.set_title("MFGP residual calibration — Q-Q across S1–S4, S7, S8")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    coverage_summary: dict[str, dict[str, float]] = {}
    residuals_by_scenario: dict[str, np.ndarray] = {}

    for name in MFGP_SCENARIOS:
        print(f"  ── {name} ──")
        gen, cnp, mfgp, data = _train_pipeline(name)

        post = OUT_DIR / f"mfgp_posterior_{name}.png"
        if gen.dim_theta == 1:
            plot_posterior_1d(name, gen, mfgp, data, post)
        else:
            plot_posterior_2d(name, gen, mfgp, data, post)
        print(f"  wrote {post}")

        cov_path = OUT_DIR / f"mfgp_coverage_{name}.png"
        coverage, result = plot_coverage(name, gen, mfgp, cnp, cov_path)
        coverage_summary[name] = coverage
        residuals_by_scenario[name] = (result["y_obs"] - result["mu"]) / result["sigma"]
        print(
            f"  wrote {cov_path}  "
            f"(1σ={coverage['1sigma']:.0%} 2σ={coverage['2sigma']:.0%} "
            f"3σ={coverage['3sigma']:.0%})"
        )

    qq_path = OUT_DIR / "mfgp_qq.png"
    plot_qq(residuals_by_scenario, qq_path)
    print(f"  wrote {qq_path}")

    print("\n  Coverage summary (target: 1σ≈68%, 2σ≈95%, 3σ≈99.7%):")
    print("    scenario   1σ      2σ      3σ")
    for name, cov in coverage_summary.items():
        s1, s2, s3 = cov["1sigma"], cov["2sigma"], cov["3sigma"]
        print(f"    {name}        {s1:5.0%}   {s2:5.0%}   {s3:5.0%}")


if __name__ == "__main__":
    main()
