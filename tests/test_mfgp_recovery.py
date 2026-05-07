"""Phase 4 acceptance gate: MFGP coverage on held-out HF observations.

End-to-end pipeline per scenario:

  1. Train a CNP on synthetic data (small budget, reused from Phase 3).
  2. Build LF / HF datasets via the CNP and the pseudo-data generator.
  3. Fit a 3-fidelity recursive co-kriging GP.
  4. Generate 100 fresh held-out HF trials at random θ, compute
     y_raw = m/N over each trial's target events.
  5. Predict the GP posterior at those θ and check what fraction of
     observations fall inside ±1σ / ±2σ / ±3σ.

Target Gaussian rates: 68.27 / 95.45 / 99.73 %. We allow a generous
tolerance (±15 percentage points at 1σ, ±10 at 2σ, ±5 at 3σ) because
the MFGP optimizer is stochastic and the held-out sample is finite.

Skips ``EVENT_ONLY`` scenarios (S5, S6) — they have no θ to drive an
MFGP. Tested scenarios: S1, S2, S3, S4, S7, S8.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from core import (
    build_cnp,
    evaluate_mfgp_coverage,
    fit_mfgp_three_fidelity,
    prepare_mfgp_datasets,
    train_cnp,
)
from data import for_scenario
from schemas.config import CNPConfig, EncoderConfig, TrainingConfig

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"GPy.*")

MFGP_SCENARIOS = ["S1", "S2", "S3", "S4", "S7", "S8"]

# Per-scenario CNP step budget (Phase 3 thresholds).
CNP_STEPS = {"S1": 500, "S2": 800, "S3": 800, "S4": 1000, "S7": 600, "S8": 600}

# MFGP coverage tolerances (percentage points off target).
TOL_1S = 0.15
TOL_2S = 0.10
TOL_3S = 0.05


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
        n_steps=CNP_STEPS[name],
        learning_rate=1.0e-3,
        batch_size=16,
        n_events_per_trial=128,
        n_mc_samples=4,
        eval_every=0,
        seed=0,
    )


@pytest.mark.parametrize("name", MFGP_SCENARIOS)
def test_mfgp_coverage(name: str) -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    gen = for_scenario(name, seed=0)
    cnp = build_cnp(_enc_cfg(), gen.dim_theta, gen.dim_phi)
    train_cnp(
        cnp, gen,
        cnp_config=_cnp_cfg(),
        training_config=_train_cfg(name),
    )

    data = prepare_mfgp_datasets(
        cnp, gen,
        n_lf_trials=200, n_lf_events=64,
        n_hf_trials=50, n_hf_events=128,
        seed=0,
    )
    mfgp = fit_mfgp_three_fidelity(data, n_restarts=10)
    result = evaluate_mfgp_coverage(
        mfgp, cnp, gen,
        n_test_trials=100, n_test_events=128, seed=12345,
    )

    s1, s2, s3 = result["1sigma"], result["2sigma"], result["3sigma"]
    target = (0.6827, 0.9545, 0.9973)
    assert abs(s1 - target[0]) <= TOL_1S, (
        f"{name}: 1σ coverage {s1:.0%} differs from target {target[0]:.0%} by > {TOL_1S:.0%}"
    )
    assert abs(s2 - target[1]) <= TOL_2S, (
        f"{name}: 2σ coverage {s2:.0%} differs from target {target[1]:.0%} by > {TOL_2S:.0%}"
    )
    assert abs(s3 - target[2]) <= TOL_3S, (
        f"{name}: 3σ coverage {s3:.0%} differs from target {target[2]:.0%} by > {TOL_3S:.0%}"
    )
