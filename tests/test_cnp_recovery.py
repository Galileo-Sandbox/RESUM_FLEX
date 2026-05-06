"""Phase 3 acceptance gate: ``MAE(β, p) < threshold[scenario]`` after training.

This is the end-to-end test that closes Phase 3. For each of S1..S8 we
build a CNP, train it on synthetic batches drawn from the
:class:`PseudoDataGenerator`, then evaluate the predicted ``β`` against
the analytical ``p`` on a fresh held-out batch. The per-scenario MAE
threshold lives in ``config.yaml`` (``mae_thresholds.s{1..8}``).

These runs are non-trivial — 0.9–3.2 s per scenario, ~16 s total — but
short enough to live in the regular pytest run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from core import build_cnp, evaluate_mae, train_cnp
from data.pseudo_generator import PseudoDataGenerator, for_scenario
from schemas.config import CNPConfig, EncoderConfig, TrainingConfig, load_config

ALL_SCENARIOS = list(PseudoDataGenerator.SCENARIOS.keys())

# Per-scenario training budgets — picked so each scenario clears its
# threshold with at least ~2× margin while keeping total test time
# under ~20 s on CPU.
_BUDGET = {
    "S1": 500,
    "S2": 800,
    "S3": 800,
    "S4": 1000,
    "S5": 600,
    "S6": 600,
    "S7": 600,
    "S8": 600,
}


def _enc_cfg() -> EncoderConfig:
    return EncoderConfig(
        type="mlp", latent_dim=32, hidden_dims=[64, 64], dropout=0.0
    )


def _cnp_cfg() -> CNPConfig:
    return CNPConfig(
        n_context_min=32,
        n_context_max=96,
        output_activation="sigmoid",
        mixup_alpha=0.1,
    )


def _train_cfg(n_steps: int) -> TrainingConfig:
    return TrainingConfig(
        n_steps=n_steps,
        learning_rate=1.0e-3,
        batch_size=16,
        n_events_per_trial=128,
        n_mc_samples=4,
        eval_every=0,
        seed=0,
    )


@pytest.mark.parametrize("name", ALL_SCENARIOS)
def test_cnp_mae_under_threshold(name: str) -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    threshold = getattr(load_config("config.yaml").mae_thresholds, name.lower())
    gen = for_scenario(name, seed=0)

    cnp = build_cnp(_enc_cfg(), gen.dim_theta, gen.dim_phi)
    train_cnp(
        cnp, gen,
        cnp_config=_cnp_cfg(),
        training_config=_train_cfg(_BUDGET[name]),
    )

    mae = evaluate_mae(
        cnp, gen,
        batch_size=64,
        n_events=256,
        n_context=128,
        seed=999,  # held-out
    )

    assert mae < threshold, (
        f"{name}: MAE={mae:.4f} did not clear threshold {threshold:.3f}. "
        f"Either training is regressing or the threshold needs revisiting."
    )
