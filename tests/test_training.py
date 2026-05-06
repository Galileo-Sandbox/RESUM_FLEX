"""Phase 3 training-loop tests.

Lightweight checks on the contract of :func:`train_cnp`,
:func:`evaluate_mae`, and the checkpoint round-trip. The serious
end-to-end MAE-vs-threshold validation across all 8 scenarios lives
in :mod:`tests.test_cnp_recovery` so it can be marked / skipped
independently when iterating fast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from core import (
    build_cnp,
    evaluate_mae,
    load_checkpoint,
    save_checkpoint,
    train_cnp,
)
from data import for_scenario
from schemas.config import CNPConfig, EncoderConfig, TrainingConfig


def _enc_cfg() -> EncoderConfig:
    return EncoderConfig(type="mlp", latent_dim=16, hidden_dims=[32, 32], dropout=0.0)


def _cnp_cfg(n_min: int = 8, n_max: int = 24) -> CNPConfig:
    return CNPConfig(
        n_context_min=n_min,
        n_context_max=n_max,
        output_activation="sigmoid",
        mixup_alpha=0.1,
    )


def _train_cfg(steps: int = 50, n_events: int = 32, eval_every: int = 0) -> TrainingConfig:
    return TrainingConfig(
        n_steps=steps,
        learning_rate=1.0e-3,
        batch_size=4,
        n_events_per_trial=n_events,
        n_mc_samples=2,
        eval_every=eval_every,
        eval_batch_size=8,
        eval_n_events=64,
        seed=0,
    )


# ---------------------------------------------------------------------------
# History contract.
# ---------------------------------------------------------------------------


def test_train_cnp_returns_history_with_expected_keys() -> None:
    gen = for_scenario("S5", seed=0)
    cnp = build_cnp(_enc_cfg(), gen.dim_theta, gen.dim_phi)
    history = train_cnp(
        cnp, gen, cnp_config=_cnp_cfg(), training_config=_train_cfg(steps=20)
    )
    assert set(history.keys()) >= {"step", "loss", "eval_step", "eval_mae"}
    assert len(history["step"]) == 20
    assert len(history["loss"]) == 20
    assert all(isinstance(s, int) for s in history["step"])
    assert all(np.isfinite(history["loss"]))


def test_train_cnp_records_eval_history() -> None:
    gen = for_scenario("S5", seed=0)
    cnp = build_cnp(_enc_cfg(), gen.dim_theta, gen.dim_phi)
    history = train_cnp(
        cnp,
        gen,
        cnp_config=_cnp_cfg(),
        training_config=_train_cfg(steps=20, eval_every=10),
    )
    # Expect 2 evaluations: at steps 9 and 19.
    assert len(history["eval_step"]) == 2
    assert all(np.isfinite(history["eval_mae"]))


def test_progress_callback_invoked() -> None:
    gen = for_scenario("S5", seed=0)
    cnp = build_cnp(_enc_cfg(), gen.dim_theta, gen.dim_phi)
    seen_steps: list[int] = []
    train_cnp(
        cnp,
        gen,
        cnp_config=_cnp_cfg(),
        training_config=_train_cfg(steps=10),
        progress_callback=lambda step, loss: seen_steps.append(step),
    )
    assert seen_steps == list(range(10))


# ---------------------------------------------------------------------------
# Loss actually decreases.
# ---------------------------------------------------------------------------


def test_train_cnp_decreases_loss() -> None:
    """Light run, 200 steps; mean of last 20 < mean of first 20."""
    torch.manual_seed(0)
    gen = for_scenario("S5", seed=0)
    cnp = build_cnp(_enc_cfg(latent_dim=32) if False else _enc_cfg(), gen.dim_theta, gen.dim_phi)
    history = train_cnp(
        cnp,
        gen,
        cnp_config=_cnp_cfg(),
        training_config=_train_cfg(steps=200, n_events=64),
    )
    early = float(np.mean(history["loss"][:20]))
    late = float(np.mean(history["loss"][-20:]))
    assert late < early - 0.05, f"loss did not drop: early={early:.3f} late={late:.3f}"


# ---------------------------------------------------------------------------
# Empty range guard.
# ---------------------------------------------------------------------------


def test_invalid_n_context_range_raises() -> None:
    gen = for_scenario("S5", seed=0)
    cnp = build_cnp(_enc_cfg(), gen.dim_theta, gen.dim_phi)
    cfg = _train_cfg(steps=5, n_events=4)  # n_events too small
    bad = CNPConfig(
        n_context_min=10,
        n_context_max=20,
        output_activation="sigmoid",
        mixup_alpha=0.1,
    )
    with pytest.raises(ValueError, match="n_context"):
        train_cnp(cnp, gen, cnp_config=bad, training_config=cfg)


# ---------------------------------------------------------------------------
# evaluate_mae contract.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["S1", "S5", "S7"])
def test_evaluate_mae_returns_finite_scalar(name: str) -> None:
    gen = for_scenario(name, seed=0)
    cnp = build_cnp(_enc_cfg(), gen.dim_theta, gen.dim_phi)
    mae = evaluate_mae(cnp, gen, batch_size=8, n_events=32, n_context=16, seed=0)
    assert isinstance(mae, float)
    assert np.isfinite(mae)
    assert mae >= 0.0


def test_evaluate_mae_drops_after_training() -> None:
    """Sanity check that evaluate_mae actually moves with training quality."""
    torch.manual_seed(0)
    gen = for_scenario("S5", seed=0)
    cnp = build_cnp(_enc_cfg(), gen.dim_theta, gen.dim_phi)
    pre = evaluate_mae(cnp, gen, batch_size=16, n_events=64, n_context=32, seed=42)
    train_cnp(
        cnp,
        gen,
        cnp_config=_cnp_cfg(),
        training_config=_train_cfg(steps=300, n_events=64),
    )
    post = evaluate_mae(cnp, gen, batch_size=16, n_events=64, n_context=32, seed=42)
    assert post < pre, f"MAE did not improve: pre={pre:.3f} post={post:.3f}"


# ---------------------------------------------------------------------------
# Checkpoint save / load round-trip.
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    """Saving and reloading must reproduce parameters bit-for-bit."""
    torch.manual_seed(0)
    gen = for_scenario("S1", seed=0)
    cnp = build_cnp(_enc_cfg(), gen.dim_theta, gen.dim_phi)
    train_cnp(
        cnp,
        gen,
        cnp_config=_cnp_cfg(),
        training_config=_train_cfg(steps=20, n_events=32),
    )

    ckpt = tmp_path / "cnp.ckpt"
    save_checkpoint(
        ckpt,
        cnp,
        encoder_config=_enc_cfg(),
        dim_theta=gen.dim_theta,
        dim_phi=gen.dim_phi,
        metadata={"scenario": "S1", "n_steps": 20},
    )
    assert ckpt.exists()

    cnp_loaded, payload = load_checkpoint(ckpt)
    assert payload["metadata"]["scenario"] == "S1"
    for (k1, p1), (k2, p2) in zip(
        cnp.state_dict().items(), cnp_loaded.state_dict().items(), strict=True
    ):
        assert k1 == k2
        torch.testing.assert_close(p1, p2, atol=0.0, rtol=0.0)


def test_load_checkpoint_predicts_identically(tmp_path: Path) -> None:
    """Predictions before save must match predictions after reload."""
    torch.manual_seed(0)
    gen = for_scenario("S5", seed=0)
    cnp = build_cnp(_enc_cfg(), gen.dim_theta, gen.dim_phi)
    train_cnp(
        cnp,
        gen,
        cnp_config=_cnp_cfg(),
        training_config=_train_cfg(steps=20, n_events=32),
    )

    eval_batch = gen.generate(n_trials=4, n_events=16)
    from core.surrogate_cnp import split_context_target
    ctx, tgt = split_context_target(eval_batch, n_context=8, seed=0)

    cnp.eval()
    with torch.no_grad():
        beta_before = cnp.predict_beta(ctx, tgt).numpy()

    ckpt = tmp_path / "cnp.ckpt"
    save_checkpoint(
        ckpt,
        cnp,
        encoder_config=_enc_cfg(),
        dim_theta=gen.dim_theta,
        dim_phi=gen.dim_phi,
    )

    cnp_loaded, _ = load_checkpoint(ckpt)
    cnp_loaded.eval()
    with torch.no_grad():
        beta_after = cnp_loaded.predict_beta(ctx, tgt).numpy()

    np.testing.assert_array_equal(beta_before, beta_after)
