"""Tests for the batch-based MFGP pipeline helpers.

The synthetic-generator wrappers (``prepare_mfgp_datasets``,
``evaluate_mfgp_coverage``) get end-to-end coverage in
``test_mfgp_recovery``; this file pins the *batch-based* user-facing
entry points: shapes, mode rejection, and parity with the wrappers.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from core import (
    build_cnp,
    evaluate_mfgp_coverage,
    evaluate_mfgp_coverage_from_batch,
    fit_mfgp_three_fidelity,
    prepare_mfgp_datasets,
    prepare_mfgp_datasets_from_batches,
    train_cnp,
)
from data import for_scenario
from schemas.config import CNPConfig, EncoderConfig, TrainingConfig

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"GPy.*")


def _quick_cnp(name: str = "S7"):
    """Cheap CNP for shape-level tests — 200 steps, tiny encoder."""
    torch.manual_seed(0)
    np.random.seed(0)
    gen = for_scenario(name, seed=0)
    enc = EncoderConfig(type="mlp", latent_dim=16, hidden_dims=[32, 32])
    cnp = build_cnp(enc, gen.dim_theta, gen.dim_phi)
    train_cnp(
        cnp, gen,
        cnp_config=CNPConfig(
            n_context_min=16, n_context_max=64,
            output_activation="sigmoid", mixup_alpha=0.1,
        ),
        training_config=TrainingConfig(
            n_steps=200, learning_rate=1.0e-3, batch_size=16,
            n_events_per_trial=64, n_mc_samples=4, eval_every=0, seed=0,
        ),
    )
    return gen, cnp


def test_prepare_from_batches_shapes() -> None:
    gen, cnp = _quick_cnp()
    lf = gen.generate(n_trials=20, n_events=32, seed=1)
    hf = gen.generate(n_trials=5, n_events=64, seed=2)

    data = prepare_mfgp_datasets_from_batches(
        cnp, lf, hf, n_mc_samples=8, seed=0,
    )

    assert data["X_lf"].shape == (20, gen.dim_theta)
    assert data["Y_lf_cnp"].shape == (20, 1)
    assert data["X_hf"].shape == (5, gen.dim_theta)
    assert data["Y_hf_cnp"].shape == (5, 1)
    assert data["Y_hf_raw"].shape == (5, 1)
    # β̄ in [0, 1]; y_raw is m/N also in [0, 1].
    assert np.all((data["Y_lf_cnp"] >= 0) & (data["Y_lf_cnp"] <= 1))
    assert np.all((data["Y_hf_cnp"] >= 0) & (data["Y_hf_cnp"] <= 1))
    assert np.all((data["Y_hf_raw"] >= 0) & (data["Y_hf_raw"] <= 1))


def test_prepare_from_batches_rejects_event_only() -> None:
    gen, cnp = _quick_cnp()
    other = for_scenario("S5", seed=0)  # EVENT_ONLY
    bad_batch = other.generate(n_trials=3, n_events=16, seed=0)
    good_batch = gen.generate(n_trials=3, n_events=16, seed=0)
    with pytest.raises(ValueError, match="lf_batch"):
        prepare_mfgp_datasets_from_batches(cnp, bad_batch, good_batch)
    with pytest.raises(ValueError, match="hf_batch"):
        prepare_mfgp_datasets_from_batches(cnp, good_batch, bad_batch)


def test_evaluate_coverage_from_batch_shapes_and_keys() -> None:
    gen, cnp = _quick_cnp()
    data = prepare_mfgp_datasets(
        cnp, gen,
        n_lf_trials=40, n_lf_events=32,
        n_hf_trials=8, n_hf_events=64,
        seed=0,
    )
    mfgp = fit_mfgp_three_fidelity(data, n_restarts=2)

    holdout = gen.generate(n_trials=15, n_events=64, seed=999)
    result = evaluate_mfgp_coverage_from_batch(mfgp, cnp, holdout, seed=999)

    assert result["theta"].shape == (15, gen.dim_theta)
    assert result["y_obs"].shape == (15,)
    assert result["mu"].shape == (15,)
    assert result["sigma"].shape == (15,)
    for key in ("1sigma", "2sigma", "3sigma"):
        assert 0.0 <= result[key] <= 1.0


def test_evaluate_coverage_from_batch_rejects_event_only() -> None:
    gen, cnp = _quick_cnp()
    data = prepare_mfgp_datasets(
        cnp, gen,
        n_lf_trials=40, n_lf_events=32,
        n_hf_trials=6, n_hf_events=64,
        seed=0,
    )
    mfgp = fit_mfgp_three_fidelity(data, n_restarts=2)
    other = for_scenario("S5", seed=0)
    bad = other.generate(n_trials=4, n_events=16, seed=0)
    with pytest.raises(ValueError, match="holdout_hf_batch"):
        evaluate_mfgp_coverage_from_batch(mfgp, cnp, bad)


def test_generator_wrapper_parity_with_batches() -> None:
    """Generator-based wrapper ≡ batch-based call on the same batches.

    Confirms the public APIs are not silently divergent: wrap = batch
    after we replay the same draws with matching seeds.
    """
    gen, cnp = _quick_cnp()
    n_lf, lf_events = 30, 32
    n_hf, hf_events = 6, 64
    seed = 7

    # MC sampling in cnp_trial_predictive consumes torch's global RNG,
    # so the two paths only match bit-for-bit if we re-seed torch
    # before each. Anchor identically before both runs.
    torch.manual_seed(42)
    via_wrapper = prepare_mfgp_datasets(
        cnp, gen,
        n_lf_trials=n_lf, n_lf_events=lf_events,
        n_hf_trials=n_hf, n_hf_events=hf_events,
        n_mc_samples=8, seed=seed,
    )
    torch.manual_seed(42)
    lf = gen.generate(n_trials=n_lf, n_events=lf_events, seed=seed)
    hf = gen.generate(n_trials=n_hf, n_events=hf_events, seed=seed + 100)
    via_batches = prepare_mfgp_datasets_from_batches(
        cnp, lf, hf,
        n_lf_context=lf_events // 2,
        n_hf_context=hf_events // 2,
        n_mc_samples=8, seed=seed,
    )
    for key in ("X_lf", "X_hf", "Y_hf_raw"):
        np.testing.assert_array_equal(via_wrapper[key], via_batches[key])
    # MC sampling in cnp_trial_predictive uses the same RNG path → CNP
    # predictions should match exactly given identical inputs.
    np.testing.assert_allclose(via_wrapper["Y_lf_cnp"], via_batches["Y_lf_cnp"])
    np.testing.assert_allclose(via_wrapper["Y_hf_cnp"], via_batches["Y_hf_cnp"])
