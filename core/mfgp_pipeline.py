"""Glue between a trained CNP and the MFGP layer.

Builds the three per-fidelity datasets the MFGP consumes:

* LF — many trials, fewer events per trial; ``β̄(θ)`` from the CNP.
* MF — fewer trials, more events per trial; ``β̄(θ)`` from the CNP.
* HF — same trials as MF; ``y_raw = m/N`` (the target metric, with
       irreducible Bernoulli noise around ``t̄(θ)``).

This module is the only place where torch and GPy meet — and only at
the boundary: ``cnp_trial_predictive`` produces numpy arrays from
torch, and those arrays are then fed to :class:`MultiFidelityGP`. The
two libraries never share imports inside any single object.

EVENT_ONLY scenarios have no ``θ`` and therefore can't drive an MFGP;
:func:`prepare_mfgp_datasets` rejects them up front.
"""

from __future__ import annotations

import numpy as np

from core.surrogate_cnp import (
    ConditionalNeuralProcess,
    split_context_target,
)
from core.surrogate_mfgp import MultiFidelityGP
from core.training import cnp_trial_predictive
from data.pseudo_generator import PseudoDataGenerator
from schemas.data_models import InputMode


def prepare_mfgp_datasets(
    cnp: ConditionalNeuralProcess,
    generator: PseudoDataGenerator,
    *,
    n_lf_trials: int = 100,
    n_lf_events: int = 64,
    n_hf_trials: int = 8,
    n_hf_events: int = 128,
    n_mc_samples: int = 50,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build LF / MF / HF datasets from a trained CNP and a generator.

    Returns a dict with five arrays, all numpy:

    ===========  ============================  ===============================
    key          shape                         meaning
    ===========  ============================  ===============================
    X_lf         ``(n_lf_trials, D_θ)``        per-trial θ for the LF dataset
    Y_lf_cnp     ``(n_lf_trials, 1)``          β̄(θ) per LF trial
    X_hf         ``(n_hf_trials, D_θ)``        per-trial θ for the HF dataset
    Y_hf_cnp     ``(n_hf_trials, 1)``          β̄(θ) per HF trial
    Y_hf_raw     ``(n_hf_trials, 1)``          ``m/N`` per HF trial
    ===========  ============================  ===============================

    Per the meta-learning paradigm, β̄ for a trial is computed by
    splitting that trial's events into a context half and a target half
    (50/50), running the CNP, and averaging predicted β over the target
    events. ``y_raw`` averages the binary ``X`` over the same target
    events so the two metrics aggregate identically.
    """
    if generator.mode is InputMode.EVENT_ONLY:
        raise ValueError(
            "MFGP needs θ as input; generator mode=EVENT_ONLY has no θ."
        )

    # Low-fidelity: many trials, few events.
    lf_batch = generator.generate(
        n_trials=n_lf_trials, n_events=n_lf_events, seed=seed,
    )
    lf_ctx, lf_tgt = split_context_target(
        lf_batch, n_context=n_lf_events // 2, seed=seed + 1,
    )
    pred_lf = cnp_trial_predictive(
        cnp, lf_ctx, lf_tgt, n_mc_samples=n_mc_samples,
    )

    # High-fidelity: fewer trials, more events. Both β̄ and m/N come from
    # the same target slice so they aggregate over the same events.
    hf_batch = generator.generate(
        n_trials=n_hf_trials, n_events=n_hf_events, seed=seed + 100,
    )
    hf_ctx, hf_tgt = split_context_target(
        hf_batch, n_context=n_hf_events // 2, seed=seed + 101,
    )
    pred_hf = cnp_trial_predictive(
        cnp, hf_ctx, hf_tgt, n_mc_samples=n_mc_samples,
    )
    y_hf_raw = hf_tgt.labels.mean(axis=1).astype(float)

    assert lf_batch.theta is not None and hf_batch.theta is not None
    return {
        "X_lf": lf_batch.theta,
        "Y_lf_cnp": pred_lf["y_cnp"].reshape(-1, 1),
        "X_hf": hf_batch.theta,
        "Y_hf_cnp": pred_hf["y_cnp"].reshape(-1, 1),
        "Y_hf_raw": y_hf_raw.reshape(-1, 1),
    }


def fit_mfgp_three_fidelity(
    data: dict[str, np.ndarray],
    *,
    kernel: str = "rbf",
    n_restarts: int = 5,
    verbose: bool = False,
) -> MultiFidelityGP:
    """Fit a 3-fidelity MFGP on ``(LF β̄, HF β̄, HF y_raw)``.

    The architecture matches the paper's recursive recipe::

        f₀ (LF)  = β̄(θ) from LF trials
        f₁ (MF)  = ρ₀·f₀ + δ₀(θ) ≈ β̄(θ) from HF trials
        f₂ (HF)  = ρ₁·f₁ + δ₁(θ) ≈ y_raw(θ) from HF trials   ← target
    """
    dim_theta = data["X_hf"].shape[1]
    X_list = [data["X_lf"], data["X_hf"], data["X_hf"]]
    Y_list = [data["Y_lf_cnp"], data["Y_hf_cnp"], data["Y_hf_raw"]]
    return MultiFidelityGP(
        n_fidelities=3, dim_theta=dim_theta, kernel=kernel,
    ).fit(X_list, Y_list, n_restarts=n_restarts, verbose=verbose)


def evaluate_mfgp_coverage(
    mfgp: MultiFidelityGP,
    cnp: ConditionalNeuralProcess,
    generator: PseudoDataGenerator,
    *,
    n_test_trials: int = 100,
    n_test_events: int = 128,
    seed: int = 12345,
) -> dict[str, np.ndarray | float]:
    """Held-out HF coverage for a fitted MFGP.

    Generates ``n_test_trials`` fresh trials at random θ, computes the
    raw rate ``m/N`` per trial (over the target half — same convention
    as :func:`prepare_mfgp_datasets`), and predicts the MFGP posterior
    at those θ. Returns observed and predicted arrays plus the 1σ/2σ/3σ
    coverage fractions (target Gaussian rates: 68.27 / 95.45 / 99.73 %).
    """
    test = generator.generate(
        n_trials=n_test_trials, n_events=n_test_events, seed=seed,
    )
    _, tgt = split_context_target(
        test, n_context=n_test_events // 2, seed=seed + 1,
    )
    assert test.theta is not None
    y_obs = tgt.labels.mean(axis=1).astype(float)
    mu, var = mfgp.predict(test.theta, fidelity=mfgp.n_fidelities - 1)
    sigma = np.sqrt(var)
    abs_diff = np.abs(y_obs - mu)
    coverage = {
        "1sigma": float((abs_diff <= 1.0 * sigma).mean()),
        "2sigma": float((abs_diff <= 2.0 * sigma).mean()),
        "3sigma": float((abs_diff <= 3.0 * sigma).mean()),
    }
    return {
        "theta": test.theta,
        "y_obs": y_obs,
        "mu": mu,
        "sigma": sigma,
        **coverage,
    }
