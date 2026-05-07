"""Glue between a trained CNP and the MFGP layer.

This module is the only place where torch and GPy meet — and only at
the boundary: ``cnp_trial_predictive`` produces numpy arrays from the
torch CNP, and those arrays are then fed to :class:`MultiFidelityGP`.
The two libraries never share imports inside any single object.

Two parallel APIs:

* **Batch-based** (canonical for users with their own data) —
  :func:`prepare_mfgp_datasets_from_batches`,
  :func:`evaluate_mfgp_coverage_from_batch`. Take pre-built
  ``StandardBatch`` objects, return numpy arrays / metric dicts. No
  synthetic-generator dependency.
* **Generator-based** (convenient for the synthetic validation
  pipeline) — :func:`prepare_mfgp_datasets`,
  :func:`evaluate_mfgp_coverage`. Thin wrappers that draw fresh
  batches from a :class:`PseudoDataGenerator` and delegate to the
  batch-based helpers.

The 3-fidelity MFGP architecture matches the paper's recursive recipe::

    f₀ (LF)  = β̄(θ) from low-fidelity trials   (cheap, broad coverage)
    f₁ (MF)  = ρ₀·f₀ + δ₀(θ) ≈ β̄(θ) from HF trials
    f₂ (HF)  = ρ₁·f₁ + δ₁(θ) ≈ y_raw(θ) from HF trials   ← target

EVENT_ONLY scenarios have no ``θ`` and therefore can't drive an MFGP;
the entry points reject them up front.
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
from schemas.data_models import InputMode, StandardBatch


# ---------------------------------------------------------------------------
# Batch-based entry points (canonical user-facing API).
# ---------------------------------------------------------------------------


def prepare_mfgp_datasets_from_batches(
    cnp: ConditionalNeuralProcess,
    lf_batch: StandardBatch,
    hf_batch: StandardBatch,
    *,
    n_lf_context: int | None = None,
    n_hf_context: int | None = None,
    n_mc_samples: int = 50,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build LF / MF / HF datasets from a trained CNP and two pre-built batches.

    Use this entry point when you have your own simulation data already
    bundled into :class:`StandardBatch` objects.

    Parameters
    ----------
    cnp
        A trained Conditional Neural Process.
    lf_batch
        Low-fidelity batch — many trials with fewer events per trial.
        Mode must be ``FULL`` or ``DESIGN_ONLY`` (must contain ``θ``).
    hf_batch
        High-fidelity batch — fewer trials with more events per trial.
        Same modality requirement as ``lf_batch``. Both ``β̄`` and
        ``y_raw`` are aggregated over the *target* half of this batch.
    n_lf_context, n_hf_context
        Number of events used as the CNP's context window per trial.
        ``None`` (default) → half of the batch's event count.
    n_mc_samples
        MC samples for the CNP predictive distribution per trial.
    seed
        Reproducibility seed for the context/target split RNG.

    Returns
    -------
    dict
        Five numpy arrays keyed as below; pass directly to
        :func:`fit_mfgp_three_fidelity`.

        ===========  ============================  ===============================
        key          shape                         meaning
        ===========  ============================  ===============================
        X_lf         ``(n_lf_trials, D_θ)``        per-trial θ for the LF dataset
        Y_lf_cnp     ``(n_lf_trials, 1)``          β̄(θ) per LF trial
        X_hf         ``(n_hf_trials, D_θ)``        per-trial θ for the HF dataset
        Y_hf_cnp     ``(n_hf_trials, 1)``          β̄(θ) per HF trial
        Y_hf_raw     ``(n_hf_trials, 1)``          ``m/N`` per HF trial
        ===========  ============================  ===============================

    Notes
    -----
    Per the meta-learning paradigm, ``β̄`` for a trial is computed by
    splitting the trial's events into a context half and a target half
    (50/50 by default), running the CNP, and averaging predicted ``β``
    over the target events. ``y_raw`` averages the binary ``X`` over
    the same target events so the two metrics aggregate identically.
    """
    _reject_event_only(lf_batch, "lf_batch")
    _reject_event_only(hf_batch, "hf_batch")

    n_lf_ctx = n_lf_context if n_lf_context is not None else lf_batch.n_events // 2
    n_hf_ctx = n_hf_context if n_hf_context is not None else hf_batch.n_events // 2

    lf_ctx, lf_tgt = split_context_target(lf_batch, n_context=n_lf_ctx, seed=seed + 1)
    pred_lf = cnp_trial_predictive(cnp, lf_ctx, lf_tgt, n_mc_samples=n_mc_samples)

    hf_ctx, hf_tgt = split_context_target(hf_batch, n_context=n_hf_ctx, seed=seed + 101)
    pred_hf = cnp_trial_predictive(cnp, hf_ctx, hf_tgt, n_mc_samples=n_mc_samples)
    y_hf_raw = hf_tgt.labels.mean(axis=1).astype(float)

    assert lf_batch.theta is not None and hf_batch.theta is not None
    return {
        "X_lf": lf_batch.theta,
        "Y_lf_cnp": pred_lf["y_cnp"].reshape(-1, 1),
        "X_hf": hf_batch.theta,
        "Y_hf_cnp": pred_hf["y_cnp"].reshape(-1, 1),
        "Y_hf_raw": y_hf_raw.reshape(-1, 1),
    }


def evaluate_mfgp_coverage_from_batch(
    mfgp: MultiFidelityGP,
    cnp: ConditionalNeuralProcess,
    holdout_hf_batch: StandardBatch,
    *,
    n_context: int | None = None,
    seed: int = 12345,
) -> dict[str, np.ndarray | float]:
    """Held-out HF coverage for a fitted MFGP, from a pre-built batch.

    Use this with your own held-out HF observations to verify the MFGP
    posterior is well-calibrated against ``y_raw = m/N``.

    Parameters
    ----------
    mfgp
        A fitted :class:`MultiFidelityGP` (typically from
        :func:`fit_mfgp_three_fidelity`).
    cnp
        The same CNP used to build the training datasets — only used
        here to keep the context/target split convention consistent.
    holdout_hf_batch
        High-fidelity batch held out from training. Must have
        ``θ`` (mode is ``FULL`` or ``DESIGN_ONLY``).
    n_context
        Context-window size for the per-trial split. ``None`` (default)
        → half of the batch's event count.
    seed
        Reproducibility seed for the split.

    Returns
    -------
    dict
        Diagnostic arrays plus three coverage fractions (target rates:
        68.27 / 95.45 / 99.73 %).

        ============  ============================  ===========================
        key           shape / type                  meaning
        ============  ============================  ===========================
        theta         ``(n_test, D_θ)``             held-out θ
        y_obs         ``(n_test,)``                 observed ``m/N``
        mu            ``(n_test,)``                 MFGP posterior mean
        sigma         ``(n_test,)``                 MFGP posterior std
        1sigma        float                         fraction within ±1σ
        2sigma        float                         fraction within ±2σ
        3sigma        float                         fraction within ±3σ
        ============  ============================  ===========================
    """
    _reject_event_only(holdout_hf_batch, "holdout_hf_batch")
    del cnp  # signature kept for parity / future per-trial diagnostics
    n_ctx = n_context if n_context is not None else holdout_hf_batch.n_events // 2

    _, tgt = split_context_target(holdout_hf_batch, n_context=n_ctx, seed=seed + 1)
    assert holdout_hf_batch.theta is not None
    y_obs = tgt.labels.mean(axis=1).astype(float)

    mu, var = mfgp.predict(holdout_hf_batch.theta, fidelity=mfgp.n_fidelities - 1)
    sigma = np.sqrt(var)
    abs_diff = np.abs(y_obs - mu)
    return {
        "theta": holdout_hf_batch.theta,
        "y_obs": y_obs,
        "mu": mu,
        "sigma": sigma,
        "1sigma": float((abs_diff <= 1.0 * sigma).mean()),
        "2sigma": float((abs_diff <= 2.0 * sigma).mean()),
        "3sigma": float((abs_diff <= 3.0 * sigma).mean()),
    }


def fit_mfgp_three_fidelity(
    data: dict[str, np.ndarray],
    *,
    kernel: str = "rbf",
    n_restarts: int = 5,
    verbose: bool = False,
) -> MultiFidelityGP:
    """Fit a 3-fidelity MFGP on ``(LF β̄, HF β̄, HF y_raw)``.

    Takes the dict produced by
    :func:`prepare_mfgp_datasets_from_batches` (or the synthetic-data
    wrapper :func:`prepare_mfgp_datasets`).
    """
    dim_theta = data["X_hf"].shape[1]
    X_list = [data["X_lf"], data["X_hf"], data["X_hf"]]
    Y_list = [data["Y_lf_cnp"], data["Y_hf_cnp"], data["Y_hf_raw"]]
    return MultiFidelityGP(
        n_fidelities=3, dim_theta=dim_theta, kernel=kernel,
    ).fit(X_list, Y_list, n_restarts=n_restarts, verbose=verbose)


# ---------------------------------------------------------------------------
# Generator-based wrappers (synthetic-validation convenience).
# ---------------------------------------------------------------------------


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
    """Synthetic-data convenience wrapper around
    :func:`prepare_mfgp_datasets_from_batches`.

    Draws fresh LF and HF batches from ``generator`` at the given
    sizes and delegates. Use this in validation / benchmarking scripts
    where the truth is known; for real data, build batches yourself
    and call :func:`prepare_mfgp_datasets_from_batches` directly.
    """
    if generator.mode is InputMode.EVENT_ONLY:
        raise ValueError(
            "MFGP needs θ as input; generator mode=EVENT_ONLY has no θ."
        )

    lf_batch = generator.generate(
        n_trials=n_lf_trials, n_events=n_lf_events, seed=seed,
    )
    hf_batch = generator.generate(
        n_trials=n_hf_trials, n_events=n_hf_events, seed=seed + 100,
    )
    return prepare_mfgp_datasets_from_batches(
        cnp, lf_batch, hf_batch,
        n_lf_context=n_lf_events // 2,
        n_hf_context=n_hf_events // 2,
        n_mc_samples=n_mc_samples,
        seed=seed,
    )


def evaluate_mfgp_coverage(
    mfgp: MultiFidelityGP,
    cnp: ConditionalNeuralProcess,
    generator: PseudoDataGenerator,
    *,
    n_test_trials: int = 100,
    n_test_events: int = 128,
    seed: int = 12345,
) -> dict[str, np.ndarray | float]:
    """Synthetic-data convenience wrapper around
    :func:`evaluate_mfgp_coverage_from_batch`.

    Generates ``n_test_trials`` fresh held-out trials and delegates.
    """
    test = generator.generate(
        n_trials=n_test_trials, n_events=n_test_events, seed=seed,
    )
    return evaluate_mfgp_coverage_from_batch(
        mfgp, cnp, test,
        n_context=n_test_events // 2,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Internal.
# ---------------------------------------------------------------------------


def _reject_event_only(batch: StandardBatch, label: str) -> None:
    if batch.mode is InputMode.EVENT_ONLY:
        raise ValueError(
            f"{label} must contain θ (mode FULL or DESIGN_ONLY); "
            f"got mode={batch.mode.value}"
        )
