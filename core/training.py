"""CNP training loop, evaluation, and checkpoint I/O.

Each step samples a fresh batch from a :class:`PseudoDataGenerator`,
draws ``n_context`` uniformly from ``[CNPConfig.n_context_min,
CNPConfig.n_context_max]`` (the meta-learning paradigm), splits the
batch into context/target sets with :func:`split_context_target`, and
runs the standard forward/backward/step.

Periodically the loop evaluates **MAE between the predicted ``β`` and
the analytical ``p``** on a held-out batch — that's the criterion that
gates Phase 3.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

import torch.nn.functional as F

from core.surrogate_cnp import (
    ConditionalNeuralProcess,
    build_cnp,
    cnp_loss,
    split_context_target,
)
from data.pseudo_generator import PseudoDataGenerator
from schemas.config import CNPConfig, EncoderConfig, TrainingConfig
from schemas.data_models import InputMode, StandardBatch


# ---------------------------------------------------------------------------
# Training history.
# ---------------------------------------------------------------------------


class TrainingHistory(dict):
    """A plain dict with a few documented keys.

    Keys:

    * ``step`` — list of training-step indices.
    * ``loss`` — per-step training loss.
    * ``eval_step`` — steps at which evaluation ran (subset of ``step``).
    * ``eval_mae`` — MAE(β, p) at each eval step.
    """


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------


def train_cnp(
    cnp: ConditionalNeuralProcess,
    generator: PseudoDataGenerator,
    *,
    cnp_config: CNPConfig,
    training_config: TrainingConfig,
    progress_callback: Callable[[int, float], None] | None = None,
) -> TrainingHistory:
    """Train a CNP via context-target meta-learning on synthetic data."""
    rng = np.random.default_rng(training_config.seed)
    torch.manual_seed(training_config.seed)
    optimizer = torch.optim.Adam(cnp.parameters(), lr=training_config.learning_rate)

    history = TrainingHistory(step=[], loss=[], eval_step=[], eval_mae=[])

    n_events = training_config.n_events_per_trial
    n_ctx_min = max(1, cnp_config.n_context_min)
    n_ctx_max = min(cnp_config.n_context_max, n_events - 1)
    if n_ctx_max < n_ctx_min:
        raise ValueError(
            f"Effective n_context range is empty: min={n_ctx_min}, max={n_ctx_max}; "
            f"n_events_per_trial={n_events} must exceed n_context_min."
        )

    cnp.train()
    for step in range(training_config.n_steps):
        # Sample a fresh batch and a random context size for this step.
        batch = generator.generate(
            n_trials=training_config.batch_size,
            n_events=n_events,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        n_ctx = int(rng.integers(n_ctx_min, n_ctx_max + 1))
        ctx, tgt = split_context_target(
            batch, n_context=n_ctx, seed=int(rng.integers(0, 2**31 - 1))
        )

        out = cnp(ctx, tgt)
        x_target = torch.as_tensor(tgt.labels, dtype=torch.float32)
        loss = cnp_loss(out, x_target, n_mc_samples=training_config.n_mc_samples)

        optimizer.zero_grad()
        loss.backward()
        if training_config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(cnp.parameters(), training_config.grad_clip)
        optimizer.step()

        history["step"].append(step)
        history["loss"].append(float(loss.item()))

        if progress_callback is not None:
            progress_callback(step, float(loss.item()))

        if (
            training_config.eval_every > 0
            and (step + 1) % training_config.eval_every == 0
        ):
            mae = evaluate_mae(
                cnp,
                generator,
                batch_size=training_config.eval_batch_size,
                n_events=training_config.eval_n_events,
                n_context=(n_ctx_min + n_ctx_max) // 2,
                seed=42 + step,
            )
            history["eval_step"].append(step)
            history["eval_mae"].append(mae)
            cnp.train()

    return history


# ---------------------------------------------------------------------------
# Evaluation: MAE between predicted β and analytical p.
# ---------------------------------------------------------------------------


def evaluate_mae(
    cnp: ConditionalNeuralProcess,
    generator: PseudoDataGenerator,
    *,
    batch_size: int,
    n_events: int,
    n_context: int,
    seed: int,
) -> float:
    """Mean absolute error between predicted ``β`` and ground-truth ``p``.

    Generates a fresh batch (with ``seed`` so the evaluation set is
    reproducible across calls) and a single fixed context/target split.
    """
    cnp.eval()
    batch = generator.generate(n_trials=batch_size, n_events=n_events, seed=seed)
    n_ctx = max(1, min(n_context, n_events - 1))
    ctx, tgt = split_context_target(batch, n_context=n_ctx, seed=seed + 1)
    with torch.no_grad():
        beta = cnp.predict_beta(ctx, tgt).cpu().numpy()
    p = _truth_at_targets(generator, tgt)
    return float(np.abs(beta - p).mean())


def cnp_trial_predictive(
    cnp: ConditionalNeuralProcess,
    ctx_batch: StandardBatch,
    target_batch: StandardBatch,
    *,
    n_mc_samples: int = 200,
    include_aleatoric: bool = True,
) -> dict[str, np.ndarray]:
    """Trial-level predictive distribution over ``y = m/N``.

    Uses the CNP decoder's per-event Gaussian over the score ``β`` to
    derive a per-trial mean ``y_CNP`` and uncertainty ``σ_CNP``. The
    propagation has two pieces:

    * **Epistemic** (from the decoder's ``σ_NN``): MC-sample
      ``β_i = sigmoid(μ_i + softplus(log σ_i)·ε_i)`` for ``K`` independent
      noise draws, take the per-sample trial mean, then the std across
      samples. Captures how confidently the CNP knows the rate.
    * **Aleatoric** (Bernoulli sampling noise on ``y_raw=m/N``):
      ``√(ŷ(1-ŷ)/N)`` with ``ŷ = y_CNP``. Captures the irreducible noise
      that ``m/N`` has around any given rate. Set
      ``include_aleatoric=False`` to inspect epistemic alone (typically
      much tighter than ``y_raw`` fluctuations and useful as a
      decoder-calibration diagnostic).

    Returns a dict with keys ``y_cnp``, ``sigma_total``,
    ``sigma_epistemic``, ``sigma_aleatoric`` — all 1-D arrays of length B.
    """
    cnp.eval()
    with torch.no_grad():
        out = cnp(ctx_batch, target_batch)
        sigma = F.softplus(out.log_sigma)                  # [B, N_t]
        eps = torch.randn(
            n_mc_samples, *out.mu_logit.shape,
            dtype=out.mu_logit.dtype, device=out.mu_logit.device,
        )
        beta_samples = torch.sigmoid(
            out.mu_logit.unsqueeze(0) + sigma.unsqueeze(0) * eps
        )                                                  # [K, B, N_t]
        y_per_sample = beta_samples.mean(dim=2)            # [K, B]
    y_cnp = y_per_sample.mean(dim=0).cpu().numpy()
    sigma_epistemic = y_per_sample.std(dim=0).cpu().numpy()

    n_t = float(target_batch.n_events)
    sigma_aleatoric = np.sqrt(np.clip(y_cnp * (1.0 - y_cnp), 0.0, None) / n_t)

    if include_aleatoric:
        sigma_total = np.sqrt(sigma_epistemic**2 + sigma_aleatoric**2)
    else:
        sigma_total = sigma_epistemic.copy()

    return {
        "y_cnp": y_cnp,
        "sigma_total": sigma_total,
        "sigma_epistemic": sigma_epistemic,
        "sigma_aleatoric": sigma_aleatoric,
    }


def _truth_at_targets(
    generator: PseudoDataGenerator, target_batch: StandardBatch
) -> np.ndarray:
    """Compute analytical ``p`` for the events in ``target_batch``.

    Returns array of shape ``[B, N_t]`` matching ``target_batch.labels``.
    """
    truth = generator.truth
    B, N_t = target_batch.batch_size, target_batch.n_events

    if truth.mode is InputMode.FULL:
        # theta:[B, D_θ] broadcast to per-event by inserting an N-axis.
        return truth.evaluate(
            theta=target_batch.theta[:, None, :],
            phi=target_batch.phi,
        )
    if truth.mode is InputMode.EVENT_ONLY:
        return truth.evaluate(theta=None, phi=target_batch.phi)
    if truth.mode is InputMode.DESIGN_ONLY:
        p_per_trial = truth.evaluate(theta=target_batch.theta, phi=None)  # [B]
        return np.broadcast_to(p_per_trial[:, None], (B, N_t))
    raise ValueError(f"unknown mode {truth.mode}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Checkpoint I/O.
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    cnp: ConditionalNeuralProcess,
    *,
    encoder_config: EncoderConfig,
    dim_theta: int | None,
    dim_phi: int | None,
    history: TrainingHistory | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist model weights + everything needed to rebuild the architecture.

    The file is a single ``torch.save`` blob containing a state dict, the
    encoder config (round-tripped through ``model_dump``), the input dims,
    and optional history / arbitrary metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": cnp.state_dict(),
        "encoder_config": encoder_config.model_dump(),
        "dim_theta": dim_theta,
        "dim_phi": dim_phi,
        "history": dict(history) if history is not None else None,
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
) -> tuple[ConditionalNeuralProcess, dict[str, Any]]:
    """Inverse of :func:`save_checkpoint`. Returns ``(cnp, payload)``."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    encoder_config = EncoderConfig(**payload["encoder_config"])
    cnp = build_cnp(
        encoder_config,
        dim_theta=payload["dim_theta"],
        dim_phi=payload["dim_phi"],
    )
    cnp.load_state_dict(payload["model_state"])
    return cnp, payload
