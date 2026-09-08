"""Conditional Neural Process surrogate at the event level.

The CNP estimates a continuous score ``β ≈ p ∈ [0, 1]`` per event,
denoising the binary ``X`` labels into a smooth probability surface.

Architecture
------------
::

    encoder       : (z_θ, z_φ_i, X_i)            ──▶ r_i              [B, N_ctx, agg]
    aggregator    : mean(r_i, dim=event_axis=1)  ──▶ r_trial          [B, agg]
    decoder       : (r_trial, z_φ_target_j)      ──▶ (μ_j, log σ_j)   [B, N_tgt]

``z_θ`` and ``z_φ`` come from :class:`UniversalEncoder` so the universal-
input contract (``θ=None`` or ``φ=None`` → learnable null token) flows
through transparently.

For binary targets the two raw decoder channels are transformed with the
same logistic-normal moment approximation as the original RESuM code. Two
explicit objectives are provided: ``theory-truth`` implements the marginalized
Bernoulli likelihood, while ``practice-truth`` reproduces the legacy code's
Normal likelihood on binary observations.

CRITICAL: the aggregator reduces along **event axis 1**, never axis 0.
Reducing axis 0 mixes unrelated trials and silently destroys learning;
the forward asserts the post-aggregation batch dim matches input ``B``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.networks import UniversalEncoder, build_encoder
from schemas.config import EncoderConfig
from schemas.data_models import StandardBatch

CnpObjective = Literal["theory-truth", "practice-truth"]


# ---------------------------------------------------------------------------
# Sub-modules.
# ---------------------------------------------------------------------------


class ContextPointEncoder(nn.Module):
    """MLP from concat ``(z_θ, z_φ_i, X_i)`` to a per-context representation ``r_i``."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_dim = 2 * latent_dim + 1  # z_θ + z_φ + X
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(
        self,
        z_theta_per_event: torch.Tensor,  # [B, N, Z]
        z_phi: torch.Tensor,              # [B, N, Z]
        x: torch.Tensor,                  # [B, N]
    ) -> torch.Tensor:
        x_in = x.unsqueeze(-1)  # [B, N, 1]
        cat = torch.cat([z_theta_per_event, z_phi, x_in], dim=-1)
        return self.net(cat)


class CnpDecoder(nn.Module):
    """MLP from ``(r_trial, z_φ_target)`` to per-target ``(μ, log σ)``."""

    def __init__(
        self,
        latent_dim: int,
        agg_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_dim = agg_dim + latent_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        # Two outputs per target: μ_logit (unconstrained) and log σ_raw.
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        r_trial: torch.Tensor,            # [B, agg]
        z_phi_target: torch.Tensor,       # [B, N_t, Z]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N_t = z_phi_target.size(1)
        r_expanded = r_trial.unsqueeze(1).expand(-1, N_t, -1)
        cat = torch.cat([r_expanded, z_phi_target], dim=-1)
        out = self.net(cat)  # [B, N_t, 2]
        return out[..., 0], out[..., 1]


# ---------------------------------------------------------------------------
# Top-level CNP.
# ---------------------------------------------------------------------------


@dataclass
class CnpOutput:
    """Decoder outputs for one forward pass.

    ``mu_logit`` and ``log_sigma`` parameterize the Gaussian over the
    pre-sigmoid score. Use :func:`cnp_loss` for training; use
    :meth:`ConditionalNeuralProcess.predict_beta` for deterministic
    evaluation (sigmoid of ``mu_logit``).
    """

    mu_logit: torch.Tensor   # [B, N_t]
    log_sigma: torch.Tensor  # [B, N_t]


def resum_binary_moments(out: CnpOutput) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the binary predictive mean/scale used by original RESuM.

    This intentionally mirrors ``sigmoid_expectation`` followed by the
    decoder's final ``softplus`` in the reference implementation. The second
    decoder channel is raw despite its historical ``log_sigma`` name.
    """
    bounded_sigma = 0.1 + 0.9 * F.softplus(out.log_sigma)
    divisor = torch.sqrt(1.0 + 3.0 / (np.pi**2) * bounded_sigma**2)
    divisor = torch.clamp(divisor, min=1e-4)
    mean = torch.sigmoid(out.mu_logit / divisor)
    variance_proxy = mean * (1.0 - mean) * (1.0 - 1.0 / divisor)
    scale = F.softplus(variance_proxy) + 1e-6
    return mean, scale


class ConditionalNeuralProcess(nn.Module):
    """Universal-input CNP composed of encoder + context-point MLP + decoder."""

    def __init__(
        self,
        encoder: UniversalEncoder,
        context_encoder: ContextPointEncoder,
        decoder: CnpDecoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.context_encoder = context_encoder
        self.decoder = decoder

    def _encode_per_event(
        self, batch: StandardBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the universal encoder and broadcast ``z_θ`` to per-event."""
        z_theta, z_phi = self.encoder(batch)            # [B, Z], [B, N, Z]
        N = z_phi.size(1)
        z_theta_per_event = z_theta.unsqueeze(1).expand(-1, N, -1)  # [B, N, Z]
        return z_theta_per_event, z_phi

    def aggregate(self, ctx_batch: StandardBatch) -> torch.Tensor:
        """Encode the context and return the per-trial summary ``r_trial``."""
        B = ctx_batch.batch_size
        z_theta_pe, z_phi = self._encode_per_event(ctx_batch)
        x_ctx = torch.as_tensor(
            ctx_batch.labels, dtype=z_phi.dtype, device=z_phi.device
        )
        r_per_event = self.context_encoder(z_theta_pe, z_phi, x_ctx)  # [B, N_ctx, agg]
        r_trial = r_per_event.mean(dim=1)                              # [B, agg]
        # Hard guard against reducing the wrong axis. If this ever fires,
        # the loss will still go down but predictions will be garbage.
        assert r_trial.shape[0] == B, (
            f"aggregator collapsed wrong axis: got {tuple(r_trial.shape)}, "
            f"expected (B={B}, *)"
        )
        return r_trial

    def forward(
        self,
        ctx_batch: StandardBatch,
        target_batch: StandardBatch,
    ) -> CnpOutput:
        """Encode context, aggregate, and decode predictions for the targets.

        ``ctx_batch`` and ``target_batch`` must have the same ``B`` and the
        same modality. They typically share ``θ`` and split a single trial's
        events into context vs target.
        """
        if ctx_batch.batch_size != target_batch.batch_size:
            raise ValueError(
                f"ctx B={ctx_batch.batch_size} != target B={target_batch.batch_size}"
            )
        if ctx_batch.mode is not target_batch.mode:
            raise ValueError(
                f"ctx mode {ctx_batch.mode} != target mode {target_batch.mode}"
            )

        r_trial = self.aggregate(ctx_batch)                   # [B, agg]
        _, z_phi_tgt = self._encode_per_event(target_batch)   # [B, N_t, Z]
        mu_logit, log_sigma = self.decoder(r_trial, z_phi_tgt)
        return CnpOutput(mu_logit=mu_logit, log_sigma=log_sigma)

    def predict_beta(
        self,
        ctx_batch: StandardBatch,
        target_batch: StandardBatch,
    ) -> torch.Tensor:
        """Deterministic ``β = sigmoid(μ_logit)`` for evaluation / plotting.

        Returns the mean of the original RESuM binary predictive transform.
        """
        out = self(ctx_batch, target_batch)
        mean, _ = resum_binary_moments(out)
        return mean


# ---------------------------------------------------------------------------
# Loss.
# ---------------------------------------------------------------------------


def cnp_loss(
    out: CnpOutput,
    x_target: torch.Tensor,
    *,
    eps: float = 1e-6,
    objective: CnpObjective = "theory-truth",
) -> torch.Tensor:
    """Dispatch to one of the two documented RESuM truth objectives.

    Both objectives are analytic and do not require Monte Carlo sampling.
    """
    if objective == "theory-truth":
        return theory_truth_loss(out, x_target, eps=eps)
    if objective == "practice-truth":
        return practice_truth_loss(out, x_target)
    raise ValueError(
        "objective must be 'theory-truth' or 'practice-truth'; "
        f"got {objective!r}"
    )


def theory_truth_loss(
    out: CnpOutput,
    x_target: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Paper-level marginalized Bernoulli negative log likelihood.

    For binary ``x``, integrating ``Bernoulli(x | p)`` over the predictive
    distribution depends only on ``E[p]``. ``resum_binary_moments`` supplies
    the reference logistic-normal approximation ``p_bar`` to that mean, so

    ``L = -mean[x log(p_bar) + (1-x) log(1-p_bar)]``.

    This is the probability-theory interpretation and the recommended default.
    """
    mean, _ = resum_binary_moments(out)
    mean = mean.clamp(eps, 1.0 - eps)
    return F.binary_cross_entropy(mean, x_target)


def practice_truth_loss(out: CnpOutput, x_target: torch.Tensor) -> torch.Tensor:
    """Literal negative log likelihood executed by the legacy RESuM code.

    The old implementation treats binary ``X`` as observations of a Normal
    distribution parameterized by the transformed mean and scale. This is
    retained for reproducibility even though it is not a Bernoulli likelihood.
    """
    mean, scale = resum_binary_moments(out)
    return -torch.distributions.Normal(mean, scale).log_prob(x_target).mean()


# ---------------------------------------------------------------------------
# Factory.
# ---------------------------------------------------------------------------


def build_cnp(
    encoder_config: EncoderConfig,
    dim_theta: int | None,
    dim_phi: int | None,
    *,
    aggregator_dim: int | None = None,
    context_hidden_dims: list[int] | None = None,
    decoder_hidden_dims: list[int] | None = None,
) -> ConditionalNeuralProcess:
    """Build a :class:`ConditionalNeuralProcess` from a typed config.

    The encoder + context + decoder MLPs all reuse ``encoder_config.hidden_dims``
    by default; pass overrides for finer control. ``aggregator_dim`` defaults
    to ``encoder_config.latent_dim`` so the per-trial summary is the same
    width as the per-event latents.
    """
    encoder = build_encoder(encoder_config, dim_theta, dim_phi)
    Z = encoder_config.latent_dim
    agg_dim = aggregator_dim or Z
    ctx_hidden = list(context_hidden_dims or encoder_config.hidden_dims)
    dec_hidden = list(decoder_hidden_dims or encoder_config.hidden_dims)
    context_encoder = ContextPointEncoder(
        latent_dim=Z,
        hidden_dims=ctx_hidden,
        out_dim=agg_dim,
        dropout=encoder_config.dropout,
    )
    decoder = CnpDecoder(
        latent_dim=Z,
        agg_dim=agg_dim,
        hidden_dims=dec_hidden,
        dropout=encoder_config.dropout,
    )
    return ConditionalNeuralProcess(
        encoder=encoder,
        context_encoder=context_encoder,
        decoder=decoder,
    )


# ---------------------------------------------------------------------------
# Context / target split (training-time helper).
# ---------------------------------------------------------------------------


def split_context_target(
    batch: StandardBatch,
    n_context: int,
    *,
    seed: int | None = None,
) -> tuple[StandardBatch, StandardBatch]:
    """Per-trial random partition of ``N`` events into context and target.

    Each trial in the input batch has ``N`` events; this function permutes
    them independently per trial and splits at ``n_context``. Both returned
    batches share the same ``θ`` (it's a trial-level parameter) but disjoint
    ``φ`` and ``labels`` slices.

    ``n_context`` must lie in ``[1, N - 1]`` so both sets are non-empty.
    """
    B, N = batch.batch_size, batch.n_events
    if not 1 <= n_context <= N - 1:
        raise ValueError(
            f"n_context must be in [1, N-1] = [1, {N - 1}]; got {n_context}"
        )

    rng = np.random.default_rng(seed)
    perm = np.stack([rng.permutation(N) for _ in range(B)], axis=0)  # [B, N]
    ctx_idx = perm[:, :n_context]
    tgt_idx = perm[:, n_context:]

    def _gather(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        # arr: [B, N] or [B, N, D]; idx: [B, k]
        if arr.ndim == 2:
            return np.take_along_axis(arr, idx, axis=1)
        if arr.ndim == 3:
            return np.take_along_axis(arr, idx[..., None], axis=1)
        raise ValueError(f"unexpected arr.ndim={arr.ndim}")

    phi_ctx = _gather(batch.phi, ctx_idx) if batch.phi is not None else None
    phi_tgt = _gather(batch.phi, tgt_idx) if batch.phi is not None else None
    labels_ctx = _gather(batch.labels, ctx_idx)
    labels_tgt = _gather(batch.labels, tgt_idx)

    ctx = StandardBatch(
        mode=batch.mode, theta=batch.theta, phi=phi_ctx, labels=labels_ctx
    )
    tgt = StandardBatch(
        mode=batch.mode, theta=batch.theta, phi=phi_tgt, labels=labels_tgt
    )
    return ctx, tgt
