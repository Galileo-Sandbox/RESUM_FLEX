"""PyTorch encoder modules.

This is the only place in the project that imports ``torch`` — by
design. The data engine, schemas, and GP/MFGP code stay numpy-native so
the deep-learning and Gaussian-process stacks don't get tangled.

The :class:`UniversalEncoder` maps a :class:`StandardBatch` into two
latent tensors:

    z_θ : [B, Z]      — one latent per design configuration
    z_φ : [B, N, Z]   — one latent per event

When the batch is missing one of the two inputs (``mode=EVENT_ONLY``
drops θ; ``mode=DESIGN_ONLY`` drops φ), the corresponding output is
filled with broadcast copies of a learnable "null token" parameter
(``theta_null`` / ``phi_null``). Every absent input therefore maps to
the **exact same** vector — verifiable by the Phase 2 latent-space plot.

The inner encoder is pluggable via :data:`_INNER_ENCODER_FACTORIES` so a
Transformer can replace the MLP without touching the universal /
null-handling logic.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from schemas.config import EncoderConfig
from schemas.data_models import StandardBatch


# ---------------------------------------------------------------------------
# Inner encoders.
# ---------------------------------------------------------------------------


class MLPEncoder(nn.Module):
    """Per-feature MLP: ``D → hidden_dims → Z`` with GELU activations."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Registry of inner-encoder factories keyed by ``EncoderConfig.type``.
# Add new entries here when implementing Transformer / etc.
_INNER_ENCODER_FACTORIES: dict[str, Callable[[int, EncoderConfig], nn.Module]] = {
    "mlp": lambda in_dim, cfg: MLPEncoder(
        in_dim=in_dim,
        hidden_dims=list(cfg.hidden_dims),
        out_dim=cfg.latent_dim,
        dropout=cfg.dropout,
    ),
}


# ---------------------------------------------------------------------------
# Universal encoder.
# ---------------------------------------------------------------------------


class UniversalEncoder(nn.Module):
    """θ-encoder + φ-encoder + learnable null embeddings."""

    def __init__(
        self,
        theta_encoder: nn.Module | None,
        phi_encoder: nn.Module | None,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.theta_encoder = theta_encoder
        self.phi_encoder = phi_encoder
        self.latent_dim = latent_dim

        # Null tokens are always present so a forward never fails to
        # produce an output of the right shape, even when the batch
        # itself omits the corresponding component.
        self.theta_null = nn.Parameter(torch.empty(latent_dim))
        self.phi_null = nn.Parameter(torch.empty(latent_dim))
        nn.init.normal_(self.theta_null, std=0.02)
        nn.init.normal_(self.phi_null, std=0.02)

    @property
    def dim_theta(self) -> int | None:
        """Input dim the θ-encoder was built for (``None`` → null only)."""
        if self.theta_encoder is None:
            return None
        return getattr(self.theta_encoder, "in_dim", None)

    @property
    def dim_phi(self) -> int | None:
        if self.phi_encoder is None:
            return None
        return getattr(self.phi_encoder, "in_dim", None)

    def forward(self, batch: StandardBatch) -> tuple[torch.Tensor, torch.Tensor]:
        B = batch.batch_size
        N = batch.n_events
        device = self.theta_null.device
        dtype = self.theta_null.dtype

        # ---- z_theta: [B, Z] ----
        if batch.theta is not None:
            if self.theta_encoder is None:
                raise ValueError(
                    "Encoder was built without a θ-encoder, but the batch "
                    "provides θ. Build the encoder with `dim_theta` matching "
                    "the batch's θ dimension."
                )
            theta_t = torch.as_tensor(batch.theta, dtype=dtype, device=device)
            z_theta = self.theta_encoder(theta_t)
        else:
            # Broadcast the learnable null token to [B, Z]. expand() returns
            # a view; backward correctly accumulates grads to theta_null.
            z_theta = self.theta_null.unsqueeze(0).expand(B, -1)

        # ---- z_phi: [B, N, Z] ----
        if batch.phi is not None:
            if self.phi_encoder is None:
                raise ValueError(
                    "Encoder was built without a φ-encoder, but the batch "
                    "provides φ. Build the encoder with `dim_phi` matching "
                    "the batch's φ dimension."
                )
            phi_t = torch.as_tensor(batch.phi, dtype=dtype, device=device)
            # Flatten the (B, N) leading axes so the per-event MLP can run
            # in one matmul, then restore the structure.
            z_phi_flat = self.phi_encoder(phi_t.reshape(B * N, -1))
            z_phi = z_phi_flat.reshape(B, N, self.latent_dim)
        else:
            z_phi = self.phi_null.unsqueeze(0).unsqueeze(0).expand(B, N, -1)

        return z_theta, z_phi


# ---------------------------------------------------------------------------
# Factory.
# ---------------------------------------------------------------------------


def build_encoder(
    config: EncoderConfig,
    dim_theta: int | None,
    dim_phi: int | None,
) -> UniversalEncoder:
    """Construct a :class:`UniversalEncoder` from a typed config.

    A non-``None`` input dimension produces an inner encoder via the
    factory registered for ``config.type``. A ``None`` dimension yields
    no inner encoder for that side, so only the null token is used.
    """
    if config.type not in _INNER_ENCODER_FACTORIES:
        raise ValueError(
            f"Unknown encoder type {config.type!r}; "
            f"registered factories: {sorted(_INNER_ENCODER_FACTORIES)}"
        )
    factory = _INNER_ENCODER_FACTORIES[config.type]
    theta_encoder = None if dim_theta is None else factory(dim_theta, config)
    phi_encoder = None if dim_phi is None else factory(dim_phi, config)
    return UniversalEncoder(
        theta_encoder=theta_encoder,
        phi_encoder=phi_encoder,
        latent_dim=config.latent_dim,
    )
