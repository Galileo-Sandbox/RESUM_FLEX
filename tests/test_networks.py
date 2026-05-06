"""Phase 2 acceptance gate: encoder shapes & null-embedding identity.

The hard requirement is that any input absence (``θ=None`` or
``φ=None``) maps to **the exact same** learnable null-token vector. We
test that bit-for-bit, not approximately.
"""

from __future__ import annotations

import pytest
import torch

from core.networks import MLPEncoder, build_encoder
from data.pseudo_generator import PseudoDataGenerator, for_scenario
from schemas.config import EncoderConfig

ALL_SCENARIOS = list(PseudoDataGenerator.SCENARIOS.keys())


def _make_config(latent_dim: int = 16) -> EncoderConfig:
    return EncoderConfig(
        type="mlp",
        latent_dim=latent_dim,
        hidden_dims=[32, 32],
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Dimension matrix: every scenario produces the same output shapes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_SCENARIOS)
def test_encoder_shapes_uniform(name: str) -> None:
    gen = for_scenario(name)
    batch = gen.generate(n_trials=4, n_events=16)
    enc = build_encoder(_make_config(latent_dim=16), gen.dim_theta, gen.dim_phi)
    enc.eval()

    with torch.no_grad():
        z_theta, z_phi = enc(batch)

    assert z_theta.shape == (4, 16)
    assert z_phi.shape == (4, 16, 16)
    assert torch.isfinite(z_theta).all()
    assert torch.isfinite(z_phi).all()


# ---------------------------------------------------------------------------
# Null-token identity.
# ---------------------------------------------------------------------------


def test_event_only_z_theta_equals_null_token() -> None:
    """Every row of z_theta must be bit-for-bit equal to theta_null."""
    gen = for_scenario("S5")
    batch = gen.generate(n_trials=8, n_events=16)
    enc = build_encoder(_make_config(), dim_theta=1, dim_phi=1)
    enc.eval()

    with torch.no_grad():
        z_theta, _ = enc(batch)
        null = enc.theta_null

    for b in range(batch.batch_size):
        torch.testing.assert_close(z_theta[b], null, atol=0.0, rtol=0.0)


def test_design_only_z_phi_equals_null_token() -> None:
    """Every (b, i) row of z_phi must be bit-for-bit equal to phi_null."""
    gen = for_scenario("S7")
    batch = gen.generate(n_trials=4, n_events=12)
    enc = build_encoder(_make_config(), dim_theta=1, dim_phi=1)
    enc.eval()

    with torch.no_grad():
        _, z_phi = enc(batch)
        null = enc.phi_null

    for b in range(batch.batch_size):
        for i in range(batch.n_events):
            torch.testing.assert_close(z_phi[b, i], null, atol=0.0, rtol=0.0)


def test_null_token_cluster_collapses_to_singleton() -> None:
    """All B output vectors collide at one point in latent space."""
    gen = for_scenario("S5")
    batch = gen.generate(n_trials=64, n_events=4)
    enc = build_encoder(_make_config(latent_dim=8), dim_theta=1, dim_phi=1)
    enc.eval()

    with torch.no_grad():
        z_theta, _ = enc(batch)

    # Pairwise distance between any two rows is exactly zero.
    diff = z_theta - z_theta[0]
    assert torch.all(diff == 0.0)


# ---------------------------------------------------------------------------
# Cross-mode flexibility.
# ---------------------------------------------------------------------------


def test_full_encoder_accepts_partial_batches() -> None:
    """An encoder built for FULL (dim_theta=1, dim_phi=1) handles S1/S5/S7 alike."""
    enc = build_encoder(_make_config(), dim_theta=1, dim_phi=1)
    enc.eval()

    s1 = for_scenario("S1").generate(n_trials=4, n_events=8)
    s5 = for_scenario("S5").generate(n_trials=4, n_events=8)
    s7 = for_scenario("S7").generate(n_trials=4, n_events=8)

    with torch.no_grad():
        for b in (s1, s5, s7):
            zt, zp = enc(b)
            assert zt.shape == (4, enc.latent_dim)
            assert zp.shape == (4, 8, enc.latent_dim)


# ---------------------------------------------------------------------------
# Misuse / errors.
# ---------------------------------------------------------------------------


def test_missing_theta_encoder_with_theta_in_batch_raises() -> None:
    """Building dim_theta=None but feeding a batch with θ should fail loudly."""
    enc = build_encoder(_make_config(), dim_theta=None, dim_phi=1)
    s1 = for_scenario("S1").generate(n_trials=2, n_events=4)
    with pytest.raises(ValueError, match=r"θ-encoder"):
        enc(s1)


def test_missing_phi_encoder_with_phi_in_batch_raises() -> None:
    enc = build_encoder(_make_config(), dim_theta=1, dim_phi=None)
    s5 = for_scenario("S5").generate(n_trials=2, n_events=4)
    with pytest.raises(ValueError, match=r"φ-encoder"):
        enc(s5)


def test_unknown_encoder_type_raises() -> None:
    """Pydantic accepts ``type='transformer'`` but the registry doesn't yet."""
    config = EncoderConfig(type="transformer", latent_dim=4, hidden_dims=[8], dropout=0.0)
    with pytest.raises(ValueError, match="Unknown encoder type"):
        build_encoder(config, 1, 1)


def test_mlp_rejects_zero_in_dim() -> None:
    with pytest.raises(ValueError):
        MLPEncoder(in_dim=0, hidden_dims=[8], out_dim=4)


# ---------------------------------------------------------------------------
# Gradient flow through null tokens.
# ---------------------------------------------------------------------------


def test_grad_flows_to_theta_null() -> None:
    gen = for_scenario("S5")
    batch = gen.generate(n_trials=2, n_events=4)
    enc = build_encoder(_make_config(), dim_theta=1, dim_phi=1)

    z_theta, _ = enc(batch)
    loss = z_theta.sum()
    loss.backward()

    assert enc.theta_null.grad is not None
    assert enc.theta_null.grad.shape == enc.theta_null.shape
    # All B rows contribute, so each null-token component receives gradient B.
    assert torch.allclose(
        enc.theta_null.grad,
        torch.full_like(enc.theta_null, float(batch.batch_size)),
    )


def test_grad_flows_to_phi_null() -> None:
    gen = for_scenario("S7")
    batch = gen.generate(n_trials=3, n_events=5)
    enc = build_encoder(_make_config(), dim_theta=1, dim_phi=1)

    _, z_phi = enc(batch)
    z_phi.sum().backward()

    assert enc.phi_null.grad is not None
    expected = float(batch.batch_size * batch.n_events)
    assert torch.allclose(enc.phi_null.grad, torch.full_like(enc.phi_null, expected))
