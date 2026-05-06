"""Phase 3 model + loss tests.

These cover the contract before training: shapes are right, the loss is
finite & sensible, the aggregator collapses the correct axis, gradients
reach the null tokens, and the loss really is Bernoulli NLL (not BCE) —
verified by checking the loss at known operating points.

End-to-end MAE vs. ground-truth ``p`` lives in a separate test
(see :mod:`tests.test_cnp_training`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from core.surrogate_cnp import (
    CnpOutput,
    build_cnp,
    cnp_loss,
    split_context_target,
)
from data.pseudo_generator import PseudoDataGenerator, for_scenario
from schemas.config import EncoderConfig

ALL_SCENARIOS = list(PseudoDataGenerator.SCENARIOS.keys())


def _config(latent_dim: int = 16) -> EncoderConfig:
    return EncoderConfig(
        type="mlp", latent_dim=latent_dim, hidden_dims=[32, 32], dropout=0.0
    )


# ---------------------------------------------------------------------------
# Shape contract.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_SCENARIOS)
def test_forward_shapes(name: str) -> None:
    gen = for_scenario(name)
    batch = gen.generate(n_trials=4, n_events=32)
    ctx, tgt = split_context_target(batch, n_context=16, seed=0)
    cnp = build_cnp(_config(), gen.dim_theta, gen.dim_phi)
    cnp.eval()
    with torch.no_grad():
        out = cnp(ctx, tgt)
    assert out.mu_logit.shape == (4, 16)
    assert out.log_sigma.shape == (4, 16)
    assert torch.isfinite(out.mu_logit).all()
    assert torch.isfinite(out.log_sigma).all()


def test_predict_beta_in_unit_interval() -> None:
    gen = for_scenario("S1")
    batch = gen.generate(n_trials=4, n_events=32)
    ctx, tgt = split_context_target(batch, n_context=16, seed=0)
    cnp = build_cnp(_config(), gen.dim_theta, gen.dim_phi)
    cnp.eval()
    with torch.no_grad():
        beta = cnp.predict_beta(ctx, tgt)
    assert beta.shape == (4, 16)
    assert torch.all((beta >= 0.0) & (beta <= 1.0))


# ---------------------------------------------------------------------------
# Aggregator-axis guard.
# ---------------------------------------------------------------------------


def test_aggregator_collapses_event_axis_not_batch() -> None:
    """Two batches with different B but same per-trial structure produce
    outputs whose first dim equals B (event axis collapsed)."""
    gen = for_scenario("S5")
    cnp = build_cnp(_config(), gen.dim_theta, gen.dim_phi)
    cnp.eval()
    for B in (3, 7, 11):
        batch = gen.generate(n_trials=B, n_events=20)
        ctx, tgt = split_context_target(batch, n_context=10, seed=0)
        with torch.no_grad():
            out = cnp(ctx, tgt)
        assert out.mu_logit.shape[0] == B


def test_ctx_target_batch_size_mismatch_raises() -> None:
    gen = for_scenario("S1")
    a = gen.generate(n_trials=4, n_events=8)
    b = gen.generate(n_trials=5, n_events=8)
    ctx_a, _ = split_context_target(a, n_context=4)
    _, tgt_b = split_context_target(b, n_context=4)
    cnp = build_cnp(_config(), gen.dim_theta, gen.dim_phi)
    with pytest.raises(ValueError, match="ctx B"):
        cnp(ctx_a, tgt_b)


def test_ctx_target_mode_mismatch_raises() -> None:
    a = for_scenario("S1").generate(n_trials=4, n_events=8)
    b = for_scenario("S5").generate(n_trials=4, n_events=8)
    ctx_a, _ = split_context_target(a, n_context=4)
    _, tgt_b = split_context_target(b, n_context=4)
    cnp = build_cnp(_config(), 1, 1)
    with pytest.raises(ValueError, match="mode"):
        cnp(ctx_a, tgt_b)


# ---------------------------------------------------------------------------
# Loss form: Bernoulli NLL, not BCE.
# ---------------------------------------------------------------------------


def test_loss_zero_when_beta_predicts_x_exactly() -> None:
    """When μ_logit is huge in the right direction and σ is tiny, β ≈ X.
    The Bernoulli NLL should approach zero.
    """
    B, N = 2, 16
    x = torch.tensor([[1.0, 0.0] * 8, [0.0, 1.0] * 8])
    mu = torch.where(x == 1.0, torch.tensor(8.0), torch.tensor(-8.0))
    log_sigma = torch.full_like(mu, -10.0)
    out = CnpOutput(mu_logit=mu, log_sigma=log_sigma)
    loss = cnp_loss(out, x, n_mc_samples=8)
    assert loss.item() < 1e-2, f"Expected ~0 loss, got {loss.item()}"


def test_loss_equals_log2_at_uniform_prediction() -> None:
    """When μ_logit=0 (so β≈0.5) and σ→0, NLL = -log(0.5) = log(2)."""
    B, N = 4, 32
    x = torch.randint(0, 2, (B, N)).float()
    mu = torch.zeros(B, N)
    log_sigma = torch.full_like(mu, -10.0)
    out = CnpOutput(mu_logit=mu, log_sigma=log_sigma)
    loss = cnp_loss(out, x, n_mc_samples=16)
    assert abs(loss.item() - math.log(2.0)) < 0.05, (
        f"Expected log(2)≈{math.log(2):.4f}, got {loss.item():.4f}"
    )


def test_loss_finite_for_random_init() -> None:
    gen = for_scenario("S5")
    batch = gen.generate(n_trials=4, n_events=32)
    ctx, tgt = split_context_target(batch, n_context=16, seed=0)
    cnp = build_cnp(_config(), gen.dim_theta, gen.dim_phi)
    out = cnp(ctx, tgt)
    x_t = torch.as_tensor(tgt.labels, dtype=torch.float32)
    loss = cnp_loss(out, x_t)
    assert torch.isfinite(loss)
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# Gradient flow.
# ---------------------------------------------------------------------------


def test_grad_flows_through_full_model() -> None:
    """Backward populates gradients on encoder, context-encoder, and decoder."""
    gen = for_scenario("S1")
    batch = gen.generate(n_trials=2, n_events=16)
    ctx, tgt = split_context_target(batch, n_context=8, seed=0)
    cnp = build_cnp(_config(), gen.dim_theta, gen.dim_phi)
    out = cnp(ctx, tgt)
    x_t = torch.as_tensor(tgt.labels, dtype=torch.float32)
    cnp_loss(out, x_t).backward()

    has_grad = [p.grad is not None and torch.any(p.grad != 0.0) for p in cnp.parameters()]
    # At least the decoder, context encoder, and one of the inner encoders must update.
    assert sum(has_grad) >= 3


def test_grad_flows_to_null_token_in_event_only_mode() -> None:
    """When θ is None, the gradient should reach theta_null."""
    gen = for_scenario("S5")
    batch = gen.generate(n_trials=4, n_events=16)
    ctx, tgt = split_context_target(batch, n_context=8, seed=0)
    cnp = build_cnp(_config(), dim_theta=1, dim_phi=1)
    out = cnp(ctx, tgt)
    x_t = torch.as_tensor(tgt.labels, dtype=torch.float32)
    cnp_loss(out, x_t).backward()
    assert cnp.encoder.theta_null.grad is not None
    assert torch.any(cnp.encoder.theta_null.grad != 0.0)


# ---------------------------------------------------------------------------
# Overfit smoke test: loss must decrease on a fixed batch (single scenario).
# ---------------------------------------------------------------------------


def test_loss_decreases_on_overfit_batch() -> None:
    """Train on a single fixed batch for a few hundred steps; the trained
    loss must be meaningfully below the constant-rate baseline NLL.

    This is the smallest end-to-end check that the math actually trains —
    if any of (forward shapes, aggregator axis, loss form, gradient flow)
    is broken, this test catches it.
    """
    torch.manual_seed(0)
    gen = for_scenario("S5", seed=0)
    batch = gen.generate(n_trials=8, n_events=128)
    ctx, tgt = split_context_target(batch, n_context=64, seed=0)
    x_t = torch.as_tensor(tgt.labels, dtype=torch.float32)

    cnp = build_cnp(_config(latent_dim=32), gen.dim_theta, gen.dim_phi)
    opt = torch.optim.Adam(cnp.parameters(), lr=2e-3)

    p_const = x_t.mean().item()
    baseline = -(
        x_t * math.log(p_const) + (1 - x_t) * math.log(1 - p_const)
    ).mean().item()

    losses = []
    for _ in range(300):
        out = cnp(ctx, tgt)
        loss = cnp_loss(out, x_t, n_mc_samples=4)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    final = float(np.mean(losses[-20:]))
    # The trained model must beat the constant-p baseline.
    assert final < baseline - 0.02, (
        f"final loss {final:.4f} did not beat baseline {baseline:.4f} — "
        f"the model is not learning p, only the marginal rate."
    )
    # And it must improve materially from random init.
    assert final < losses[0] - 0.1
