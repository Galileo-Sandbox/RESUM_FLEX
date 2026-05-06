"""Tests for the synthetic ground-truth generator (Phase 1 acceptance gate).

Two flavors of test:

* **Coverage** — every S1..S8 scenario produces a valid ``StandardBatch``
  whose shapes line up with the requested dimensionality and mode.
* **Bernoulli round-trip** — given enough samples, the empirical rate
  ``mean(X)`` approaches the analytical ``mean(t)``, demonstrating that
  the labels are genuinely a Bernoulli realization of the truth (not just
  random noise).
"""

from __future__ import annotations

import numpy as np
import pytest

from data.pseudo_generator import (
    GaussianBumpTruth,
    PseudoDataGenerator,
    for_scenario,
)
from schemas.data_models import InputMode

ALL_SCENARIOS = list(PseudoDataGenerator.SCENARIOS.keys())


# ---------------------------------------------------------------------------
# Coverage: shape & mode checks across S1–S8.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_SCENARIOS)
def test_scenario_produces_valid_batch(name: str) -> None:
    gen = for_scenario(name)
    spec = PseudoDataGenerator.SCENARIOS[name]
    batch = gen.generate(n_trials=4, n_events=64)

    assert batch.mode is spec["mode"]
    assert batch.batch_size == 4
    assert batch.n_events == 64

    if spec["dim_theta"] is None:
        assert batch.theta is None
        assert not batch.mask_theta
    else:
        assert batch.theta is not None
        assert batch.theta.shape == (4, spec["dim_theta"])

    if spec["dim_phi"] is None:
        assert batch.phi is None
        assert not batch.mask_phi
    else:
        assert batch.phi is not None
        assert batch.phi.shape == (4, 64, spec["dim_phi"])


def test_unknown_scenario_raises() -> None:
    with pytest.raises(KeyError):
        for_scenario("S99")


# ---------------------------------------------------------------------------
# Truth boundedness.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_SCENARIOS)
def test_truth_in_unit_interval(name: str) -> None:
    """``t(θ, φ) ∈ [0, 1]`` for any inputs in the sampling domain."""
    gen = for_scenario(name)
    rng = np.random.default_rng(0)
    n = 2000
    theta = (
        rng.uniform(-1.0, 1.0, size=(n, gen.dim_theta)) if gen.dim_theta else None
    )
    phi = (
        rng.uniform(-1.0, 1.0, size=(n, gen.dim_phi)) if gen.dim_phi else None
    )
    p = gen.truth.evaluate(theta=theta, phi=phi)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert p.max() <= gen.truth.t_max + 1e-9
    # The peak must be reachable: the bump max occurs at the peak point,
    # so evaluating exactly at the peak should hit ~t_max.
    if gen.dim_theta and gen.dim_phi:
        peak_p = gen.truth.evaluate(
            theta=gen.truth.theta_peak[None, :], phi=gen.truth.phi_peak[None, :]
        )
    elif gen.dim_theta:
        peak_p = gen.truth.evaluate(theta=gen.truth.theta_peak[None, :], phi=None)
    else:
        peak_p = gen.truth.evaluate(theta=None, phi=gen.truth.phi_peak[None, :])
    assert np.allclose(peak_p, gen.truth.t_max)


# ---------------------------------------------------------------------------
# Bernoulli round-trip: empirical rate ≈ mean of analytical p.
# ---------------------------------------------------------------------------

def test_event_only_rate_matches_truth() -> None:
    """S5 with many events: ``mean(X)`` ≈ ``mean(t(φ))`` over the sampled φ."""
    gen = for_scenario("S5", seed=42)
    batch = gen.generate(n_trials=1, n_events=200_000)
    assert batch.phi is not None
    p = gen.truth.evaluate(theta=None, phi=batch.phi)  # [1, 200000]
    obs = float(batch.labels.mean())
    expected = float(p.mean())
    assert abs(obs - expected) < 0.005, f"|obs-expected|={abs(obs - expected):.4f}"


def test_design_only_per_trial_rate() -> None:
    """S7: each trial's ``mean(X_k)`` ≈ ``t(θ_k)`` since events share one ``p``."""
    gen = for_scenario("S7", seed=0)
    batch = gen.generate(n_trials=64, n_events=4_000)
    assert batch.theta is not None
    p_per_trial = gen.truth.evaluate(theta=batch.theta, phi=None)  # [64]
    obs_per_trial = batch.labels.mean(axis=1)  # [64]
    # σ of obs ≈ √(p(1-p)/N). With p~0.4 and N=4000, σ ≈ 0.0077, so allow 4σ.
    tolerance = 0.04
    assert np.all(np.abs(obs_per_trial - p_per_trial) < tolerance)


def test_full_mode_rate_matches_truth() -> None:
    gen = for_scenario("S1", seed=7)
    batch = gen.generate(n_trials=4, n_events=50_000)
    assert batch.theta is not None and batch.phi is not None
    p = gen.truth.evaluate(theta=batch.theta[:, None, :], phi=batch.phi)  # [4, 50000]
    obs = float(batch.labels.mean())
    expected = float(p.mean())
    assert abs(obs - expected) < 0.005


# ---------------------------------------------------------------------------
# Mode contracts on the truth itself.
# ---------------------------------------------------------------------------

def test_event_only_truth_rejects_theta() -> None:
    truth = for_scenario("S5").truth
    with pytest.raises(ValueError):
        truth.evaluate(theta=np.array([[0.0]]), phi=np.array([[0.0]]))


def test_design_only_truth_rejects_phi() -> None:
    truth = for_scenario("S7").truth
    with pytest.raises(ValueError):
        truth.evaluate(theta=np.array([[0.0]]), phi=np.array([[0.0]]))


def test_full_truth_requires_both() -> None:
    truth = for_scenario("S1").truth
    with pytest.raises(ValueError):
        truth.evaluate(theta=None, phi=np.array([[0.0]]))
    with pytest.raises(ValueError):
        truth.evaluate(theta=np.array([[0.0]]), phi=None)


def test_invalid_t_max_rejected() -> None:
    with pytest.raises(ValueError):
        GaussianBumpTruth(
            mode=InputMode.EVENT_ONLY,
            theta_peak=None,
            phi_peak=np.array([0.0]),
            sigma_theta=0.4,
            sigma_phi=0.4,
            t_max=1.5,
        )


def test_truth_construction_mode_consistency() -> None:
    """A truth constructed with the wrong combo of peaks fails fast."""
    with pytest.raises(ValueError):
        GaussianBumpTruth(
            mode=InputMode.EVENT_ONLY,
            theta_peak=np.array([0.0]),  # forbidden
            phi_peak=np.array([0.0]),
            sigma_theta=0.4,
            sigma_phi=0.4,
            t_max=0.5,
        )
    with pytest.raises(ValueError):
        GaussianBumpTruth(
            mode=InputMode.FULL,
            theta_peak=None,  # required
            phi_peak=np.array([0.0]),
            sigma_theta=0.4,
            sigma_phi=0.4,
            t_max=0.5,
        )


# ---------------------------------------------------------------------------
# Reproducibility.
# ---------------------------------------------------------------------------

def test_same_seed_gives_same_batch() -> None:
    gen = for_scenario("S1", seed=123)
    a = gen.generate(n_trials=4, n_events=32, seed=99)
    b = gen.generate(n_trials=4, n_events=32, seed=99)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert a.theta is not None and b.theta is not None
    np.testing.assert_array_equal(a.theta, b.theta)


def test_different_seeds_differ() -> None:
    gen = for_scenario("S1")
    a = gen.generate(n_trials=4, n_events=32, seed=1)
    b = gen.generate(n_trials=4, n_events=32, seed=2)
    assert not np.array_equal(a.labels, b.labels)
