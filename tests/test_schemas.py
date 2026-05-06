"""Phase 0 acceptance gate.

Verifies that ``StandardBatch`` can be instantiated for the three required
modalities (S1 FULL, S5 EVENT_ONLY, S7 DESIGN_ONLY) across multiple θ/φ
dimensionalities, and that obvious mis-shapes / mode-mismatches fail fast.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from schemas.config import load_config
from schemas.data_models import (
    DesignPoint,
    EventBatch,
    InputMode,
    ModelPrediction,
    StandardBatch,
)

B, N = 4, 32


def _binary_labels(rng: np.random.Generator, b: int = B, n: int = N) -> np.ndarray:
    return rng.integers(0, 2, size=(b, n))


# ---------------------------------------------------------------------------
# Acceptance gate: S1 / S5 / S7 over multiple dimensionalities.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d_theta,d_phi", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_s1_full_mode_arbitrary_dims(d_theta: int, d_phi: int) -> None:
    """S1: FULL mode — both θ and φ present, across S1–S4 dimensionalities."""
    rng = np.random.default_rng(0)
    batch = StandardBatch(
        mode=InputMode.FULL,
        theta=rng.standard_normal((B, d_theta)),
        phi=rng.standard_normal((B, N, d_phi)),
        labels=_binary_labels(rng),
    )
    assert batch.mask_theta and batch.mask_phi
    assert batch.batch_size == B
    assert batch.n_events == N
    assert batch.theta.shape == (B, d_theta)
    assert batch.phi.shape == (B, N, d_phi)


@pytest.mark.parametrize("d_phi", [1, 2])
def test_s5_event_only(d_phi: int) -> None:
    """S5/S6: EVENT_ONLY mode — θ is None, φ is present."""
    rng = np.random.default_rng(1)
    batch = StandardBatch(
        mode=InputMode.EVENT_ONLY,
        phi=rng.standard_normal((B, N, d_phi)),
        labels=_binary_labels(rng),
    )
    assert not batch.mask_theta
    assert batch.mask_phi
    assert batch.theta is None
    assert batch.phi.shape == (B, N, d_phi)


@pytest.mark.parametrize("d_theta", [1, 2])
def test_s7_design_only(d_theta: int) -> None:
    """S7/S8: DESIGN_ONLY mode — θ is present, φ is None."""
    rng = np.random.default_rng(2)
    batch = StandardBatch(
        mode=InputMode.DESIGN_ONLY,
        theta=rng.standard_normal((B, d_theta)),
        labels=_binary_labels(rng),
    )
    assert batch.mask_theta
    assert not batch.mask_phi
    assert batch.phi is None
    assert batch.theta.shape == (B, d_theta)


# ---------------------------------------------------------------------------
# Negative tests: malformed batches must fail at construction.
# ---------------------------------------------------------------------------

def test_full_mode_requires_both() -> None:
    rng = np.random.default_rng(3)
    with pytest.raises(ValidationError):
        StandardBatch(
            mode=InputMode.FULL,
            theta=rng.standard_normal((B, 2)),
            labels=_binary_labels(rng),
        )


def test_event_only_rejects_theta() -> None:
    rng = np.random.default_rng(4)
    with pytest.raises(ValidationError):
        StandardBatch(
            mode=InputMode.EVENT_ONLY,
            theta=rng.standard_normal((B, 2)),
            phi=rng.standard_normal((B, N, 2)),
            labels=_binary_labels(rng),
        )


def test_design_only_rejects_phi() -> None:
    rng = np.random.default_rng(5)
    with pytest.raises(ValidationError):
        StandardBatch(
            mode=InputMode.DESIGN_ONLY,
            theta=rng.standard_normal((B, 2)),
            phi=rng.standard_normal((B, N, 2)),
            labels=_binary_labels(rng),
        )


def test_batch_dim_mismatch_raises() -> None:
    rng = np.random.default_rng(6)
    with pytest.raises(ValidationError):
        StandardBatch(
            mode=InputMode.FULL,
            theta=rng.standard_normal((B + 1, 2)),
            phi=rng.standard_normal((B, N, 2)),
            labels=_binary_labels(rng),
        )


def test_event_count_mismatch_raises() -> None:
    rng = np.random.default_rng(7)
    with pytest.raises(ValidationError):
        StandardBatch(
            mode=InputMode.FULL,
            theta=rng.standard_normal((B, 2)),
            phi=rng.standard_normal((B, N + 5, 2)),
            labels=_binary_labels(rng),
        )


def test_non_binary_labels_raises() -> None:
    rng = np.random.default_rng(8)
    bad_labels = rng.integers(0, 5, size=(B, N))  # values 0..4
    with pytest.raises(ValidationError):
        StandardBatch(
            mode=InputMode.EVENT_ONLY,
            phi=rng.standard_normal((B, N, 2)),
            labels=bad_labels,
        )


def test_beta_out_of_range_raises() -> None:
    rng = np.random.default_rng(9)
    with pytest.raises(ValidationError):
        StandardBatch(
            mode=InputMode.EVENT_ONLY,
            phi=rng.standard_normal((B, N, 2)),
            labels=_binary_labels(rng),
            beta=rng.standard_normal((B, N)),  # not bounded to [0, 1]
        )


def test_beta_in_range_ok() -> None:
    rng = np.random.default_rng(10)
    batch = StandardBatch(
        mode=InputMode.EVENT_ONLY,
        phi=rng.standard_normal((B, N, 2)),
        labels=_binary_labels(rng),
        beta=rng.uniform(0.0, 1.0, size=(B, N)),
    )
    assert batch.beta is not None
    assert batch.beta.shape == (B, N)


# ---------------------------------------------------------------------------
# DesignPoint / EventBatch / ModelPrediction.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d_theta", [1, 2, 3, 5])
def test_design_point_arbitrary_dim(d_theta: int) -> None:
    DesignPoint(theta=np.arange(d_theta, dtype=float))


def test_design_point_rejects_2d() -> None:
    with pytest.raises(ValidationError):
        DesignPoint(theta=np.zeros((2, 2)))


def test_event_batch_n_consistency() -> None:
    rng = np.random.default_rng(11)
    EventBatch(
        phi=rng.standard_normal((10, 3)),
        labels=rng.integers(0, 2, size=10),
    )
    with pytest.raises(ValidationError):
        EventBatch(
            phi=rng.standard_normal((10, 3)),
            labels=rng.integers(0, 2, size=8),
        )


def test_model_prediction_ok() -> None:
    rng = np.random.default_rng(12)
    pred = ModelPrediction(
        mean=rng.standard_normal(B),
        variance=np.abs(rng.standard_normal(B)),
        theta_query=rng.standard_normal((B, 2)),
    )
    assert pred.mean.shape == (B,)
    assert np.all(pred.variance >= 0)


def test_model_prediction_rejects_negative_variance() -> None:
    rng = np.random.default_rng(13)
    with pytest.raises(ValidationError):
        ModelPrediction(
            mean=rng.standard_normal(B),
            variance=-np.abs(rng.standard_normal(B)),
            theta_query=rng.standard_normal((B, 2)),
        )


# ---------------------------------------------------------------------------
# Coercion: list / tuple inputs become ndarray.
# ---------------------------------------------------------------------------

def test_list_coercion() -> None:
    batch = StandardBatch(
        mode=InputMode.DESIGN_ONLY,
        theta=[[0.1, 0.2], [0.3, 0.4]],
        labels=[[0, 1, 0], [1, 0, 1]],
    )
    assert isinstance(batch.theta, np.ndarray)
    assert isinstance(batch.labels, np.ndarray)
    assert batch.theta.shape == (2, 2)
    assert batch.labels.shape == (2, 3)


# ---------------------------------------------------------------------------
# Config loader.
# ---------------------------------------------------------------------------

def test_config_loads() -> None:
    cfg = load_config("config.yaml")
    assert cfg.encoder.latent_dim == 64
    assert cfg.cnp.mixup_alpha == 0.1
    assert cfg.mfgp.n_fidelities == 3
    assert cfg.mae_thresholds.s4 > cfg.mae_thresholds.s1  # higher-dim looser
