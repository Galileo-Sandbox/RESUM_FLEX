"""Tests for ``core/scaling.py`` and the ``StandardBatch``
scale-imbalance warning.

These guard against regressions of the "silent killer" reported when
applying the framework to real physical inputs (Energy ∈ [500, 3000]
keV alongside Threshold ∈ [0, 1]): the CNP encoder went scale-blind
to the small-magnitude dimension. The framework now (a) provides a
:class:`MinMaxScaler` to fix it explicitly, and (b) warns at batch
construction when the imbalance is detectable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from core import MinMaxScaler
from schemas.data_models import (
    InputMode,
    ScaleImbalanceWarning,
    StandardBatch,
)


# ---------------------------------------------------------------------------
# MinMaxScaler.
# ---------------------------------------------------------------------------


def test_from_bounds_forward_and_inverse_roundtrip() -> None:
    sc = MinMaxScaler.from_bounds(low=[500.0, 0.0], high=[3000.0, 1.0])
    raw = np.array([[500.0, 0.0], [1750.0, 0.5], [3000.0, 1.0]])
    scaled = sc.transform(raw)
    np.testing.assert_allclose(scaled, [[-1, -1], [0, 0], [1, 1]])
    np.testing.assert_allclose(sc.inverse_transform(scaled), raw)


def test_fit_2d_recovers_targets() -> None:
    rng = np.random.default_rng(0)
    raw = rng.uniform([10.0, -3.0], [20.0, 7.0], size=(50, 2))
    sc = MinMaxScaler.fit(raw)
    scaled = sc.transform(raw)
    # All feature mins map to target_low, maxes to target_high.
    np.testing.assert_allclose(scaled.max(axis=0), [1.0, 1.0])
    np.testing.assert_allclose(scaled.min(axis=0), [-1.0, -1.0])


def test_fit_3d_phi_reduces_over_leading_axes() -> None:
    rng = np.random.default_rng(0)
    raw = rng.uniform([100.0, 0.0], [200.0, 0.5], size=(4, 8, 2))
    sc = MinMaxScaler.fit(raw)
    scaled = sc.transform(raw)
    np.testing.assert_allclose(scaled.max(axis=(0, 1)), [1.0, 1.0])
    np.testing.assert_allclose(scaled.min(axis=(0, 1)), [-1.0, -1.0])


def test_constant_feature_maps_to_midpoint_and_inverts() -> None:
    sc = MinMaxScaler.from_bounds(low=[1.0, 5.0], high=[3.0, 5.0])
    # The constant feature (index 1) goes to the target midpoint regardless
    # of input value.
    forward = sc.transform(np.array([[2.0, 5.0], [3.0, 5.0]]))
    np.testing.assert_allclose(forward[:, 1], 0.0)
    np.testing.assert_allclose(forward[:, 0], [0.0, 1.0])
    # Inverse for the constant feature returns fmin (== fmax) regardless
    # of the scaled value — there's nothing to disambiguate.
    inverse = sc.inverse_transform(np.array([[0.0, -0.7], [1.0, 0.4]]))
    np.testing.assert_allclose(inverse[:, 1], 5.0)
    np.testing.assert_allclose(inverse[:, 0], [2.0, 3.0])


def test_custom_target_interval() -> None:
    sc = MinMaxScaler.from_bounds(
        low=[0.0], high=[10.0], target_low=0.0, target_high=1.0,
    )
    np.testing.assert_allclose(
        sc.transform(np.array([[0.0], [5.0], [10.0]])).flatten(),
        [0.0, 0.5, 1.0],
    )


def test_invalid_bounds_raise() -> None:
    with pytest.raises(ValueError, match="1-D"):
        MinMaxScaler.from_bounds(low=[[1.0]], high=[[2.0]])
    with pytest.raises(ValueError, match="<= feature_max"):
        MinMaxScaler.from_bounds(low=[2.0], high=[1.0])
    with pytest.raises(ValueError, match="target_low"):
        MinMaxScaler.from_bounds(low=[0.0], high=[1.0],
                                  target_low=1.0, target_high=0.0)


def test_dim_mismatch_raises() -> None:
    sc = MinMaxScaler.from_bounds(low=[0.0, 0.0], high=[1.0, 1.0])
    with pytest.raises(ValueError, match="incompatible"):
        sc.transform(np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# StandardBatch scale-imbalance warning.
# ---------------------------------------------------------------------------


def _binary_labels(B: int = 8, N: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.binomial(1, 0.3, size=(B, N)).astype(np.int8)


def test_warns_on_imbalanced_theta() -> None:
    rng = np.random.default_rng(0)
    theta = np.column_stack([
        rng.uniform(500.0, 3000.0, size=12),  # range ~2500
        rng.uniform(0.0, 1.0, size=12),       # range ~1   → ratio ~2500×
    ])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        StandardBatch(
            mode=InputMode.DESIGN_ONLY,
            theta=theta, phi=None,
            labels=_binary_labels(B=12),
        )
    matches = [item for item in w if issubclass(item.category, ScaleImbalanceWarning)]
    assert len(matches) == 1
    assert "theta" in str(matches[0].message)


def test_warns_on_imbalanced_phi() -> None:
    rng = np.random.default_rng(0)
    phi = np.stack([
        rng.uniform(100.0, 200.0, size=(8, 6)),  # range ~100
        rng.uniform(0.0, 0.5,    size=(8, 6)),   # range ~0.5 → ratio ~200×
    ], axis=-1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        StandardBatch(
            mode=InputMode.EVENT_ONLY,
            theta=None, phi=phi,
            labels=_binary_labels(B=8, N=6),
        )
    matches = [item for item in w if issubclass(item.category, ScaleImbalanceWarning)]
    assert len(matches) == 1
    assert "phi" in str(matches[0].message)


def test_no_warning_on_normalized_inputs() -> None:
    sc = MinMaxScaler.from_bounds(low=[500.0, 0.0], high=[3000.0, 1.0])
    rng = np.random.default_rng(0)
    raw = np.column_stack([
        rng.uniform(500.0, 3000.0, size=12),
        rng.uniform(0.0, 1.0, size=12),
    ])
    theta = sc.transform(raw)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        StandardBatch(
            mode=InputMode.DESIGN_ONLY,
            theta=theta, phi=None,
            labels=_binary_labels(B=12),
        )
    matches = [item for item in w if issubclass(item.category, ScaleImbalanceWarning)]
    assert len(matches) == 0


def test_no_warning_for_single_feature() -> None:
    """A 1-D θ has no comparator dimension to be 'imbalanced' against."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, 1e6, size=(10, 1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        StandardBatch(
            mode=InputMode.DESIGN_ONLY,
            theta=theta, phi=None,
            labels=_binary_labels(B=10),
        )
    matches = [item for item in w if issubclass(item.category, ScaleImbalanceWarning)]
    assert len(matches) == 0


def test_no_warning_when_ranges_are_close() -> None:
    """A 5× range gap should not trigger the 10× threshold."""
    rng = np.random.default_rng(0)
    theta = np.column_stack([
        rng.uniform(0.0, 5.0, size=15),  # range ~5
        rng.uniform(0.0, 1.0, size=15),  # range ~1 → ratio ~5×
    ])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        StandardBatch(
            mode=InputMode.DESIGN_ONLY,
            theta=theta, phi=None,
            labels=_binary_labels(B=15),
        )
    matches = [item for item in w if issubclass(item.category, ScaleImbalanceWarning)]
    assert len(matches) == 0
