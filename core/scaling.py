"""Feature scaling utilities for real-world physical inputs.

The MFGP (Phase 4) and especially the CNP encoder (Phase 2/3) assume
``θ`` and ``φ`` components live on a similar numerical scale across
dimensions. When physical inputs span very different ranges — e.g.
Energy ∈ [500, 3000] keV alongside Threshold ∈ [0, 1] — the gradient
imbalance makes the MLP encoder *scale-blind*: it locks onto the
high-magnitude dimension and ignores the small one (often by 10²–10³ ×).
The MFGP can compensate via ARD lengthscales, but the CNP cannot
without explicit normalization.

This module provides :class:`MinMaxScaler` for explicit, reversible
input scaling. Two construction paths:

* :meth:`MinMaxScaler.fit` — fit min/max from a representative array.
* :meth:`MinMaxScaler.from_bounds` — build from known physical
  bounds (recommended when the design space is constrained a priori).

The scaler operates on the *last* axis of any array, so it works
uniformly for ``θ`` ``[B, D]`` and ``φ`` ``[B, N, D]``. Apply it
*before* constructing :class:`schemas.data_models.StandardBatch`, and
keep it alongside your CNP / MFGP checkpoint so predictions can be
inverse-transformed back to physical units.

Recommended workflow::

    theta_scaler = MinMaxScaler.from_bounds(low=[500.0, 0.0], high=[3000.0, 1.0])
    theta_scaled = theta_scaler.transform(theta_raw)
    batch = StandardBatch(mode=..., theta=theta_scaled, phi=..., labels=...)
    # ... train CNP, fit MFGP ...
    # At predict time, scale the query with the *same* scaler:
    mu, var = mfgp.predict(theta_scaler.transform(theta_query))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MinMaxScaler:
    """Per-feature affine scaler that maps inputs to a target interval.

    The forward map is::

        y = target_low + (x - feature_min) * (target_high - target_low) / (feature_max - feature_min)

    applied independently along the last axis of the input array.
    Constant features (where ``feature_min == feature_max``) are left
    at the midpoint of the target interval to keep the transform well
    defined.

    Attributes
    ----------
    feature_min, feature_max
        1-D arrays of length ``D``, the input ranges per feature.
        ``feature_min[i] <= feature_max[i]`` is required.
    target_low, target_high
        Output interval. Defaults to ``[-1, 1]`` to match the synthetic
        pseudo-data convention used by the rest of the framework.
    """

    feature_min: np.ndarray
    feature_max: np.ndarray
    target_low: float = -1.0
    target_high: float = 1.0

    def __post_init__(self) -> None:
        fmin = np.asarray(self.feature_min, dtype=float)
        fmax = np.asarray(self.feature_max, dtype=float)
        if fmin.shape != fmax.shape or fmin.ndim != 1:
            raise ValueError(
                f"feature_min / feature_max must be matching 1-D arrays, "
                f"got shapes {fmin.shape} and {fmax.shape}"
            )
        if np.any(fmin > fmax):
            raise ValueError(
                "feature_min must be <= feature_max element-wise; "
                f"violated at indices {np.where(fmin > fmax)[0].tolist()}"
            )
        if not self.target_low < self.target_high:
            raise ValueError(
                f"target_low ({self.target_low}) must be < target_high "
                f"({self.target_high})"
            )
        object.__setattr__(self, "feature_min", fmin)
        object.__setattr__(self, "feature_max", fmax)

    @property
    def dim(self) -> int:
        return int(self.feature_min.shape[0])

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        *,
        target_low: float = -1.0,
        target_high: float = 1.0,
    ) -> "MinMaxScaler":
        """Fit the scaler to the per-feature min / max of ``X``.

        ``X`` may be 2-D ``[B, D]`` or 3-D ``[B, N, D]``; min/max are
        reduced over all axes except the last one.
        """
        if X.ndim < 2:
            raise ValueError(
                f"X must have ≥ 2 dimensions (last axis is the feature "
                f"axis); got shape {X.shape}"
            )
        reduce_axes = tuple(range(X.ndim - 1))
        return cls(
            feature_min=X.min(axis=reduce_axes),
            feature_max=X.max(axis=reduce_axes),
            target_low=target_low,
            target_high=target_high,
        )

    @classmethod
    def from_bounds(
        cls,
        low: np.ndarray | list[float],
        high: np.ndarray | list[float],
        *,
        target_low: float = -1.0,
        target_high: float = 1.0,
    ) -> "MinMaxScaler":
        """Build a scaler from known physical bounds.

        Preferred when the design space is constrained a priori (e.g.
        Energy ∈ [500, 3000] keV, Threshold ∈ [0, 1]) — fitting on data
        could yield a scaler narrower than the actual feasible region.
        """
        return cls(
            feature_min=np.asarray(low, dtype=float),
            feature_max=np.asarray(high, dtype=float),
            target_low=target_low,
            target_high=target_high,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the forward affine map along the last axis."""
        self._check_dim(X)
        span = self.feature_max - self.feature_min
        nonzero = span > 0
        # Constant feature → output sits at the target midpoint; the
        # input is by definition immovable so any slope works there.
        midpoint = 0.5 * (self.target_low + self.target_high)
        safe_span = np.where(nonzero, span, 1.0)
        linear = (
            self.target_low
            + (X - self.feature_min)
            * (self.target_high - self.target_low)
            / safe_span
        )
        return np.where(nonzero, linear, midpoint)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Map scaled values back to the original feature space."""
        self._check_dim(Y)
        span = self.feature_max - self.feature_min
        nonzero = span > 0
        # Constant feature → snap back to fmin (== fmax) regardless of Y.
        target_span = self.target_high - self.target_low
        safe_target = np.where(nonzero, target_span, 1.0)
        linear = self.feature_min + (Y - self.target_low) * span / safe_target
        return np.where(nonzero, linear, self.feature_min)

    def _check_dim(self, X: np.ndarray) -> None:
        if X.ndim < 2 or X.shape[-1] != self.dim:
            raise ValueError(
                f"input shape {X.shape} incompatible with scaler dim "
                f"{self.dim}; last axis must equal scaler dim"
            )
