"""Data contracts for the RESUM_FLEX pipeline.

All array fields are `numpy.ndarray` — framework-neutral. PyTorch conversion
happens inside ``core/networks.py``; ``core/surrogate_mfgp.py`` (GPy/Emukit)
consumes these arrays directly. The schema layer therefore has no torch or
GPy dependency.

The pipeline supports three input modalities (see CLAUDE.md Validation
Matrix S1–S8):

* ``FULL``         — both ``theta`` and ``phi`` are present.
* ``EVENT_ONLY``   — only ``phi``; ``theta`` is ``None``.
* ``DESIGN_ONLY``  — only ``theta``; ``phi`` is ``None``.

The downstream encoder uses learnable null embeddings to handle the
missing component, so absence is communicated via ``None`` (and the
``mask_*`` properties), not by zero-filled tensors.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, model_validator


class InputMode(str, Enum):
    """The three input modalities the pipeline must support."""

    FULL = "full"
    EVENT_ONLY = "event_only"
    DESIGN_ONLY = "design_only"


def _coerce_ndarray(value: Any) -> Any:
    """Coerce list/tuple inputs into ``np.ndarray``; pass ``None`` through."""
    if value is None or isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


NDArray = Annotated[np.ndarray, BeforeValidator(_coerce_ndarray)]
OptionalNDArray = Annotated[np.ndarray | None, BeforeValidator(_coerce_ndarray)]


class DesignPoint(BaseModel):
    """A single design configuration ``θ``.

    Attributes
    ----------
    theta : np.ndarray
        Shape ``[D_θ]``. ``D_θ`` is arbitrary (≥ 1) to support 1D, 2D, … design
        spaces from the validation matrix.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    theta: NDArray

    @model_validator(mode="after")
    def _check_shape(self) -> DesignPoint:
        if self.theta.ndim != 1:
            raise ValueError(
                f"DesignPoint.theta must be 1D [D_θ], got shape {self.theta.shape}"
            )
        return self


class EventBatch(BaseModel):
    """A collection of events generated under a single (or unspecified) design.

    Attributes
    ----------
    phi : np.ndarray
        Shape ``[N, D_φ]``. Event-specific parameters.
    labels : np.ndarray
        Shape ``[N]``. Binary Bernoulli draws ``X ∈ {0, 1}``.
    theta : np.ndarray | None
        Shape ``[D_θ]`` if known, else ``None`` (event-only mode).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    phi: NDArray
    labels: NDArray
    theta: OptionalNDArray = None

    @model_validator(mode="after")
    def _check_shapes(self) -> EventBatch:
        if self.phi.ndim != 2:
            raise ValueError(f"phi must be [N, D_φ], got shape {self.phi.shape}")
        if self.labels.ndim != 1:
            raise ValueError(f"labels must be [N], got shape {self.labels.shape}")
        if self.phi.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"phi N={self.phi.shape[0]} does not match labels N={self.labels.shape[0]}"
            )
        if self.theta is not None and self.theta.ndim != 1:
            raise ValueError(f"theta must be 1D [D_θ], got shape {self.theta.shape}")
        _check_binary(self.labels)
        return self


class StandardBatch(BaseModel):
    """Primary pipeline carrier.

    Shapes
    ------
    theta  : ``[B, D_θ]``     (optional; absent in EVENT_ONLY mode)
    phi    : ``[B, N, D_φ]``  (optional; absent in DESIGN_ONLY mode)
    labels : ``[B, N]``       (binary X — always present)
    beta   : ``[B, N]``       (CNP-reconstructed continuous score in [0, 1];
                                populated in Phase 3, ``None`` before that)

    The ``mode`` field declares which modality this batch represents and is
    cross-checked against the presence of ``theta`` / ``phi`` so a malformed
    batch fails fast at construction time.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: InputMode
    labels: NDArray
    theta: OptionalNDArray = None
    phi: OptionalNDArray = None
    beta: OptionalNDArray = None

    @property
    def mask_theta(self) -> bool:
        return self.theta is not None

    @property
    def mask_phi(self) -> bool:
        return self.phi is not None

    @property
    def batch_size(self) -> int:
        return int(self.labels.shape[0])

    @property
    def n_events(self) -> int:
        return int(self.labels.shape[1])

    @model_validator(mode="after")
    def _check_consistency(self) -> StandardBatch:
        if self.labels.ndim != 2:
            raise ValueError(f"labels must be [B, N], got shape {self.labels.shape}")
        B, N = self.labels.shape
        _check_binary(self.labels)

        # Modality / presence cross-check.
        if self.mode is InputMode.FULL:
            if self.theta is None or self.phi is None:
                raise ValueError("FULL mode requires both theta and phi to be present.")
        elif self.mode is InputMode.EVENT_ONLY:
            if self.theta is not None:
                raise ValueError("EVENT_ONLY mode requires theta=None.")
            if self.phi is None:
                raise ValueError("EVENT_ONLY mode requires phi to be present.")
        elif self.mode is InputMode.DESIGN_ONLY:
            if self.phi is not None:
                raise ValueError("DESIGN_ONLY mode requires phi=None.")
            if self.theta is None:
                raise ValueError("DESIGN_ONLY mode requires theta to be present.")

        # Shape consistency.
        if self.theta is not None:
            if self.theta.ndim != 2:
                raise ValueError(f"theta must be [B, D_θ], got shape {self.theta.shape}")
            if self.theta.shape[0] != B:
                raise ValueError(
                    f"theta B={self.theta.shape[0]} does not match labels B={B}"
                )
        if self.phi is not None:
            if self.phi.ndim != 3:
                raise ValueError(f"phi must be [B, N, D_φ], got shape {self.phi.shape}")
            if self.phi.shape[0] != B or self.phi.shape[1] != N:
                raise ValueError(
                    f"phi shape {self.phi.shape} inconsistent with labels [B={B}, N={N}]"
                )
        if self.beta is not None:
            if self.beta.shape != (B, N):
                raise ValueError(
                    f"beta must have shape (B, N) = {(B, N)}, got {self.beta.shape}"
                )
            if np.any((self.beta < 0.0) | (self.beta > 1.0)):
                raise ValueError("beta values must lie in [0, 1].")
        return self


class ModelPrediction(BaseModel):
    """MFGP posterior at a set of query design points.

    Attributes
    ----------
    mean : np.ndarray
        Shape ``[B]``. Posterior mean ``μ_θ``.
    variance : np.ndarray
        Shape ``[B]``. Posterior variance ``σ²_θ`` (non-negative).
    theta_query : np.ndarray
        Shape ``[B, D_θ]``. The design points the prediction refers to.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean: NDArray
    variance: NDArray
    theta_query: NDArray

    @model_validator(mode="after")
    def _check_shapes(self) -> ModelPrediction:
        if self.mean.ndim != 1:
            raise ValueError(f"mean must be 1D [B], got shape {self.mean.shape}")
        if self.variance.shape != self.mean.shape:
            raise ValueError(
                f"variance shape {self.variance.shape} != mean shape {self.mean.shape}"
            )
        if self.theta_query.ndim != 2:
            raise ValueError(
                f"theta_query must be [B, D_θ], got shape {self.theta_query.shape}"
            )
        if self.theta_query.shape[0] != self.mean.shape[0]:
            raise ValueError(
                f"theta_query B={self.theta_query.shape[0]} != mean B={self.mean.shape[0]}"
            )
        if np.any(self.variance < 0):
            raise ValueError("variance must be non-negative.")
        return self


def _check_binary(labels: np.ndarray) -> None:
    """Raise if ``labels`` contains anything other than {0, 1}."""
    if labels.size == 0:
        return
    unique = np.unique(labels)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            f"labels must be binary {{0, 1}} (raw Bernoulli draws); found values {unique}"
        )
