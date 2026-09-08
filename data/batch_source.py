"""Formal batch-source contracts and an adapter for in-memory datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from schemas.data_models import InputMode, StandardBatch


class BatchSource(Protocol):
    """Anything that can supply reproducible CNP training batches."""

    mode: InputMode
    dim_theta: int | None
    dim_phi: int | None

    def generate(self, n_trials: int, n_events: int, seed: int) -> StandardBatch: ...


@dataclass(frozen=True)
class FixedBatchSource:
    """Resample trials and paired events from a fixed ``StandardBatch``."""

    batch: StandardBatch
    replace_trials: bool = True
    replace_events: bool = False

    @property
    def mode(self) -> InputMode:
        return self.batch.mode

    @property
    def dim_theta(self) -> int | None:
        return None if self.batch.theta is None else self.batch.theta.shape[1]

    @property
    def dim_phi(self) -> int | None:
        return None if self.batch.phi is None else self.batch.phi.shape[2]

    def generate(self, n_trials: int, n_events: int, seed: int) -> StandardBatch:
        if n_trials <= 0 or n_events <= 0:
            raise ValueError("n_trials and n_events must be positive")
        if not self.replace_trials and n_trials > self.batch.batch_size:
            raise ValueError("n_trials exceeds available trials without replacement")
        if not self.replace_events and n_events > self.batch.n_events:
            raise ValueError("n_events exceeds available events without replacement")

        rng = np.random.default_rng(seed)
        trial_idx = rng.choice(
            self.batch.batch_size, size=n_trials, replace=self.replace_trials
        )
        event_idx = np.stack(
            [
                rng.choice(self.batch.n_events, size=n_events, replace=self.replace_events)
                for _ in range(n_trials)
            ]
        )
        labels_by_trial = self.batch.labels[trial_idx]
        labels = np.take_along_axis(labels_by_trial, event_idx, axis=1)
        phi = None
        if self.batch.phi is not None:
            phi_by_trial = self.batch.phi[trial_idx]
            phi = np.take_along_axis(phi_by_trial, event_idx[..., None], axis=1)
        theta = None if self.batch.theta is None else self.batch.theta[trial_idx]
        return StandardBatch(mode=self.mode, theta=theta, phi=phi, labels=labels)
