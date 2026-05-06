"""Synthetic ground-truth generator for the validation matrix.

The pseudo-data generator builds a controllable analytical
``t(θ, φ) ∈ [0, 1]`` and samples binary outcomes ``X ~ Bernoulli(t)``
from it. Subsequent phases (encoder, CNP, MFGP) train against these
batches and grade themselves against the analytical ``t`` they were
generated from — without it we cannot verify that ``β`` reconstructs
``p`` rather than just fitting ``X`` directly.

The default truth is a single Gaussian bump:

    t(θ, φ) = t_max · exp(-||θ - θ_peak||² / 2 σ_θ²)
                       · exp(-||φ - φ_peak||² / 2 σ_φ²)

For modes missing a parameter, the corresponding factor is dropped (i.e.
treated as 1). Peaks live near the origin and inputs are sampled
uniformly from ``[-1, 1]`` along each axis, so the bump occupies a
visible fraction of the domain — useful for the Phase 1 sanity plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from schemas.data_models import InputMode, StandardBatch


@dataclass(frozen=True, eq=False)
class GaussianBumpTruth:
    """Analytical ``t(θ, φ)`` used to generate pseudo data.

    Attributes
    ----------
    mode
        Which of the three modalities this truth describes.
    theta_peak, phi_peak
        Peak locations in input space. Shape ``(D_θ,)`` / ``(D_φ,)``;
        ``None`` for the parameter that is absent in the given ``mode``.
    sigma_theta, sigma_phi
        Bump widths along the corresponding axis.
    t_max
        Probability at the peak. Must lie in ``(0, 1]``. The paper's
        true rate is ~2·10⁻⁵ but for synthetic validation we use a
        moderate value (~0.3) so a few thousand events already produce
        enough positives to learn from.
    """

    mode: InputMode
    theta_peak: np.ndarray | None
    phi_peak: np.ndarray | None
    sigma_theta: float
    sigma_phi: float
    t_max: float

    def __post_init__(self) -> None:
        if not 0.0 < self.t_max <= 1.0:
            raise ValueError(f"t_max must be in (0, 1], got {self.t_max}")
        needs_theta = self.mode in (InputMode.FULL, InputMode.DESIGN_ONLY)
        needs_phi = self.mode in (InputMode.FULL, InputMode.EVENT_ONLY)
        if needs_theta and self.theta_peak is None:
            raise ValueError(f"mode={self.mode.value} requires theta_peak")
        if not needs_theta and self.theta_peak is not None:
            raise ValueError(f"mode={self.mode.value} forbids theta_peak")
        if needs_phi and self.phi_peak is None:
            raise ValueError(f"mode={self.mode.value} requires phi_peak")
        if not needs_phi and self.phi_peak is not None:
            raise ValueError(f"mode={self.mode.value} forbids phi_peak")

    @property
    def dim_theta(self) -> int | None:
        return None if self.theta_peak is None else int(self.theta_peak.shape[0])

    @property
    def dim_phi(self) -> int | None:
        return None if self.phi_peak is None else int(self.phi_peak.shape[0])

    def evaluate(
        self,
        theta: np.ndarray | None = None,
        phi: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute ``t(θ, φ)`` element-wise.

        Each input array's **last axis is the dimension axis**; preceding
        axes broadcast in the usual numpy way. The returned array's shape
        is the broadcast of the leading axes.

        Examples
        --------
        ``evaluate(theta=[B, 1, D_θ], phi=[B, N, D_φ])`` → ``[B, N]``
        ``evaluate(theta=[G, D_θ])``                     → ``[G]`` (DESIGN_ONLY)
        ``evaluate(phi=[G, D_φ])``                       → ``[G]`` (EVENT_ONLY)
        """
        log_p: np.ndarray | float = 0.0

        if theta is not None:
            if self.theta_peak is None:
                raise ValueError(
                    "Truth has no θ dependence (mode=EVENT_ONLY); pass theta=None"
                )
            diff = np.asarray(theta, dtype=float) - self.theta_peak
            log_p = log_p + (-0.5 * (diff**2).sum(axis=-1) / self.sigma_theta**2)
        elif self.mode in (InputMode.FULL, InputMode.DESIGN_ONLY):
            raise ValueError(f"mode={self.mode.value} requires theta")

        if phi is not None:
            if self.phi_peak is None:
                raise ValueError(
                    "Truth has no φ dependence (mode=DESIGN_ONLY); pass phi=None"
                )
            diff = np.asarray(phi, dtype=float) - self.phi_peak
            log_p = log_p + (-0.5 * (diff**2).sum(axis=-1) / self.sigma_phi**2)
        elif self.mode in (InputMode.FULL, InputMode.EVENT_ONLY):
            raise ValueError(f"mode={self.mode.value} requires phi")

        return self.t_max * np.exp(log_p)


# ---------------------------------------------------------------------------
# Generator.
# ---------------------------------------------------------------------------

@dataclass
class PseudoDataGenerator:
    """Wraps a :class:`GaussianBumpTruth` and produces ``StandardBatch``es."""

    truth: GaussianBumpTruth
    seed: int = 0

    SCENARIOS: ClassVar[dict[str, dict]] = {
        "S1": {"mode": InputMode.FULL,        "dim_theta": 1,    "dim_phi": 1},
        "S2": {"mode": InputMode.FULL,        "dim_theta": 2,    "dim_phi": 1},
        "S3": {"mode": InputMode.FULL,        "dim_theta": 1,    "dim_phi": 2},
        "S4": {"mode": InputMode.FULL,        "dim_theta": 2,    "dim_phi": 2},
        "S5": {"mode": InputMode.EVENT_ONLY,  "dim_theta": None, "dim_phi": 1},
        "S6": {"mode": InputMode.EVENT_ONLY,  "dim_theta": None, "dim_phi": 2},
        "S7": {"mode": InputMode.DESIGN_ONLY, "dim_theta": 1,    "dim_phi": None},
        "S8": {"mode": InputMode.DESIGN_ONLY, "dim_theta": 2,    "dim_phi": None},
    }

    @property
    def dim_theta(self) -> int | None:
        return self.truth.dim_theta

    @property
    def dim_phi(self) -> int | None:
        return self.truth.dim_phi

    @property
    def mode(self) -> InputMode:
        return self.truth.mode

    def generate(self, n_trials: int, n_events: int, seed: int | None = None) -> StandardBatch:
        """Draw a fresh pseudo batch of size ``[n_trials, n_events]``."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        B, N = n_trials, n_events

        theta: np.ndarray | None = None
        phi: np.ndarray | None = None

        if self.truth.dim_theta is not None:
            theta = rng.uniform(-1.0, 1.0, size=(B, self.truth.dim_theta))
        if self.truth.dim_phi is not None:
            phi = rng.uniform(-1.0, 1.0, size=(B, N, self.truth.dim_phi))

        if self.truth.mode is InputMode.FULL:
            # theta: [B, D_θ] → broadcast to per-event by inserting an N-axis.
            p = self.truth.evaluate(theta=theta[:, None, :], phi=phi)  # [B, N]
        elif self.truth.mode is InputMode.EVENT_ONLY:
            p = self.truth.evaluate(theta=None, phi=phi)  # [B, N]
        elif self.truth.mode is InputMode.DESIGN_ONLY:
            p_per_trial = self.truth.evaluate(theta=theta, phi=None)  # [B]
            p = np.broadcast_to(p_per_trial[:, None], (B, N))
        else:  # pragma: no cover — exhaustive
            raise ValueError(f"unknown mode {self.truth.mode}")

        labels = (rng.uniform(size=(B, N)) < p).astype(np.int8)

        return StandardBatch(
            mode=self.truth.mode,
            theta=theta,
            phi=phi,
            labels=labels,
        )


def for_scenario(
    name: str,
    *,
    seed: int = 0,
    t_max: float = 0.4,
    sigma: float = 0.4,
) -> PseudoDataGenerator:
    """Build the canonical generator for one of the validation-matrix scenarios.

    Peak locations are deterministic (not seeded) so heatmaps and curves
    are interpretable across runs: ``θ_peak = +0.2 · 1`` and
    ``φ_peak = -0.2 · 1``.
    """
    key = name.upper()
    if key not in PseudoDataGenerator.SCENARIOS:
        raise KeyError(
            f"unknown scenario {name!r}; valid: {sorted(PseudoDataGenerator.SCENARIOS)}"
        )
    spec = PseudoDataGenerator.SCENARIOS[key]
    mode: InputMode = spec["mode"]
    d_theta: int | None = spec["dim_theta"]
    d_phi: int | None = spec["dim_phi"]

    theta_peak = None if d_theta is None else np.full(d_theta, 0.2)
    phi_peak = None if d_phi is None else np.full(d_phi, -0.2)

    truth = GaussianBumpTruth(
        mode=mode,
        theta_peak=theta_peak,
        phi_peak=phi_peak,
        sigma_theta=sigma,
        sigma_phi=sigma,
        t_max=t_max,
    )
    return PseudoDataGenerator(truth=truth, seed=seed)
