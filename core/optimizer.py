"""IVR active-learning optimizer for the MFGP.

Given a fitted :class:`MultiFidelityGP`, this module picks the next
design point ``θ_next`` that — if observed at the highest fidelity — is
expected to shrink the integrated posterior variance the most:

    R(θ*) = ∫_Θ [σ²(θ) - σ²_new(θ | θ*)] dθ
          ≈ Σ_m [K_post(θ_m, θ*)]² / σ²(θ*)

The closed-form rank-1 variance update means we **never refit** during
acquisition scoring — we only need the current GP's posterior kernel on
arbitrary point pairs. We get it directly from
``K(x1, x2) - K(x1, X_train) (K + σ² I)⁻¹ K(X_train, x2)``; GPy stores
the precomputed inverse as ``model.posterior.woodbury_inv``.

The acquisition runs at a fixed fidelity (default: highest, i.e. the
``y_raw`` level we ultimately want low-variance predictions of).

Constraints: a callable ``feasibility_fn(theta) -> bool[N]`` zeros out
infeasible candidates so the optimizer never picks a θ outside the
allowed set. Box bounds are passed separately and used for candidate /
MC-integration sampling.

The active-learning loop drives this in a 5-step cycle: acquire → query
truth → refit → repeat. For the synthetic pseudo-data generator,
"querying the truth" means sampling a fresh batch of events at the
chosen θ and pushing both ``β̄(θ)`` (from the CNP) and ``y_raw = m/N``
into the MFGP's HF dataset.

This module is **numpy + torch** (uses the CNP for β̄). It does not
import GPy directly, but it does poke at ``mfgp.model.kern`` and
``mfgp.model.posterior`` — that coupling is documented and intentional;
the cost of the closed-form update is dwarfed by the alternative of
refitting per candidate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from core.surrogate_cnp import ConditionalNeuralProcess, split_context_target
from core.surrogate_mfgp import MultiFidelityGP
from core.training import cnp_trial_predictive
from data.pseudo_generator import PseudoDataGenerator
from schemas.data_models import InputMode, StandardBatch


# ---------------------------------------------------------------------------
# Posterior covariance helper.
# ---------------------------------------------------------------------------


def posterior_covariance(
    mfgp: MultiFidelityGP,
    X1: np.ndarray,
    X2: np.ndarray,
    *,
    fidelity: int | None = None,
) -> np.ndarray:
    """GP posterior cross-covariance ``K_post(X1, X2)`` at one fidelity.

    Both inputs are ``(n, dim_theta)`` arrays in the original θ space;
    the fidelity column is appended internally so Emukit's joint kernel
    sees them at the requested level. Returns shape ``(n1, n2)``.

    GPy's :meth:`posterior_covariance_between_points` would do the same
    thing in one line, but it routes through the mixed-noise likelihood
    which raises on ``Y_metadata=None`` — so we compute the closed form
    ourselves from ``kern.K`` and the cached ``woodbury_inv``.
    """
    if not mfgp.is_fitted:
        raise RuntimeError("mfgp not fitted; call .fit() first")
    if X1.ndim != 2 or X1.shape[1] != mfgp.dim_theta:
        raise ValueError(f"X1.shape={X1.shape}; expected (n, {mfgp.dim_theta})")
    if X2.ndim != 2 or X2.shape[1] != mfgp.dim_theta:
        raise ValueError(f"X2.shape={X2.shape}; expected (n, {mfgp.dim_theta})")
    f = mfgp._resolve_fidelity(fidelity)

    X1_aug = np.concatenate(
        [X1, np.full((X1.shape[0], 1), f, dtype=float)], axis=1
    )
    X2_aug = np.concatenate(
        [X2, np.full((X2.shape[0], 1), f, dtype=float)], axis=1
    )
    gp = mfgp.model
    K12 = gp.kern.K(X1_aug, X2_aug)
    K1X = gp.kern.K(X1_aug, gp.X)
    KX2 = gp.kern.K(gp.X, X2_aug)
    W = gp.posterior.woodbury_inv
    if W.ndim == 3:
        W = W[..., 0]
    return np.asarray(K12 - K1X @ W @ KX2)


# ---------------------------------------------------------------------------
# Domain & sampling.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoxBounds:
    """Axis-aligned box constraint on θ.

    ``low``, ``high`` are 1-D arrays of length ``dim_theta`` with
    ``low < high`` element-wise. The ``contains`` predicate is
    vectorized over the batch axis.
    """

    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        low = np.asarray(self.low, dtype=float)
        high = np.asarray(self.high, dtype=float)
        if low.shape != high.shape or low.ndim != 1:
            raise ValueError(
                f"low/high must be 1-D and same shape; got {low.shape} / {high.shape}"
            )
        if not np.all(low < high):
            raise ValueError(f"need low < high element-wise; got {low} / {high}")
        # Replace via object.__setattr__ since frozen.
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def dim(self) -> int:
        return int(self.low.shape[0])

    @property
    def volume(self) -> float:
        return float(np.prod(self.high - self.low))

    def contains(self, theta: np.ndarray) -> np.ndarray:
        if theta.ndim != 2 or theta.shape[1] != self.dim:
            raise ValueError(
                f"theta.shape={theta.shape}; expected (n, {self.dim})"
            )
        return ((theta >= self.low) & (theta <= self.high)).all(axis=1)

    def sample_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=(n, self.dim))

    def grid_1d(self, n: int) -> np.ndarray:
        if self.dim != 1:
            raise ValueError("grid_1d requires dim == 1")
        return np.linspace(self.low[0], self.high[0], n)[:, None]

    def grid_2d(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.dim != 2:
            raise ValueError("grid_2d requires dim == 2")
        ax0 = np.linspace(self.low[0], self.high[0], n)
        ax1 = np.linspace(self.low[1], self.high[1], n)
        g0, g1 = np.meshgrid(ax0, ax1, indexing="ij")
        flat = np.stack([g0.ravel(), g1.ravel()], axis=1)
        return ax0, ax1, flat


# ---------------------------------------------------------------------------
# Acquisition.
# ---------------------------------------------------------------------------


@dataclass
class IvrAcquisition:
    """Integrated-variance-reduction acquisition over a fitted MFGP.

    Parameters
    ----------
    mfgp
        A fitted :class:`MultiFidelityGP`.
    bounds
        Box bounds on θ. MC integration points are drawn from this box
        (uniform), so this is the domain over which we integrate the
        variance.
    n_mc_samples
        Number of MC integration points used to approximate the
        ``∫_Θ`` term. Higher → smoother surface, slower scoring.
    fidelity
        Fidelity level at which we score posterior variance reduction.
        Default: highest fidelity (the y_raw target).
    feasibility_fn
        Optional ``(theta:[n,D]) -> bool[n]`` predicate. Infeasible
        candidates get acquisition score 0.
    seed
        RNG seed for the MC integration sample.
    """

    mfgp: MultiFidelityGP
    bounds: BoxBounds
    n_mc_samples: int = 1000
    fidelity: int | None = None
    feasibility_fn: Callable[[np.ndarray], np.ndarray] | None = None
    seed: int = 0
    _mc_points: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bounds.dim != self.mfgp.dim_theta:
            raise ValueError(
                f"bounds.dim={self.bounds.dim} != mfgp.dim_theta={self.mfgp.dim_theta}"
            )
        if self.n_mc_samples <= 0:
            raise ValueError(f"n_mc_samples must be > 0, got {self.n_mc_samples}")
        rng = np.random.default_rng(self.seed)
        self._mc_points = self.bounds.sample_uniform(self.n_mc_samples, rng)

    def score(self, theta_candidates: np.ndarray) -> np.ndarray:
        """Variance-reduction acquisition for each candidate (higher = better).

        Returns a 1-D array of length ``n_candidates``. Infeasible
        candidates (outside the box, or ruled out by ``feasibility_fn``)
        return ``0.0``.
        """
        if theta_candidates.ndim != 2 or theta_candidates.shape[1] != self.bounds.dim:
            raise ValueError(
                f"theta_candidates.shape={theta_candidates.shape}; "
                f"expected (n, {self.bounds.dim})"
            )
        n = theta_candidates.shape[0]

        feas = self.bounds.contains(theta_candidates)
        if self.feasibility_fn is not None:
            feas &= np.asarray(self.feasibility_fn(theta_candidates), dtype=bool)
        if not feas.any():
            return np.zeros(n)

        feasible_idx = np.where(feas)[0]
        feasible_theta = theta_candidates[feasible_idx]

        # K_post[m, c] = posterior covariance between MC point m and
        # candidate c at the chosen fidelity.
        K_mc_cand = posterior_covariance(
            self.mfgp, self._mc_points, feasible_theta, fidelity=self.fidelity,
        )                                                        # [M, C]
        _, var_cand = self.mfgp.predict(feasible_theta, fidelity=self.fidelity)
        # σ² can be vanishingly small near training points; clamp to avoid
        # divide-by-zero blowing up acquisition there.
        var_cand = np.clip(var_cand, 1e-12, None)
        red = (K_mc_cand**2).sum(axis=0) / var_cand              # [C]

        out = np.zeros(n)
        out[feasible_idx] = red
        return out

    def best(
        self, theta_candidates: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Return ``(theta_next, best_score, all_scores)``."""
        scores = self.score(theta_candidates)
        if not np.any(scores > 0):
            raise RuntimeError(
                "All candidates have zero acquisition; relax feasibility or "
                "widen the candidate set."
            )
        best_idx = int(np.argmax(scores))
        return theta_candidates[best_idx].copy(), float(scores[best_idx]), scores


@dataclass
class ExpectedImprovementAcquisition:
    """Expected Improvement acquisition over a fitted MFGP.

    For minimization (``target='min'``) with current incumbent
    ``y_best`` and Gaussian posterior ``f(θ*) ~ N(μ, σ²)``:

        EI(θ*) = (y_best − μ)·Φ(z) + σ·φ(z),   z = (y_best − μ) / σ

    For maximization (``target='max'``):

        EI(θ*) = (μ − y_best)·Φ(z) + σ·φ(z),   z = (μ − y_best) / σ

    A small ``xi`` shift trades exploration vs exploitation (larger
    ``xi`` → more exploratory). Default ``xi=0.0`` matches the textbook
    formula. Φ, φ are the standard-normal CDF / PDF.

    Unlike :class:`IvrAcquisition` (pure exploration), EI is an
    exploitation-leaning acquisition: it concentrates probes where the
    posterior predicts an optimum. Stars cluster near the predicted
    optimum once σ is small, and explore high-σ regions while σ is
    large — both governed by the same closed-form.

    Parameters
    ----------
    mfgp
        A fitted :class:`MultiFidelityGP`.
    bounds
        Box bounds on θ. Used only for filtering candidates.
    incumbent
        Current best observed value (in the y-space the GP predicts at
        ``fidelity``). For our 3-fidelity setup the highest fidelity is
        ``y_raw = m/N``; pass the appropriate min / max of the HF
        observations.
    target
        ``'min'`` to find a minimum, ``'max'`` to find a maximum.
    xi
        Exploration trade-off shift. ``0.0`` is the default text-book
        formulation.
    fidelity
        Fidelity level at which the GP posterior is queried (default:
        highest, the y_raw target).
    feasibility_fn
        Optional ``(theta:[n,D]) -> bool[n]`` predicate; infeasible
        candidates score ``0.0``.
    """

    mfgp: MultiFidelityGP
    bounds: BoxBounds
    incumbent: float
    target: str = "max"
    xi: float = 0.0
    fidelity: int | None = None
    feasibility_fn: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.bounds.dim != self.mfgp.dim_theta:
            raise ValueError(
                f"bounds.dim={self.bounds.dim} != mfgp.dim_theta={self.mfgp.dim_theta}"
            )
        if self.target not in ("max", "min"):
            raise ValueError(f"target must be 'max' or 'min', got {self.target!r}")

    def score(self, theta_candidates: np.ndarray) -> np.ndarray:
        """EI score per candidate (higher = better).

        ``EI ≥ 0`` everywhere by construction; the standard-normal CDF
        and PDF are both non-negative, and the formula combines them
        linearly with non-negative coefficients.
        """
        if theta_candidates.ndim != 2 or theta_candidates.shape[1] != self.bounds.dim:
            raise ValueError(
                f"theta_candidates.shape={theta_candidates.shape}; "
                f"expected (n, {self.bounds.dim})"
            )
        n = theta_candidates.shape[0]
        feas = self.bounds.contains(theta_candidates)
        if self.feasibility_fn is not None:
            feas &= np.asarray(self.feasibility_fn(theta_candidates), dtype=bool)
        if not feas.any():
            return np.zeros(n)

        feasible_idx = np.where(feas)[0]
        feasible_theta = theta_candidates[feasible_idx]

        mu, var = self.mfgp.predict(feasible_theta, fidelity=self.fidelity)
        sigma = np.sqrt(np.clip(var, 0.0, None))

        # Direction sign: minimization wants μ < incumbent;
        # maximization wants μ > incumbent.
        if self.target == "min":
            improvement = (self.incumbent - self.xi) - mu
        else:
            improvement = mu - (self.incumbent + self.xi)

        # Gaussian-EI closed form. Where σ ≈ 0 the analytical EI is
        # exactly max(improvement, 0); we mirror that to dodge 0/0.
        ei = np.zeros_like(mu)
        positive_sigma = sigma > 1.0e-12
        z = np.where(positive_sigma, improvement / np.where(positive_sigma, sigma, 1.0), 0.0)
        # Standard-normal CDF / PDF without scipy dependency at call site.
        from math import erf, sqrt as _sqrt
        phi = np.exp(-0.5 * z * z) / _sqrt(2.0 * np.pi)
        cdf = 0.5 * (1.0 + np.vectorize(erf)(z / _sqrt(2.0)))
        ei_pos = improvement * cdf + sigma * phi
        ei[positive_sigma] = np.clip(ei_pos[positive_sigma], 0.0, None)
        ei[~positive_sigma] = np.clip(improvement[~positive_sigma], 0.0, None)

        out = np.zeros(n)
        out[feasible_idx] = ei
        return out

    def best(
        self, theta_candidates: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Return ``(theta_next, best_score, all_scores)``."""
        scores = self.score(theta_candidates)
        # Even a fully-explored GP with σ ≈ 0 can have EI ≈ 0 across all
        # candidates; in that case fall back to the closest tie.
        best_idx = int(np.argmax(scores))
        return theta_candidates[best_idx].copy(), float(scores[best_idx]), scores


def integrated_variance(
    mfgp: MultiFidelityGP,
    bounds: BoxBounds,
    *,
    n_mc_samples: int = 2000,
    fidelity: int | None = None,
    seed: int = 0,
) -> float:
    """Estimate ``∫_Θ σ²(θ) dθ`` by Monte Carlo over the box.

    The pass-criterion for the active-learning loop is that this number
    decreases across iterations.
    """
    rng = np.random.default_rng(seed)
    pts = bounds.sample_uniform(n_mc_samples, rng)
    _, var = mfgp.predict(pts, fidelity=fidelity)
    return float(var.mean()) * bounds.volume


# ---------------------------------------------------------------------------
# "Truth query" — sampling a fresh HF batch at a fixed θ_next.
# ---------------------------------------------------------------------------


def simulate_at_theta(
    generator: PseudoDataGenerator,
    cnp: ConditionalNeuralProcess,
    theta: np.ndarray,
    *,
    n_events: int,
    seed: int,
    n_mc_samples: int = 50,
) -> tuple[float, float]:
    """Run a fresh single-trial pseudo-simulation at a chosen ``θ``.

    Returns ``(beta_bar, y_raw)`` for the new HF trial. ``beta_bar`` is
    the CNP-aggregated continuous score; ``y_raw = m/N`` is the raw
    Bernoulli rate. Both are computed over the trial's *target* events
    (50/50 context/target split) — same convention as
    :func:`prepare_mfgp_datasets`, so the new HF observation is on the
    same scale as the existing dataset.

    EVENT_ONLY scenarios are rejected: there is no θ to fix.
    """
    if generator.mode is InputMode.EVENT_ONLY:
        raise ValueError(
            "simulate_at_theta needs a θ; generator mode=EVENT_ONLY has no θ."
        )
    if theta.ndim != 1 or theta.shape[0] != (generator.dim_theta or 0):
        raise ValueError(
            f"theta.shape={theta.shape}; expected ({generator.dim_theta},)"
        )

    truth = generator.truth
    rng = np.random.default_rng(seed)
    theta_row = theta.reshape(1, -1)

    if truth.mode is InputMode.FULL:
        phi = rng.uniform(-1.0, 1.0, size=(1, n_events, truth.dim_phi))
        p = truth.evaluate(theta=theta_row[:, None, :], phi=phi)         # [1, N]
        labels = (rng.uniform(size=(1, n_events)) < p).astype(np.int8)
        batch = StandardBatch(mode=InputMode.FULL, theta=theta_row, phi=phi, labels=labels)
    else:  # DESIGN_ONLY
        phi = None
        p_per_trial = truth.evaluate(theta=theta_row, phi=None)          # [1]
        p = np.broadcast_to(p_per_trial[:, None], (1, n_events))
        labels = (rng.uniform(size=(1, n_events)) < p).astype(np.int8)
        batch = StandardBatch(
            mode=InputMode.DESIGN_ONLY, theta=theta_row, phi=phi, labels=labels,
        )

    ctx, tgt = split_context_target(batch, n_context=n_events // 2, seed=seed + 1)
    pred = cnp_trial_predictive(cnp, ctx, tgt, n_mc_samples=n_mc_samples)
    beta_bar = float(pred["y_cnp"][0])
    y_raw = float(tgt.labels.mean(axis=1)[0])
    return beta_bar, y_raw


# ---------------------------------------------------------------------------
# Active-learning loop.
# ---------------------------------------------------------------------------


@dataclass
class ActiveLearningStep:
    """Snapshot of one IVR active-learning iteration.

    Records the chosen θ, the acquisition / variance surfaces it picked
    from, and the integrated variance *after* refitting — so the
    "variance shrinks across steps" gate can be evaluated post-hoc.

    Surface arrays are 1-D for ``dim_theta == 1`` and 2-D (``[G0, G1]``)
    for ``dim_theta == 2``. Higher-dim θ records ``None`` for the
    surfaces (no canonical 2-D projection); the loop still runs.
    """

    step: int
    theta_next: np.ndarray
    sampled_theta: np.ndarray
    grid_axes: list[np.ndarray] | None
    acquisition: np.ndarray | None
    sigma: np.ndarray | None
    integrated_variance_before: float
    integrated_variance_after: float
    beta_bar_obs: float
    y_raw_obs: float


@dataclass
class ActiveLearningLoop:
    """Drives an active-learning iteration over a fitted MFGP.

    Two acquisition modes are supported:

    * ``acquisition='ivr'`` (default) — pure exploration, picks the θ
      that minimizes integrated posterior variance. Direction-agnostic;
      the ``target`` flag is ignored at acquisition time.
    * ``acquisition='ei'`` — Expected Improvement, exploitation-leaning.
      The ``target`` flag (``'max'`` or ``'min'``) controls the
      direction of optimization; the incumbent for EI is the running
      best of ``Y_hf_raw`` in the appropriate direction.

    Each step:

    1. Score the chosen acquisition over a candidate set (grid for dim
       ≤ 2, uniform sample for higher-dim).
    2. Query the pseudo-truth at ``θ_next`` for one fresh HF trial.
    3. Append ``β̄`` and ``y_raw`` to the MFGP's HF datasets and refit.
    4. Record an :class:`ActiveLearningStep` with the surfaces & metrics.

    LF data stays fixed across steps — active learning only spends HF
    "budget", which matches the paper's framing where HF events are the
    expensive thing.
    """

    mfgp: MultiFidelityGP
    generator: PseudoDataGenerator
    cnp: ConditionalNeuralProcess
    bounds: BoxBounds
    data: dict[str, np.ndarray]
    n_hf_events: int = 128
    n_mc_samples: int = 1000
    n_candidates_per_axis: int = 50
    seed: int = 0
    refit_n_restarts: int = 5
    acquisition: str = "ivr"
    target: str = "max"
    ei_xi: float = 0.0
    _step: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.bounds.dim != self.mfgp.dim_theta:
            raise ValueError(
                f"bounds.dim={self.bounds.dim} != mfgp.dim_theta={self.mfgp.dim_theta}"
            )
        if self.acquisition not in ("ivr", "ei"):
            raise ValueError(
                f"acquisition must be 'ivr' or 'ei', got {self.acquisition!r}"
            )
        if self.target not in ("max", "min"):
            raise ValueError(f"target must be 'max' or 'min', got {self.target!r}")

    def _make_candidates(
        self,
    ) -> tuple[np.ndarray, list[np.ndarray] | None, tuple[int, ...] | None]:
        """Build candidate set + (optional) grid axes for plotting.

        For dim ≤ 2 we use a regular grid so the recorded surfaces are
        directly plottable; for higher-dim we fall back to a uniform
        sample and return ``None`` axes.
        """
        d = self.bounds.dim
        if d == 1:
            cands = self.bounds.grid_1d(self.n_candidates_per_axis)
            return cands, [cands[:, 0]], (self.n_candidates_per_axis,)
        if d == 2:
            ax0, ax1, flat = self.bounds.grid_2d(self.n_candidates_per_axis)
            return flat, [ax0, ax1], (
                self.n_candidates_per_axis, self.n_candidates_per_axis,
            )
        rng = np.random.default_rng(self.seed + self._step + 1)
        n = self.n_candidates_per_axis ** 2
        return self.bounds.sample_uniform(n, rng), None, None

    def _build_acquisition(self):
        """Build a fresh acquisition instance over the current MFGP."""
        if self.acquisition == "ivr":
            return IvrAcquisition(
                self.mfgp, self.bounds,
                n_mc_samples=self.n_mc_samples, fidelity=None,
                seed=self.seed + self._step,
            )
        # EI — incumbent is the current best of Y_hf_raw in the target's
        # direction (this is the y-space the GP predicts at the highest
        # fidelity, which matches what we'd compare against).
        y_hf = self.data["Y_hf_raw"].ravel()
        incumbent = float(y_hf.min() if self.target == "min" else y_hf.max())
        return ExpectedImprovementAcquisition(
            self.mfgp, self.bounds, incumbent=incumbent,
            target=self.target, xi=self.ei_xi, fidelity=None,
        )

    def step(self) -> ActiveLearningStep:
        """Run one acquire → query → refit cycle."""
        iv_before = integrated_variance(
            self.mfgp, self.bounds,
            n_mc_samples=2000, fidelity=None, seed=self.seed,
        )
        cands, axes, shape = self._make_candidates()
        acq = self._build_acquisition()
        theta_next, _, scores = acq.best(cands)
        _, var_cand = self.mfgp.predict(cands, fidelity=None)
        sigma_cand = np.sqrt(var_cand)

        beta_bar, y_raw = simulate_at_theta(
            self.generator, self.cnp, theta_next,
            n_events=self.n_hf_events,
            seed=self.seed + 1000 + self._step,
        )

        new_theta = theta_next.reshape(1, -1)
        self.data = {
            "X_lf": self.data["X_lf"],
            "Y_lf_cnp": self.data["Y_lf_cnp"],
            "X_hf": np.vstack([self.data["X_hf"], new_theta]),
            "Y_hf_cnp": np.vstack([self.data["Y_hf_cnp"], [[beta_bar]]]),
            "Y_hf_raw": np.vstack([self.data["Y_hf_raw"], [[y_raw]]]),
        }
        self.mfgp = MultiFidelityGP(
            n_fidelities=self.mfgp.n_fidelities,
            dim_theta=self.mfgp.dim_theta,
            kernel=self.mfgp.kernel_name,
            ard=self.mfgp.ard,
        ).fit(
            [self.data["X_lf"], self.data["X_hf"], self.data["X_hf"]],
            [self.data["Y_lf_cnp"], self.data["Y_hf_cnp"], self.data["Y_hf_raw"]],
            n_restarts=self.refit_n_restarts,
        )
        iv_after = integrated_variance(
            self.mfgp, self.bounds,
            n_mc_samples=2000, fidelity=None, seed=self.seed,
        )

        record = ActiveLearningStep(
            step=self._step,
            theta_next=theta_next,
            sampled_theta=self.data["X_hf"].copy(),
            grid_axes=axes,
            acquisition=scores.reshape(shape) if shape is not None else None,
            sigma=sigma_cand.reshape(shape) if shape is not None else None,
            integrated_variance_before=iv_before,
            integrated_variance_after=iv_after,
            beta_bar_obs=beta_bar,
            y_raw_obs=y_raw,
        )
        self._step += 1
        return record

    def run(self, n_steps: int) -> list[ActiveLearningStep]:
        """Run ``n_steps`` and return the list of step records."""
        return [self.step() for _ in range(n_steps)]
