"""Multi-Fidelity Gaussian Process surrogate (design-level).

Wraps Emukit's :class:`GPyLinearMultiFidelityModel` (linear recursive
co-kriging) into a small, schema-friendly API. The standard 3-fidelity
RESuM stack is:

  * f_0 (LF)   — ``β̄(θ)`` from low-fidelity events
  * f_1 (MF)   — ``β̄(θ)`` from high-fidelity events
  * f_2 (HF)   — ``y_raw = m/N`` from high-fidelity events  (target)

with the recursion ``f_{i+1}(θ) = ρ_i · f_i(θ) + δ_i(θ)`` learned by
co-kriging. The module is **numpy / GPy only** — no torch import here,
per the project's hard decoupling rule.

A fitted MFGP can be persisted with :func:`save_mfgp` / :func:`load_mfgp`
— pickle-backed, so use only with files from trusted sources.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import GPy
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel

from schemas.data_models import ModelPrediction


_KERNEL_FACTORIES = {
    "rbf": lambda d, ard: GPy.kern.RBF(d, ARD=ard),
    "matern52": lambda d, ard: GPy.kern.Matern52(d, ARD=ard),
}


class MultiFidelityGP:
    """Linear recursive co-kriging GP across ``n_fidelities`` levels.

    Construction is two-step: ``__init__`` declares the architecture,
    ``fit(X_list, Y_list)`` consumes the per-fidelity datasets and runs
    Emukit's optimizer with restarts.
    """

    def __init__(
        self,
        n_fidelities: int,
        dim_theta: int,
        *,
        kernel: str = "rbf",
        ard: bool = True,
    ) -> None:
        if n_fidelities < 2:
            raise ValueError(f"n_fidelities must be ≥ 2, got {n_fidelities}")
        if kernel not in _KERNEL_FACTORIES:
            raise ValueError(
                f"unknown kernel {kernel!r}; supported: {sorted(_KERNEL_FACTORIES)}"
            )
        if dim_theta <= 0:
            raise ValueError(f"dim_theta must be positive, got {dim_theta}")
        self.n_fidelities = n_fidelities
        self.dim_theta = dim_theta
        self.kernel_name = kernel
        self.ard = ard
        self._model: GPyLinearMultiFidelityModel | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> GPyLinearMultiFidelityModel:
        if self._model is None:
            raise RuntimeError("model not fitted; call .fit() first")
        return self._model

    def _build_kernel(self) -> LinearMultiFidelityKernel:
        factory = _KERNEL_FACTORIES[self.kernel_name]
        kernels = [factory(self.dim_theta, self.ard) for _ in range(self.n_fidelities)]
        return LinearMultiFidelityKernel(kernels)

    def fit(
        self,
        X_list: list[np.ndarray],
        Y_list: list[np.ndarray],
        *,
        n_restarts: int = 5,
        verbose: bool = False,
    ) -> "MultiFidelityGP":
        """Fit the GP on per-fidelity ``(X, Y)`` arrays.

        ``X_list[i]`` has shape ``(n_i, dim_theta)`` and ``Y_list[i]``
        has shape ``(n_i, 1)``. ``i`` indexes fidelity from low (0) to
        high (``n_fidelities - 1``).
        """
        if len(X_list) != self.n_fidelities or len(Y_list) != self.n_fidelities:
            raise ValueError(
                f"expected {self.n_fidelities} X / Y arrays, "
                f"got {len(X_list)} / {len(Y_list)}"
            )
        for i, (X, Y) in enumerate(zip(X_list, Y_list, strict=True)):
            if X.ndim != 2 or X.shape[1] != self.dim_theta:
                raise ValueError(
                    f"X_list[{i}].shape={X.shape}; expected (n_i, {self.dim_theta})"
                )
            if Y.ndim != 2 or Y.shape[1] != 1 or Y.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Y_list[{i}].shape={Y.shape}; expected ({X.shape[0]}, 1)"
                )

        X_train, Y_train = convert_xy_lists_to_arrays(X_list, Y_list)
        model = GPyLinearMultiFidelityModel(
            X_train, Y_train, self._build_kernel(), n_fidelities=self.n_fidelities,
        )
        model.optimize_restarts(num_restarts=n_restarts, verbose=verbose)
        self._model = model
        return self

    # ---- prediction --------------------------------------------------------

    def predict(
        self, X_new: np.ndarray, fidelity: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and variance at fidelity ``f`` (default: highest).

        Returns 1-D arrays of length ``n``. Variance is the GP's posterior
        variance at the fidelity level — non-negative by construction.
        """
        if X_new.ndim != 2 or X_new.shape[1] != self.dim_theta:
            raise ValueError(
                f"X_new.shape={X_new.shape}; expected (n, {self.dim_theta})"
            )
        f = self._resolve_fidelity(fidelity)
        fid_col = np.full((X_new.shape[0], 1), f, dtype=float)
        X_aug = np.concatenate([X_new, fid_col], axis=1)
        # GPy's mixed-noise likelihood needs the per-row fidelity index in
        # Y_metadata or it raises NoneType errors during predict.
        mean, var = self.model.predict(
            X_aug, Y_metadata={"output_index": fid_col.astype(int)},
        )
        return mean.flatten(), np.clip(var.flatten(), 0.0, None)

    def predict_as_model_prediction(
        self, X_new: np.ndarray, fidelity: int | None = None,
    ) -> ModelPrediction:
        """Convenience wrapper returning the project's typed prediction schema."""
        mean, var = self.predict(X_new, fidelity)
        return ModelPrediction(mean=mean, variance=var, theta_query=X_new)

    def _resolve_fidelity(self, fidelity: int | None) -> int:
        if self._model is None:
            raise RuntimeError("model not fitted; call .fit() first")
        if fidelity is None:
            return self.n_fidelities - 1
        if not 0 <= fidelity < self.n_fidelities:
            raise ValueError(
                f"fidelity must be in [0, {self.n_fidelities - 1}], got {fidelity}"
            )
        return fidelity


# ---------------------------------------------------------------------------
# Persistence.
# ---------------------------------------------------------------------------


def save_mfgp(path: str | Path, mfgp: MultiFidelityGP) -> Path:
    """Pickle a fitted :class:`MultiFidelityGP` to ``path``.

    Captures the full object including the underlying GPy model state,
    so :func:`load_mfgp` returns a model that produces identical
    ``predict`` outputs without needing to refit.

    .. warning::
       Pickle is unsafe to load from untrusted sources. Use only with
       files you produced (or trust the producer of).

    Returns the resolved :class:`Path` for chaining.
    """
    if not mfgp.is_fitted:
        raise RuntimeError("cannot save an unfitted MFGP; call .fit() first")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(mfgp, f)
    return path


def load_mfgp(path: str | Path) -> MultiFidelityGP:
    """Load a pickled :class:`MultiFidelityGP` from ``path``.

    Validates that the unpickled object is a fitted ``MultiFidelityGP``
    so a wrong-type file fails loudly instead of returning silently
    broken state.
    """
    path = Path(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, MultiFidelityGP):
        raise TypeError(
            f"file {path} did not contain a MultiFidelityGP "
            f"(got {type(obj).__name__})"
        )
    if not obj.is_fitted:
        raise RuntimeError(f"loaded MFGP from {path} is not fitted")
    return obj
