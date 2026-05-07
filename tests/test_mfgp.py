"""Phase 4 contract tests for MultiFidelityGP.

The big "does it actually fit physical data" test lives in
:mod:`tests.test_mfgp_recovery` (the coverage gate). These tests
exercise the small surface: shape checks, prediction smoke tests on a
toy 2-fidelity sin(x) problem, and the ``ModelPrediction`` adapter.

GPy / paramz emit a flock of LaTeX-docstring SyntaxWarnings on import
in Python 3.12 — they're not our problem. The pytest config in
``pyproject.toml`` filters them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from core import MultiFidelityGP
from schemas.data_models import ModelPrediction

# Suppress noise from GPy / paramz LaTeX docstrings that survives the
# pyproject filter on some platforms.
warnings.filterwarnings("ignore", category=SyntaxWarning)


# ---------------------------------------------------------------------------
# Constructor.
# ---------------------------------------------------------------------------


def test_invalid_n_fidelities_raises() -> None:
    with pytest.raises(ValueError, match="n_fidelities"):
        MultiFidelityGP(n_fidelities=1, dim_theta=1)


def test_invalid_kernel_raises() -> None:
    with pytest.raises(ValueError, match="unknown kernel"):
        MultiFidelityGP(n_fidelities=2, dim_theta=1, kernel="cubic")


def test_invalid_dim_theta_raises() -> None:
    with pytest.raises(ValueError, match="dim_theta"):
        MultiFidelityGP(n_fidelities=2, dim_theta=0)


def test_predict_before_fit_raises() -> None:
    gp = MultiFidelityGP(n_fidelities=2, dim_theta=1)
    with pytest.raises(RuntimeError, match="not fitted"):
        gp.predict(np.array([[0.0]]))


# ---------------------------------------------------------------------------
# Toy 2-fidelity smoke test.
# ---------------------------------------------------------------------------


def _toy_two_fidelity_data(n_lf: int = 25, n_hf: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_lf = np.linspace(-1, 1, n_lf).reshape(-1, 1)
    Y_lf = np.sin(2 * X_lf) + 0.05 * rng.standard_normal((n_lf, 1))
    X_hf = np.linspace(-1, 1, n_hf).reshape(-1, 1)
    Y_hf = np.sin(2 * X_hf) + 0.3 * X_hf + 0.02 * rng.standard_normal((n_hf, 1))
    return X_lf, Y_lf, X_hf, Y_hf


def test_fit_and_predict_shapes() -> None:
    X_lf, Y_lf, X_hf, Y_hf = _toy_two_fidelity_data()
    gp = MultiFidelityGP(n_fidelities=2, dim_theta=1).fit(
        [X_lf, X_hf], [Y_lf, Y_hf], n_restarts=2,
    )
    assert gp.is_fitted

    X_test = np.linspace(-1, 1, 30).reshape(-1, 1)
    mean, var = gp.predict(X_test)
    assert mean.shape == (30,)
    assert var.shape == (30,)
    assert np.all(var >= 0)
    assert np.all(np.isfinite(mean))


def test_predict_at_lf_and_hf_differ() -> None:
    """For a problem where LF ≠ HF, predictions at fidelities 0 and 1 differ."""
    X_lf, Y_lf, X_hf, Y_hf = _toy_two_fidelity_data()
    gp = MultiFidelityGP(n_fidelities=2, dim_theta=1).fit(
        [X_lf, X_hf], [Y_lf, Y_hf], n_restarts=2,
    )
    X_test = np.linspace(-1, 1, 30).reshape(-1, 1)
    mu_lf, _ = gp.predict(X_test, fidelity=0)
    mu_hf, _ = gp.predict(X_test, fidelity=1)
    # They should differ on average — the HF data was offset by 0.3·x.
    assert np.mean(np.abs(mu_lf - mu_hf)) > 0.05


def test_predict_at_invalid_fidelity_raises() -> None:
    X_lf, Y_lf, X_hf, Y_hf = _toy_two_fidelity_data()
    gp = MultiFidelityGP(n_fidelities=2, dim_theta=1).fit(
        [X_lf, X_hf], [Y_lf, Y_hf], n_restarts=2,
    )
    with pytest.raises(ValueError, match="fidelity must be"):
        gp.predict(np.array([[0.0]]), fidelity=2)
    with pytest.raises(ValueError, match="fidelity must be"):
        gp.predict(np.array([[0.0]]), fidelity=-1)


def test_predict_dim_mismatch_raises() -> None:
    X_lf, Y_lf, X_hf, Y_hf = _toy_two_fidelity_data()
    gp = MultiFidelityGP(n_fidelities=2, dim_theta=1).fit(
        [X_lf, X_hf], [Y_lf, Y_hf], n_restarts=2,
    )
    with pytest.raises(ValueError, match="X_new.shape"):
        gp.predict(np.zeros((5, 2)))  # wrong dim


def test_fit_dim_or_count_mismatch_raises() -> None:
    X_lf, Y_lf, X_hf, Y_hf = _toy_two_fidelity_data()
    gp = MultiFidelityGP(n_fidelities=2, dim_theta=1)
    # Wrong number of fidelities passed
    with pytest.raises(ValueError, match="expected 2"):
        gp.fit([X_lf], [Y_lf])
    # Y has wrong second dim
    bad_Y = Y_hf.flatten()
    with pytest.raises(ValueError, match=r"Y_list\[1\]"):
        gp.fit([X_lf, X_hf], [Y_lf, bad_Y])
    # X has wrong feature dim
    with pytest.raises(ValueError, match=r"X_list\[0\]"):
        gp.fit([np.zeros((5, 2)), X_hf], [np.zeros((5, 1)), Y_hf])


# ---------------------------------------------------------------------------
# Interpolation sanity: fitted model passes near training HF points.
# ---------------------------------------------------------------------------


def test_predict_interpolates_hf_training_points() -> None:
    """At HF training θ, posterior mean should sit within a few σ of Y_hf."""
    X_lf, Y_lf, X_hf, Y_hf = _toy_two_fidelity_data()
    gp = MultiFidelityGP(n_fidelities=2, dim_theta=1).fit(
        [X_lf, X_hf], [Y_lf, Y_hf], n_restarts=3,
    )
    mu, var = gp.predict(X_hf, fidelity=1)
    sigma = np.sqrt(var) + 1e-3
    z = np.abs(mu - Y_hf.flatten()) / sigma
    # Even with optimization stochasticity, every HF training point should
    # be inside ±5σ of the posterior. (Usually much tighter.)
    assert np.all(z < 5.0), f"max |z| = {z.max():.3f}"


# ---------------------------------------------------------------------------
# ModelPrediction adapter.
# ---------------------------------------------------------------------------


def test_predict_as_model_prediction_returns_schema() -> None:
    X_lf, Y_lf, X_hf, Y_hf = _toy_two_fidelity_data()
    gp = MultiFidelityGP(n_fidelities=2, dim_theta=1).fit(
        [X_lf, X_hf], [Y_lf, Y_hf], n_restarts=2,
    )
    X_test = np.linspace(-1, 1, 7).reshape(-1, 1)
    pred = gp.predict_as_model_prediction(X_test)
    assert isinstance(pred, ModelPrediction)
    assert pred.mean.shape == (7,)
    assert pred.variance.shape == (7,)
    assert pred.theta_query.shape == (7, 1)
    assert np.all(pred.variance >= 0)


# ---------------------------------------------------------------------------
# 2-D θ.
# ---------------------------------------------------------------------------


def test_fit_and_predict_2d_theta() -> None:
    rng = np.random.default_rng(0)
    n_lf, n_hf = 40, 8
    X_lf = rng.uniform(-1, 1, size=(n_lf, 2))
    Y_lf = (np.sin(2 * X_lf[:, 0]) * np.cos(2 * X_lf[:, 1])).reshape(-1, 1)
    X_hf = rng.uniform(-1, 1, size=(n_hf, 2))
    Y_hf = (np.sin(2 * X_hf[:, 0]) * np.cos(2 * X_hf[:, 1])).reshape(-1, 1) + 0.2

    gp = MultiFidelityGP(n_fidelities=2, dim_theta=2).fit(
        [X_lf, X_hf], [Y_lf, Y_hf], n_restarts=2,
    )
    X_test = rng.uniform(-1, 1, size=(20, 2))
    mean, var = gp.predict(X_test)
    assert mean.shape == (20,)
    assert var.shape == (20,)
    assert np.all(var >= 0)


# ---------------------------------------------------------------------------
# save_mfgp / load_mfgp.
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_predicts_identically(tmp_path) -> None:
    from core import load_mfgp, save_mfgp
    X_lf, Y_lf, X_hf, Y_hf = _toy_two_fidelity_data()
    gp = MultiFidelityGP(n_fidelities=2, dim_theta=1).fit(
        [X_lf, X_hf], [Y_lf, Y_hf], n_restarts=2,
    )
    path = save_mfgp(tmp_path / "mfgp.pkl", gp)
    assert path.exists()
    gp2 = load_mfgp(path)
    X_test = np.linspace(-1.0, 1.0, 9).reshape(-1, 1)
    mu1, var1 = gp.predict(X_test)
    mu2, var2 = gp2.predict(X_test)
    np.testing.assert_allclose(mu1, mu2)
    np.testing.assert_allclose(var1, var2)


def test_save_unfitted_raises(tmp_path) -> None:
    from core import save_mfgp
    gp = MultiFidelityGP(n_fidelities=2, dim_theta=1)
    with pytest.raises(RuntimeError, match="unfitted"):
        save_mfgp(tmp_path / "x.pkl", gp)


def test_load_wrong_type_raises(tmp_path) -> None:
    import pickle
    from core import load_mfgp
    path = tmp_path / "wrong.pkl"
    with open(path, "wb") as f:
        pickle.dump({"not": "an mfgp"}, f)
    with pytest.raises(TypeError, match="MultiFidelityGP"):
        load_mfgp(path)
