"""Совместимость с scikit-learn."""

import numpy as np
import pytest
from conftest import make_data
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from trosna_linear_regression import TrosnaLinearRegression


@parametrize_with_checks(
    [
        TrosnaLinearRegression(),
        TrosnaLinearRegression(alpha=0.1, l1_ratio=0.5, repulsion_ratio=0.3, kernel="cauchy"),
        TrosnaLinearRegression(alpha=3.0, repulsion_ratio=1.0, repulsion_point=1.0),
        TrosnaLinearRegression(alpha=0.0, fit_intercept=False),
        TrosnaLinearRegression(alpha=1.0, repulsion_ratio=0.5, kernel="log_barrier"),
        TrosnaLinearRegression(alpha=0.1, l1_ratio=0.3, repulsion_ratio=0.5, kernel="inverse_barrier"),
        TrosnaLinearRegression(
            alpha=0.5, repulsion_ratio=0.5, kernel="log_barrier", repulsion_side="above", repulsion_point=-10.0
        ),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_get_set_params_and_clone():
    m = TrosnaLinearRegression(alpha=2.0, l1_ratio=0.3, repulsion_point=5.0)
    params = m.get_params()
    assert params["alpha"] == 2.0
    assert params["repulsion_point"] == 5.0
    assert set(params) == {
        "alpha",
        "l1_ratio",
        "repulsion_ratio",
        "repulsion_point",
        "repulsion_width",
        "kernel",
        "repulsion_side",
        "fit_intercept",
        "tol",
        "max_iter",
    }
    c = clone(m.set_params(kernel="cauchy"))
    assert c.kernel == "cauchy"
    assert not hasattr(c, "coef_")


def test_pipeline_and_grid_search(rng):
    X, y = make_data(rng, n=150)
    pipe = make_pipeline(StandardScaler(), TrosnaLinearRegression())
    grid = GridSearchCV(
        pipe,
        {
            "trosnalinearregression__alpha": [0.01, 0.1, 1.0],
            "trosnalinearregression__l1_ratio": [0.0, 0.3],
            "trosnalinearregression__repulsion_ratio": [0.0, 0.5],
        },
        cv=3,
    ).fit(X, y)
    assert grid.best_score_ > 0.8
    assert cross_val_score(TrosnaLinearRegression(alpha=0.01), X, y, cv=3).mean() > 0.8


def test_fitted_attributes(rng):
    X, y = make_data(rng)
    m = TrosnaLinearRegression(l1_ratio=0.2, repulsion_ratio=0.3).fit(X, y)
    assert m.coef_.shape == (X.shape[1],)
    assert isinstance(m.intercept_, float)
    assert m.l2_ratio_ == pytest.approx(0.5)
    assert m.n_features_in_ == X.shape[1]
    assert m.n_iter_ >= 1
    assert np.isfinite(m.objective_)


def test_predict_before_fit():
    with pytest.raises(NotFittedError):
        TrosnaLinearRegression().predict(np.ones((2, 2)))


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"l1_ratio": 0.7, "repulsion_ratio": 0.5}, "не может превышать 1"),
        ({"l1_ratio": -0.1}, "неотрицательными"),
        ({"alpha": -1.0}, "alpha"),
        ({"repulsion_width": -1.0}, "ширина"),
        ({"kernel": "box"}, "ядро"),
        ({"tol": 0.0}, "tol"),
        ({"max_iter": 0}, "max_iter"),
        ({"fit_intercept": "yes"}, "fit_intercept"),
    ],
)
def test_invalid_params(params, match):
    with pytest.raises(ValueError, match=match):
        TrosnaLinearRegression(**params).fit(np.eye(3), np.arange(3.0))


@pytest.mark.parametrize("w", [np.ones(2), -np.ones(3), np.zeros(3), np.array([1.0, np.nan, 1.0])])
def test_invalid_sample_weight(w):
    with pytest.raises(ValueError, match="sample_weight"):
        TrosnaLinearRegression().fit(np.eye(3), np.arange(3.0), w)


def test_convergence_warning(rng):
    X, y = make_data(rng)
    with pytest.warns(ConvergenceWarning):
        TrosnaLinearRegression(alpha=3.0, repulsion_ratio=1.0, max_iter=1).fit(X, y)
