"""Без отталкивания модель совпадает с МНК, Ridge, Lasso и ElasticNet из scikit-learn."""

import numpy as np
import pytest
from conftest import make_data
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from trosna_linear_regression import TrosnaLinearRegression

ATOL = 1e-6


def assert_same(ours, ref):
    np.testing.assert_allclose(ours.coef_, ref.coef_, atol=ATOL)
    assert ours.intercept_ == pytest.approx(ref.intercept_, abs=ATOL)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_ols(rng, fit_intercept, weighted):
    X, y = make_data(rng)
    w = rng.uniform(0.2, 3.0, size=y.size) if weighted else None
    ours = TrosnaLinearRegression(alpha=0.0, fit_intercept=fit_intercept).fit(X, y, w)
    assert_same(ours, LinearRegression(fit_intercept=fit_intercept).fit(X, y, w))


@pytest.mark.parametrize("alpha", [0.01, 0.5, 5.0])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_ridge(rng, alpha, fit_intercept):
    """½/n ‖r‖² + ½ α ‖β‖²  ⇔  Ridge: ‖r‖² + nα ‖β‖²."""
    X, y = make_data(rng)
    ours = TrosnaLinearRegression(alpha=alpha, repulsion_ratio=0.0, fit_intercept=fit_intercept).fit(X, y)
    ref = Ridge(alpha=alpha * y.size, fit_intercept=fit_intercept).fit(X, y)
    assert_same(ours, ref)


@pytest.mark.parametrize("alpha", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_lasso(rng, alpha, fit_intercept):
    X, y = make_data(rng, p=6)
    ours = TrosnaLinearRegression(alpha=alpha, l1_ratio=1.0, repulsion_ratio=0.0, fit_intercept=fit_intercept)
    ref = Lasso(alpha=alpha, fit_intercept=fit_intercept, tol=1e-12, max_iter=100_000)
    ours.fit(X, y)
    ref.fit(X, y)
    assert_same(ours, ref)
    np.testing.assert_array_equal(ours.coef_ == 0, ref.coef_ == 0)


@pytest.mark.parametrize("l1_ratio", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("weighted", [False, True])
def test_elastic_net(rng, l1_ratio, weighted):
    X, y = make_data(rng, p=5)
    w = rng.uniform(0.2, 3.0, size=y.size) if weighted else None
    ours = TrosnaLinearRegression(alpha=0.2, l1_ratio=l1_ratio, repulsion_ratio=0.0).fit(X, y, w)
    ref = ElasticNet(alpha=0.2, l1_ratio=l1_ratio, tol=1e-12, max_iter=100_000).fit(X, y, w)
    assert_same(ours, ref)


def test_far_point_has_no_effect(rng):
    """Если точка далеко от всех предсказаний, отталкивание практически не действует."""
    X, y = make_data(rng)
    ref = TrosnaLinearRegression(alpha=0.0).fit(X, y)
    far = TrosnaLinearRegression(alpha=1.0, repulsion_ratio=1.0, repulsion_point=1e3, repulsion_width=1.0).fit(X, y)
    assert_same(far, ref)
