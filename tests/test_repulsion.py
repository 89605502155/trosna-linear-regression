"""Поведение отталкивающей регуляризации."""

import numpy as np
import pytest
from conftest import make_data

from trosna_linear_regression import TrosnaLinearRegression, get_kernel


def mean_repulsion(model, X, point):
    pred = model.predict(X)
    return get_kernel(model.kernel).value(pred - point, model.repulsion_width_).mean()


@pytest.mark.parametrize("kernel", ["gaussian", "cauchy"])
@pytest.mark.parametrize("point", [0.0, 1.5])
def test_stronger_repulsion_moves_predictions_away(rng, kernel, point):
    """Штраф отталкивания в решении не растёт с увеличением его силы (свойство оптимума)."""
    X, y = make_data(rng, n=200)
    values = []
    for alpha in [0.0, 0.25, 0.5, 1.0]:
        model = TrosnaLinearRegression(alpha=alpha, repulsion_ratio=1.0, repulsion_point=point, kernel=kernel)
        values.append(mean_repulsion(model.fit(X, y), X, point))
    assert np.all(np.diff(values) <= 1e-10)
    assert values[-1] < values[0]


def test_fewer_predictions_near_point(rng):
    """Слабая зависимость от признака: отталкивание уводит всю прямую от точки 0."""
    X = rng.normal(size=(300, 1))
    y = 0.4 + 0.3 * X[:, 0] + rng.normal(scale=0.3, size=300)
    base = TrosnaLinearRegression(alpha=0.0).fit(X, y).predict(X)
    rep = TrosnaLinearRegression(alpha=3.0, repulsion_ratio=1.0, repulsion_width=0.5).fit(X, y).predict(X)
    assert np.mean(np.abs(rep) < 0.2) < 0.5 * np.mean(np.abs(base) < 0.2)
    assert rep.mean() > base.mean()


def test_shift_equivariance(rng):
    """Сдвиг y и точки на одну константу сдвигает только свободный член."""
    X, y = make_data(rng)
    a = TrosnaLinearRegression(alpha=2.0, l1_ratio=0.2, repulsion_ratio=0.6, repulsion_point=0.5).fit(X, y)
    b = TrosnaLinearRegression(alpha=2.0, l1_ratio=0.2, repulsion_ratio=0.6, repulsion_point=10.5).fit(X, y + 10)
    np.testing.assert_allclose(b.coef_, a.coef_, atol=1e-6)
    assert b.intercept_ == pytest.approx(a.intercept_ + 10, abs=1e-6)


@pytest.mark.parametrize("kernel", ["gaussian", "cauchy"])
def test_scale_equivariance(rng, kernel):
    """Без L1 и с шириной по умолчанию: y → k·y, c → k·c даёт β → k·β, b → k·b."""
    X, y = make_data(rng)
    params = {"alpha": 3.0, "repulsion_ratio": 0.7, "kernel": kernel}
    a = TrosnaLinearRegression(repulsion_point=0.4, **params).fit(X, y)
    b = TrosnaLinearRegression(repulsion_point=0.4 * 7, **params).fit(X, 7 * y)
    assert b.repulsion_width_ == pytest.approx(7 * a.repulsion_width_)
    np.testing.assert_allclose(b.coef_, 7 * a.coef_, rtol=1e-5, atol=1e-6)
    assert b.intercept_ == pytest.approx(7 * a.intercept_, rel=1e-5, abs=1e-6)


def test_default_width_is_weighted_std(rng):
    X, y = make_data(rng)
    w = rng.uniform(0.1, 2.0, size=y.size)
    m = TrosnaLinearRegression().fit(X, y, w)
    mean = np.average(y, weights=w)
    assert m.repulsion_width_ == pytest.approx(np.sqrt(np.average((y - mean) ** 2, weights=w)))
    assert TrosnaLinearRegression().fit(X, np.full(y.size, 3.0)).repulsion_width_ == 1.0


def test_integer_weights_equal_repeated_rows(rng):
    X, y = make_data(rng, n=40)
    w = rng.integers(1, 4, size=40)
    params = {"alpha": 4.0, "l1_ratio": 0.1, "repulsion_ratio": 0.8, "repulsion_width": 1.0}
    a = TrosnaLinearRegression(**params).fit(X, y, w)
    b = TrosnaLinearRegression(**params).fit(np.repeat(X, w, axis=0), np.repeat(y, w))
    np.testing.assert_allclose(a.coef_, b.coef_, atol=1e-6)
    assert a.intercept_ == pytest.approx(b.intercept_, abs=1e-6)


def test_zero_weight_equals_dropping_rows(rng):
    X, y = make_data(rng, n=50)
    w = np.ones(50)
    w[::5] = 0
    params = {"alpha": 0.5, "repulsion_ratio": 0.5, "repulsion_width": 1.0}
    a = TrosnaLinearRegression(**params).fit(X, y, w)
    b = TrosnaLinearRegression(**params).fit(X[w > 0], y[w > 0])
    np.testing.assert_allclose(a.coef_, b.coef_, atol=1e-6)
