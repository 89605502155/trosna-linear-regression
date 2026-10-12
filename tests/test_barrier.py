"""Барьерное отталкивание: от «чуть-чуть» до «точки быть не может»."""

import numpy as np
import pytest
from conftest import make_data

from trosna_linear_regression import Penalty, TrosnaLinearRegression, fit, get_kernel, objective

BARRIERS = ["log_barrier", "inverse_barrier"]


def barrier_model(kernel, alpha, width=0.5, point=0.0):
    return TrosnaLinearRegression(
        alpha=alpha, repulsion_ratio=1.0, kernel=kernel, repulsion_width=width, repulsion_point=point
    )


@pytest.mark.parametrize("kernel", BARRIERS)
def test_no_effect_when_predictions_are_outside_zone(rng, kernel):
    """Если МНК и так держит все предсказания вне зоны, барьер ничего не меняет."""
    X, y = make_data(rng)
    ols = TrosnaLinearRegression(alpha=0.0).fit(X, y)
    gap = np.min(np.abs(ols.predict(X) + 50.0))
    model = barrier_model(kernel, alpha=10.0, width=0.9 * gap, point=-50.0).fit(X, y)
    np.testing.assert_allclose(model.coef_, ols.coef_, atol=1e-8)


@pytest.mark.parametrize("kernel", BARRIERS)
def test_weak_barrier_barely_changes_fit(rng, kernel):
    """Малая сила — отталкивание «чуть-чуть»: качество почти как у МНК."""
    X, y = make_data(rng, n=300)
    r2_ols = TrosnaLinearRegression(alpha=0.0).fit(X, y).score(X, y)
    r2 = barrier_model(kernel, alpha=1e-4).fit(X, y).score(X, y)
    assert r2 == pytest.approx(r2_ols, abs=1e-3)


@pytest.mark.parametrize("kernel", BARRIERS)
def test_stronger_barrier_clears_zone(rng, kernel):
    """Прямая проходит рядом с точкой: с ростом силы штраф в решении не растёт (стороны общие), зона пустеет."""
    X = rng.normal(size=(300, 1))
    y = 1.0 + 0.4 * X[:, 0] + rng.normal(scale=0.2, size=300)
    penalty, in_zone = [], []
    for alpha in [0.0, 0.1, 1.0, 10.0]:
        model = barrier_model(kernel, alpha, width=0.6).fit(X, y)
        pred = model.predict(X)
        penalty.append(get_kernel(kernel).value(pred, 0.6).mean())
        in_zone.append(np.mean(np.abs(pred) < 0.6))
    assert np.all(np.diff(penalty) <= 1e-12 * penalty[0])
    assert in_zone[0] > 0.05
    assert in_zone[-1] < 0.8 * in_zone[0]


@pytest.mark.parametrize("kernel", BARRIERS)
def test_predictions_never_cross_point(rng, kernel):
    """Каждое обучающее предсказание остаётся по ту же сторону от точки, что и у старта (МНК)."""
    X, y = make_data(rng, n=200, shift=0.0)
    side0 = np.sign(TrosnaLinearRegression(alpha=0.0).fit(X, y).predict(X))
    for alpha in [0.01, 1.0, 100.0]:
        pred = barrier_model(kernel, alpha).fit(X, y).predict(X)
        np.testing.assert_array_equal(np.sign(pred), side0)
        assert np.min(np.abs(pred)) > 0


@pytest.mark.parametrize("kernel", BARRIERS)
def test_barrier_objective_is_finite_and_not_worse_than_start(rng, kernel):
    X, y = make_data(rng, shift=0.0)
    pen = Penalty(alpha=5.0, l1_ratio=0.1, l2_ratio=0.0, repulsion_ratio=0.9, width=0.4, kernel=kernel)
    assert not pen.is_convex
    start = fit(X, y, pen.without_repulsion())
    res = fit(X, y, pen)
    assert res.converged
    assert np.isfinite(res.objective)
    assert res.objective <= objective(X, y, start.coef, start.intercept, pen)


@pytest.mark.parametrize("kernel", BARRIERS)
def test_unique_solution_given_sides(rng, kernel):
    """При фиксированных сторонах задача выпукла: разные старты с теми же сторонами сходятся в одно."""
    X = rng.normal(size=(150, 2))
    y = 1.0 + X @ [0.3, -0.2] + rng.normal(scale=0.2, size=150)
    pen = Penalty(alpha=2.0, l1_ratio=0.0, l2_ratio=0.2, repulsion_ratio=0.8, width=0.5, kernel=kernel)
    starts = [(np.zeros(2), 1.0), (np.array([0.05, 0.05]), 3.0), (np.array([-0.05, 0.02]), 0.8)]
    sols = []
    for coef0, b0 in starts:
        assert np.all(X @ coef0 + b0 > 0)  # все стартовые предсказания над точкой
        sols.append(fit(X, y, pen, tol=1e-12, coef_init=coef0, intercept_init=b0))
    for other in sols[1:]:
        np.testing.assert_allclose(other.coef, sols[0].coef, atol=1e-6)
        assert other.intercept == pytest.approx(sols[0].intercept, abs=1e-6)


def test_inverse_barrier_keeps_further_than_log(rng):
    X = rng.normal(size=(300, 1))
    y = 0.3 + 0.4 * X[:, 0] + rng.normal(scale=0.2, size=300)
    near = {}
    for kernel in BARRIERS:
        pred = barrier_model(kernel, alpha=1.0, width=0.3).fit(X, y).predict(X)
        near[kernel] = np.mean(np.abs(pred) < 0.1)
    assert near["inverse_barrier"] <= near["log_barrier"]


def test_unavoidable_point_stays_finite():
    """Без свободного члена нулевая строка всегда предсказывает 0 = c: штраф огромен, но конечен."""
    X = np.vstack([np.zeros((1, 2)), np.random.default_rng(0).normal(size=(30, 2))])
    y = X @ [1.0, -1.0] + 2.0
    model = TrosnaLinearRegression(alpha=1.0, repulsion_ratio=0.5, kernel="inverse_barrier", fit_intercept=False)
    model.fit(X, y)
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.objective_)
