"""Сходимость: диагностика, согласие решателей, честные предупреждения."""

import numpy as np
import pytest
from conftest import make_data
from sklearn.exceptions import ConvergenceWarning

from trosna_linear_regression import Penalty, TrosnaLinearRegression, fit
from trosna_linear_regression.core import _lbfgs_minimize, _newton_minimize, _Problem


@pytest.mark.parametrize(
    "pen",
    [
        Penalty(alpha=0.5, l1_ratio=0.0, l2_ratio=0.5, repulsion_ratio=0.5),
        Penalty(alpha=0.9, l1_ratio=0.3, l2_ratio=0.0, repulsion_ratio=0.7, kernel="cauchy"),
        Penalty(alpha=0.2, l1_ratio=0.5, l2_ratio=0.5, repulsion_ratio=0.0),
    ],
)
def test_newton_and_lbfgs_agree_on_convex_problems(rng, pen):
    """Два независимых решателя приходят в одну точку — значит, оба действительно сошлись."""
    X, y = make_data(rng, p=5)
    s = np.full(y.size, 1.0 / y.size)
    problem = _Problem(X, y, s, pen, True)
    x0 = np.zeros(6)
    xn, _, ok_n, _ = _newton_minimize(problem, x0, 1e-11, 1000, [])
    xl, _, ok_l, _ = _lbfgs_minimize(problem, x0, 1e-11, 10_000, [])
    assert ok_n
    assert ok_l
    np.testing.assert_allclose(xn, xl, atol=1e-7)


@pytest.mark.parametrize(
    ("params", "solver"),
    [
        ({"alpha": 0.5}, "newton"),
        ({"alpha": 5.0, "repulsion_ratio": 1.0}, "lbfgs"),
        ({"alpha": 5.0, "repulsion_ratio": 1.0, "kernel": "log_barrier"}, "newton"),
        ({"alpha": 0.5, "repulsion_ratio": 0.5}, "newton"),
    ],
)
def test_diagnostics(rng, params, solver):
    X, y = make_data(rng)
    m = TrosnaLinearRegression(**params).fit(X, y)
    assert m.converged_
    assert m.solver_ == solver
    assert 0 <= m.optimality_ <= 1e-8 * 1e3 * max(1.0, np.sqrt(np.mean(y**2)))


def test_soft_kernel_with_forecast_points_uses_lbfgs(rng):
    """На точках без отклика мягкое ядро даёт отрицательную кривизну — Ньютон не применяется."""
    X, y = make_data(rng)
    m = TrosnaLinearRegression(alpha=0.5, repulsion_ratio=0.5).fit(X, y, repulsion_X=X[:3] + 1)
    assert m.solver_ == "lbfgs"
    assert m.converged_


@pytest.mark.parametrize("kernel", ["gaussian", "log_barrier"])
def test_iteration_limit_is_reported(rng, kernel):
    X, y = make_data(rng)
    with pytest.warns(ConvergenceWarning, match="нарушение условий оптимальности"):
        m = TrosnaLinearRegression(alpha=3.0, repulsion_ratio=1.0, kernel=kernel, max_iter=1, tol=1e-14).fit(X, y)
    assert not m.converged_
    assert m.optimality_ > 0


def test_newton_converges_quadratically(rng):
    """Число итераций Ньютона почти не растёт с требуемой точностью (квадратичная сходимость)."""
    X, y = make_data(rng, p=5)
    pen = Penalty(alpha=0.8, l1_ratio=0.0, l2_ratio=0.2, repulsion_ratio=0.8, width=0.5, kernel="log_barrier")
    iters = [fit(X, y, pen, tol=tol).n_iter for tol in (1e-4, 1e-8, 1e-12)]
    assert iters[-1] - iters[0] <= 4


@pytest.mark.parametrize("kernel", ["gaussian", "log_barrier"])
def test_history_ends_at_reported_optimality(rng, kernel):
    X, y = make_data(rng)
    m = TrosnaLinearRegression(alpha=3.0, repulsion_ratio=1.0, kernel=kernel).fit(X, y)
    h = m.optimality_history_
    assert h.size >= 2
    assert h[-1] == pytest.approx(m.optimality_, rel=1e-6, abs=1e-12)
    assert h[-1] < h[0]
