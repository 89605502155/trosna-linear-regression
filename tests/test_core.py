"""Целевая функция, градиент, оптимальность решения."""

import numpy as np
import pytest
import scipy.optimize
from conftest import make_data, reference_objective

from trosna_linear_regression import Penalty, fit, get_kernel, objective
from trosna_linear_regression.core import _Problem


def penalty_kwargs(p: Penalty) -> dict:
    return {
        "alpha": p.alpha,
        "l1_ratio": p.l1_ratio,
        "l2_ratio": p.l2_ratio,
        "repulsion_ratio": p.repulsion_ratio,
        "point": p.point,
        "width": p.width,
        "kernel": p.kernel,
    }


PENALTIES = [
    Penalty(alpha=0.0, l1_ratio=0.0, l2_ratio=1.0, repulsion_ratio=0.0),
    Penalty(alpha=0.3, l1_ratio=0.2, l2_ratio=0.3, repulsion_ratio=0.5, point=1.0, width=2.0),
    Penalty(alpha=3.0, l1_ratio=0.0, l2_ratio=0.0, repulsion_ratio=1.0, point=-0.5, width=1.5),
    Penalty(alpha=4.0, l1_ratio=0.1, l2_ratio=0.1, repulsion_ratio=0.8, kernel="cauchy", width=0.7),
    Penalty(alpha=2.0, l1_ratio=0.2, l2_ratio=0.0, repulsion_ratio=0.8, kernel="log_barrier", width=1.5),
    Penalty(alpha=0.5, l1_ratio=0.0, l2_ratio=0.5, repulsion_ratio=0.5, kernel="inverse_barrier", width=1.0, point=0.5),
]


@pytest.mark.parametrize("pen", PENALTIES)
def test_objective_matches_reference(rng, pen):
    X, y = make_data(rng, n=30, p=3)
    coef, b = rng.normal(size=3), 0.4
    w = rng.uniform(0.1, 3.0, size=y.size)
    assert objective(X, y, coef, b, pen) == pytest.approx(reference_objective(X, y, coef, b, **penalty_kwargs(pen)))
    assert objective(X, y, coef, b, pen, sample_weight=w) == pytest.approx(
        reference_objective(X, y, coef, b, w=w, **penalty_kwargs(pen))
    )


def make_problem(rng, pen, fit_intercept, *, n=40, p=3, extra=0):
    X, y = make_data(rng, n=n, p=p)
    s = np.full(y.size, 1.0 / y.size)
    q = rng.normal(size=(extra, p)) if extra else None
    return X, y, q, _Problem(X, y, s, pen, fit_intercept, extra=q)


@pytest.mark.parametrize("pen", PENALTIES)
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("extra", [0, 3])
def test_gradient_and_hessian_match_finite_difference(rng, pen, fit_intercept, extra):
    _, _, _, problem = make_problem(rng, pen, fit_intercept, extra=extra)
    x = rng.normal(size=problem.p + int(fit_intercept))
    _, g, H = problem.smooth(x)
    err = scipy.optimize.check_grad(lambda v: problem.smooth(v, hessian=False)[0], lambda v: problem.smooth(v)[1], x)
    assert err < 1e-4 * (1 + np.linalg.norm(g))
    h = 1e-6
    numeric = np.column_stack(
        [
            (problem.smooth(x + h * e, hessian=False)[1] - problem.smooth(x - h * e, hessian=False)[1]) / (2 * h)
            for e in np.eye(x.size)
        ]
    )
    np.testing.assert_allclose(H, numeric, rtol=1e-4, atol=1e-5 * (1 + np.abs(H).max()))


@pytest.mark.parametrize("pen", PENALTIES)
def test_lbfgs_variables_match_smooth_part(rng, pen):
    """Расщеплённые переменные L-BFGS-B дают ту же J, что и гладкая часть плюс L1."""
    _, _, _, problem = make_problem(rng, pen, True)
    x = rng.normal(size=problem.p + 1)
    value, _ = problem.value_and_grad(problem.x_to_theta(x))
    smooth, _, _ = problem.smooth(x, hessian=False)
    assert value == pytest.approx(smooth + problem.l1 @ np.abs(x[: problem.p]))
    np.testing.assert_allclose(problem.theta_to_x(problem.x_to_theta(x)), x)


@pytest.mark.parametrize("pen", PENALTIES)
@pytest.mark.parametrize("extra", [0, 2])
def test_smooth_value_equals_objective(rng, pen, extra):
    """Значение в переменных оптимизатора (центрирование, масштаб) совпадает с J(β, b)."""
    X, y, q, problem = make_problem(rng, pen, True, n=25, extra=extra)
    coef, b_centered = rng.normal(size=3), 0.7
    x = problem.to_x(coef, b_centered)
    value, _, _ = problem.smooth(x, hessian=False)
    coef2, intercept = problem.from_x(x)
    np.testing.assert_allclose(coef2, coef)
    total = value + problem.l1 @ np.abs(x[:3])
    assert total == pytest.approx(objective(X, y, coef, intercept, pen, repulsion_X=q))


@pytest.mark.parametrize("pen", PENALTIES)
def test_solution_is_stationary(rng, pen):
    """Условия оптимальности: ∇f + α ρ_l1 ∂‖β‖₁ ∋ 0, производная по b равна нулю."""
    X, y = make_data(rng, n=80, p=4)
    res = fit(X, y, pen, tol=1e-10)
    assert res.converged
    s = np.full(y.size, 1.0 / y.size)
    pred = X @ res.coef + res.intercept
    g_pred = s * (pred - y) + pen.repulsion * s * get_kernel(pen.kernel).derivative(pred - pen.point, pen.width)
    g = X.T @ g_pred + pen.l2 * res.coef
    tol = 1e-6 * (1 + np.abs(X.T @ (s * y)).max())
    assert abs(g_pred.sum()) < tol
    nz = res.coef != 0
    np.testing.assert_allclose(g[nz], -pen.l1 * np.sign(res.coef[nz]), atol=tol)
    assert np.all(np.abs(g[~nz]) <= pen.l1 + tol)


def test_lasso_gives_exact_zeros(rng):
    X = rng.normal(size=(100, 8))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.1, size=100)
    res = fit(X, y, Penalty(alpha=0.3, l1_ratio=0.8, l2_ratio=0.0, repulsion_ratio=0.2))
    assert np.all(res.coef[2:] == 0.0)
    assert np.all(res.coef[:2] != 0.0)


@pytest.mark.parametrize("kernel", ["gaussian", "cauchy"])
def test_convex_regime_has_unique_solution(rng, kernel):
    """При α ρ_rep ≤ 1 и ρ_l2 > 0 задача строго выпукла: решение не зависит от старта."""
    X, y = make_data(rng, n=60, p=3)
    pen = Penalty(alpha=1.0, l1_ratio=0.0, l2_ratio=0.1, repulsion_ratio=0.9, kernel=kernel)
    assert pen.is_convex
    sols = [fit(X, y, pen, tol=1e-12, coef_init=rng.normal(scale=5, size=3), intercept_init=c) for c in (-10, 0, 10)]
    for other in sols[1:]:
        np.testing.assert_allclose(other.coef, sols[0].coef, atol=1e-6)
        assert other.intercept == pytest.approx(sols[0].intercept, abs=1e-6)


def test_nonconvex_improves_on_start(rng):
    """Невыпуклая задача стартует из решения без отталкивания и только улучшает J."""
    X, y = make_data(rng)
    pen = Penalty(alpha=6.0, l1_ratio=0.0, l2_ratio=0.0, repulsion_ratio=1.0, width=1.0)
    assert not pen.is_convex
    start = fit(X, y, pen.without_repulsion())
    res = fit(X, y, pen)
    assert res.objective <= objective(X, y, start.coef, start.intercept, pen) + 1e-12


def test_penalty_validation():
    with pytest.raises(ValueError, match="в сумме 1"):
        Penalty(l1_ratio=0.5, l2_ratio=0.5, repulsion_ratio=0.5)
    with pytest.raises(ValueError, match="неотрицательными"):
        Penalty(l1_ratio=-0.1, l2_ratio=0.6, repulsion_ratio=0.5)
    with pytest.raises(ValueError, match="alpha"):
        Penalty(alpha=-1.0)
    with pytest.raises(ValueError, match="ширина"):
        Penalty(width=0.0)
    with pytest.raises(ValueError, match="конечным"):
        Penalty(point=np.nan)
    with pytest.raises(ValueError, match="ядро"):
        Penalty(kernel="box")


def test_zero_sample_weight_sum():
    with pytest.raises(ValueError, match="сумма весов"):
        fit(np.ones((3, 1)), np.ones(3), Penalty(), sample_weight=np.zeros(3))
