"""Свойства на случайных входах (hypothesis)."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from trosna_linear_regression import Penalty, fit, get_kernel, objective


@st.composite
def problems(draw):
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    n = draw(st.integers(2, 40))
    p = draw(st.integers(1, 5))
    X = rng.normal(size=(n, p)) * draw(st.sampled_from([0.1, 1.0, 10.0]))
    y = X @ rng.normal(size=p) + rng.normal(scale=draw(st.sampled_from([0.01, 1.0, 5.0])), size=n)
    l1 = draw(st.sampled_from([0.0, 0.2, 0.5]))
    rep = draw(st.sampled_from([0.0, 0.3, 0.5]))
    pen = Penalty(
        alpha=draw(st.sampled_from([0.0, 0.1, 1.0, 5.0])),
        l1_ratio=l1,
        l2_ratio=1.0 - l1 - rep,
        repulsion_ratio=rep,
        point=draw(st.sampled_from([0.0, 1.0, -3.0])),
        width=draw(st.sampled_from([0.3, 1.0, 4.0])),
        kernel=draw(st.sampled_from(["gaussian", "cauchy", "log_barrier", "inverse_barrier"])),
    )
    return X, y, pen, rng


@settings(max_examples=150, deadline=None)
@given(problems())
def test_solution_is_local_minimum(problem):
    """J в решении не больше, чем в случайных соседних точках, и чем в нуле.

    С барьером решение оптимально при закреплённых сторонах, поэтому сравнение
    идёт только с точками, где все предсказания по тем же сторонам от ``c``.
    """
    X, y, pen, rng = problem
    res = fit(X, y, pen, tol=1e-10)
    assert np.all(np.isfinite(res.coef))
    barrier = get_kernel(pen.kernel).barrier and pen.repulsion > 0
    sides = np.sign(X @ res.coef + res.intercept - pen.point)

    def comparable(coef, b):
        return not barrier or np.array_equal(np.sign(X @ coef + b - pen.point), sides)

    scale = 1.0 + abs(res.objective)
    zero, mean = np.zeros(X.shape[1]), float(np.mean(y))
    if comparable(zero, mean):
        assert res.objective <= objective(X, y, zero, mean, pen) + 1e-9 * scale
    for eps in (1e-3, 1e-2):
        for _ in range(5):
            coef = res.coef + eps * rng.normal(size=res.coef.size)
            b = res.intercept + eps * rng.normal()
            if comparable(coef, b):
                assert res.objective <= objective(X, y, coef, b, pen) + 1e-9 * scale


@settings(max_examples=100, deadline=None)
@given(problems())
def test_convex_regime_is_global_minimum(problem):
    """При α ρ_rep ≤ 1 решение не хуже далёких случайных точек."""
    X, y, pen, rng = problem
    if not pen.is_convex:
        return
    res = fit(X, y, pen, tol=1e-10)
    scale = 1.0 + abs(res.objective)
    for _ in range(10):
        coef = res.coef + rng.normal(scale=3.0, size=res.coef.size)
        assert res.objective <= objective(X, y, coef, res.intercept + rng.normal(scale=3.0), pen) + 1e-9 * scale
