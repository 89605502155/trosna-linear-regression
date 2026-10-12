"""Физический предел: одна разрешённая сторона и точки прогноза."""

import numpy as np
import pytest

from trosna_linear_regression import TrosnaLinearRegression

BARRIERS = ["log_barrier", "inverse_barrier"]


def falling_window():
    """Окно из 5 замеров памяти после резкого спада: прямая МНК уходит ниже минимума ВМ."""
    t = np.arange(5.0)[:, None]
    mem = np.array([2600.0, 2550.0, 1500.0, 900.0, 640.0])
    return t, mem


def weights(t, t_next, power=1.0):
    return np.abs(t_next - t[:, 0]) ** -power


@pytest.mark.parametrize("kernel", BARRIERS)
def test_forecast_stays_above_physical_limit(kernel):
    t, mem = falling_window()
    t_next = np.array([[5.0]])
    limit = 512.0
    w = weights(t, 5.0)
    ols = TrosnaLinearRegression(alpha=0.0).fit(t, mem, w)
    assert ols.predict(t_next)[0] < limit  # обычная взвешенная регрессия «засыпает»
    model = TrosnaLinearRegression(
        alpha=1.0,
        repulsion_ratio=1.0,
        kernel=kernel,
        repulsion_point=limit,
        repulsion_side="above",
        repulsion_width=200.0,
    ).fit(t, mem, w, repulsion_X=t_next)
    assert model.converged_
    assert model.predict(t_next)[0] > limit
    # Прямая во времени: условие в окне и в момент прогноза выполняется на всём отрезке.
    grid = np.linspace(0, 5, 101)[:, None]
    assert np.all(model.predict(grid) > limit)


@pytest.mark.parametrize("kernel", BARRIERS)
def test_side_below(kernel):
    t, mem = falling_window()
    model = TrosnaLinearRegression(
        alpha=1.0,
        repulsion_ratio=1.0,
        kernel=kernel,
        repulsion_point=2000.0,
        repulsion_side="below",
        repulsion_width=300.0,
    ).fit(t, mem, repulsion_X=[[-1.0]])
    assert np.all(model.predict(np.vstack([t, [[-1.0]]])) < 2000.0)


def test_infeasible_start_is_pulled_to_allowed_side():
    """Старт (МНК) нарушает условие сразу во многих точках — выпуклое продолжение барьера всё равно сходится."""
    t, mem = falling_window()
    model = TrosnaLinearRegression(
        alpha=1.0,
        repulsion_ratio=1.0,
        kernel="log_barrier",
        repulsion_point=1600.0,
        repulsion_side="above",
        repulsion_width=100.0,
    ).fit(t, mem)
    assert model.converged_
    assert np.all(model.predict(t) > 1600.0)


def test_side_requires_barrier():
    with pytest.raises(ValueError, match="барьерного"):
        TrosnaLinearRegression(repulsion_side="above").fit(np.eye(3), np.arange(3.0))
    with pytest.raises(ValueError, match="сторона"):
        TrosnaLinearRegression(kernel="log_barrier", repulsion_side="left").fit(np.eye(3), np.arange(3.0))


def test_repulsion_x_shape():
    with pytest.raises(ValueError, match="признаков"):
        TrosnaLinearRegression().fit(np.eye(3), np.arange(3.0), repulsion_X=np.ones((2, 2)))


def test_forecast_points_do_not_change_fit_without_repulsion():
    t, mem = falling_window()
    a = TrosnaLinearRegression(alpha=0.0).fit(t, mem)
    b = TrosnaLinearRegression(alpha=0.0).fit(t, mem, repulsion_X=[[5.0], [6.0]])
    np.testing.assert_allclose(a.coef_, b.coef_)
