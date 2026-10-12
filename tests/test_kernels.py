"""Свойства ядер отталкивания."""

import numpy as np
import pytest

from trosna_linear_regression import KERNELS, get_kernel

NAMES = sorted(KERNELS)
SOFT = sorted(name for name, k in KERNELS.items() if not k.barrier)
BARRIERS = sorted(name for name, k in KERNELS.items() if k.barrier)
WIDTHS = [0.1, 1.0, 7.5]
d = np.linspace(-10, 10, 2001)


def test_kernel_groups():
    assert SOFT == ["cauchy", "gaussian"]
    assert BARRIERS == ["inverse_barrier", "log_barrier"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("width", WIDTHS)
def test_maximum_at_point_symmetric_and_decreasing(name, width):
    k = get_kernel(name)
    v = k.value(d * width, width)
    assert np.argmax(v) == d.size // 2
    np.testing.assert_allclose(v, v[::-1])
    assert np.all(np.diff(v[d.size // 2 :]) <= 0)
    assert np.all(v >= 0)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("width", WIDTHS)
def test_derivative_matches_finite_difference(name, width):
    k = get_kernel(name)
    x = d * width / 3
    x = x[np.abs(x) > 0.05 * width]  # у барьеров производная у точки огромна
    h = 1e-7 * width
    numeric = (k.value(x + h, width) - k.value(x - h, width)) / (2 * h)
    np.testing.assert_allclose(k.derivative(x, width), numeric, rtol=1e-5, atol=1e-6 * width)


@pytest.mark.parametrize("name", SOFT)
@pytest.mark.parametrize("width", WIDTHS)
def test_soft_curvature_is_normalized_to_minus_one(name, width):
    """Наименьшая кривизна -1 (в самой точке) — основа порога выпуклости ``α ρ_rep ≤ 1``."""
    k = get_kernel(name)
    x = np.linspace(-5 * width, 5 * width, 20001)
    curvature = np.diff(k.derivative(x, width)) / (x[1] - x[0])
    assert curvature.min() == pytest.approx(-1.0, abs=1e-3)
    assert k.min_curvature == -1.0


@pytest.mark.parametrize("name", SOFT)
def test_soft_kernels_are_bounded(name):
    k = get_kernel(name)
    assert np.max(k.value(np.array([0.0, 1e-300, -1e-300]), 2.0)) <= 4.0


@pytest.mark.parametrize("name", BARRIERS)
@pytest.mark.parametrize("width", WIDTHS)
def test_barrier_grows_toward_point(name, width):
    """Чем ближе к точке, тем дороже; к порогу δ = 1e-6 σ штраф уже велик, в самой точке конечен."""
    k = get_kernel(name)
    dist = width * np.logspace(-1, -10, 10)
    v = k.value(dist, width)
    assert np.all(np.diff(v) > 0)
    assert v[5] > 10 * width**2  # |d| = 1e-6 σ
    assert np.isfinite(k.value(np.array([0.0]), width)[0])


@pytest.mark.parametrize("name", BARRIERS)
@pytest.mark.parametrize("width", WIDTHS)
def test_one_sided_barrier_forbids_crossing(name, width):
    """За точкой односторонний барьер растёт дальше: пересечение стоит огромно, ψ выпукла на всей оси."""
    k = get_kernel(name)
    z = width * np.linspace(-1, 2, 300001)
    v, g, h = k.one_sided(z, width)
    assert np.all(h >= 0)
    assert k.one_sided(np.array([-0.1 * width]), width)[0][0] > 1e4 * width**2
    assert np.all(np.diff(v[z < 0]) < 0)
    assert np.all(np.diff(g) >= -1e-9 * np.abs(g).max())
    pos = z > 1e-3 * width
    np.testing.assert_allclose(v[pos], k.value(z[pos], width))


@pytest.mark.parametrize("name", BARRIERS)
@pytest.mark.parametrize("width", WIDTHS)
def test_barrier_is_zero_outside_zone_and_smooth_at_edge(name, width):
    """Вне зоны ``|d| ≥ σ`` штрафа нет; на границе значение и производная непрерывны (обе 0)."""
    k = get_kernel(name)
    outside = width * np.array([1.0, 1.5, 10.0, -1.0, -3.0])
    np.testing.assert_array_equal(k.value(outside, width), 0.0)
    np.testing.assert_array_equal(k.derivative(outside, width), 0.0)
    edge = width * (1 - 1e-6)
    assert k.value(np.array([edge]), width)[0] < 1e-9 * width**2
    assert abs(k.derivative(np.array([edge]), width)[0]) < 1e-4 * width


@pytest.mark.parametrize("name", BARRIERS)
@pytest.mark.parametrize("width", WIDTHS)
def test_barrier_is_convex_on_each_side(name, width):
    """Выпуклость по каждую сторону: при фиксированных сторонах задача выпукла."""
    k = get_kernel(name)
    for x in (np.linspace(1e-3 * width, 3 * width, 30001), -np.linspace(3 * width, 1e-3 * width, 30001)):
        g = k.derivative(x, width)
        assert np.all(np.diff(g) >= -1e-9 * np.abs(g).max())
    assert k.min_curvature == 0.0


@pytest.mark.parametrize("name", NAMES)
def test_scales_with_width_squared(name):
    """``φ(kd; kσ) = k² φ(d; σ)`` — штраф меняется вместе с единицами ``y``, как и МНК."""
    k = get_kernel(name)
    np.testing.assert_allclose(k.value(3.0 * d, 3.0 * 2.0), 9.0 * k.value(d, 2.0))


def test_cauchy_has_heavier_tails():
    g, c = get_kernel("gaussian"), get_kernel("cauchy")
    far = np.array([5.0, 10.0])
    assert np.all(c.value(far, 1.0) > g.value(far, 1.0))


def test_inverse_barrier_is_steeper_than_log():
    near = np.array([1e-2, 1e-4])
    assert np.all(get_kernel("inverse_barrier").value(near, 1.0) > get_kernel("log_barrier").value(near, 1.0))


def test_unknown_kernel():
    with pytest.raises(ValueError, match="неизвестное ядро"):
        get_kernel("laplace")
