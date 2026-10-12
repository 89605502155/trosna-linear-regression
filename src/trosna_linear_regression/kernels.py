"""Ядра отталкивающего штрафа.

Отталкивающий штраф наказывает модель за то, что предсказание ``ŷ`` оказалось
близко к заданной точке ``c``. Он зависит только от отклонения ``d = ŷ - c``
и ширины ``σ > 0``, симметричен (``φ(d) = φ(-d)``), монотонно убывает с ростом
``|d|`` и стремится к нулю при ``|d| → ∞`` — далёкие предсказания практически
не штрафуются. Масштаб ``σ²`` делает штраф согласованным с МНК при пересчёте
единиц ``y``: ``φ(kd; kσ) = k² φ(d; σ)``.

Мягкие ядра (``"gaussian"``, ``"cauchy"``) ограничены: точку можно занять,
заплатив конечный штраф. Они нормированы на кривизну ``φ''(0) = -1``, что
делает силу штрафа сравнимой с квадратичной ошибкой МНК (её кривизна по ``ŷ``
равна ``+1``): при ``a ≤ 1`` слагаемое ``½ (y - ŷ)² + a φ(ŷ - c)`` выпукло по
``ŷ``, а при ``a > 1`` возле точки появляется вогнутый участок, который
«расталкивает» предсказания.

Барьерные ядра (``"log_barrier"``, ``"inverse_barrier"``) неограниченны:
``φ(d) → ∞`` при ``d → 0``, т.е. сколь угодно близко подойти к точке можно,
но цена растёт без предела. Для применений, где значение ``c`` физически
невозможно. Барьер действует только в запретной зоне ``|d| < σ`` и гладко
(с непрерывной производной) обращается в ноль на её границе: предсказания вне
зоны не штрафуются вовсе. При обучении каждому предсказанию назначается своя
сторона от точки (или одна разрешённая сторона для всех), и штраф становится
односторонним (``one_sided``): за точкой он продолжает расти, так что
пересечь её нельзя. Односторонний барьер выпукл, поэтому при назначенных
сторонах вся задача выпукла.

Все ядра изотропны: штраф зависит только от расстояния ``|ŷ - c|`` и убывает
от точки одинаково в обе стороны, как высота от вершины горы.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RepulsionKernel:
    """Ядро штрафа: значение ``φ(d; σ)``, первая и вторая производные по ``d``."""

    name: str
    value: Callable[[FloatArray, float], FloatArray]
    derivative: Callable[[FloatArray, float], FloatArray]
    curvature: Callable[[FloatArray, float], FloatArray]
    #: Барьер: ``φ(d) → ∞`` при ``d → 0``; иначе ядро ограничено.
    barrier: bool = False
    #: Наименьшая кривизна: по всей оси для мягких ядер (нормирована на -1),
    #: по каждую сторону от точки для барьерных.
    min_curvature: float = -1.0
    #: Для барьеров: односторонний вариант ``ψ(z) -> (ψ, ψ', ψ'')``, где
    #: ``z > 0`` — разрешённая сторона; выпуклый на всей оси.
    one_sided: Callable[[FloatArray, float], tuple[FloatArray, FloatArray, FloatArray]] | None = None


def gaussian_value(d: FloatArray, width: float) -> FloatArray:
    """``φ(d) = σ² exp(-d² / 2σ²)``."""
    u = d / width
    return width**2 * np.exp(-0.5 * u * u)


def gaussian_derivative(d: FloatArray, width: float) -> FloatArray:
    """``φ'(d) = -d exp(-d² / 2σ²)``."""
    u = d / width
    return -d * np.exp(-0.5 * u * u)


def gaussian_curvature(d: FloatArray, width: float) -> FloatArray:
    """``φ''(d) = (d²/σ² - 1) exp(-d² / 2σ²)``, наименьшее значение -1 в точке."""
    u = d / width
    return (u * u - 1.0) * np.exp(-0.5 * u * u)


def cauchy_value(d: FloatArray, width: float) -> FloatArray:
    """``φ(d) = σ² / (2 (1 + d²/σ²))`` — тяжёлые хвосты, штраф убывает как ``1/d²``."""
    u = d / width
    return 0.5 * width**2 / (1.0 + u * u)


def cauchy_derivative(d: FloatArray, width: float) -> FloatArray:
    """``φ'(d) = -d / (1 + d²/σ²)²``."""
    u = d / width
    return -d / (1.0 + u * u) ** 2


def cauchy_curvature(d: FloatArray, width: float) -> FloatArray:
    """``φ''(d) = (3d²/σ² - 1) / (1 + d²/σ²)³``, наименьшее значение -1 в точке."""
    u = d / width
    return (3.0 * u * u - 1.0) / (1.0 + u * u) ** 3


#: Порог барьера в долях σ. Ближе ``δ = BARRIER_FLOOR · σ`` к точке (и за
#: ней) барьер продолжается выпуклой квадратичной функцией по Тейлору из ``δ``:
#: значения остаются конечными (бесконечность ломает линейный поиск), но
#: пересечение точки стоит порядка ``σ²/δ · |z|`` и больше.
BARRIER_FLOOR = 1e-6

BarrierProfile = Callable[[FloatArray, float], tuple[FloatArray, FloatArray, FloatArray]]


def _log_profile(z: FloatArray, width: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``σ² (-ln u - (1 - u²)/2)``, ``u = z/σ``, и две производные при ``z > 0``; 0 вне зоны."""
    inside = z < width
    zi = np.where(inside, z, width)
    u = zi / width
    value = width**2 * (-np.log(u) - 0.5 * (1.0 - u * u))
    d1 = zi - width**2 / zi
    d2 = 1.0 + width**2 / (zi * zi)
    return np.where(inside, value, 0.0), np.where(inside, d1, 0.0), np.where(inside, d2, 0.0)


def _inverse_profile(z: FloatArray, width: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``σ²/2 (1/u - 1)²``, ``u = z/σ``, и две производные при ``z > 0``; 0 вне зоны."""
    inside = z < width
    zi = np.where(inside, z, width)
    r = width / zi
    value = 0.5 * width**2 * (r - 1.0) ** 2
    d1 = -width * r * r * (r - 1.0)
    d2 = r**3 * (3.0 * r - 2.0)
    return np.where(inside, value, 0.0), np.where(inside, d1, 0.0), np.where(inside, d2, 0.0)


OneSided = Callable[[FloatArray, float], tuple[FloatArray, FloatArray, FloatArray]]


def _one_sided(profile: BarrierProfile) -> OneSided:
    """Односторонний барьер ``ψ(z)``, ``z`` — расстояние до точки со знаком «своей» стороны.

    При ``z ≥ δ`` — сам профиль, при ``z < δ`` (вплотную к точке и за ней) —
    его квадратичное продолжение. ``ψ`` выпукла на всей оси. Возвращает
    значение, первую и вторую производные.
    """

    def psi(z: FloatArray, width: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        z = np.asarray(z, dtype=np.float64)
        delta = BARRIER_FLOOR * width
        v, d1, d2 = profile(np.maximum(z, delta), width)
        t = np.minimum(z - delta, 0.0)
        return v + d1 * t + 0.5 * d2 * t * t, d1 + d2 * t, d2

    return psi


def _symmetric(psi: OneSided):
    """Симметричное ядро ``φ(d) = ψ(|d|)``, ``φ' = sign(d) ψ'(|d|)``, ``φ'' = ψ''(|d|)``."""

    def value(d: FloatArray, width: float) -> FloatArray:
        return psi(np.abs(d), width)[0]

    def derivative(d: FloatArray, width: float) -> FloatArray:
        return np.sign(d) * psi(np.abs(d), width)[1]

    def curvature(d: FloatArray, width: float) -> FloatArray:
        return psi(np.abs(d), width)[2]

    return value, derivative, curvature


log_barrier_one_sided = _one_sided(_log_profile)
inverse_barrier_one_sided = _one_sided(_inverse_profile)
log_barrier_value, log_barrier_derivative, log_barrier_curvature = _symmetric(log_barrier_one_sided)
inverse_barrier_value, inverse_barrier_derivative, inverse_barrier_curvature = _symmetric(inverse_barrier_one_sided)


KERNELS: dict[str, RepulsionKernel] = {
    "gaussian": RepulsionKernel("gaussian", gaussian_value, gaussian_derivative, gaussian_curvature),
    "cauchy": RepulsionKernel("cauchy", cauchy_value, cauchy_derivative, cauchy_curvature),
    "log_barrier": RepulsionKernel(
        "log_barrier",
        log_barrier_value,
        log_barrier_derivative,
        log_barrier_curvature,
        barrier=True,
        min_curvature=0.0,
        one_sided=log_barrier_one_sided,
    ),
    "inverse_barrier": RepulsionKernel(
        "inverse_barrier",
        inverse_barrier_value,
        inverse_barrier_derivative,
        inverse_barrier_curvature,
        barrier=True,
        min_curvature=0.0,
        one_sided=inverse_barrier_one_sided,
    ),
}


def get_kernel(name: str) -> RepulsionKernel:
    """Ядро по имени: ``"gaussian"``, ``"cauchy"``, ``"log_barrier"``, ``"inverse_barrier"``."""
    try:
        return KERNELS[name]
    except (KeyError, TypeError):
        raise ValueError(f"неизвестное ядро {name!r}, доступны: {sorted(KERNELS)}") from None
