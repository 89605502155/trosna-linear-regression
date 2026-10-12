"""Линейная регрессия МНК с L1-, L2- (тихоновской) и отталкивающей регуляризацией.

Минимизируется

    J(β, b) = Σ_i s̃_i ½ (y_i - ŷ_i)²
              + α ρ_rep [ Σ_i s̃_i φ(ŷ_i - c) + 1/n Σ_q φ(ŷ_q - c) ]
              + α ρ_l1 ‖β‖₁ + ½ α ρ_l2 ‖β‖²,

где ``ŷ = xᵀβ + b``, ``s̃_i = s_i / Σ s_j`` — нормированные веса наблюдений,
``φ`` — ядро отталкивания от точки ``c`` (см. ``kernels``), доли
``ρ_l1 + ρ_l2 + ρ_rep = 1``. Индекс ``q`` пробегает дополнительные точки
``repulsion_X`` без отклика (например, моменты прогноза): на них действует
только отталкивание, каждая весит как одно наблюдение среднего веса.

L2-штраф — регуляризация Тихонова (Тихонов, 1963) нулевого порядка. Без L1
и отталкивания решение даётся нормальной системой ``(Xᵀ S X + λ I) β = Xᵀ S y``
с ``λ = α ρ_l2``.
Свободный член ``b`` L1/L2 не штрафуется, но в отталкивании участвует.

* ``ρ_rep = 0`` — в точности ``sklearn.linear_model.ElasticNet``;
* ``α = 0`` — обычный МНК;
* мягкое ядро и ``α ρ_rep ≤ 1`` без дополнительных точек — задача выпукла;
* мягкое ядро и ``α ρ_rep > 1`` — возле точки штраф вогнут, предсказания
  расталкиваются от неё; задача невыпукла, ищется локальный минимум из
  решения задачи без отталкивания;
* барьерное ядро — штраф неограниченно растёт у самой точки, и каждому
  предсказанию назначается сторона: своя для каждого по стартовому решению
  (``side="auto"``) или одна для всех (``"above"``, ``"below"``). Штраф
  односторонний, пересечь точку нельзя, при назначенных сторонах задача
  выпукла.

Решатели. Выпуклые задачи решаются методом Ньютона с явным гессианом
``Aᵀ W A + λ S⁻²`` (матричная форма регуляризации Тихонова), дроблением
шага по Армихо и L1 по ортантам (как в OWL-QN). Для чисто квадратичной задачи
это одна итерация, то есть точное решение нормальной системы. Невыпуклые
задачи с мягким ядром решаются L-BFGS-B, где при L1 коэффициенты расщепляются
``β = β⁺ - β⁻``, ``β± ≥ 0``. Сходимость в обоих случаях проверяется по одному
и тому же критерию: нарушению условий оптимальности (ККТ) не больше ``tol``
в единицах ``y``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.optimize
from numpy.typing import ArrayLike, NDArray

from .kernels import RepulsionKernel, get_kernel

FloatArray = NDArray[np.float64]

SIDES = ("auto", "above", "below")


@dataclass(frozen=True)
class Penalty:
    """Настройки регуляризации.

    ``alpha`` — общая сила; ``l1_ratio``, ``l2_ratio``, ``repulsion_ratio`` —
    доли трёх видов штрафа, в сумме 1. ``point`` — точка ``c``, от которой
    отталкиваются предсказания, ``width`` — ширина ``σ`` зоны отталкивания,
    ``side`` — разрешённая сторона для барьеров.
    """

    alpha: float = 1.0
    l1_ratio: float = 0.0
    l2_ratio: float = 0.5
    repulsion_ratio: float = 0.5
    point: float = 0.0
    width: float = 1.0
    kernel: str = "gaussian"
    side: str = "auto"

    def __post_init__(self) -> None:
        for name in ("alpha", "l1_ratio", "l2_ratio", "repulsion_ratio", "point", "width"):
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f"{name} должен быть конечным числом, получено {value}")
        if self.alpha < 0:
            raise ValueError(f"alpha должен быть неотрицательным, получено {self.alpha}")
        ratios = (self.l1_ratio, self.l2_ratio, self.repulsion_ratio)
        if min(ratios) < 0:
            raise ValueError(f"доли штрафов должны быть неотрицательными, получено {ratios}")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError(f"доли штрафов должны давать в сумме 1, получено {sum(ratios)}")
        if self.width <= 0:
            raise ValueError(f"ширина зоны отталкивания должна быть положительной, получено {self.width}")
        kernel = get_kernel(self.kernel)
        if self.side not in SIDES:
            raise ValueError(f"сторона должна быть одной из {SIDES}, получено {self.side!r}")
        if self.side != "auto" and not kernel.barrier:
            raise ValueError("разрешённую сторону можно задать только для барьерного ядра")

    @property
    def l1(self) -> float:
        return self.alpha * self.l1_ratio

    @property
    def l2(self) -> float:
        return self.alpha * self.l2_ratio

    @property
    def repulsion(self) -> float:
        return self.alpha * self.repulsion_ratio

    @property
    def barrier(self) -> bool:
        """Действует ли барьер (тогда предсказаниям назначаются стороны)."""
        return self.repulsion > 0 and get_kernel(self.kernel).barrier

    @property
    def is_convex(self) -> bool:
        """Выпукла ли задача по обучающим точкам.

        Мягкое ядро: кривизна ``½r² + aφ`` по ``ŷ`` не меньше ``1 - a``. Барьер:
        выпукл при назначенных сторонах, а при ``side="auto"`` стороны берутся
        из решения без отталкивания, т.е. задача в целом не выпукла.
        """
        if self.repulsion == 0:
            return True
        if get_kernel(self.kernel).barrier:
            return self.side != "auto"
        return self.repulsion <= 1.0

    def without_repulsion(self) -> Penalty:
        """Та же задача без отталкивающего слагаемого: силы ``α ρ_l1`` и ``α ρ_l2`` сохраняются.

        Стартовая точка для невыпуклой задачи; для барьера по ней назначаются стороны.
        """
        rest = self.l1_ratio + self.l2_ratio
        if rest == 0:
            return replace(self, alpha=0.0, l1_ratio=0.0, l2_ratio=1.0, repulsion_ratio=0.0, side="auto")
        return replace(
            self,
            alpha=self.alpha * rest,
            l1_ratio=self.l1_ratio / rest,
            l2_ratio=self.l2_ratio / rest,
            repulsion_ratio=0.0,
            side="auto",
        )


@dataclass(frozen=True)
class FitResult:
    coef: FloatArray
    intercept: float
    objective: float
    n_iter: int
    converged: bool
    message: str
    #: Нарушение условий оптимальности (ККТ) в найденной точке, в единицах y.
    optimality: float = 0.0
    #: Допуск, с которым сравнивалось ``optimality``.
    tolerance: float = 0.0
    #: ``"newton"`` или ``"lbfgs"``.
    solver: str = "newton"
    #: Нарушение условий оптимальности после каждой итерации (первый элемент — в старте).
    history: tuple[float, ...] = ()


def _normalized_weights(n: int, sample_weight: ArrayLike | None) -> FloatArray:
    if sample_weight is None:
        return np.full(n, 1.0 / n)
    s = np.asarray(sample_weight, dtype=np.float64)
    total = s.sum()
    if total <= 0:
        raise ValueError("сумма весов наблюдений должна быть положительной")
    return s / total


def _extra_points(repulsion_X: ArrayLike | None, n_features: int) -> FloatArray:
    if repulsion_X is None:
        return np.zeros((0, n_features))
    q = np.asarray(repulsion_X, dtype=np.float64)
    if q.ndim == 1:
        q = q.reshape(1, -1) if q.size == n_features else q.reshape(-1, 1)
    if q.ndim != 2 or q.shape[1] != n_features or not np.all(np.isfinite(q)):
        raise ValueError(f"repulsion_X должен быть конечной матрицей (m, {n_features}), получено {q.shape}")
    return q


def objective(
    X: ArrayLike,
    y: ArrayLike,
    coef: ArrayLike,
    intercept: float,
    penalty: Penalty,
    *,
    sample_weight: ArrayLike | None = None,
    repulsion_X: ArrayLike | None = None,
) -> float:
    """Значение ``J(β, b)`` прямо по формуле, без оптимизаций (ядро симметричное)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64)
    s = _normalized_weights(y.size, sample_weight)
    q = _extra_points(repulsion_X, X.shape[1])
    kernel = get_kernel(penalty.kernel)
    pred = X @ coef + intercept
    value = s @ (0.5 * (y - pred) ** 2)
    if penalty.repulsion > 0:
        rep = s @ kernel.value(pred - penalty.point, penalty.width)
        if q.shape[0]:
            rep += kernel.value(q @ coef + intercept - penalty.point, penalty.width).sum() / y.size
        value += penalty.repulsion * rep
    return float(value + penalty.l1 * np.abs(coef).sum() + 0.5 * penalty.l2 * coef @ coef)


class _Problem:
    """Задача в масштабированных переменных.

    Признаки центрируются и масштабируются к единичному взвешенному СКО
    (диагональное предобусловливание). Переменные ``x = (γ, b)``, ``γ = β·σ_x``,
    ``b`` — свободный член для центрированных признаков. Строки отталкивания —
    обучающие точки и дополнительные ``repulsion_X``.
    """

    def __init__(
        self,
        X: FloatArray,
        y: FloatArray,
        s: FloatArray,
        penalty: Penalty,
        fit_intercept: bool,
        *,
        extra: FloatArray | None = None,
        sides: FloatArray | None = None,
    ) -> None:
        n, self.p = X.shape
        self.fit_intercept = fit_intercept
        # Центрирование по взвешенному среднему: предсказания не меняются,
        # а обусловленность задачи по свободному члену улучшается.
        self.x_mean = s @ X if fit_intercept else np.zeros(self.p)
        Xc = X - self.x_mean
        scale = np.sqrt(s @ (Xc * Xc))
        self.scale = np.where(scale > 0, scale, 1.0)
        self.Xs = Xc / self.scale
        extra = np.zeros((0, self.p)) if extra is None else extra
        self.R = np.vstack([self.Xs, (extra - self.x_mean) / self.scale])
        self.rw = np.concatenate([s, np.full(extra.shape[0], 1.0 / n)])
        self.y = y
        self.s = s
        self.penalty = penalty
        self.kernel: RepulsionKernel = get_kernel(penalty.kernel)
        self.sides = sides
        # L2 в масштабированных переменных: ‖β‖² = γᵀ S⁻² γ.
        self.l2w = 1.0 / self.scale**2
        self.l1 = penalty.l1 / self.scale
        self.split = penalty.l1 > 0

    # --- переменные ---------------------------------------------------------

    def to_x(self, coef: FloatArray, b_centered: float) -> FloatArray:
        gamma = coef * self.scale
        return np.concatenate([gamma, [b_centered]]) if self.fit_intercept else gamma

    def from_x(self, x: FloatArray) -> tuple[FloatArray, float]:
        """``β`` и свободный член в исходных (нецентрированных) признаках."""
        coef = x[: self.p] / self.scale
        b = float(x[self.p]) if self.fit_intercept else 0.0
        return coef, (b - float(self.x_mean @ coef)) if self.fit_intercept else 0.0

    def _predictions(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        gamma = x[: self.p]
        b = float(x[self.p]) if self.fit_intercept else 0.0
        return self.Xs @ gamma + b, self.R @ gamma + b

    def _repulsion(self, pred_rep: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Штраф отталкивания на строках ``R`` и две его производные по предсказанию."""
        pen = self.penalty
        d = pred_rep - pen.point
        if self.sides is not None:
            v, d1, d2 = self.kernel.one_sided(self.sides * d, pen.width)
            return v, self.sides * d1, d2
        k = self.kernel
        return k.value(d, pen.width), k.derivative(d, pen.width), k.curvature(d, pen.width)

    # --- гладкая часть ------------------------------------------------------

    def smooth(self, x: FloatArray, *, hessian: bool = True) -> tuple[float, FloatArray, FloatArray | None]:
        """Гладкая часть ``J`` (без L1), её градиент и гессиан по ``x = (γ, b)``."""
        pen = self.penalty
        p = self.p
        gamma = x[:p]
        pred, pred_rep = self._predictions(x)
        r = pred - self.y
        value = float(self.s @ (0.5 * r * r) + 0.5 * pen.l2 * gamma @ (self.l2w * gamma))
        g_pred = self.s * r
        g_gamma = self.Xs.T @ g_pred + pen.l2 * self.l2w * gamma
        g_b = g_pred.sum()
        if pen.repulsion > 0:
            v, d1, d2 = self._repulsion(pred_rep)
            value += pen.repulsion * float(self.rw @ v)
            g_rep = pen.repulsion * self.rw * d1
            g_gamma = g_gamma + self.R.T @ g_rep
            g_b += g_rep.sum()
        grad = np.concatenate([g_gamma, [g_b]]) if self.fit_intercept else g_gamma
        if not hessian:
            return value, grad, None
        A = np.hstack([self.Xs, np.ones((self.Xs.shape[0], 1))]) if self.fit_intercept else self.Xs
        H = A.T @ (self.s[:, None] * A)
        if pen.repulsion > 0:
            B = np.hstack([self.R, np.ones((self.R.shape[0], 1))]) if self.fit_intercept else self.R
            H += B.T @ ((pen.repulsion * self.rw * d2)[:, None] * B)
        H[:p, :p] += np.diag(pen.l2 * self.l2w)
        return value, grad, H

    def terms(self, x: FloatArray) -> tuple[FloatArray, FloatArray, float]:
        """Поэлементные слагаемые (данные, отталкивание) и тихоновское — для точных разностей J."""
        pen = self.penalty
        gamma = x[: self.p]
        pred, pred_rep = self._predictions(x)
        r = pred - self.y
        rep = pen.repulsion * self._repulsion(pred_rep)[0] if pen.repulsion > 0 else np.zeros(pred_rep.size)
        return 0.5 * r * r, rep, 0.5 * pen.l2 * float(gamma @ (self.l2w * gamma))

    def change(self, x_old: FloatArray, x_new: FloatArray) -> float:
        """``J(x_new) - J(x_old)``, посчитанное по слагаемым.

        Огромные постоянные вклады (строка, чьё предсказание не может уйти от
        точки) сокращаются точно, и разность не тонет в ошибке округления.
        """
        d_old, r_old, t_old = self.terms(x_old)
        d_new, r_new, t_new = self.terms(x_new)
        p = self.p
        l1_part = float(self.l1 @ (np.abs(x_new[:p]) - np.abs(x_old[:p])))
        return float(self.s @ (d_new - d_old) + self.rw @ (r_new - r_old)) + (t_new - t_old) + l1_part

    def kkt_residual(self, x: FloatArray) -> float:
        _, g, _ = self.smooth(x, hessian=False)
        return _kkt_residual(x, g, self.l1, self.p)

    # --- переменные L-BFGS-B (расщепление γ = γ⁺ - γ⁻ при L1) ----------------

    def theta_to_x(self, theta: FloatArray) -> FloatArray:
        p = self.p
        gamma = theta[:p] - theta[p : 2 * p] if self.split else theta[:p]
        return np.concatenate([gamma, theta[-1:]]) if self.fit_intercept else gamma

    def x_to_theta(self, x: FloatArray) -> FloatArray:
        gamma = x[: self.p]
        parts = [np.maximum(gamma, 0.0), np.maximum(-gamma, 0.0)] if self.split else [gamma]
        if self.fit_intercept:
            parts.append(x[self.p :])
        return np.concatenate(parts)

    def value_and_grad(self, theta: FloatArray) -> tuple[float, FloatArray]:
        p = self.p
        value, g, _ = self.smooth(self.theta_to_x(theta), hessian=False)
        g_gamma = g[:p]
        parts = [g_gamma]
        if self.split:
            value += float(self.l1 @ (theta[:p] + theta[p : 2 * p]))
            parts = [g_gamma + self.l1, -g_gamma + self.l1]
        if self.fit_intercept:
            parts.append(g[p:])
        return value, np.concatenate(parts)

    def bounds(self) -> list[tuple[float | None, float | None]] | None:
        if not self.split:
            return None
        return [(0.0, None)] * (2 * self.p) + [(None, None)] * int(self.fit_intercept)

    def projected_gradient(self, theta: FloatArray) -> FloatArray:
        """Градиент с обнулёнными компонентами, упирающимися в границу ``γ± ≥ 0``."""
        _, g = self.value_and_grad(theta)
        if self.split:
            at_bound = np.zeros(g.size, dtype=bool)
            at_bound[: 2 * self.p] = (theta[: 2 * self.p] <= 0) & (g[: 2 * self.p] > 0)
            g = np.where(at_bound, 0.0, g)
        return g


#: Сколько раз перезапускать L-BFGS-B, если он остановился, не достигнув допуска.
_MAX_RESTARTS = 10

#: Во сколько раз нарушение условий оптимальности может превышать допуск, когда
#: решатель упёрся в предел точности вычислений, чтобы решение считалось найденным.
_STALL_FACTOR = 1e3

#: Порог декремента Ньютона относительно ``1 + mean(y²)`` (порядка машинной точности).
_DECREMENT_FLOOR = 1e-15


def _gradient_tolerance(problem: _Problem, tol: float) -> float:
    # Признаки отмасштабированы, поэтому градиент измеряется в единицах y.
    return tol * max(1.0, float(np.sqrt(problem.s @ (problem.y * problem.y))))


def _kkt_residual(x: FloatArray, g: FloatArray, l1: FloatArray, p: int) -> float:
    """Нарушение условий оптимальности ``0 ∈ g + l1 ∂|γ|`` (по свободному члену ``g_b = 0``)."""
    gamma, g_gamma = x[:p], g[:p]
    res = np.where(gamma != 0, g_gamma + l1 * np.sign(gamma), np.maximum(np.abs(g_gamma) - l1, 0.0))
    return float(np.max(np.abs(np.concatenate([res, g[p:]])), initial=0.0))


def _lbfgs_minimize(
    problem: _Problem, x0: FloatArray, tol: float, max_iter: int, history: list[float]
) -> tuple[FloatArray, int, bool, str]:
    """L-BFGS-B с перезапусками (для невыпуклых задач с мягким ядром)."""
    gtol = _gradient_tolerance(problem, tol)
    theta, n_iter = problem.x_to_theta(x0), 0
    history.append(problem.kkt_residual(x0))
    value = problem.value_and_grad(theta)[0]
    for _ in range(_MAX_RESTARTS):
        res = scipy.optimize.minimize(
            problem.value_and_grad,
            theta,
            jac=True,
            method="L-BFGS-B",
            bounds=problem.bounds(),
            callback=lambda th: history.append(problem.kkt_residual(problem.theta_to_x(th))),
            options={
                "maxiter": max(1, max_iter - n_iter),
                "maxfun": 2 * max_iter,
                "gtol": gtol,
                "ftol": 0.0,
                "maxcor": 20,
            },
        )
        n_iter += int(res.nit)
        improved = res.fun < value
        theta, value = res.x, float(res.fun)
        pg = float(np.max(np.abs(problem.projected_gradient(theta)), initial=0.0))
        if not res.success:
            # status 2 (ABNORMAL): линейный поиск не смог уменьшить J. Это предел
            # точности вычислений, а не расходимость, если градиент почти в допуске.
            stalled = res.status == 2 and pg <= _STALL_FACTOR * gtol
            return problem.theta_to_x(theta), n_iter, stalled, str(res.message)
        if pg <= gtol:
            return problem.theta_to_x(theta), n_iter, True, str(res.message)
        if not improved:
            return problem.theta_to_x(theta), n_iter, pg <= _STALL_FACTOR * gtol, str(res.message)
        if n_iter >= max_iter:
            return problem.theta_to_x(theta), n_iter, False, str(res.message)
    return problem.theta_to_x(theta), n_iter, True, str(res.message)


def _orthant_newton_direction(
    x: FloatArray, g: FloatArray, H: FloatArray, l1: FloatArray, p: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Шаг Ньютона по ортанту (как в OWL-QN): знаки ненулевых ``γ_j`` фиксируются.

    Нули, у которых ``|g_j| ≤ l1_j``, остаются нулями. На остальных координатах
    ``‖γ‖₁`` линеен (``l1_j σ_j γ_j``), и шаг — точное решение линейной системы
    с гессианом, в который входит матрица Тихонова. Возвращает шаг, знаки
    ортанта и псевдоградиент.
    """
    gamma, g_gamma = x[:p], g[:p]
    sigma = np.where(gamma != 0, np.sign(gamma), -np.sign(g_gamma))
    pseudo = g.copy()
    pseudo[:p] = g_gamma + l1 * sigma
    free = np.ones(x.size, dtype=bool)
    free[:p] = (gamma != 0) | (np.abs(g_gamma) > l1)
    pseudo[:p][~free[:p]] = 0.0
    d = np.zeros(x.size)
    if np.any(free):
        # Псевдообращение (lstsq с отсечением малых сингулярных чисел): у вырожденного
        # гессиана (p > n, нет L2) это шаг наименьшей нормы, без усиления ошибок
        # округления вдоль ядра. Составляющую градиента вдоль ядра, если она есть,
        # убирает шаг по антиградиенту в _newton_minimize.
        d[free] = np.linalg.lstsq(H[np.ix_(free, free)], -pseudo[free], rcond=None)[0]
    if pseudo @ d >= 0:  # численно вырожденный гессиан — откат к антиградиенту
        d = -pseudo
    return d, sigma, pseudo


def _newton_minimize(
    problem: _Problem, x0: FloatArray, tol: float, max_iter: int, history: list[float]
) -> tuple[FloatArray, int, bool, str]:
    """Ньютон по ортантам с дроблением шага (условие Армихо) для выпуклых задач.

    Гессиан положительно полуопределён (выпуклость), поэтому шаг Ньютона —
    направление спуска. У крутой стенки барьера метод устойчив, в отличие от
    квазиньютоновских.
    """
    p = problem.p
    l1 = problem.l1
    gtol = _gradient_tolerance(problem, tol)
    # Естественный масштаб J — среднее y² (сам J может содержать огромные постоянные вклады).
    j_scale = 1.0 + float(problem.s @ (problem.y * problem.y))
    x = x0
    for it in range(max_iter):
        _, g, H = problem.smooth(x)
        residual = _kkt_residual(x, g, l1, p)
        history.append(residual)
        if residual <= gtol:
            return x, it, True, "Ньютон: условия оптимальности выполнены"
        d, sigma, pseudo = _orthant_newton_direction(x, g, H, l1, p)
        decrease = float(pseudo @ d)
        if -decrease <= _DECREMENT_FLOOR * j_scale:
            if residual <= _STALL_FACTOR * gtol:
                # Декремент Ньютона оценивает J - J*: задача выпукла, и J уже минимальна с машинной
                # точностью. Градиент у крутой стенки при этом может не опуститься до допуска из-за
                # ошибок округления в слагаемых порядка σ⁴/z³.
                return x, it, True, "Ньютон: декремент на уровне машинной точности"
            # Шаг Ньютона выродился, а условия оптимальности не выполнены — шаг по антиградиенту.
            d, decrease = -pseudo, -float(pseudo @ pseudo)
        step = 1.0
        while True:
            trial = x + step * d
            # Проекция на ортант: координаты, сменившие знак, обнуляются.
            trial[:p] = np.where(trial[:p] * sigma > 0, trial[:p], 0.0)
            if problem.change(x, trial) <= 1e-4 * step * decrease or step < 1e-20:
                break
            step *= 0.5
        if decrease >= 0 or step < 1e-20:
            # Ни шаг Ньютона, ни его доли не уменьшают J: предел точности вычислений.
            return x, it, residual <= _STALL_FACTOR * gtol, "Ньютон: шаг не уменьшает функцию"
        x = trial
    return x, max_iter, problem.kkt_residual(x) <= gtol, "Ньютон: достигнуто max_iter"


def fit(
    X: ArrayLike,
    y: ArrayLike,
    penalty: Penalty,
    *,
    sample_weight: ArrayLike | None = None,
    repulsion_X: ArrayLike | None = None,
    fit_intercept: bool = True,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    coef_init: ArrayLike | None = None,
    intercept_init: float | None = None,
) -> FitResult:
    """Минимизировать ``J(β, b)``.

    Старт: ``coef_init``, иначе для выпуклой задачи без барьера — ноль, иначе
    решение той же задачи без отталкивания (ElasticNet / гребневая / МНК). Для
    барьера с ``side="auto"`` по стартовому решению назначаются стороны, а при
    одной разрешённой стороне старт сдвигается внутрь допустимой области.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    s = _normalized_weights(y.size, sample_weight)
    extra = _extra_points(repulsion_X, X.shape[1])
    n_iter = 0

    if coef_init is not None:
        coef0 = np.asarray(coef_init, dtype=np.float64)
        intercept0 = float(s @ y) if intercept_init is None else float(intercept_init)
    elif penalty.is_convex and not penalty.barrier:
        coef0, intercept0 = np.zeros(X.shape[1]), float(s @ y)
    else:
        start = fit(
            X,
            y,
            penalty.without_repulsion(),
            sample_weight=sample_weight,
            fit_intercept=fit_intercept,
            tol=tol,
            max_iter=max_iter,
        )
        n_iter += start.n_iter
        coef0, intercept0 = start.coef, start.intercept
    if not fit_intercept:
        intercept0 = 0.0

    sides = None
    if penalty.barrier:
        if penalty.side == "auto":
            # Сторона каждого предсказания — по стартовому решению; ровно в точке — «сверху».
            pred0 = np.vstack([X, extra]) @ coef0 + intercept0
            sides = np.where(pred0 < penalty.point, -1.0, 1.0)
        else:
            sides = np.full(y.size + extra.shape[0], 1.0 if penalty.side == "above" else -1.0)
            if fit_intercept and coef_init is None:
                # Старт внутри допустимой области: сдвиг свободного члена так, чтобы все
                # предсказания были не ближе σ/2 к точке. Из точки у самой стенки (или за
                # ней) Ньютон на логарифмическом барьере лишь удваивает расстояние до стенки
                # за итерацию, а изнутри сходится за несколько шагов.
                z0 = sides * (np.vstack([X, extra]) @ coef0 + intercept0 - penalty.point)
                lack = 0.5 * penalty.width - float(z0.min())
                if lack > 0:
                    intercept0 += sides[0] * lack

    problem = _Problem(X, y, s, penalty, fit_intercept, extra=extra, sides=sides)
    x0 = problem.to_x(coef0, intercept0 + float(problem.x_mean @ coef0))
    # Ньютон — для выпуклых задач. Мягкое ядро на дополнительных точках (без
    # слагаемого данных) даёт отрицательную кривизну, поэтому там L-BFGS-B.
    newton = penalty.barrier or penalty.repulsion == 0 or (penalty.is_convex and extra.shape[0] == 0)
    history: list[float] = []
    if newton:
        x, it, converged, message = _newton_minimize(problem, x0, tol, max_iter, history)
    else:
        x, it, converged, message = _lbfgs_minimize(problem, x0, tol, max_iter, history)
    coef, intercept = problem.from_x(x)
    return FitResult(
        coef=coef.copy(),
        intercept=intercept,
        objective=_barrier_objective(problem, x)
        if penalty.barrier
        else objective(X, y, coef, intercept, penalty, sample_weight=s, repulsion_X=extra),
        n_iter=n_iter + it,
        converged=converged,
        message=message,
        optimality=problem.kkt_residual(x),
        tolerance=_gradient_tolerance(problem, tol),
        solver="newton" if newton else "lbfgs",
        history=tuple(history),
    )


def _barrier_objective(problem: _Problem, x: FloatArray) -> float:
    """``J`` с односторонним барьером (совпадает с ``objective``, пока стороны соблюдены)."""
    data, rep, tikh = problem.terms(x)
    return float(problem.s @ data + problem.rw @ rep) + tikh + float(problem.l1 @ np.abs(x[: problem.p]))
