"""Оценщик в стиле scikit-learn."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from .core import Penalty, fit


class TrosnaLinearRegression(RegressorMixin, BaseEstimator):
    """Линейная регрессия МНК с L1-, тихоновской (L2) и отталкивающей регуляризацией.

    Минимизирует

        1/(2n) Σ (y_i - ŷ_i)²  +  α ρ_rep · 1/n Σ φ(ŷ_i - c)
        +  α ρ_l1 ‖β‖₁  +  ½ α ρ_l2 ‖β‖²,

    где ``ρ_l1 = l1_ratio``, ``ρ_rep = repulsion_ratio``,
    ``ρ_l2 = 1 - l1_ratio - repulsion_ratio``. Отталкивающий штраф ``φ`` тем
    больше, чем ближе предсказание к точке ``c = repulsion_point``; он
    максимален в самой точке и пренебрежимо мал дальше нескольких
    ``repulsion_width`` от неё.

    L2-штраф — регуляризация Тихонова нулевого порядка (гребневая регрессия).
    Отталкивание действует и на предсказания в точках ``repulsion_X``,
    переданных в ``fit`` (например, в моментах прогноза).

    Частные случаи: ``repulsion_ratio=0`` — ``ElasticNet(alpha, l1_ratio)``,
    ``l1_ratio=1`` — ``Lasso(alpha)``, ``alpha=0`` — ``LinearRegression``.

    Parameters
    ----------
    alpha : float, default=1.0
        Общая сила регуляризации, ``α ≥ 0``.
    l1_ratio : float, default=0.0
        Доля L1-штрафа.
    repulsion_ratio : float, default=0.5
        Доля отталкивающего штрафа. Оставшаяся доля
        ``1 - l1_ratio - repulsion_ratio`` приходится на L2; сумма долей L1 и
        отталкивания не может превышать 1. Для мягких ядер при
        ``alpha * repulsion_ratio ≤ 1`` задача выпукла, при больших значениях
        предсказания расталкиваются от точки и решение — локальный минимум.
    repulsion_point : float, default=0.0
        Точка ``c``, от которой отталкиваются предсказания.
    repulsion_width : float or None, default=None
        Ширина ``σ`` зоны отталкивания в единицах ``y`` (для барьеров —
        радиус запретной зоны). ``None`` — взвешенное стандартное отклонение
        ``y`` (1, если ``y`` постоянен).
    kernel : {"gaussian", "cauchy", "log_barrier", "inverse_barrier"}, default="gaussian"
        Форма штрафа ``φ(d)``, ``d = ŷ - c``.

        Мягкие ядра ограничены: подойти к точке можно, заплатив конечный штраф.
        ``"gaussian"`` — ``σ² exp(-d²/2σ²)``; ``"cauchy"`` — ``σ²/(2(1 + d²/σ²))``,
        с тяжёлыми хвостами, т.е. отталкивание ощутимо дальше от точки.

        Барьерные ядра неограниченны: штраф уходит в бесконечность при
        ``ŷ → c`` и равен нулю вне зоны ``|d| < σ``. Для случаев, когда
        значение ``c`` физически невозможно. ``"log_barrier"`` —
        ``σ² (-ln u - (1 - u²)/2)``, ``u = |d|/σ``, растёт у точки как
        ``-ln|d|``; ``"inverse_barrier"`` — ``σ²/2 (1/u - 1)²``, растёт как
        ``1/d²``. Сторону от точки задаёт ``repulsion_side``, пересечь точку
        предсказание не может, при назначенных сторонах задача выпукла. Сила
        ``alpha`` задаёт, насколько широко барьер «чувствуется»: при малой силе
        он почти не меняет МНК.
    repulsion_side : {"auto", "above", "below"}, default="auto"
        Только для барьерных ядер. ``"auto"`` — каждое предсказание остаётся по
        той стороне от точки, где было в решении без отталкивания. ``"above"``
        (``"below"``) — все предсказания, включая ``repulsion_X``, должны быть
        выше (ниже) точки: значение ``c`` и всё за ним физически невозможно.
    fit_intercept : bool, default=True
        Оценивать ли свободный член (L1/L2 его не штрафуют).
    tol : float, default=1e-8
        Допуск по нарушению условий оптимальности (ККТ), относительно СКО ``y``.
    max_iter : int, default=10000
        Наибольшее число итераций решателя (Ньютон для выпуклых задач, L-BFGS-B
        для невыпуклых).

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    l2_ratio_ : float
        Доля L2-штрафа ``1 - l1_ratio - repulsion_ratio``.
    repulsion_width_ : float
        Фактически использованная ширина ``σ``.
    objective_ : float
        Значение минимизируемой функции в найденном решении.
    converged_ : bool
        Сошёлся ли решатель. Если нет, выдаётся ``ConvergenceWarning``.
    optimality_ : float
        Нарушение условий оптимальности (ККТ) в найденном решении, в единицах ``y``.
    solver_ : {"newton", "lbfgs"}
        Каким методом решена задача.
    tolerance_ : float
        Допуск, с которым сравнивается ``optimality_`` (``tol`` в единицах ``y``).
    optimality_history_ : ndarray
        Нарушение условий оптимальности по итерациям основного решателя
        (первый элемент — в стартовой точке). Позволяет проследить сходимость.
    n_iter_ : int
    n_features_in_ : int
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Есть, только если ``X`` имел имена признаков.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        l1_ratio: float = 0.0,
        repulsion_ratio: float = 0.5,
        repulsion_point: float = 0.0,
        repulsion_width: float | None = None,
        kernel: str = "gaussian",
        repulsion_side: str = "auto",
        fit_intercept: bool = True,
        tol: float = 1e-8,
        max_iter: int = 10_000,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.repulsion_ratio = repulsion_ratio
        self.repulsion_point = repulsion_point
        self.repulsion_width = repulsion_width
        self.kernel = kernel
        self.repulsion_side = repulsion_side
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter

    def _penalty(self, y: NDArray[np.float64], s: NDArray[np.float64] | None) -> Penalty:
        l2_ratio = 1.0 - self.l1_ratio - self.repulsion_ratio
        if -1e-12 < l2_ratio < 0:
            l2_ratio = 0.0
        if l2_ratio < 0:
            raise ValueError(
                f"l1_ratio + repulsion_ratio не может превышать 1, получено {self.l1_ratio + self.repulsion_ratio}"
            )
        width = self.repulsion_width
        if width is None:
            mean = np.average(y, weights=s)
            width = float(np.sqrt(np.average((y - mean) ** 2, weights=s)))
            if not width > 1e-12 * max(1.0, abs(mean)):
                width = 1.0
        return Penalty(
            alpha=float(self.alpha),
            l1_ratio=float(self.l1_ratio),
            l2_ratio=l2_ratio,
            repulsion_ratio=float(self.repulsion_ratio),
            point=float(self.repulsion_point),
            width=float(width),
            kernel=self.kernel,
            side=self.repulsion_side,
        )

    def _check_params(self) -> None:
        if not isinstance(self.fit_intercept, bool | np.bool_):
            raise ValueError(f"fit_intercept должен быть bool, получено {self.fit_intercept!r}")
        if not (np.isfinite(self.tol) and self.tol > 0):
            raise ValueError(f"tol должен быть положительным, получено {self.tol}")
        if not (isinstance(self.max_iter, int | np.integer) and self.max_iter >= 1):
            raise ValueError(f"max_iter должен быть целым ≥ 1, получено {self.max_iter!r}")

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        repulsion_X: ArrayLike | None = None,
    ) -> TrosnaLinearRegression:
        """Обучить модель.

        ``sample_weight`` — неотрицательные веса наблюдений. ``repulsion_X`` —
        точки без отклика, предсказания в которых тоже отталкиваются от
        ``repulsion_point`` (например, моменты будущего прогноза).
        """
        self._check_params()
        X, y = validate_data(self, X, y, dtype=np.float64, y_numeric=True)
        s = None
        if sample_weight is not None:
            s = np.asarray(sample_weight, dtype=np.float64)
            if s.ndim == 0:
                s = np.full(y.shape[0], float(s))
            if s.shape != (y.shape[0],):
                raise ValueError(f"sample_weight должен иметь форму {(y.shape[0],)}, получено {s.shape}")
            if not np.all(np.isfinite(s)) or np.any(s < 0):
                raise ValueError("sample_weight должен быть конечным и неотрицательным")
            if s.sum() <= 0:
                raise ValueError("sample_weight: все веса нулевые (sample weights are all zero)")
        penalty = self._penalty(y, s)
        if repulsion_X is not None:
            repulsion_X = check_array(repulsion_X, dtype=np.float64)
            if repulsion_X.shape[1] != X.shape[1]:
                raise ValueError(
                    f"repulsion_X должен иметь {X.shape[1]} признаков, как X, получено {repulsion_X.shape[1]}"
                )

        result = fit(
            X,
            y,
            penalty,
            sample_weight=s,
            repulsion_X=repulsion_X,
            fit_intercept=self.fit_intercept,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        if not result.converged:
            warnings.warn(
                f"Решатель не сошёлся (итераций: {result.n_iter}, max_iter={self.max_iter}, нарушение условий "
                f"оптимальности {result.optimality:.3g} при допуске {result.tolerance:.3g}): {result.message}. "
                "Увеличьте max_iter или tol, либо отмасштабируйте признаки.",
                ConvergenceWarning,
                stacklevel=2,
            )
        self.coef_ = result.coef
        self.intercept_ = result.intercept
        self.l2_ratio_ = penalty.l2_ratio
        self.repulsion_width_ = penalty.width
        self.objective_ = result.objective
        self.converged_ = result.converged
        self.optimality_ = result.optimality
        self.solver_ = result.solver
        self.tolerance_ = result.tolerance
        self.optimality_history_ = np.asarray(result.history)
        self.n_iter_ = result.n_iter
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return X @ self.coef_ + self.intercept_
