# trosna-linear-regression

[![PyPI](https://img.shields.io/pypi/v/trosna-linear-regression?color=orange)](https://pypi.org/project/trosna-linear-regression/)
![Python](https://img.shields.io/pypi/pyversions/trosna-linear-regression?color=blueviolet)
![License](https://img.shields.io/pypi/l/trosna-linear-regression?color=blueviolet)

[Русский](#русский) · [English](#english)

---

## Русский

Линейная регрессия МНК с тремя видами регуляризации, которые можно сочетать в любой пропорции:

* **L1** (лассо): отбор признаков, нулевые коэффициенты получаются точными нулями;
* **L2 по Тихонову** (гребневая регрессия): штраф `½λ‖β‖²`, решение без L1 и отталкивания — нормальная система
  `(XᵀSX + λI)β = XᵀSy`;
* **отталкивание от заданной точки**: чем ближе предсказание к точке `c`, тем больше штраф. Мягкие ядра лишь
  отталкивают, а барьерные делают штраф бесконечным в самой точке: предсказание может подойти к ней сколь угодно
  близко, но цена растёт без предела.

Эстиматор совместим со scikit-learn (`BaseEstimator`, `RegressorMixin`, проходит `check_estimator`) и работает в
`Pipeline`, `GridSearchCV`, `cross_val_score`.

```bash
pip install trosna-linear-regression
```

### Зачем

В задачах управления часть значений физически невозможна: объём памяти виртуальной машины не может быть меньше
минимума, при котором работает её ОС, нагрузка не может быть отрицательной. Обычная (в том числе взвешенная)
регрессия об этом не знает. После резкого спада старые точки окна тянут прямую вниз, прогноз уходит в
невозможную область, и система отбирает у машины память, без которой та «засыпает». Барьер в точке предела
держит прогноз в допустимой области и почти не меняет модель, пока до предела далеко.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from trosna_linear_regression import TrosnaLinearRegression

t = np.arange(5.0)[:, None]  # 5 последних мониторингов
mem = np.array([2600, 2550, 1500, 900, 640.0])  # потребление памяти ВМ, МБ: резкий спад
w = 1 / np.abs(5.0 - t[:, 0])  # вес точки по близости к моменту прогноза
t_next = [[5.0]]

LinearRegression().fit(t, mem, sample_weight=w).predict(t_next)  # 28 МБ: ниже минимума 512 МБ

model = TrosnaLinearRegression(
    alpha=1.0,
    repulsion_ratio=1.0,  # только отталкивание
    kernel="log_barrier",  # штраф бесконечен в самой точке
    repulsion_point=512.0,  # минимум памяти ВМ
    repulsion_side="above",  # допустимы только значения выше
    repulsion_width=200.0,  # барьер действует в полосе 512…712 МБ
).fit(t, mem, w, repulsion_X=t_next)  # отталкивать и прогноз в момент t_next
model.predict(t_next)  # 564 МБ
model.converged_, model.solver_, model.optimality_  # (True, 'newton', 2.1e-07)
```

Для прямой во времени условие, выполненное в точках окна и в момент прогноза, выполняется и на всём отрезке
между ними. Подробные примеры с графиками лежат в репозитории
[trosna-linear-regression-examples](https://github.com/89605502155/trosna-linear-regression-examples).

### Постановка

```
J(β, b) = Σ s̃ᵢ ½(yᵢ − ŷᵢ)²
        + α ρ_rep [ Σ s̃ᵢ φ(ŷᵢ − c) + 1/n Σ_q φ(ŷ_q − c) ]
        + α ρ_l1 ‖β‖₁ + ½ α ρ_l2 ‖β‖²
```

`ŷ = xᵀβ + b`, `s̃ᵢ` — нормированные веса наблюдений (`sample_weight`), `q` пробегает точки `repulsion_X` без
отклика (моменты прогноза). Доли `ρ_l1 = l1_ratio`, `ρ_rep = repulsion_ratio`, `ρ_l2 = 1 − ρ_l1 − ρ_rep`,
общая сила `α = alpha`, как в `ElasticNet`. Свободный член `b` L1/L2 не штрафуется.

### Ядра отталкивания

Все ядра изотропны: штраф зависит только от расстояния `|ŷ − c|` и одинаково убывает в обе стороны от точки.

| `kernel` | `φ(d)`, `u = \|d\|/σ` | В точке | Когда брать |
|---|---|---|---|
| `"gaussian"` | `σ² exp(−u²/2)` | `σ²` | лёгкое отталкивание |
| `"cauchy"` | `σ² / (2(1 + u²))` | `σ²/2` | лёгкое отталкивание, действует дальше |
| `"log_barrier"` | `σ²(−ln u − (1−u²)/2)` при `u < 1`, иначе 0 | ∞, как `−ln\|d\|` | точка запрещена, мягкая стенка |
| `"inverse_barrier"` | `σ²/2 (1/u − 1)²` при `u < 1`, иначе 0 | ∞, как `1/d²` | точка запрещена, жёсткая стенка |

Мягкие ядра нормированы на кривизну `φ''(0) = −1`. Поэтому при `α ρ_rep ≤ 1` задача выпукла, а при больших
значениях предсказания расталкиваются от точки (задача невыпукла, ищется локальный минимум из решения без
отталкивания). Барьерные ядра равны нулю вне зоны `|d| < σ` и выпуклы по каждую сторону от точки. Сторону
задаёт `repulsion_side`: `"auto"` оставляет каждое предсказание там, где оно было в решении без отталкивания,
а `"above"` и `"below"` требуют одной стороны для всех. При назначенных сторонах задача выпукла.

### Физическая аналогия

Идея пришла из электростатики. Предсказание ведёт себя как заряженная частица в силовом поле между двумя
электродами: наблюдение притягивает её как пружина (квадрат отклонения), а одноимённо заряженный электрод в точке
`c` отталкивает тем сильнее, чем она ближе (ядро `φ`). Функционал `J` играет роль потенциальной энергии, решение
является положением равновесия, а обучение описывает движение частицы к нему. Логарифмический барьер повторяет
потенциал заряженной нити (`−ln|d|`), а `inverse_barrier` с точностью до множителя совпадает с отталкивающим
потенциалом Хатиба из метода потенциальных полей в робототехнике (Khatib, 1986), где `σ` играет роль радиуса
влияния препятствия. Чтобы коснуться электрода, нужна бесконечная энергия, поэтому точка недостижима. Для
регрессии все предсказания лежат на одной прямой и движутся вместе, как бусины на жёстком стержне: поле, действующее
на прогноз, поднимает всю прямую.

### Сходимость

Выпуклые задачи решаются методом Ньютона с явным гессианом `AᵀWA + λI` (матричная форма регуляризации
Тихонова), дроблением шага по Армихо и L1 по ортантам (как в OWL-QN). Для квадратичной задачи это одна
итерация. Невыпуклые задачи с мягким ядром решаются L-BFGS-B. В обоих случаях сходимость проверяется по
одному критерию: нарушение условий оптимальности (ККТ) не больше `tol`, отнесённого к масштабу `y`. После
обучения доступны `converged_`, `optimality_` (нарушение условий оптимальности в найденной точке), `solver_`
и `n_iter_`. Если решатель не сошёлся, выдаётся `ConvergenceWarning` с величиной нарушения.

Корректность проверяется тестами: совпадение с `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, совпадение
двух независимых решателей, проверка градиента и гессиана конечными разностями, свойства на случайных задачах (hypothesis) и полный набор проверок scikit-learn.

### Параметры

| Параметр | По умолчанию | Смысл |
|---|---|---|
| `alpha` | `1.0` | общая сила регуляризации |
| `l1_ratio` | `0.0` | доля L1 |
| `repulsion_ratio` | `0.5` | доля отталкивания, L2 получает остаток |
| `repulsion_point` | `0.0` | точка `c` |
| `repulsion_width` | `None` | ширина `σ` (радиус зоны барьера), `None` — СКО `y` |
| `kernel` | `"gaussian"` | ядро отталкивания |
| `repulsion_side` | `"auto"` | разрешённая сторона для барьеров |
| `fit_intercept` | `True` | оценивать свободный член |
| `tol`, `max_iter` | `1e-8`, `10000` | допуск по условиям оптимальности и предел итераций |

`fit(X, y, sample_weight=None, repulsion_X=None)`.

---

## English

Least-squares linear regression with three kinds of regularization that can be mixed in any proportion:

* **L1** (lasso): feature selection with exact zeros;
* **Tikhonov L2** (ridge): penalty `½λ‖β‖²`; without L1 and repulsion the solution is the normal system
  `(XᵀSX + λI)β = XᵀSy`;
* **repulsion from a given point**: the closer a prediction is to the point `c`, the larger the penalty. Soft
  kernels only push predictions away, while barrier kernels make the penalty infinite at the point itself: a
  prediction may come arbitrarily close, but the cost grows without bound.

The estimator is scikit-learn compatible (`BaseEstimator`, `RegressorMixin`, passes `check_estimator`) and works
with `Pipeline`, `GridSearchCV` and `cross_val_score`.

```bash
pip install trosna-linear-regression
```

### Why

In control problems some values are physically impossible: a virtual machine cannot run with less memory than its
OS needs, a load cannot be negative. Ordinary (including weighted) regression does not know that. After a sharp
drop the old points of the window pull the line down, the forecast enters the impossible region and the system
takes away memory the machine needs, so the machine "falls asleep". A barrier at the limit keeps the forecast in
the feasible region and barely changes the model while the limit is far away.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from trosna_linear_regression import TrosnaLinearRegression

t = np.arange(5.0)[:, None]  # last 5 monitoring steps
mem = np.array([2600, 2550, 1500, 900, 640.0])  # VM memory usage, MB: a sharp drop
w = 1 / np.abs(5.0 - t[:, 0])  # weight by closeness to the forecast time
t_next = [[5.0]]

LinearRegression().fit(t, mem, sample_weight=w).predict(t_next)  # 28 MB: below the 512 MB minimum

model = TrosnaLinearRegression(
    alpha=1.0,
    repulsion_ratio=1.0,  # repulsion only
    kernel="log_barrier",  # infinite penalty at the point
    repulsion_point=512.0,  # VM memory minimum
    repulsion_side="above",  # only values above are allowed
    repulsion_width=200.0,  # the barrier acts within 512…712 MB
).fit(t, mem, w, repulsion_X=t_next)  # also repel the forecast at t_next
model.predict(t_next)  # 564 MB
model.converged_, model.solver_, model.optimality_  # (True, 'newton', 2.1e-07)
```

For a line in time, a condition that holds at the window points and at the forecast time holds on the whole
segment between them. Detailed examples with plots are in
[trosna-linear-regression-examples](https://github.com/89605502155/trosna-linear-regression-examples).

### Problem

```
J(β, b) = Σ s̃ᵢ ½(yᵢ − ŷᵢ)²
        + α ρ_rep [ Σ s̃ᵢ φ(ŷᵢ − c) + 1/n Σ_q φ(ŷ_q − c) ]
        + α ρ_l1 ‖β‖₁ + ½ α ρ_l2 ‖β‖²
```

`ŷ = xᵀβ + b`, `s̃ᵢ` are normalized sample weights, `q` runs over the response-free points `repulsion_X`
(forecast times). The shares are `ρ_l1 = l1_ratio`, `ρ_rep = repulsion_ratio`, `ρ_l2 = 1 − ρ_l1 − ρ_rep`, and
the overall strength is `α = alpha`, as in `ElasticNet`. The intercept `b` is not penalized by L1/L2.

### Repulsion kernels

All kernels are isotropic: the penalty depends only on `|ŷ − c|` and decays equally on both sides of the point.

| `kernel` | `φ(d)`, `u = \|d\|/σ` | At the point | Use for |
|---|---|---|---|
| `"gaussian"` | `σ² exp(−u²/2)` | `σ²` | mild repulsion |
| `"cauchy"` | `σ² / (2(1 + u²))` | `σ²/2` | mild repulsion with a longer reach |
| `"log_barrier"` | `σ²(−ln u − (1−u²)/2)` for `u < 1`, else 0 | ∞, like `−ln\|d\|` | forbidden point, soft wall |
| `"inverse_barrier"` | `σ²/2 (1/u − 1)²` for `u < 1`, else 0 | ∞, like `1/d²` | forbidden point, hard wall |

Soft kernels are normalized to curvature `φ''(0) = −1`. Hence the problem is convex for `α ρ_rep ≤ 1`, while
larger values push predictions apart (the problem is non-convex, a local minimum is sought starting from the
solution without repulsion). Barrier kernels vanish outside the zone `|d| < σ` and are convex on each side of
the point. The side is set by `repulsion_side`: `"auto"` keeps every prediction where it was in the solution
without repulsion, while `"above"` and `"below"` require one side for all. With the sides fixed, the problem is
convex.

### Physical analogy

The idea comes from electrostatics. A prediction behaves like a charged particle in a force field between two
electrodes: the observation pulls it like a spring (the squared residual), while a like-charged electrode at `c`
pushes it away the harder the closer it gets (the kernel `φ`). The objective `J` is the potential energy, the
solution is the equilibrium, and fitting is the particle moving towards it. The log barrier is the potential of a
charged wire (`−ln|d|`), and `inverse_barrier` coincides up to a factor with Khatib's repulsive potential from the
potential field method in robotics (Khatib, 1986), with `σ` as the obstacle's radius of influence. Reaching the
electrode takes infinite energy, so the point is unreachable. In regression all predictions lie on one line and
move together like beads on a rigid rod: the field acting on the forecast lifts the whole line.

### Convergence

Convex problems are solved by Newton's method with the explicit Hessian `AᵀWA + λI` (the matrix form of
Tikhonov regularization), Armijo backtracking and orthant-wise handling of L1 (as in OWL-QN). For a quadratic
problem this takes one iteration. Non-convex problems with a soft kernel are solved by L-BFGS-B. Both use the same
stopping rule: the violation of the optimality (KKT) conditions must not exceed `tol` relative to the scale of
`y`. After fitting, `converged_`, `optimality_` (the KKT violation at the solution), `solver_` and `n_iter_` are
available. If the solver did not converge, a `ConvergenceWarning` reports the violation.

Correctness is covered by tests: agreement with `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, agreement of two
independent solvers, finite-difference checks of the gradient and Hessian, property-based tests on
random problems (hypothesis) and the full scikit-learn estimator checks.

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `alpha` | `1.0` | overall regularization strength |
| `l1_ratio` | `0.0` | L1 share |
| `repulsion_ratio` | `0.5` | repulsion share, L2 gets the rest |
| `repulsion_point` | `0.0` | the point `c` |
| `repulsion_width` | `None` | width `σ` (barrier zone radius), `None` means the std of `y` |
| `kernel` | `"gaussian"` | repulsion kernel |
| `repulsion_side` | `"auto"` | allowed side for barriers |
| `fit_intercept` | `True` | fit the intercept |
| `tol`, `max_iter` | `1e-8`, `10000` | optimality tolerance and iteration limit |

`fit(X, y, sample_weight=None, repulsion_X=None)`.

---

## Разработка / Development

```bash
uv sync
uv run pytest                  # 618 тестов / tests
uv run ruff check . && uv run ruff format --check .
```

Python 3.11–3.14; тесты проходят и с минимальными numpy 1.26, scipy 1.11, scikit-learn 1.6.
Tests also pass with the minimum supported numpy 1.26, scipy 1.11 and scikit-learn 1.6.

## Лицензия / License

MIT, см. / see [LICENSE](LICENSE).
