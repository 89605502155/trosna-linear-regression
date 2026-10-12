"""Общие данные и эталоны для тестов (в лоб, без оптимизаций)."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261011)


def make_data(rng: np.random.Generator, n: int = 120, p: int = 4, noise: float = 0.5, shift: float = 0.3):
    X = rng.normal(size=(n, p))
    coef = rng.normal(scale=2.0, size=p)
    y = X @ coef + shift + rng.normal(scale=noise, size=n)
    return X, y


def reference_objective(
    X, y, coef, intercept, *, alpha, l1_ratio, l2_ratio, repulsion_ratio, point, width, kernel, w=None
):
    """Целевая функция, выписанная поэлементно циклом — независимо от библиотеки."""
    n = y.size
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    total = 0.0
    for i in range(n):
        pred = float(X[i] @ coef + intercept)
        d = pred - point
        u = abs(d) / width
        if kernel == "gaussian":
            phi = width**2 * np.exp(-(d**2) / (2 * width**2))
        elif kernel == "cauchy":
            phi = width**2 / (2 * (1 + d**2 / width**2))
        elif kernel == "log_barrier":
            phi = width**2 * (-np.log(u) - (1 - u**2) / 2) if u < 1 else 0.0
        else:
            phi = width**2 / 2 * (1 / u - 1) ** 2 if u < 1 else 0.0
        total += w[i] * (0.5 * (y[i] - pred) ** 2 + alpha * repulsion_ratio * phi)
    total /= w.sum()
    return total + alpha * l1_ratio * np.sum(np.abs(coef)) + 0.5 * alpha * l2_ratio * coef @ coef
