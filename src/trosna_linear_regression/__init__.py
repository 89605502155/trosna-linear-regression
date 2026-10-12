"""Линейная регрессия МНК с L1-, тихоновской и отталкивающей от заданной точки регуляризацией."""

from .core import FitResult, Penalty, fit, objective
from .estimator import TrosnaLinearRegression
from .kernels import KERNELS, RepulsionKernel, get_kernel

__all__ = [
    "KERNELS",
    "FitResult",
    "Penalty",
    "RepulsionKernel",
    "TrosnaLinearRegression",
    "fit",
    "get_kernel",
    "objective",
]
