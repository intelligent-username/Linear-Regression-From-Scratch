"""naivelinear: Minimal naive closed-form linear regression.

Public API:
- linear_regression(points): closed form (normal equation) solver
- LinearRegressionNaive: estimator class wrapper
- mean_squared_error: basic metric
"""
from .core import linear_regression, LinearRegressionNaive
from .metrics import mean_squared_error

__all__ = [
    "linear_regression",
    "LinearRegressionNaive",
    "mean_squared_error",
]

__version__ = "0.1.0"
