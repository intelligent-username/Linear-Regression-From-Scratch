"""Metrics for naive linear regression."""
from __future__ import annotations
from typing import Sequence, Union
import numpy as np

Number = Union[int, float]
ArrayLike1D = Union[Sequence[Number], np.ndarray]

__all__ = ["mean_squared_error"]

def mean_squared_error(y_true: ArrayLike1D, y_pred: ArrayLike1D) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true_arr.size == 0:
        raise ValueError("y_true and y_pred must be non-empty")
    diff = y_true_arr - y_pred_arr
    return float(np.mean(diff * diff))
