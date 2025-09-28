"""Error metrics (legacy module).

This file is kept for backward compatibility. New code should import:

    from naivelinear.metrics import mean_squared_error

Only Mean Squared Error (MSE) is provided.

Design choices:
- Accepts sequences or numpy arrays; coerces to numpy internally.
- Returns a float (Python float) rather than a 0-dim ndarray for convenience.
- Performs basic validation on matching lengths.
"""
from __future__ import annotations
from typing import Sequence, Union
import numpy as np

Number = Union[int, float]
ArrayLike1D = Union[Sequence[Number], np.ndarray]


def mean_squared_error(y_true: ArrayLike1D, y_pred: ArrayLike1D) -> float:
    """Compute the Mean Squared Error between true and predicted values.

    Parameters
    ----------
    y_true : sequence or np.ndarray
        Ground truth target values.
    y_pred : sequence or np.ndarray
        Predicted target values.

    Returns
    -------
    float
        The mean of squared differences.

    Raises
    ------
    ValueError
        If the inputs differ in length or are empty.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true_arr.size == 0:
        raise ValueError("y_true and y_pred must be non-empty")

    diff = y_true_arr - y_pred_arr
    return float(np.mean(diff * diff))

__all__ = ["mean_squared_error"]
