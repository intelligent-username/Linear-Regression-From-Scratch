"""Naive closed-form linear regression.

Design goals:
- Keep strictly to normal equation solution (no gradient descent, no regularization).
- Provide both a functional interface w/ a small estimator-style class.
- Minimal input validation (for speed & clarity).
- Make the 'brains' behind linear regression techniques accessible.

Input format for the functional API:
    points: iterable of (x1, x2, ..., x_m, y)
Returns:
    numpy.ndarray of shape (m+1,) with parameters (bias first).
"""
from __future__ import annotations
from typing import Iterable, Sequence, Union, List, Tuple
import numpy as np

Num = Union[int, float]
Point = Union[Sequence[Num], np.ndarray]
PointsType = Union[Iterable[Point], np.ndarray]

__all__ = ["linear_regression", "LinearRegressionNaive"]


def _coerce(points: PointsType) -> np.ndarray:
    data = np.asarray(list(points), dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Expect 2D data with at least one feature plus target")
    return data


def linear_regression(points: PointsType) -> np.ndarray:
    """Closed-form ordinary least squares via normal equation.

    Parameters
    ----------
    points : iterable of (x1,...,x_m, y)
        Final column is treated as target.

    Returns
    -------
    np.ndarray
        Parameter vector (bias first) of length m+1.

    Raises
    ------
    ValueError
        If insufficient points (need at least m+1) or degenerate design matrix.
    """
    data = _coerce(points)
    n_features = data.shape[1] - 1
    if data.shape[0] < n_features + 1:
        raise ValueError(
            f"At least {n_features + 1} points required (got {data.shape[0]})"
        )

    X_no_bias = data[:, :-1]
    y = data[:, -1]
    X = np.hstack([np.ones((data.shape[0], 1)), X_no_bias])

    # Normal equation (explicit inverse is acceptable for small educational demos)
    gram = X.T @ X
    try:
        params = np.linalg.inv(gram) @ X.T @ y
    except np.linalg.LinAlgError as e:
        raise ValueError("Design matrix is singular (collinearity or insufficient variation)") from e
    return params


class LinearRegressionNaive:
    """Minimal estimator-style wrapper around the normal equation solution.

    Attributes
    ----------
    params_ : np.ndarray | None
        Learned parameter vector (bias first) after fit.
    n_features_ : int
        Number of feature columns (excluding target). Set after fit.
    fitted_ : bool
        Indicates whether fit() has been called successfully.
    """

    def __init__(self) -> None:
        self.params_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.fitted_: bool = False

    def fit(self, points: PointsType) -> "LinearRegressionNaive":
        data = _coerce(points)
        self.n_features_ = data.shape[1] - 1
        self.params_ = linear_regression(data)
        self.fitted_ = True
        return self

    def predict(self, X_features: Union[np.ndarray, Sequence[Sequence[Num]]]) -> np.ndarray:
        if not self.fitted_ or self.params_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_arr = np.asarray(X_features, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X_arr.shape[1]}"
            )
        bias = np.ones((X_arr.shape[0], 1))
        X_full = np.hstack([bias, X_arr])
        return X_full @ self.params_

    def mean_squared_error(self, points: PointsType) -> float:
        """Convenience: compute MSE on provided points (reuses internal params)."""
        from .metrics import mean_squared_error  # local import to avoid cycle

        data = _coerce(points)
        y_true = data[:, -1]
        preds = self.predict(data[:, :-1])
        return mean_squared_error(y_true, preds)

    def __repr__(self) -> str:  # pragma: no cover
        if self.fitted_ and self.params_ is not None:
            return f"LinearRegressionNaive(params={self.params_!r})"
        return "LinearRegressionNaive(unfitted)"
