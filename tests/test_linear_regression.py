import numpy as np
import pytest
from naivelinear import linear_regression, mean_squared_error, LinearRegressionNaive


def generate_synthetic(points: int, features: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    # true parameters (bias + feature weights)
    true_params = rng.normal(size=features + 1)
    X_feats = rng.normal(size=(points, features))
    bias = np.ones((points, 1))
    X = np.hstack([bias, X_feats])
    noise = rng.normal(scale=0.01, size=points)
    y = X @ true_params + noise
    # Return points in the expected format: (x1,...,xf,y)
    pts = np.hstack([X_feats, y[:, None]])
    return pts, true_params


def test_closed_form_matches_lstsq_small_function():
    pts, _ = generate_synthetic(points=10, features=2, seed=42)
    params = linear_regression(pts)
    data = np.array(pts)
    X = np.hstack([np.ones((data.shape[0], 1)), data[:, :-1]])
    y = data[:, -1]
    lstsq_params, *_ = np.linalg.lstsq(X, y, rcond=None)
    np.testing.assert_allclose(params, lstsq_params, rtol=1e-6, atol=1e-6)


def test_closed_form_matches_lstsq_small_class():
    pts, _ = generate_synthetic(points=12, features=3, seed=7)
    model = LinearRegressionNaive().fit(pts)
    data = np.array(pts)
    X = np.hstack([np.ones((data.shape[0], 1)), data[:, :-1]])
    y = data[:, -1]
    lstsq_params, *_ = np.linalg.lstsq(X, y, rcond=None)
    np.testing.assert_allclose(model.params_, lstsq_params, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("features,points,seed", [
    (1, 8, 0),
    (2, 15, 1),
    (3, 40, 2),
])
def test_random_multiple_runs(features, points, seed):
    pts, _ = generate_synthetic(points=points, features=features, seed=seed)
    params = linear_regression(pts)
    data = np.array(pts)
    X = np.hstack([np.ones((data.shape[0], 1)), data[:, :-1]])
    y = data[:, -1]
    lstsq_params, *_ = np.linalg.lstsq(X, y, rcond=None)
    np.testing.assert_allclose(params, lstsq_params, rtol=1e-6, atol=1e-6)


def test_mse_zero_when_perfect():
    # Construct perfectly linear data: y = 3 + 2x1 - x2
    X_feats = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, -1.0],
        [3.0, 4.0],
        [4.0, -2.0],
    ])
    true_params = np.array([3.0, 2.0, -1.0])  # bias, w1, w2
    bias = np.ones((X_feats.shape[0], 1))
    y = (bias @ true_params[0:1] + X_feats @ true_params[1:]).ravel()
    pts = np.hstack([X_feats, y[:, None]])

    params = linear_regression(pts)
    # predictions (functional)
    X = np.hstack([bias, X_feats])
    y_pred = X @ params
    mse = mean_squared_error(y, y_pred)
    assert mse < 1e-12
    np.testing.assert_allclose(params, true_params, rtol=1e-8, atol=1e-8)
    # class consistency
    model = LinearRegressionNaive().fit(pts)
    class_pred = model.predict(X_feats)
    np.testing.assert_allclose(class_pred, y, rtol=1e-8, atol=1e-8)
    assert model.mean_squared_error(pts) < 1e-12


def test_error_on_insufficient_points():
    pts, _ = generate_synthetic(points=2, features=3, seed=1)  # need at least features+1=4
    with pytest.raises(ValueError):
        linear_regression(pts)
