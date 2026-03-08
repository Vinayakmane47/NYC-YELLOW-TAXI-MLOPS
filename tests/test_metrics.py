"""Tests for src.utils.metrics."""

import numpy as np
import pandas as pd

from src.utils.metrics import compute_regression_metrics


class TestComputeRegressionMetrics:
    def test_perfect_predictions(self):
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = compute_regression_metrics(y_true, y_pred)

        assert result["rmse"] == 0.0
        assert result["mae"] == 0.0
        assert result["r2"] == 1.0

    def test_known_values(self):
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])

        result = compute_regression_metrics(y_true, y_pred)

        assert abs(result["rmse"] - 0.5) < 1e-6
        assert abs(result["mae"] - 0.5) < 1e-6
        assert result["r2"] > 0.0

    def test_returns_float_types(self):
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        result = compute_regression_metrics(y_true, y_pred)

        assert isinstance(result["rmse"], float)
        assert isinstance(result["mae"], float)
        assert isinstance(result["r2"], float)

    def test_all_keys_present(self):
        y_true = pd.Series([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])

        result = compute_regression_metrics(y_true, y_pred)

        assert set(result.keys()) == {"rmse", "mae", "r2"}
