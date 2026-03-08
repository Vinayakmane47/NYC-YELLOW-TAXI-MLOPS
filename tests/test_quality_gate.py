"""Tests for src.utils.quality_gate."""

from src.utils.quality_gate import QualityGate


class TestQualityGate:
    def test_passes_good_metrics(self):
        gate = QualityGate(min_r2=0.5, max_rmse=10.0, max_mae=8.0)
        result = gate.evaluate({"r2": 0.8, "rmse": 5.0, "mae": 4.0})
        assert result.passed
        assert result.violations == []

    def test_fails_low_r2(self):
        gate = QualityGate(min_r2=0.5, max_rmse=10.0, max_mae=8.0)
        result = gate.evaluate({"r2": 0.3, "rmse": 5.0, "mae": 4.0})
        assert not result.passed
        assert len(result.violations) == 1
        assert "R2" in result.violations[0]

    def test_fails_high_rmse(self):
        gate = QualityGate(min_r2=0.0, max_rmse=5.0, max_mae=100.0)
        result = gate.evaluate({"r2": 0.8, "rmse": 10.0, "mae": 4.0})
        assert not result.passed
        assert len(result.violations) == 1
        assert "RMSE" in result.violations[0]

    def test_fails_high_mae(self):
        gate = QualityGate(min_r2=0.0, max_rmse=100.0, max_mae=3.0)
        result = gate.evaluate({"r2": 0.8, "rmse": 5.0, "mae": 4.0})
        assert not result.passed
        assert len(result.violations) == 1
        assert "MAE" in result.violations[0]

    def test_multiple_violations(self):
        gate = QualityGate(min_r2=0.9, max_rmse=1.0, max_mae=1.0)
        result = gate.evaluate({"r2": 0.3, "rmse": 10.0, "mae": 8.0})
        assert not result.passed
        assert len(result.violations) == 3

    def test_edge_case_exact_thresholds(self):
        gate = QualityGate(min_r2=0.5, max_rmse=5.0, max_mae=4.0)
        result = gate.evaluate({"r2": 0.5, "rmse": 5.0, "mae": 4.0})
        assert result.passed

    def test_thresholds_recorded(self):
        gate = QualityGate(min_r2=0.5, max_rmse=10.0, max_mae=8.0)
        result = gate.evaluate({"r2": 0.8, "rmse": 5.0, "mae": 4.0})
        assert result.thresholds_used == {
            "min_r2": 0.5,
            "max_rmse": 10.0,
            "max_mae": 8.0,
        }
