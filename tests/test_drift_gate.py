"""Tests for src.utils.drift_gate."""

from src.utils.drift_gate import DriftGate


class TestDriftGate:
    def test_passes_no_drift(self):
        gate = DriftGate(max_share_drifted=0.3, target_drift_pvalue=0.05)
        result = gate.evaluate({"share_drifted": 0.1, "target_pvalue": 0.5})
        assert result.passed
        assert result.violations == []

    def test_fails_high_share_drifted(self):
        gate = DriftGate(max_share_drifted=0.3, target_drift_pvalue=0.05)
        result = gate.evaluate({"share_drifted": 0.5, "target_pvalue": 0.5})
        assert not result.passed
        assert len(result.violations) == 1
        assert "share_of_drifted_columns" in result.violations[0]

    def test_fails_low_target_pvalue(self):
        gate = DriftGate(max_share_drifted=0.3, target_drift_pvalue=0.05)
        result = gate.evaluate({"share_drifted": 0.1, "target_pvalue": 0.01})
        assert not result.passed
        assert len(result.violations) == 1
        assert "target_drift" in result.violations[0]

    def test_passes_with_none_target_pvalue(self):
        gate = DriftGate(max_share_drifted=0.3, target_drift_pvalue=0.05)
        result = gate.evaluate({"share_drifted": 0.1, "target_pvalue": None})
        assert result.passed

    def test_multiple_violations(self):
        gate = DriftGate(max_share_drifted=0.1, target_drift_pvalue=0.05)
        result = gate.evaluate({"share_drifted": 0.5, "target_pvalue": 0.01})
        assert not result.passed
        assert len(result.violations) == 2

    def test_thresholds_recorded(self):
        gate = DriftGate(max_share_drifted=0.3, target_drift_pvalue=0.05)
        result = gate.evaluate({"share_drifted": 0.1, "target_pvalue": 0.5})
        assert result.thresholds_used == {
            "max_share_drifted": 0.3,
            "target_drift_pvalue": 0.05,
        }
