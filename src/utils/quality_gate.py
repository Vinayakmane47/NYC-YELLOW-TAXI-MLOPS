from dataclasses import dataclass
from typing import Dict, List


@dataclass
class QualityGateResult:
    passed: bool
    violations: List[str]
    metrics_checked: Dict[str, float]
    thresholds_used: Dict[str, float]


class QualityGate:
    def __init__(self, min_r2: float, max_rmse: float, max_mae: float) -> None:
        self.min_r2 = min_r2
        self.max_rmse = max_rmse
        self.max_mae = max_mae

    def evaluate(self, metrics: Dict[str, float]) -> QualityGateResult:
        violations: List[str] = []

        if metrics["r2"] < self.min_r2:
            violations.append(f"R2 {metrics['r2']:.4f} < min_r2 threshold {self.min_r2}")
        if metrics["rmse"] > self.max_rmse:
            violations.append(f"RMSE {metrics['rmse']:.4f} > max_rmse threshold {self.max_rmse}")
        if metrics["mae"] > self.max_mae:
            violations.append(f"MAE {metrics['mae']:.4f} > max_mae threshold {self.max_mae}")

        return QualityGateResult(
            passed=len(violations) == 0,
            violations=violations,
            metrics_checked=metrics,
            thresholds_used={
                "min_r2": self.min_r2,
                "max_rmse": self.max_rmse,
                "max_mae": self.max_mae,
            },
        )
