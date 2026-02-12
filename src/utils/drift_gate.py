from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DriftGateResult:
    passed: bool
    violations: List[str]
    metrics_checked: Dict[str, float]
    thresholds_used: Dict[str, float]


class DriftGate:
    def __init__(self, max_share_drifted: float, target_drift_pvalue: float) -> None:
        self.max_share_drifted = max_share_drifted
        self.target_drift_pvalue = target_drift_pvalue

    def evaluate(self, drift_metrics: Dict[str, float]) -> DriftGateResult:
        violations: List[str] = []

        share = drift_metrics["share_drifted"]
        if share > self.max_share_drifted:
            violations.append(
                f"share_of_drifted_columns {share:.2%} > "
                f"threshold {self.max_share_drifted:.2%}"
            )

        target_pval = drift_metrics.get("target_pvalue")
        if target_pval is not None and target_pval < self.target_drift_pvalue:
            violations.append(
                f"target_drift p-value {target_pval:.4f} < "
                f"threshold {self.target_drift_pvalue}"
            )

        return DriftGateResult(
            passed=len(violations) == 0,
            violations=violations,
            metrics_checked=drift_metrics,
            thresholds_used={
                "max_share_drifted": self.max_share_drifted,
                "target_drift_pvalue": self.target_drift_pvalue,
            },
        )
