"""
Data Drift Detection Pipeline - Airflow DAG stage.

Compares reference data (training split) against current data (validation split)
using Evidently AI. Detects feature drift and target drift. If significant drift
is found, the retrain_decider in DAG A triggers DAG B for model retraining.

Usage::

    python -m src.drift_detection
"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import ChampionConfig, PipelineConfig
from src.utils.base_pipeline import BasePipeline, mlflow
from src.utils.drift_gate import DriftGate
from src.utils.spark_utils import (
    SparkUtils,
    build_stage_metadata,
    write_stage_metadata,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DriftResult:
    """Output of the drift detection pipeline."""

    drift_detected: bool
    share_drifted: float
    n_drifted: int
    n_columns: int
    drifted_columns: List[str]
    target_drift_pvalue: float | None
    drift_gate_passed: bool
    violations: List[str]
    report_path: str
    run_id: str
    timestamp_utc: str


# ---------------------------------------------------------------------------
# Drift Detection Pipeline
# ---------------------------------------------------------------------------


class DriftDetectionPipeline(BasePipeline):
    """Compares reference vs current data distributions using Evidently.

    Runs after ml_transformation and before model_training. If drift
    is detected beyond configured thresholds, the pipeline signals
    that retraining is justified.
    """

    DRIFT_THRESHOLD = 0.05  # default p-value for per-column drift

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        champion_config: ChampionConfig,
    ) -> None:
        super().__init__(pipeline_config, champion_config)
        self.spark = SparkUtils(app_name="nyc-taxi-pipeline-drift").spark

    # -- Data loading --------------------------------------------------------

    def _load_split(self, path: str, max_rows: int, split_name: str) -> pd.DataFrame:
        """Read parquet with Spark, limit rows, convert to pandas."""
        sdf = self.spark.read.parquet(path)
        if max_rows > 0:
            sdf = sdf.limit(max_rows)
        pdf = sdf.toPandas()
        print(f"[drift] {split_name}: {len(pdf):,} rows, {len(pdf.columns)} cols")
        return pdf

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load reference (train) and current (val) data."""
        drift_cfg = self.config.drift
        reference = self._load_split(
            self.config.data.train_path,
            drift_cfg.max_rows_reference,
            "reference (train)",
        )
        current = self._load_split(
            self.config.data.val_path,
            drift_cfg.max_rows_current,
            "current (val)",
        )
        return reference, current

    # -- Evidently report ----------------------------------------------------

    def _run_drift_report(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> "Report":
        """Run Evidently DataDriftPreset on reference vs current."""
        report = Report([DataDriftPreset()])
        snapshot = report.run(reference_data=reference, current_data=current)
        return snapshot

    def _extract_metrics(self, snapshot) -> Dict:
        """Parse Evidently snapshot into structured drift metrics."""
        results = snapshot.metric_results

        n_drifted = 0
        share_drifted = 0.0
        per_column: Dict[str, Dict] = {}
        drifted_columns: List[str] = []
        target_pvalue: float | None = None
        target_col = self.config.data.target_col

        for _key, val in results.items():
            name = val.display_name

            if "Count of Drifted" in name:
                n_drifted = int(val.count.value)
                share_drifted = float(val.share.value)

            elif "Value drift for" in name:
                col = name.replace("Value drift for ", "")
                pvalue = float(val.value)
                is_drifted = pvalue < self.DRIFT_THRESHOLD
                per_column[col] = {"pvalue": pvalue, "drifted": is_drifted}
                if is_drifted:
                    drifted_columns.append(col)
                if col == target_col:
                    target_pvalue = pvalue

        return {
            "dataset_drift": n_drifted > 0,
            "share_drifted": share_drifted,
            "n_drifted": n_drifted,
            "n_columns": len(per_column),
            "drifted_columns": drifted_columns,
            "target_pvalue": target_pvalue,
            "per_column": per_column,
        }

    # -- Report saving -------------------------------------------------------

    def _save_html_report(self, snapshot, pipeline_run_id: str) -> Path:
        """Save Evidently HTML report to metadata dir."""
        report_dir = Path("src/metadata") / f"pipeline_{pipeline_run_id}"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "drift_report.html"
        snapshot.save_html(str(report_path))
        print(f"[drift] HTML report saved to {report_path}")
        return report_path

    # -- Filesystem ----------------------------------------------------------

    def _metadata_path(self) -> Path:
        return (
            Path("src/metadata")
            / f"pipeline_{self.pipeline_run_id}"
            / "drift_detection.json"
        )

    # -- Main pipeline -------------------------------------------------------

    def run(self) -> DriftResult:
        """Run drift detection, evaluate gate, log to MLflow."""
        print(f"[drift] starting drift detection at {self.timestamp}")

        # 1. Load data
        reference, current = self._load_data()

        # 2. Run Evidently report
        print("[drift] running Evidently DataDriftPreset...")
        snapshot = self._run_drift_report(reference, current)
        metrics = self._extract_metrics(snapshot)

        print(
            f"[drift] drifted {metrics['n_drifted']}/{metrics['n_columns']} columns "
            f"({metrics['share_drifted']:.1%})"
        )
        if metrics["drifted_columns"]:
            print(f"[drift] drifted columns: {metrics['drifted_columns'][:10]}")
        if metrics["target_pvalue"] is not None:
            print(f"[drift] target ({self.config.data.target_col}) p-value: {metrics['target_pvalue']:.4f}")

        # 3. Evaluate drift gate
        drift_cfg = self.config.drift
        gate = DriftGate(drift_cfg.max_share_drifted, drift_cfg.target_drift_pvalue)
        gate_result = gate.evaluate({
            "share_drifted": metrics["share_drifted"],
            "target_pvalue": metrics["target_pvalue"],
        })
        print(f"[drift] drift gate passed (no significant drift): {gate_result.passed}")
        if not gate_result.passed:
            for v in gate_result.violations:
                print(f"[drift]   violation: {v}")

        # 4. Save HTML report
        report_path = self._save_html_report(snapshot, self.pipeline_run_id)

        # 5. Log to MLflow as nested child run
        parent_run_id = self._get_or_create_parent_run()
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(
            run_name=f"drift_{self.timestamp}",
            tags={"mlflow.parentRunId": parent_run_id},
        ) as run:
            mlflow.set_tag("project", "NYC-YELLOW-TAXI-MLOPS")
            mlflow.set_tag("repo", self.config.tracking.repo_name)
            mlflow.set_tag("pipeline_stage", "drift_detection")
            mlflow.set_tag("model_family", self.champion.model_family)
            mlflow.set_tag("pipeline_run_id", self.pipeline_run_id)
            mlflow.set_tag("drift_detected", str(metrics["dataset_drift"]))
            mlflow.set_tag("drift_gate_passed", str(gate_result.passed))

            mlflow.log_metric("share_drifted", metrics["share_drifted"])
            mlflow.log_metric("n_drifted_columns", metrics["n_drifted"])
            mlflow.log_metric("n_total_columns", metrics["n_columns"])
            if metrics["target_pvalue"] is not None:
                mlflow.log_metric("target_drift_pvalue", metrics["target_pvalue"])

            mlflow.log_param("reference_rows", len(reference))
            mlflow.log_param("current_rows", len(current))
            mlflow.log_param("max_share_drifted_threshold", drift_cfg.max_share_drifted)
            mlflow.log_param("target_pvalue_threshold", drift_cfg.target_drift_pvalue)

            mlflow.log_artifact(str(report_path))

            # 6. Save metadata JSON
            result = DriftResult(
                drift_detected=metrics["dataset_drift"],
                share_drifted=metrics["share_drifted"],
                n_drifted=metrics["n_drifted"],
                n_columns=metrics["n_columns"],
                drifted_columns=metrics["drifted_columns"],
                target_drift_pvalue=metrics["target_pvalue"],
                drift_gate_passed=gate_result.passed,
                violations=gate_result.violations,
                report_path=str(report_path),
                run_id=run.info.run_id,
                timestamp_utc=self.timestamp,
            )

            metadata = build_stage_metadata(
                stage="drift_detection",
                pipeline_run_id=self.pipeline_run_id,
                run_id=run.info.run_id,
                created_at_utc=self.timestamp,
                data_rows={
                    "reference_rows": len(reference),
                    "current_rows": len(current),
                },
                metrics={
                    "dataset_drift": metrics["dataset_drift"],
                    "share_drifted": metrics["share_drifted"],
                    "n_drifted": metrics["n_drifted"],
                    "n_columns": metrics["n_columns"],
                    "drifted_columns": metrics["drifted_columns"],
                    "target_pvalue": metrics["target_pvalue"],
                    "drift_gate_passed": gate_result.passed,
                    "violations": gate_result.violations,
                },
                artifacts={
                    "drift_report_html": str(report_path),
                },
                status="success",
                error=None,
            )
            meta_path = write_stage_metadata(
                stage_file_name="drift_detection.json",
                metadata=metadata,
                pipeline_run_id=self.pipeline_run_id,
            )
            mlflow.log_artifact(str(meta_path))

            print(f"[drift] metadata saved to {meta_path}")
            print(f"[drift] MLflow run_id={run.info.run_id}")

        return result

    def close(self) -> None:
        """Stop Spark session."""
        if self.spark:
            self.spark.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> DriftResult:
    """Entry point for Airflow PythonOperator or CLI."""
    config = PipelineConfig.from_yaml("src/config/settings.yaml")
    champion_path = config.training.champion_config_path
    champion = ChampionConfig.load(champion_path)

    pipeline = DriftDetectionPipeline(config, champion)
    try:
        result = pipeline.run()
        print("\n" + "=" * 80)
        print("DRIFT DETECTION RESULT")
        print("=" * 80)
        print(json.dumps(asdict(result), indent=2))
        return result
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
