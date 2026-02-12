"""
Model Evaluation Pipeline - Main Airflow DAG stage.

Loads the newly trained champion model and compares it against the
currently registered Production model in MLflow Model Registry.
Uses multiple metrics (RMSE, MAE, R2) for comparison. Decides whether
the new model should be promoted.

Usage::

    python -m src.model_evaluation
"""

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import ChampionConfig, PipelineConfig
from src.utils.base_pipeline import BasePipeline, mlflow
from src.utils.quality_gate import QualityGate
from src.utils.spark_utils import (
    SparkUtils,
    build_stage_metadata,
    write_stage_metadata,
)

importlib.import_module("mlflow.sklearn")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MetricComparison:
    """Comparison detail for a single metric."""

    metric: str
    new_value: float
    old_value: Optional[float]
    improved: bool
    delta: Optional[float]


@dataclass
class ComparisonResult:
    """Outcome of comparing new vs registered model."""

    is_new_better: bool
    metric_comparisons: List[MetricComparison]
    summary: str
    decision_reason: str


@dataclass
class EvaluationResult:
    """Output of the evaluation pipeline."""

    new_model_metrics: Dict[str, float]
    old_model_metrics: Optional[Dict[str, float]]
    comparison: ComparisonResult
    quality_gate_passed: bool
    violations: List[str]
    should_register: bool
    decision_reason: str
    evaluation_report_path: str
    run_id: str
    timestamp_utc: str


# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------


class ModelEvaluationPipeline(BasePipeline):
    """Evaluates the newly trained model against the registered Production model.

    Compares RMSE, MAE, R2 and runs the quality gate. Decides whether
    the new model should be promoted to Production in the registry.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        champion_config: ChampionConfig,
    ) -> None:
        super().__init__(pipeline_config, champion_config)
        self.spark = SparkUtils(app_name="nyc-taxi-pipeline-evaluation").spark

    # -- Data loading --------------------------------------------------------

    def _load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load test split via Spark and split into X, y."""
        sdf = self.spark.read.parquet(self.config.data.test_path)
        target = self.config.data.target_col

        if target not in sdf.columns:
            raise ValueError(f"Target column '{target}' missing in test data")

        feature_cols = [c for c in sdf.columns if c != target]
        sdf = sdf.select(*feature_cols, target)

        max_rows = self.config.data.max_rows_test
        if max_rows > 0:
            sdf = sdf.limit(max_rows)

        pdf = sdf.toPandas()
        print(f"[evaluation] test data: {len(pdf):,} rows, {len(pdf.columns)} cols")

        x = pdf.drop(columns=[target])
        y = pdf[target]
        return x, y

    # -- Model loading -------------------------------------------------------

    def _load_new_model(self) -> Any:
        """Load newly trained model path from training metadata."""
        training_meta_path = (
            Path("src/metadata")
            / f"pipeline_{self.pipeline_run_id}"
            / "model_training.json"
        )
        if not training_meta_path.exists():
            raise FileNotFoundError(
                f"Training metadata not found at '{training_meta_path}'. "
                "Run model_training first."
            )
        with training_meta_path.open("r", encoding="utf-8") as f:
            training_meta = json.load(f)
        model_path = training_meta.get("artifacts", {}).get("model_path", "")
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(
                f"New model artifact not found at '{model_path}'. "
                "Run model_training first."
            )
        print(f"[evaluation] loading new model from {model_path}")
        return joblib.load(model_path)

    def _load_registered_model(self) -> Optional[Any]:
        """Load the current Production model from MLflow Model Registry.

        Returns None if no model is registered yet (first run).
        """
        try:
            model_uri = f"models:/{self.REGISTERED_MODEL_NAME}/Production"
            print(f"[evaluation] loading registered model from {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)
            print("[evaluation] registered Production model loaded successfully")
            return model
        except Exception as e:
            print(f"[evaluation] no registered Production model found: {e}")
            return None

    # -- Metrics -------------------------------------------------------------

    def _compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    # -- Model comparison ----------------------------------------------------

    def _compare_models(
        self,
        new_metrics: Dict[str, float],
        old_metrics: Optional[Dict[str, float]],
    ) -> ComparisonResult:
        """Compare new model vs registered model using multiple metrics.

        New model wins if all of:
          - RMSE is lower (or equal)
          - MAE is lower (or equal)
          - R2 is higher (or equal)

        If no old model exists, new model wins by default.
        """
        if old_metrics is None:
            comparisons = [
                MetricComparison(
                    metric=m,
                    new_value=new_metrics[m],
                    old_value=None,
                    improved=True,
                    delta=None,
                )
                for m in ("rmse", "mae", "r2")
            ]
            return ComparisonResult(
                is_new_better=True,
                metric_comparisons=comparisons,
                summary="No registered model found. New model wins by default.",
                decision_reason="no_production_model",
            )

        comparisons: List[MetricComparison] = []
        policy = self.config.evaluation
        rmse_delta = old_metrics["rmse"] - new_metrics["rmse"]
        mae_delta = old_metrics["mae"] - new_metrics["mae"]
        r2_delta = new_metrics["r2"] - old_metrics["r2"]

        # For RMSE and MAE: lower is better
        for metric in ("rmse", "mae"):
            new_val = new_metrics[metric]
            old_val = old_metrics[metric]
            if metric == "rmse":
                improved = rmse_delta >= policy.min_rmse_improvement
            else:
                improved = mae_delta >= -policy.max_mae_regression
            comparisons.append(MetricComparison(
                metric=metric,
                new_value=new_val,
                old_value=old_val,
                improved=improved,
                delta=old_val - new_val,
            ))

        # For R2: higher is better
        new_r2 = new_metrics["r2"]
        old_r2 = old_metrics["r2"]
        comparisons.append(MetricComparison(
            metric="r2",
            new_value=new_r2,
            old_value=old_r2,
            improved=r2_delta >= -policy.max_r2_regression,
            delta=r2_delta,
        ))

        is_new_better = all(c.improved for c in comparisons)

        improvements = [c.metric for c in comparisons if c.improved]
        regressions = [c.metric for c in comparisons if not c.improved]

        summary = (
            f"Promotion policy check -> rmse_delta={rmse_delta:.6f}, "
            f"mae_delta={mae_delta:.6f}, r2_delta={r2_delta:.6f}, "
            f"improved={improvements or ['none']}, regressed={regressions or ['none']}."
        )
        decision_reason = (
            "meets_rmse_improvement_and_no_material_mae_r2_regression"
            if is_new_better
            else "fails_promotion_policy_thresholds"
        )

        return ComparisonResult(
            is_new_better=is_new_better,
            metric_comparisons=comparisons,
            summary=summary,
            decision_reason=decision_reason,
        )

    # -- Filesystem ----------------------------------------------------------

    def _metadata_path(self) -> Path:
        return Path("src/metadata") / f"pipeline_{self.pipeline_run_id}" / "model_evaluation.json"

    # -- Main pipeline -------------------------------------------------------

    def run(self) -> EvaluationResult:
        """Evaluate new model, compare with registered, decide promotion."""
        print(f"[evaluation] starting evaluation at {self.timestamp}")

        # 1. Load test data
        x_test, y_test = self._load_test_data()

        # 2. Load models
        new_model = self._load_new_model()
        old_model = self._load_registered_model()

        # 3. Predict with new model
        print("[evaluation] predicting with new model...")
        new_pred = new_model.predict(x_test)
        new_metrics = self._compute_metrics(y_test, new_pred)
        print(
            f"[evaluation] new model: "
            f"rmse={new_metrics['rmse']:.4f} "
            f"mae={new_metrics['mae']:.4f} "
            f"r2={new_metrics['r2']:.4f}"
        )

        # 4. Predict with old model (if exists)
        old_metrics: Optional[Dict[str, float]] = None
        if old_model is not None:
            print("[evaluation] predicting with registered model...")
            old_pred = old_model.predict(x_test)
            old_metrics = self._compute_metrics(y_test, old_pred)
            print(
                f"[evaluation] registered model: "
                f"rmse={old_metrics['rmse']:.4f} "
                f"mae={old_metrics['mae']:.4f} "
                f"r2={old_metrics['r2']:.4f}"
            )

        # 5. Compare models
        comparison = self._compare_models(new_metrics, old_metrics)
        print(f"[evaluation] comparison: {comparison.summary}")

        # 6. Quality gate
        thresholds = self.config.training.metric_thresholds
        gate = QualityGate(thresholds.min_r2, thresholds.max_rmse, thresholds.max_mae)
        gate_result = gate.evaluate(new_metrics)
        print(f"[evaluation] quality gate passed: {gate_result.passed}")
        if not gate_result.passed:
            for v in gate_result.violations:
                print(f"[evaluation]   violation: {v}")

        should_register = comparison.is_new_better and gate_result.passed
        print(f"[evaluation] should register: {should_register}")

        # 7. Log to MLflow
        parent_run_id = self._get_or_create_parent_run()
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(
            run_name=f"evaluation_{self.timestamp}",
            tags={"mlflow.parentRunId": parent_run_id},
        ) as run:
            mlflow.set_tag("project", "NYC-YELLOW-TAXI-MLOPS")
            mlflow.set_tag("repo", self.config.tracking.repo_name)
            mlflow.set_tag("pipeline_stage", "model_evaluation")
            mlflow.set_tag("model_family", self.champion.model_family)
            mlflow.set_tag("pipeline_run_id", self.pipeline_run_id)
            mlflow.set_tag("quality_gate_passed", str(gate_result.passed))
            mlflow.set_tag("is_new_model_better", str(comparison.is_new_better))
            mlflow.set_tag("should_register", str(should_register))
            mlflow.set_tag("decision_reason", comparison.decision_reason)

            # New model metrics
            mlflow.log_metrics({
                "new_test_rmse": new_metrics["rmse"],
                "new_test_mae": new_metrics["mae"],
                "new_test_r2": new_metrics["r2"],
            })

            # Old model metrics (if available)
            if old_metrics is not None:
                mlflow.log_metrics({
                    "registered_test_rmse": old_metrics["rmse"],
                    "registered_test_mae": old_metrics["mae"],
                    "registered_test_r2": old_metrics["r2"],
                })

                # Deltas
                mlflow.log_metrics({
                    "delta_rmse": old_metrics["rmse"] - new_metrics["rmse"],
                    "delta_mae": old_metrics["mae"] - new_metrics["mae"],
                    "delta_r2": new_metrics["r2"] - old_metrics["r2"],
                })

            mlflow.log_param("test_rows", len(x_test))
            mlflow.log_param("champion_model_family", self.champion.model_family)

            # 8. Save evaluation metadata
            meta_path = self._metadata_path()
            meta_path.parent.mkdir(parents=True, exist_ok=True)

            result = EvaluationResult(
                new_model_metrics=new_metrics,
                old_model_metrics=old_metrics,
                comparison=comparison,
                quality_gate_passed=gate_result.passed,
                violations=gate_result.violations,
                should_register=should_register,
                decision_reason=comparison.decision_reason,
                evaluation_report_path=str(meta_path),
                run_id=run.info.run_id,
                timestamp_utc=self.timestamp,
            )

            metadata = build_stage_metadata(
                stage="model_evaluation",
                pipeline_run_id=self.pipeline_run_id,
                run_id=run.info.run_id,
                created_at_utc=self.timestamp,
                data_rows={"test_rows": len(x_test)},
                metrics={
                    "new_model": new_metrics,
                    "registered_model": old_metrics,
                    "deltas": {
                        "rmse": (old_metrics["rmse"] - new_metrics["rmse"]) if old_metrics else None,
                        "mae": (old_metrics["mae"] - new_metrics["mae"]) if old_metrics else None,
                        "r2": (new_metrics["r2"] - old_metrics["r2"]) if old_metrics else None,
                    },
                    "quality_gate_passed": gate_result.passed,
                    "is_new_better": comparison.is_new_better,
                    "should_register": should_register,
                    "decision_reason": comparison.decision_reason,
                },
                artifacts={
                    "compared_against_model": self.REGISTERED_MODEL_NAME,
                },
                status="success",
                error=None,
            )
            meta_path = write_stage_metadata(
                stage_file_name="model_evaluation.json",
                metadata=metadata,
                pipeline_run_id=self.pipeline_run_id,
            )
            mlflow.log_artifact(str(meta_path))

            print(f"[evaluation] metadata saved to {meta_path}")
            print(f"[evaluation] MLflow run_id={run.info.run_id}")

        return result

    def close(self) -> None:
        """Stop Spark session."""
        if self.spark:
            self.spark.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Airflow PythonOperator or CLI."""
    config = PipelineConfig.from_yaml()
    champion = ChampionConfig.load(config.training.champion_config_path)

    pipeline = ModelEvaluationPipeline(config, champion)
    try:
        result = pipeline.run()
        print("\n" + "=" * 80)
        print("EVALUATION RESULT")
        print("=" * 80)
        print(f"New model better: {result.comparison.is_new_better}")
        print(f"Quality gate passed: {result.quality_gate_passed}")
        print(f"Should register: {result.should_register}")
        print(f"New metrics: {result.new_model_metrics}")
        if result.old_model_metrics:
            print(f"Old metrics: {result.old_model_metrics}")
        print(f"Comparison: {result.comparison.summary}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
