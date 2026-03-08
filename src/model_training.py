"""
Model Training Pipeline - Main Airflow DAG stage.

Reads champion.json (output of offline screening/HPO) to get the
winning model family + hyperparameters, then trains that exact model on
full production data (~3M rows). Logs everything to MLflow experiment "pipeline".

Usage::

    python -m src.model_training
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd

from src.config.config import ChampionConfig, PipelineConfig
from src.models.model_factory import build_model
from src.utils.base_pipeline import BasePipeline, mlflow
from src.utils.metrics import compute_regression_metrics
from src.utils.mlflow_pyfunc import SklearnJoblibPyfuncModel
from src.utils.spark_utils import (
    SparkUtils,
    build_stage_metadata,
    get_pipeline_metadata_dir,
    write_stage_metadata,
)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Output of the model training pipeline."""

    model_family: str
    model_path: str
    val_metrics: Dict[str, float]
    train_rows: int
    val_rows: int
    run_id: str
    experiment_name: str
    timestamp_utc: str
    params: Dict[str, Any] = field(default_factory=dict)
    metadata_path: str = ""


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------


class ModelTrainingPipeline(BasePipeline):
    """Trains the champion model (from HPO) on full production data.

    Reads champion.json for model family + params, loads full
    training data via Spark, fits the model, and logs to MLflow
    experiment "pipeline".
    """

    MAX_RECOMMENDED_PANDAS_ROWS = 2_000_000

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        champion_config: ChampionConfig,
    ) -> None:
        super().__init__(pipeline_config, champion_config)
        self.spark = SparkUtils(app_name="nyc-taxi-pipeline-training").spark

    # -- Data loading --------------------------------------------------------

    def _load_split(self, path: str, split_name: str, max_rows: int) -> pd.DataFrame:
        """Read parquet with Spark, limit rows, convert to pandas."""
        sdf = self.spark.read.parquet(path)
        target = self.config.data.target_col

        if target not in sdf.columns:
            raise ValueError(f"Target column '{target}' missing in {split_name}: {path}")

        feature_cols = [c for c in sdf.columns if c != target]
        sdf = sdf.select(*feature_cols, target)

        sample_frac = self.config.data.sample_fraction
        if sample_frac is not None and 0 < sample_frac < 1:
            sdf = sdf.sample(
                withReplacement=False,
                fraction=sample_frac,
                seed=self.config.training.random_state,
            )

        if max_rows > 0:
            sdf = sdf.limit(max_rows)

        pdf = sdf.toPandas()
        print(f"[training] {split_name}: {len(pdf):,} rows, {len(pdf.columns)} cols")
        return pdf

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and val splits."""
        train_df = self._load_split(
            self.config.data.train_path,
            split_name="train",
            max_rows=self.config.data.max_rows_train,
        )
        val_df = self._load_split(
            self.config.data.val_path,
            split_name="val",
            max_rows=self.config.data.max_rows_val,
        )
        return train_df, val_df

    def _validate_feature_set(self, x_train: pd.DataFrame, x_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Validate and align features against champion contract when provided."""
        if not self.champion.feature_list:
            return x_train, x_val

        expected = list(self.champion.feature_list)
        missing = [col for col in expected if col not in x_train.columns]
        if missing:
            raise ValueError(f"Champion feature_list has missing columns in training data: {missing[:10]}")

        x_train_aligned = x_train[expected]
        x_val_aligned = x_val[expected]
        return x_train_aligned, x_val_aligned

    def _enforce_data_guardrails(self, train_rows: int) -> None:
        if train_rows > self.MAX_RECOMMENDED_PANDAS_ROWS:
            raise ValueError(
                "Loaded training data is too large for pandas-based training "
                f"({train_rows:,} rows). Reduce data.max_rows_train/sample_fraction or "
                "switch to a distributed trainer."
            )

    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        target = self.config.data.target_col
        return df.drop(columns=[target]), df[target]

    # -- Model building ------------------------------------------------------

    def _build_model(self) -> Any:
        """Build the champion model from the shared model factory."""
        return build_model(self.champion.model_family, self.champion.params)

    # -- Filesystem ----------------------------------------------------------

    def _ensure_dirs(self) -> None:
        Path(self.config.training.models_dir).mkdir(parents=True, exist_ok=True)
        get_pipeline_metadata_dir(self.pipeline_run_id)

    def _metadata_path(self) -> Path:
        return Path("src/metadata") / f"pipeline_{self.pipeline_run_id}" / "model_training.json"

    def _model_artifact_paths(self) -> Path:
        metadata_dir = get_pipeline_metadata_dir(self.pipeline_run_id)
        joblib_path = (metadata_dir / f"{self.champion.model_family}.joblib").resolve()
        return joblib_path

    # -- Main pipeline -------------------------------------------------------

    def run(self) -> TrainingResult:
        """Train champion model on full data, log to MLflow."""
        self._ensure_dirs()
        print(f"[training] champion model={self.champion.model_family}")
        print(f"[training] params={json.dumps(self.champion.params, indent=2)}")

        # 1. Load data
        print("[training] loading data...")
        train_df, val_df = self._load_data()
        self._enforce_data_guardrails(len(train_df))
        x_train, y_train = self._prepare_xy(train_df)
        x_val, y_val = self._prepare_xy(val_df)
        x_train, x_val = self._validate_feature_set(x_train, x_val)

        # 2. Build model
        model = self._build_model()

        # 3. Train and log to MLflow
        parent_run_id = self._get_or_create_parent_run()
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(
            run_name=f"training_{self.pipeline_run_id}",
            tags={"mlflow.parentRunId": parent_run_id},
        ) as run:
            mlflow.set_tag("project", "NYC-YELLOW-TAXI-MLOPS")
            mlflow.set_tag("repo", self.config.tracking.repo_name)
            mlflow.set_tag("pipeline_stage", "model_training")
            mlflow.set_tag("model_family", self.champion.model_family)
            mlflow.set_tag("pipeline_run_id", self.pipeline_run_id)

            # Log params
            mlflow.log_params(self.champion.params)
            mlflow.log_param("target_col", self.config.data.target_col)
            mlflow.log_param("trained_on_range", self.config.data.trained_on_range)
            mlflow.log_param("train_rows", len(train_df))
            mlflow.log_param("val_rows", len(val_df))

            # Fit
            print(f"[training] fitting on {len(x_train):,} rows...")
            start_ts = time.time()
            model.fit(x_train, y_train)
            elapsed = time.time() - start_ts
            print(f"[training] fit completed in {elapsed:.2f}s")

            mlflow.log_metric("training_time_sec", elapsed)

            # Val metrics
            val_pred = model.predict(x_val)
            val_metrics = compute_regression_metrics(y_val, val_pred)
            print(
                f"[training] val metrics: "
                f"rmse={val_metrics['rmse']:.4f} "
                f"mae={val_metrics['mae']:.4f} "
                f"r2={val_metrics['r2']:.4f}"
            )
            mlflow.log_metrics(
                {
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                    "val_r2": val_metrics["r2"],
                }
            )

            # Log feature importance
            if hasattr(model, "feature_importances_"):
                importance = sorted(
                    zip(x_train.columns, model.feature_importances_),
                    key=lambda item: item[1],
                    reverse=True,
                )
                for i, (feat, imp) in enumerate(importance[:20]):
                    mlflow.log_metric(f"feat_imp_{i:02d}_{feat[:40]}", float(imp))
                imp_path = get_pipeline_metadata_dir(self.pipeline_run_id) / "feature_importance.json"
                with imp_path.open("w", encoding="utf-8") as f:
                    json.dump(importance, f, indent=2)
                mlflow.log_artifact(str(imp_path))
                print(f"[training] top 5 features: {[f[0] for f in importance[:5]]}")
            elif hasattr(model, "coef_"):
                importance = sorted(
                    zip(x_train.columns, [abs(c) for c in model.coef_]),
                    key=lambda item: item[1],
                    reverse=True,
                )
                for i, (feat, imp) in enumerate(importance[:20]):
                    mlflow.log_metric(f"feat_imp_{i:02d}_{feat[:40]}", float(imp))
                print(f"[training] top 5 features (coef): {[f[0] for f in importance[:5]]}")

            # Save model artifact
            model_path = self._model_artifact_paths()
            joblib.dump(model, model_path, compress=3)
            print(f"[training] model saved to {model_path}")

            # Upload a lightweight MLflow pyfunc wrapper that references the
            # compressed joblib artifact. This keeps the remote artifact small
            # enough for DagShub while preserving a registry-compatible runs:/ URI.
            mlflow.pyfunc.log_model(
                artifact_path="trained_model",
                python_model=SklearnJoblibPyfuncModel(),
                artifacts={"model_file": str(model_path)},
            )
            registered_model_source = f"runs:/{run.info.run_id}/trained_model"
            print(f"[training] remote MLflow model logged at {registered_model_source}")

            # Save training metadata
            meta_path = self._metadata_path()
            result = TrainingResult(
                model_family=self.champion.model_family,
                model_path=str(model_path),
                val_metrics=val_metrics,
                train_rows=len(train_df),
                val_rows=len(val_df),
                run_id=run.info.run_id,
                experiment_name=self.experiment_name,
                timestamp_utc=self.timestamp,
                params=self.champion.params,
                metadata_path=str(meta_path),
            )

            metadata = build_stage_metadata(
                stage="model_training",
                pipeline_run_id=self.pipeline_run_id,
                run_id=result.run_id,
                created_at_utc=self.timestamp,
                data_rows={
                    "train_rows": result.train_rows,
                    "val_rows": result.val_rows,
                },
                metrics={
                    "validation": result.val_metrics,
                    "training_time_sec": elapsed,
                },
                artifacts={
                    "model_path": result.model_path,
                    "registered_model_source": registered_model_source,
                },
                status="success",
                error=None,
            )
            meta_path = write_stage_metadata(
                stage_file_name="model_training.json",
                metadata=metadata,
                pipeline_run_id=self.pipeline_run_id,
            )
            mlflow.log_artifact(str(meta_path))

            print(f"[training] metadata saved to {meta_path}")
            print(f"[training] MLflow run_id={run.info.run_id}")

        return result

    def close(self) -> None:
        """Stop Spark session."""
        try:
            if self.spark:
                self.spark.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Airflow PythonOperator or CLI."""
    config = PipelineConfig.from_yaml("src/config/settings.yaml")
    champion_path = config.training.champion_config_path
    champion = ChampionConfig.load(champion_path)

    pipeline = ModelTrainingPipeline(config, champion)
    try:
        result = pipeline.run()
        print("\n" + "=" * 80)
        print("TRAINING RESULT")
        print("=" * 80)
        print(json.dumps(asdict(result), indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
