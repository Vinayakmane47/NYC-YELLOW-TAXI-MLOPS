"""
Model Registry Pipeline - Main Airflow DAG stage.

Registers the best model to MLflow Model Registry with stage transitions
(Staging -> Production). Reads evaluation results to decide whether to
promote the new model.

Usage::

    python -m src.model_registry
"""

import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import ChampionConfig, PipelineConfig
from src.utils.base_pipeline import BasePipeline, mlflow
from src.utils.spark_utils import build_stage_metadata, write_stage_metadata

mlflow_client = importlib.import_module("mlflow.tracking").MlflowClient


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RegistryResult:
    """Output of the model registry pipeline."""

    registered: bool
    model_name: str
    version: Optional[int]
    stage: str
    previous_version: Optional[int]
    run_id: str
    timestamp_utc: str
    reason: str


# ---------------------------------------------------------------------------
# Model Registry Pipeline
# ---------------------------------------------------------------------------


class ModelRegistryPipeline(BasePipeline):
    """Registers the best model to MLflow Model Registry.

    If evaluation says the new model should be promoted:
      1. Register the model from the training MLflow run
      2. Transition to Staging
      3. If quality gate passed, transition to Production
      4. Archive old Production version
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        champion_config: ChampionConfig,
    ) -> None:
        super().__init__(pipeline_config, champion_config)
        self.client = mlflow_client(tracking_uri=self.config.tracking.tracking_uri)

    # -- Evaluation result loading -------------------------------------------

    def _load_evaluation_metadata(self) -> Dict[str, Any]:
        """Load evaluation_metadata.json from the pipeline experiments dir."""
        meta_path = (
            Path("src/metadata")
            / f"pipeline_{self.pipeline_run_id}"
            / "model_evaluation.json"
        )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Evaluation metadata not found at {meta_path}. "
                "Run model_evaluation first."
            )
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _find_training_run_id(self) -> str:
        """Find the training run ID from training_metadata.json."""
        meta_path = (
            Path("src/metadata")
            / f"pipeline_{self.pipeline_run_id}"
            / "model_training.json"
        )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Training metadata not found at {meta_path}. "
                "Run model_training first."
            )
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data["run_id"]

    # -- Registry operations -------------------------------------------------

    def _get_current_production_version(self) -> Optional[int]:
        """Get the current Production version number, if any."""
        try:
            versions = self.client.get_latest_versions(
                self.REGISTERED_MODEL_NAME, stages=["Production"]
            )
            if versions:
                return int(versions[0].version)
        except Exception:
            pass
        return None

    def _archive_old_production(self, current_version: int) -> None:
        """Transition the current Production version to Archived."""
        print(f"[registry] archiving old Production version {current_version}")
        self.client.transition_model_version_stage(
            name=self.REGISTERED_MODEL_NAME,
            version=current_version,
            stage="Archived",
        )

    def _find_existing_version_for_run(self, training_run_id: str) -> Optional[Dict[str, Any]]:
        versions = self.client.search_model_versions(f"name='{self.REGISTERED_MODEL_NAME}'")
        for version in versions:
            if version.run_id == training_run_id:
                return {
                    "version": int(version.version),
                    "current_stage": version.current_stage,
                }
        return None

    def _transition_if_needed(self, version_num: int, target_stage: str) -> None:
        details = self.client.get_model_version(self.REGISTERED_MODEL_NAME, str(version_num))
        if details.current_stage == target_stage:
            print(f"[registry] version {version_num} already in {target_stage}")
            return
        self.client.transition_model_version_stage(
            name=self.REGISTERED_MODEL_NAME,
            version=version_num,
            stage=target_stage,
        )
        print(f"[registry] version {version_num} -> {target_stage}")

    # -- Filesystem ----------------------------------------------------------

    def _metadata_path(self) -> Path:
        return Path("src/metadata") / f"pipeline_{self.pipeline_run_id}" / "model_registry.json"

    def _validate_evaluation_preflight(self, eval_meta: Dict[str, Any]) -> None:
        required = ["stage", "pipeline_run_id", "metrics"]
        missing = [k for k in required if k not in eval_meta]
        if missing:
            raise ValueError(f"Evaluation metadata missing required fields: {missing}")
        if eval_meta["stage"] != "model_evaluation":
            raise ValueError(f"Unexpected evaluation stage value: {eval_meta['stage']}")
        if eval_meta["pipeline_run_id"] != self.pipeline_run_id:
            raise ValueError(
                "Pipeline run mismatch between champion.json and evaluation metadata: "
                f"{self.pipeline_run_id} != {eval_meta['pipeline_run_id']}"
            )

    # -- Main pipeline -------------------------------------------------------

    def run(self) -> RegistryResult:
        """Register model if evaluation says it should be promoted."""
        print(f"[registry] starting at {self.timestamp}")

        # 1. Load evaluation decision
        eval_meta = self._load_evaluation_metadata()
        self._validate_evaluation_preflight(eval_meta)
        eval_metrics = eval_meta.get("metrics", {})
        should_register = bool(eval_metrics.get("should_register", False))
        quality_gate_passed = bool(eval_metrics.get("quality_gate_passed", False))

        if not should_register:
            reason = (
                "Evaluation decided not to register. "
                f"New model better: {eval_metrics.get('is_new_better')}. "
                f"Quality gate passed: {quality_gate_passed}."
            )
            print(f"[registry] skipping registration: {reason}")
            result = RegistryResult(
                registered=False,
                model_name=self.REGISTERED_MODEL_NAME,
                version=None,
                stage="None",
                previous_version=self._get_current_production_version(),
                run_id="",
                timestamp_utc=self.timestamp,
                reason=reason,
            )
            self._save_metadata(result)
            return result

        # 2. Find the training run to register from
        training_run_id = self._find_training_run_id()
        model_uri = f"runs:/{training_run_id}/trained_model"
        print(f"[registry] registering model from {model_uri}")

        # 3. Get current production version (for archiving later)
        previous_version = self._get_current_production_version()
        existing_version = self._find_existing_version_for_run(training_run_id)

        # 4. Register model
        parent_run_id = self._get_or_create_parent_run()
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(
            run_name=f"registry_{self.timestamp}",
            tags={"mlflow.parentRunId": parent_run_id},
        ) as run:
            mlflow.set_tag("project", "NYC-YELLOW-TAXI-MLOPS")
            mlflow.set_tag("repo", self.config.tracking.repo_name)
            mlflow.set_tag("pipeline_stage", "model_registry")
            mlflow.set_tag("model_family", self.champion.model_family)
            mlflow.set_tag("pipeline_run_id", self.pipeline_run_id)

            # Register the model
            if existing_version is not None:
                version_num = int(existing_version["version"])
                print(
                    f"[registry] model for training run already exists as v{version_num} "
                    f"(stage={existing_version['current_stage']})"
                )
            else:
                # Use create_model_version (lower-level API) for compatibility
                # with DagShub and older MLflow servers that don't support
                # the logged_model API used by mlflow.register_model in 3.x.
                try:
                    self.client.create_registered_model(self.REGISTERED_MODEL_NAME)
                except Exception:
                    pass  # already exists
                model_version = self.client.create_model_version(
                    name=self.REGISTERED_MODEL_NAME,
                    source=model_uri,
                    run_id=training_run_id,
                )
                version_num = int(model_version.version)
                print(f"[registry] registered as version {version_num}")

            # 5. Transition to Staging first
            self._transition_if_needed(version_num, "Staging")

            final_stage = "Staging"

            # 6. If quality gate passed, promote to Production
            if quality_gate_passed:
                self._transition_if_needed(version_num, "Production")
                final_stage = "Production"

                # Archive old production
                if previous_version is not None and previous_version != version_num:
                    self._archive_old_production(previous_version)
            else:
                print(
                    f"[registry] quality gate failed, "
                    f"version {version_num} stays in Staging"
                )

            # Log registry info
            mlflow.log_param("registered_model_name", self.REGISTERED_MODEL_NAME)
            mlflow.log_param("model_version", version_num)
            mlflow.log_param("final_stage", final_stage)
            mlflow.log_param("previous_version", str(previous_version))
            mlflow.log_param("pipeline_run_id", self.pipeline_run_id)
            mlflow.set_tag("registered_version", str(version_num))
            mlflow.set_tag("final_stage", final_stage)

            result = RegistryResult(
                registered=True,
                model_name=self.REGISTERED_MODEL_NAME,
                version=version_num,
                stage=final_stage,
                previous_version=previous_version,
                run_id=run.info.run_id,
                timestamp_utc=self.timestamp,
                reason=f"Model registered as v{version_num} in {final_stage}.",
            )

            self._save_metadata(result)
            mlflow.log_artifact(str(self._metadata_path()))

            print(f"[registry] MLflow run_id={run.info.run_id}")

        return result

    def _save_metadata(self, result: RegistryResult) -> None:
        """Save registry metadata JSON."""
        meta_path = self._metadata_path()
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "registered": result.registered,
            "model_name": result.model_name,
            "version": result.version,
            "final_stage": result.stage,
            "previous_version": result.previous_version,
            "reason": result.reason,
        }
        payload = build_stage_metadata(
            stage="model_registry",
            pipeline_run_id=self.pipeline_run_id,
            run_id=result.run_id,
            created_at_utc=result.timestamp_utc,
            data_rows={},
            metrics=metadata,
            artifacts={},
            status="success" if result.registered else "skipped",
            error=None,
        )
        meta_path = write_stage_metadata(
            stage_file_name="model_registry.json",
            metadata=payload,
            pipeline_run_id=self.pipeline_run_id,
        )
        print(f"[registry] metadata saved to {meta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Airflow PythonOperator or CLI."""
    config = PipelineConfig.from_yaml()
    champion = ChampionConfig.load(config.training.champion_config_path)

    pipeline = ModelRegistryPipeline(config, champion)
    try:
        result = pipeline.run()
        print("\n" + "=" * 80)
        print("REGISTRY RESULT")
        print("=" * 80)
        print(json.dumps(asdict(result), indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
