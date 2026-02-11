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
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import ChampionConfig, PipelineConfig
from src.mlflow.mlflow import _load_external_mlflow

mlflow = _load_external_mlflow()
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


class ModelRegistryPipeline:
    """Registers the best model to MLflow Model Registry.

    If evaluation says the new model should be promoted:
      1. Register the model from the training MLflow run
      2. Transition to Staging
      3. If quality gate passed, transition to Production
      4. Archive old Production version
    """

    EXPERIMENT_NAME = "pipeline"
    REGISTERED_MODEL_NAME = "nyc-taxi-trip-duration"

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        champion_config: ChampionConfig,
    ) -> None:
        self.config = pipeline_config
        self.champion = champion_config
        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.pipeline_run_id = (
            self.champion.mlflow.pipeline_run_id
            if self.champion.mlflow and self.champion.mlflow.pipeline_run_id
            else (
                self.champion.mlflow.experiment_name
                if self.champion.mlflow and self.champion.mlflow.experiment_name
                else f"pipeline_{self.timestamp}"
            )
        )
        self._setup_mlflow()
        self.client = mlflow_client(tracking_uri=self.config.tracking.tracking_uri)

    # -- MLflow setup --------------------------------------------------------

    def _load_env_token(self) -> str:
        key = self.config.tracking.token_env_key
        token = os.getenv(key)
        if token:
            return token.strip()

        env_path = Path(self.config.tracking.env_file_path)
        if not env_path.exists():
            raise ValueError(f"Missing token and {env_path} not found. Set {key} in env or .env.")

        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            k, v = stripped.split("=", 1)
            if k.strip() == key:
                token = v.strip().strip('"').strip("'")
                if token:
                    return token
                break

        raise ValueError(f"{key} is missing/empty in {env_path}.")

    def _setup_mlflow(self) -> None:
        token = self._load_env_token()
        os.environ["MLFLOW_TRACKING_USERNAME"] = self.config.tracking.username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(self.config.tracking.tracking_uri)

    # -- Evaluation result loading -------------------------------------------

    def _load_evaluation_metadata(self) -> Dict[str, Any]:
        """Load evaluation_metadata.json from the pipeline experiments dir."""
        meta_path = (
            Path(self.config.training.champion_config_path).parent
            / "experiments"
            / self.pipeline_run_id
            / "evaluation_metadata.json"
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
            Path(self.config.training.champion_config_path).parent
            / "experiments"
            / self.pipeline_run_id
            / "training_metadata.json"
        )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Training metadata not found at {meta_path}. "
                "Run model_training first."
            )
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data["mlflow_run_id"]

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
        return (
            Path(self.config.training.champion_config_path).parent
            / "experiments"
            / self.pipeline_run_id
            / "registry_metadata.json"
        )

    def _validate_evaluation_preflight(self, eval_meta: Dict[str, Any]) -> None:
        required = ["stage", "pipeline_run_id", "should_register"]
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
        should_register = eval_meta.get("should_register", False)
        quality_gate_passed = bool(eval_meta.get("quality_gate_passed", False))

        if not should_register:
            reason = (
                "Evaluation decided not to register. "
                f"New model better: {eval_meta.get('comparison', {}).get('is_new_better')}. "
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
        mlflow.set_experiment(self.EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"registry_{self.timestamp}") as run:
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
                model_version = mlflow.register_model(
                    model_uri=model_uri,
                    name=self.REGISTERED_MODEL_NAME,
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
            "stage": "model_registry",
            "pipeline_run_id": self.pipeline_run_id,
            "run_id": result.run_id,
            "experiment_name": self.EXPERIMENT_NAME,
            "created_at_utc": result.timestamp_utc,
            "data_rows": {},
            "metrics": {},
            "artifacts": {
                "metadata_path": str(meta_path),
            },
            "registered": result.registered,
            "model_name": result.model_name,
            "version": result.version,
            "final_stage": result.stage,
            "previous_version": result.previous_version,
            "reason": result.reason,
        }

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=False)
            f.write("\n")
        print(f"[registry] metadata saved to {meta_path}")

    def close(self) -> None:
        pass


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
