"""
Base pipeline class with shared MLflow setup and pipeline run ID resolution.

Eliminates duplicated _setup_mlflow / pipeline_run_id logic across
model_evaluation.py and model_registry.py.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow

from src.config.config import ChampionConfig, PipelineConfig
from src.utils.mlflow_auth import setup_mlflow
from src.utils.spark_utils import get_pipeline_run_id


class BasePipeline:
    """Common setup shared by evaluation and registry pipeline stages."""

    REGISTERED_MODEL_NAME = "nyc-taxi-trip-duration"
    EXPERIMENT_NAME = "nyc-taxi-pipeline"

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        champion_config: ChampionConfig,
    ) -> None:
        self.config = pipeline_config
        self.champion = champion_config
        self._now = datetime.now(timezone.utc)
        self.timestamp = self._now.isoformat()
        self.pipeline_run_id = self._resolve_pipeline_run_id()
        self.experiment_name = self.EXPERIMENT_NAME
        self.run_name = f"pipeline_{self.pipeline_run_id}"
        self._setup_mlflow()

    def _resolve_pipeline_run_id(self) -> str:
        """Resolve pipeline run ID.

        Priority:
        1. PIPELINE_RUN_ID env var (set by Airflow DAG)
        2. Auto-generated YYYYMMDD_HHMMSS fallback (for CLI runs)
        """
        return get_pipeline_run_id(strict=False)

    # -- Parent run management -----------------------------------------------

    def _parent_run_path(self) -> Path:
        return Path("src/metadata") / f"pipeline_{self.pipeline_run_id}" / "parent_run.json"

    def _save_parent_run_id(self, parent_run_id: str) -> None:
        """Save the parent MLflow run ID so subsequent stages can nest under it."""
        path = self._parent_run_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"parent_run_id": parent_run_id}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[pipeline] parent run ID saved to {path}")

    def _load_parent_run_id(self) -> Optional[str]:
        """Load a previously saved parent run ID (written by model_training)."""
        path = self._parent_run_path()
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("parent_run_id")

    def _get_or_create_parent_run(self) -> str:
        """Return the shared parent run ID, creating it if this is the first stage.

        The parent run is a lightweight container: experiment=pipeline_DDMMYYYY,
        run_name=run_HHMM. Each stage creates a nested child run under it.
        """
        existing = self._load_parent_run_id()
        if existing:
            print(f"[pipeline] reusing parent run {existing}")
            return existing

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.run_name) as parent:
            mlflow.set_tag("project", "NYC-YELLOW-TAXI-MLOPS")
            mlflow.set_tag("repo", self.config.tracking.repo_name)
            mlflow.set_tag("pipeline_run_id", self.pipeline_run_id)
            mlflow.set_tag("model_family", self.champion.model_family)
            parent_run_id = parent.info.run_id

        self._save_parent_run_id(parent_run_id)
        print(f"[pipeline] created parent run {parent_run_id}")
        return parent_run_id

    def _setup_mlflow(self) -> None:
        setup_mlflow(
            tracking_uri=self.config.tracking.tracking_uri,
            username=self.config.tracking.username,
            token_env_key=self.config.tracking.token_env_key,
            env_file_paths=[self.config.tracking.env_file_path],
        )

    def close(self) -> None:
        """No-op for base pipeline (Spark-using subclasses override)."""
