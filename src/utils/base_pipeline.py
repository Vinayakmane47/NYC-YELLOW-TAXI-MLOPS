"""
Base pipeline class with shared MLflow setup and pipeline run ID resolution.

Eliminates duplicated _load_env_token / _setup_mlflow / pipeline_run_id
logic across model_evaluation.py and model_registry.py.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from src.config.config import ChampionConfig, PipelineConfig
from src.utils.spark_utils import get_pipeline_run_id


class BasePipeline:
    """Common setup shared by evaluation and registry pipeline stages."""

    REGISTERED_MODEL_NAME = "nyc-taxi-trip-duration"

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        champion_config: ChampionConfig,
    ) -> None:
        self.config = pipeline_config
        self.champion = champion_config
        self._now = datetime.now(timezone.utc)
        self.timestamp = self._now.strftime("%H%M%d%m%Y")
        self.experiment_name = f"pipeline_{self._now.strftime('%d%m%Y')}"
        self.run_name = f"run_{self._now.strftime('%H%M')}"
        self.pipeline_run_id = self._resolve_pipeline_run_id()
        self._setup_mlflow()

    def _resolve_pipeline_run_id(self) -> str:
        run_id = get_pipeline_run_id(strict=False)
        if run_id:
            return run_id
        if self.champion.mlflow and self.champion.mlflow.pipeline_run_id:
            return self.champion.mlflow.pipeline_run_id
        if self.champion.mlflow and self.champion.mlflow.experiment_name:
            return self.champion.mlflow.experiment_name
        return self.timestamp

    # -- Parent run management -----------------------------------------------

    def _parent_run_path(self) -> Path:
        return (
            Path("src/metadata")
            / f"pipeline_{self.pipeline_run_id}"
            / "parent_run.json"
        )

    def _save_parent_run_id(self, parent_run_id: str) -> None:
        """Save the parent MLflow run ID so subsequent stages can nest under it."""
        path = self._parent_run_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"parent_run_id": parent_run_id}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[pipeline] parent run ID saved to {path}")

    def _load_parent_run_id(self) -> str | None:
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

    def close(self) -> None:
        pass
