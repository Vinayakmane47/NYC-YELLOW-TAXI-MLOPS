"""Model loader for inference - downloads and caches model from MLflow registry."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.utils.mlflow_auth import setup_mlflow

REGISTERED_MODEL_NAME = "nyc-taxi-trip-duration"
DEFAULT_TRACKING_URI = "https://dagshub.com/Vinayakmane47/NYC-YELLOW-TAXI-MLOPS.mlflow/"
DEFAULT_USERNAME = "Vinayakmane47"
MODEL_CACHE_DIR = "artifacts/mlflow_cache"


class ModelLoader:
    """Downloads the raw joblib model artifact from MLflow registry and caches locally.

    Instead of using mlflow.pyfunc.load_model (which requires the custom
    SklearnJoblibPyfuncModel class from src.utils), this downloads the raw
    .joblib artifact directly and loads it with joblib.
    """

    def __init__(
        self,
        model_name: str = REGISTERED_MODEL_NAME,
        model_stage: str = "Production",
        champion_path: str = "src/hpo/champion.json",
        cache_dir: str = MODEL_CACHE_DIR,
    ):
        self.model_name = model_name
        self.model_stage = model_stage
        self.champion_path = Path(champion_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model: Any = None
        self._model_uri: Optional[str] = None
        self._model_version: Optional[str] = None
        self._champion_info: Optional[Dict[str, Any]] = None
        self._source: str = "unknown"

        self._load_champion_info()
        self._load_model()

    def _load_champion_info(self) -> None:
        """Load champion.json for model metadata."""
        if self.champion_path.exists():
            with open(self.champion_path, encoding="utf-8") as f:
                self._champion_info = json.load(f)

    def _load_model(self) -> None:
        """Download model from MLflow registry, fall back to cached version."""
        try:
            self._load_from_registry()
            return
        except Exception as e:
            print(f"[model_loader] MLflow registry load failed: {e}")

        # Fall back to cached joblib
        try:
            self._load_from_cache()
            return
        except Exception as e:
            print(f"[model_loader] Cache load failed: {e}")

        raise RuntimeError(
            f"Cannot load model '{self.model_name}' from MLflow registry or cache. "
            "Ensure the model is registered and DAGSHUB_TOKEN is set."
        )

    def _load_from_registry(self) -> None:
        """Download the raw joblib artifact from MLflow and cache locally."""
        mlflow = setup_mlflow(DEFAULT_TRACKING_URI, DEFAULT_USERNAME)
        client = mlflow.tracking.MlflowClient()

        # Find the Production version and its run_id
        versions = client.get_latest_versions(self.model_name, stages=[self.model_stage])
        if not versions:
            raise RuntimeError(f"No model version found in stage '{self.model_stage}' for '{self.model_name}'")

        version_info = versions[0]
        self._model_version = version_info.version
        run_id = version_info.run_id

        print(f"[model_loader] Found {self.model_name} v{self._model_version} (run_id={run_id}) in {self.model_stage}")

        # Find the joblib artifact inside trained_model/artifacts/
        artifacts = client.list_artifacts(run_id, "trained_model/artifacts")
        joblib_artifact = None
        for a in artifacts:
            if a.path.endswith(".joblib"):
                joblib_artifact = a.path
                break

        if not joblib_artifact:
            raise RuntimeError(f"No .joblib artifact found in run {run_id} under trained_model/artifacts/")

        # Download the artifact to cache
        cache_path = self.cache_dir / f"{self.model_name}_v{self._model_version}.joblib"

        if cache_path.exists():
            print(f"[model_loader] Using cached artifact: {cache_path}")
        else:
            print(f"[model_loader] Downloading {joblib_artifact} ...")
            download_dir = client.download_artifacts(run_id, joblib_artifact, str(self.cache_dir))
            # download_artifacts returns the local path of the downloaded file
            downloaded = Path(download_dir)
            if downloaded != cache_path:
                downloaded.rename(cache_path)
            print(f"[model_loader] Cached to {cache_path}")

        self._model = joblib.load(cache_path)
        self._model_uri = f"models:/{self.model_name}/{self.model_stage} (v{self._model_version})"
        self._source = "mlflow_registry"
        print(f"[model_loader] Loaded {self.model_name} v{self._model_version} from {self.model_stage}")

    def _load_from_cache(self) -> None:
        """Load model from latest cached joblib file."""
        joblib_files = sorted(
            self.cache_dir.glob(f"{self.model_name}_v*.joblib"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not joblib_files:
            raise FileNotFoundError(f"No cached model files in {self.cache_dir}")

        cache_path = joblib_files[0]
        self._model = joblib.load(cache_path)
        self._model_uri = f"cache:{cache_path}"
        self._source = "local_cache"
        print(f"[model_loader] Loaded from cache: {cache_path}")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Run prediction on model-ready features."""
        return self._model.predict(features)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> Optional[str]:
        return self._model_uri

    @property
    def model_family(self) -> str:
        if self._champion_info:
            return self._champion_info.get("model_family", "unknown")
        return "unknown"

    @property
    def model_version(self) -> str:
        if self._model_version:
            return f"v{self._model_version}"
        return "unknown"

    @property
    def model_metrics(self) -> Dict[str, Any]:
        if self._champion_info:
            return self._champion_info.get("metrics", {})
        return {}

    @property
    def model_params(self) -> Dict[str, Any]:
        if self._champion_info:
            return self._champion_info.get("params", {})
        return {}

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata for API responses."""
        return {
            "model_family": self.model_family,
            "model_version": self.model_version,
            "model_name": self.model_name,
            "model_stage": self.model_stage,
            "source": self._source,
            "model_uri": self._model_uri,
            "metrics": self.model_metrics,
            "params": self.model_params,
            "loaded": self.is_loaded,
        }

    def reload(self) -> None:
        """Reload the model from MLflow registry."""
        self._model_version = None
        self._load_champion_info()
        self._load_model()
