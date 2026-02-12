from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class TrackingConfig(BaseModel):
    tracking_uri: str
    username: str
    repo_name: str
    token_env_key: str = "DAGSHUB_TOKEN"
    env_file_path: str = ".env"
    model_config = {"extra": "forbid"}


class DataConfig(BaseModel):
    train_path: str
    val_path: str
    test_path: str
    target_col: str = "trip_duration_min"
    trained_on_range: str = ""
    max_rows_train: int = 300_000
    max_rows_val: int = 100_000
    max_rows_test: int = 100_000
    sample_fraction: Optional[float] = None
    model_config = {"extra": "forbid"}


class MetricThresholds(BaseModel):
    min_r2: float = 0.0
    max_rmse: float = 999_999.0
    max_mae: float = 999_999.0
    model_config = {"extra": "forbid"}


class TrainingConfig(BaseModel):
    n_trials_per_model: int = 25
    random_state: int = 42
    champion_config_path: str = "src/mlflow/champion.json"
    metric_thresholds: MetricThresholds = Field(default_factory=MetricThresholds)
    model_config = {"extra": "forbid"}


class ScreeningConfig(BaseModel):
    top_n_for_hpo: int = 3
    row_cap_train: int = 100_000
    row_cap_val: int = 50_000
    model_config = {"extra": "forbid"}


class ModelConfig(BaseModel):
    name: str
    enabled: bool = True
    screening_fixed_params: Dict[str, Any] = Field(default_factory=dict)
    search_space: Dict[str, Any] = Field(default_factory=dict)
    fixed_params: Dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}


class MlflowPipelineConfig(BaseModel):
    tracking: TrackingConfig
    data: DataConfig
    training: TrainingConfig
    screening: ScreeningConfig
    models_to_train: List[ModelConfig] = Field(default_factory=list)
    screening_models: List[ModelConfig] = Field(default_factory=list)
    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: str = "src/mlflow/config_mlflow.yaml") -> "MlflowPipelineConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.model_validate(raw)

    @property
    def enabled_hpo_models(self) -> List[ModelConfig]:
        return [m for m in self.models_to_train if m.enabled and m.search_space]

    @property
    def enabled_screening_models(self) -> List[ModelConfig]:
        return [m for m in self.screening_models if m.enabled]
