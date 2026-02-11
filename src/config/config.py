"""
Pydantic configuration models for the NYC Yellow Taxi MLOps pipeline.

Single source of truth: loads and validates config_mlflow.yaml and champion.json.
All pipeline stages (training, evaluation, registry) use these typed models
instead of raw dict access.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class TrackingConfig(BaseModel):
    """MLflow / DagShub tracking connection settings."""

    tracking_uri: str
    username: str
    repo_name: str
    token_env_key: str = "DAGSHUB_TOKEN"
    env_file_path: str = ".env"
    model_config = {"extra": "forbid"}


class DataConfig(BaseModel):
    """Paths and limits for train / val / test splits."""

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
    """Quality gate thresholds for model promotion."""

    min_r2: float = 0.0
    max_rmse: float = 999_999.0
    max_mae: float = 999_999.0
    model_config = {"extra": "forbid"}


class TrainingConfig(BaseModel):
    """HPO and artifact settings."""

    n_trials_per_model: int = 25
    random_state: int = 42
    models_dir: str = "src/mlflow/models"
    champion_config_path: str = "src/mlflow/champion.json"
    metric_thresholds: MetricThresholds = Field(default_factory=MetricThresholds)
    model_config = {"extra": "forbid"}


class ScreeningConfig(BaseModel):
    """Model screening stage settings."""

    top_n_for_hpo: int = 3
    row_cap_train: int = 100_000
    row_cap_val: int = 50_000
    model_config = {"extra": "forbid"}


class ModelConfig(BaseModel):
    """Configuration for a single model family."""

    name: str
    enabled: bool = True
    screening_fixed_params: Dict[str, Any] = Field(default_factory=dict)
    search_space: Dict[str, Any] = Field(default_factory=dict)
    fixed_params: Dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}


class EvaluationPolicyConfig(BaseModel):
    """
    Deterministic promotion rule thresholds:
    - new RMSE must improve by at least min_rmse_improvement
    - MAE regression must not exceed max_mae_regression
    - R2 regression must not exceed max_r2_regression
    """

    min_rmse_improvement: float = 0.0
    max_mae_regression: float = 0.0
    max_r2_regression: float = 0.0
    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Root pipeline config
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Root config loaded from config_mlflow.yaml.

    Usage::

        config = PipelineConfig.from_yaml("src/mlflow/config_mlflow.yaml")
        print(config.tracking.tracking_uri)
    """

    tracking: TrackingConfig
    data: DataConfig
    training: TrainingConfig
    screening: ScreeningConfig
    evaluation: EvaluationPolicyConfig = Field(default_factory=EvaluationPolicyConfig)
    models_to_train: List[ModelConfig] = Field(default_factory=list)
    screening_models: List[ModelConfig] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: str = "src/mlflow/config_mlflow.yaml") -> "PipelineConfig":
        """Load, parse, and validate the YAML config file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    @property
    def enabled_hpo_models(self) -> List[ModelConfig]:
        """Return only enabled models that have a search space (HPO-eligible)."""
        return [m for m in self.models_to_train if m.enabled and m.search_space]

    @property
    def enabled_screening_models(self) -> List[ModelConfig]:
        """Return only enabled screening-only models."""
        return [m for m in self.screening_models if m.enabled]

    @property
    def all_enabled_models(self) -> List[ModelConfig]:
        """Return all enabled models (HPO + screening-only)."""
        return [m for m in self.models_to_train if m.enabled] + self.enabled_screening_models


# ---------------------------------------------------------------------------
# Champion config (output of screening / HPO)
# ---------------------------------------------------------------------------


class ChampionMetricBlock(BaseModel):
    rmse: float
    mae: float
    r2: float
    model_config = {"extra": "allow"}


class ChampionMetrics(BaseModel):
    validation: ChampionMetricBlock
    test: Optional[ChampionMetricBlock] = None
    model_config = {"extra": "allow"}


class ChampionData(BaseModel):
    train_rows: Optional[int] = None
    val_rows: Optional[int] = None
    test_rows: Optional[int] = None
    target_col: Optional[str] = None
    trained_on_range: Optional[str] = None
    data_schema_hash: Optional[str] = None
    model_config = {"extra": "allow"}


class ChampionQualityGate(BaseModel):
    passed: bool
    violations: List[str] = Field(default_factory=list)
    thresholds: Optional[MetricThresholds] = None
    model_config = {"extra": "allow"}


class ChampionMlflowInfo(BaseModel):
    experiment_name: str
    screening_run_id: str
    hpo_run_id: str
    pipeline_run_id: Optional[str] = None
    model_config = {"extra": "allow"}


class ChampionConfig(BaseModel):
    """Loaded from champion.json produced by the HPO pipeline.

    Usage::

        champion = ChampionConfig.load("src/mlflow/champion.json")
        print(champion.model_family, champion.params)
    """

    model_family: str = Field(min_length=1)
    params: Dict[str, Any] = Field(default_factory=dict, min_length=1)
    metrics: ChampionMetrics
    data: ChampionData = Field(default_factory=ChampionData)
    quality_gate: Optional[ChampionQualityGate] = None
    mlflow: Optional[ChampionMlflowInfo] = None
    created_at_utc: str = Field(min_length=1)

    # Backward-compatible optional extension blocks for downstream scripts.
    feature_list: List[str] = Field(default_factory=list)
    data_schema_hash: str = ""
    trained_on_range: str = ""
    metric_thresholds: MetricThresholds = Field(default_factory=MetricThresholds)
    dataset_info: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, str] = Field(default_factory=dict)
    training_info: Dict[str, Any] = Field(default_factory=dict)
    leaderboard: List[Dict[str, Any]] = Field(default_factory=list)
    screening_results: List[Dict[str, Any]] = Field(default_factory=list)
    hpo_candidates: List[str] = Field(default_factory=list)
    mlflow_info: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}

    @classmethod
    def load(cls, path: str = "src/mlflow/champion.json") -> "ChampionConfig":
        """Load and validate champion.json."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Champion config not found: {path}")
        with config_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.model_validate(raw)

    def save(self, path: str = "src/mlflow/champion.json") -> None:
        """Write current state back to JSON."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, sort_keys=False)
            f.write("\n")
