import hashlib
import importlib
import json
import os
import sys
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# Ensure project root is importable when run as `python src/mlflow/mlflow.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.spark_utils import SparkUtils

if TYPE_CHECKING:
    from src.config.config import PipelineConfig


def _load_external_mlflow():
    """
    Load pip-installed mlflow package and avoid local shadowing from src/mlflow.
    """
    current_dir = str(Path(__file__).resolve().parent)
    parent_src_dir = str(Path(__file__).resolve().parents[1])
    removed_paths: List[str] = []

    for candidate in (current_dir, parent_src_dir):
        if candidate in sys.path:
            sys.path.remove(candidate)
            removed_paths.append(candidate)

    local_mod = sys.modules.get("mlflow")
    if local_mod is not None:
        mod_file = getattr(local_mod, "__file__", "")
        if mod_file and str(Path(mod_file).resolve()).startswith(current_dir):
            del sys.modules["mlflow"]

    external_mlflow = importlib.import_module("mlflow")

    for candidate in reversed(removed_paths):
        sys.path.insert(0, candidate)

    return external_mlflow


mlflow = _load_external_mlflow()
importlib.import_module("mlflow.sklearn")


DEFAULT_CONFIG_PATH = "src/mlflow/config_mlflow.yaml"


# ---------------------------------------------------------------------------
# Model Registry + Strategy Pattern
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Type["ModelStrategy"]] = {}


def register_model(name: str):
    """Decorator that auto-registers a ModelStrategy subclass by name."""

    def decorator(cls: Type["ModelStrategy"]) -> Type["ModelStrategy"]:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


class ModelStrategy(ABC):
    """Encapsulates a single model family: search space, fixed params, and
    how to instantiate the underlying sklearn-compatible estimator."""

    def __init__(self, model_config: Dict[str, Any], random_state: int) -> None:
        self.search_space: Dict[str, Any] = model_config.get("search_space", {})
        self.fixed_params: Dict[str, Any] = model_config.get("fixed_params", {})
        self.screening_fixed_params: Dict[str, Any] = model_config.get(
            "screening_fixed_params",
            self.fixed_params,
        )
        self.random_state = random_state

    @property
    def has_search_space(self) -> bool:
        return len(self.search_space) > 0

    def default_params(self) -> Dict[str, Any]:
        """Return fixed params + random_state for screening (no Optuna)."""
        return {**self.screening_fixed_params, "random_state": self.random_state}

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters from the config-driven search space."""
        params: Dict[str, Any] = {}
        for param_name, spec in self.search_space.items():
            param_type = spec.get("type")
            if param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name, int(spec["low"]), int(spec["high"])
                )
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    float(spec["low"]),
                    float(spec["high"]),
                    log=bool(spec.get("log", False)),
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, spec["choices"])
            else:
                raise ValueError(f"Unsupported param type '{param_type}' for {param_name}")

        params.update(self.fixed_params)
        params["random_state"] = self.random_state
        return params

    @abstractmethod
    def build_model(self, params: Dict[str, Any]) -> Any:
        """Instantiate the sklearn-compatible estimator with the given params."""
        ...


# -- HPO-eligible models (have search_space in YAML) ------------------------

@register_model("lightgbm")
class LightGBMStrategy(ModelStrategy):
    def build_model(self, params: Dict[str, Any]) -> LGBMRegressor:
        return LGBMRegressor(**params)


@register_model("random_forest")
class RandomForestStrategy(ModelStrategy):
    def build_model(self, params: Dict[str, Any]) -> RandomForestRegressor:
        return RandomForestRegressor(**params)


@register_model("extra_trees")
class ExtraTreesStrategy(ModelStrategy):
    def build_model(self, params: Dict[str, Any]) -> ExtraTreesRegressor:
        return ExtraTreesRegressor(**params)


# -- Screening-only models (default params, no HPO) -------------------------

@register_model("ridge")
class RidgeStrategy(ModelStrategy):
    def build_model(self, params: Dict[str, Any]) -> Ridge:
        return Ridge(**params)


@register_model("lasso")
class LassoStrategy(ModelStrategy):
    def build_model(self, params: Dict[str, Any]) -> Lasso:
        return Lasso(**params)


@register_model("elasticnet")
class ElasticNetStrategy(ModelStrategy):
    def build_model(self, params: Dict[str, Any]) -> ElasticNet:
        return ElasticNet(**params)


@register_model("gradient_boosting")
class GradientBoostingStrategy(ModelStrategy):
    def build_model(self, params: Dict[str, Any]) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(**params)


@register_model("hist_gradient_boosting")
class HistGradientBoostingStrategy(ModelStrategy):
    def build_model(self, params: Dict[str, Any]) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(**params)


@register_model("adaboost")
class AdaBoostStrategy(ModelStrategy):
    def build_model(self, params: Dict[str, Any]) -> AdaBoostRegressor:
        return AdaBoostRegressor(**params)


@register_model("decision_tree")
class DecisionTreeStrategy(ModelStrategy):
    def build_model(self, params: Dict[str, Any]) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(**params)


# ---------------------------------------------------------------------------
# Quality Gate
# ---------------------------------------------------------------------------


@dataclass
class QualityGateResult:
    """Outcome of running the champion model through metric thresholds."""

    passed: bool
    violations: List[str]
    metrics_checked: Dict[str, float]
    thresholds_used: Dict[str, float]


class QualityGate:
    """MLOps quality gate that validates model metrics against thresholds.

    Default thresholds (r2>=0, rmse/mae<=999999) are intentionally permissive
    so the gate is opt-in: tighten values in config_mlflow.yaml when ready.
    """

    def __init__(self, min_r2: float, max_rmse: float, max_mae: float) -> None:
        self.min_r2 = min_r2
        self.max_rmse = max_rmse
        self.max_mae = max_mae

    def evaluate(self, metrics: Dict[str, float]) -> QualityGateResult:
        violations: List[str] = []

        if metrics["r2"] < self.min_r2:
            violations.append(f"R2 {metrics['r2']:.4f} < min_r2 threshold {self.min_r2}")
        if metrics["rmse"] > self.max_rmse:
            violations.append(f"RMSE {metrics['rmse']:.4f} > max_rmse threshold {self.max_rmse}")
        if metrics["mae"] > self.max_mae:
            violations.append(f"MAE {metrics['mae']:.4f} > max_mae threshold {self.max_mae}")

        return QualityGateResult(
            passed=len(violations) == 0,
            violations=violations,
            metrics_checked=metrics,
            thresholds_used={
                "min_r2": self.min_r2,
                "max_rmse": self.max_rmse,
                "max_mae": self.max_mae,
            },
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class HyperparameterTuner:
    def __init__(
        self,
        config: Optional["PipelineConfig"] = None,
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> None:
        # Lazy import to avoid circular dependency at module level.
        from src.config.config import PipelineConfig

        if config is not None:
            self.config = config
        else:
            self.config = PipelineConfig.from_yaml(config_path)

        self.config_path = config_path

        # Typed access via Pydantic config
        self.tracking_uri: str = self.config.tracking.tracking_uri
        self.dagshub_username: str = self.config.tracking.username
        self.dagshub_repo_name: str = self.config.tracking.repo_name
        self.dagshub_token_env_key: str = self.config.tracking.token_env_key
        self.env_path: str = self.config.tracking.env_file_path

        self.train_path: str = self.config.data.train_path
        self.val_path: str = self.config.data.val_path
        self.test_path: str = self.config.data.test_path
        self.target_col: str = self.config.data.target_col
        self.trained_on_range: str = self.config.data.trained_on_range
        self.max_rows_train: int = self.config.data.max_rows_train
        self.max_rows_val: int = self.config.data.max_rows_val
        self.max_rows_test: int = self.config.data.max_rows_test
        self.sample_fraction: Optional[float] = self.config.data.sample_fraction

        self.n_trials_per_model: int = self.config.training.n_trials_per_model
        self.random_state: int = self.config.training.random_state
        self.champion_config_path = Path(self.config.training.champion_config_path)
        self.min_r2: float = self.config.training.metric_thresholds.min_r2
        self.max_rmse: float = self.config.training.metric_thresholds.max_rmse
        self.max_mae: float = self.config.training.metric_thresholds.max_mae

        self.top_n_for_hpo: int = self.config.screening.top_n_for_hpo
        self.screening_row_cap_train: int = self.config.screening.row_cap_train
        self.screening_row_cap_val: int = self.config.screening.row_cap_val

        hpo_model_configs = [m.model_dump() for m in self.config.enabled_hpo_models]
        screening_model_configs = [m.model_dump() for m in self.config.enabled_screening_models]

        if not hpo_model_configs:
            raise ValueError(f"No enabled HPO models found in {config_path}")
        if not screening_model_configs:
            raise ValueError(
                f"No enabled screening models found in {config_path}. "
                "Add models under 'screening_models'."
            )

        self.hpo_strategies: Dict[str, ModelStrategy] = self._build_strategies(hpo_model_configs)
        self.screening_strategies: Dict[str, ModelStrategy] = self._build_strategies(
            screening_model_configs
        )

        self.pipeline_timestamp = datetime.now(timezone.utc).strftime("%H%M%d%m%Y")
        self.experiment_name = f"run_{self.pipeline_timestamp}"
        self.pipeline_run_id = self.experiment_name
        self.screen_run_name = "screening"
        self.hpo_run_name = "HPO"
        self.spark = SparkUtils(app_name="nyc-taxi-mlflow-training").spark
        self._setup_mlflow()

    def _build_strategies(self, models_config: List[Dict[str, Any]]) -> Dict[str, ModelStrategy]:
        strategies: Dict[str, ModelStrategy] = {}
        for model_cfg in models_config:
            name = model_cfg["name"]
            if name not in MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown model '{name}'. "
                    f"Registered models: {list(MODEL_REGISTRY.keys())}"
                )
            strategies[name] = MODEL_REGISTRY[name](model_cfg, self.random_state)
        return strategies

    # -- MLflow setup --------------------------------------------------------

    def _load_env_token(self) -> str:
        token = os.getenv(self.dagshub_token_env_key)
        if token:
            return token.strip()

        env_path = Path(self.env_path)
        if not env_path.exists():
            raise ValueError(
                f"Missing token and {env_path} not found. "
                f"Set {self.dagshub_token_env_key} in env or .env."
            )

        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == self.dagshub_token_env_key:
                token = value.strip().strip('"').strip("'")
                if token:
                    return token
                break

        raise ValueError(
            f"{self.dagshub_token_env_key} is missing/empty in {env_path}."
        )

    def _setup_mlflow(self) -> None:
        token = self._load_env_token()
        os.environ["MLFLOW_TRACKING_USERNAME"] = self.dagshub_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(self.tracking_uri)

    def _safe_mlflow_call(
        self,
        func: Any,
        *args: Any,
        retries: int = 3,
        backoff_sec: float = 0.75,
        **kwargs: Any,
    ) -> Any:
        """
        Retry transient MLflow logging calls and avoid failing the full pipeline
        on non-critical tracking hiccups.
        """
        attempt = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt > retries:
                    warnings.warn(
                        (
                            f"MLflow call failed after {retries} retries: "
                            f"{getattr(func, '__name__', str(func))} -> {exc}"
                        ),
                        UserWarning,
                        stacklevel=2,
                    )
                    return None
                sleep_for = backoff_sec * attempt
                print(
                    "[mlflow-retry] "
                    f"fn={getattr(func, '__name__', str(func))} attempt={attempt}/{retries} "
                    f"sleep={sleep_for:.2f}s error={exc}"
                )
                time.sleep(sleep_for)

    def _close_parent_run(self, run_id: str, status: str) -> None:
        active = mlflow.active_run()
        if active is None or active.info.run_id != run_id:
            return
        try:
            mlflow.end_run(status=status)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Failed to close MLflow parent run {run_id}: {exc}",
                UserWarning,
                stacklevel=2,
            )

    # -- Data ----------------------------------------------------------------

    def _load_split_with_spark(self, path: str, split_name: str, max_rows: int) -> pd.DataFrame:
        """
        Read parquet with Spark, reduce rows/columns, then convert to pandas.
        """
        sdf = self.spark.read.parquet(path)

        if self.target_col not in sdf.columns:
            raise ValueError(
                f"Target column '{self.target_col}' is missing in {split_name} dataset: {path}"
            )

        feature_cols = [c for c in sdf.columns if c != self.target_col]
        selected_cols = feature_cols + [self.target_col]
        sdf = sdf.select(*selected_cols)

        if self.sample_fraction is not None and 0 < self.sample_fraction < 1:
            sdf = sdf.sample(withReplacement=False, fraction=self.sample_fraction, seed=self.random_state)

        if max_rows > 0:
            sdf = sdf.limit(max_rows)

        pdf = sdf.toPandas()
        print(
            f"[load_data] {split_name}: loaded {len(pdf):,} rows, "
            f"{len(pdf.columns):,} columns (Spark -> pandas)"
        )
        return pdf

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = self._load_split_with_spark(
            self.train_path, split_name="train", max_rows=self.max_rows_train
        )
        val_df = self._load_split_with_spark(
            self.val_path, split_name="val", max_rows=self.max_rows_val
        )
        test_df = self._load_split_with_spark(
            self.test_path, split_name="test", max_rows=self.max_rows_test
        )
        return train_df, val_df, test_df

    def prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' is missing.")
        x = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        return x, y

    # -- Metrics & hashing ---------------------------------------------------

    def _metric_dict(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"rmse": rmse, "mae": mae, "r2": r2}

    def _schema_hash(self, df: pd.DataFrame) -> str:
        schema_payload = [
            {"name": col, "dtype": str(df[col].dtype)}
            for col in df.columns
        ]
        raw = json.dumps(schema_payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    # -- Stage 1: Screening --------------------------------------------------

    def screen_models(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> List[Dict[str, Any]]:
        """Stage 1: Train each model once with default params, rank by val RMSE."""
        results: List[Dict[str, Any]] = []

        for name, strategy in self.screening_strategies.items():
            print(f"[screening] starting model={name}")
            start_ts = time.time()
            params = strategy.default_params()
            model = strategy.build_model(params)
            model.fit(x_train, y_train)
            metrics = self._metric_dict(y_val, model.predict(x_val))
            elapsed = time.time() - start_ts
            print(
                f"[screening] finished model={name} "
                f"rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} r2={metrics['r2']:.4f} "
                f"elapsed_sec={elapsed:.2f}"
            )

            with mlflow.start_run(run_name=f"{name}", nested=True):
                self._safe_mlflow_call(mlflow.log_params, params)
                self._safe_mlflow_call(
                    mlflow.log_metrics,
                    {f"val_{k}": v for k, v in metrics.items()},
                )
                self._safe_mlflow_call(mlflow.set_tag, "model_family", name)
                self._safe_mlflow_call(mlflow.set_tag, "stage", "screening")
                self._safe_mlflow_call(
                    mlflow.set_tag, "hpo_eligible", str(name in self.hpo_strategies)
                )

            results.append({
                "model_family": name,
                "val_metrics": metrics,
                "hpo_eligible": name in self.hpo_strategies,
            })

        results.sort(key=lambda r: r["val_metrics"]["rmse"])
        return results

    # -- Stage 2: HPO --------------------------------------------------------

    def tune_one_family(
        self,
        family: str,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        strategy = self.hpo_strategies[family]
        print(f"[hpo] starting family={family} trials={self.n_trials_per_model}")

        def objective(trial: optuna.Trial) -> float:
            params = strategy.suggest_params(trial)
            model = strategy.build_model(params)
            model.fit(x_train, y_train)
            metrics = self._metric_dict(y_val, model.predict(x_val))
            trial.set_user_attr("metrics", metrics)

            with mlflow.start_run(run_name=f"{family}_trial_{trial.number}", nested=True):
                self._safe_mlflow_call(mlflow.log_params, params)
                self._safe_mlflow_call(
                    mlflow.log_metrics,
                    {f"val_{k}": v for k, v in metrics.items()},
                )
                self._safe_mlflow_call(mlflow.set_tag, "model_family", family)
                self._safe_mlflow_call(mlflow.set_tag, "trial_number", str(trial.number))
                self._safe_mlflow_call(mlflow.set_tag, "stage", "hpo_trial")

            return metrics["rmse"]

        study = optuna.create_study(direction="minimize", study_name=f"{family}_study")
        study.optimize(objective, n_trials=self.n_trials_per_model, show_progress_bar=False)

        best_params = {**study.best_trial.params}
        best_params.update(strategy.fixed_params)
        best_params["random_state"] = self.random_state

        best_model = strategy.build_model(best_params)
        best_model.fit(x_train, y_train)
        best_metrics = self._metric_dict(y_val, best_model.predict(x_val))

        with mlflow.start_run(run_name=f"{family}_best", nested=True):
            self._safe_mlflow_call(mlflow.set_tag, "model_family", family)
            self._safe_mlflow_call(mlflow.set_tag, "stage", "hpo_family_best")
            self._safe_mlflow_call(mlflow.log_param, "n_trials", self.n_trials_per_model)
            self._safe_mlflow_call(mlflow.log_params, best_params)
            self._safe_mlflow_call(
                mlflow.log_metrics,
                {
                    "val_rmse": best_metrics["rmse"],
                    "val_mae": best_metrics["mae"],
                    "val_r2": best_metrics["r2"],
                },
            )

        print(
            f"[hpo] finished family={family} best_rmse={best_metrics['rmse']:.4f} "
            f"best_mae={best_metrics['mae']:.4f} best_r2={best_metrics['r2']:.4f}"
        )

        return {
            "model_family": family,
            "study_best_value_rmse": float(study.best_value),
            "params": best_params,
            "val_metrics": best_metrics,
            "model": best_model,
            "n_trials": self.n_trials_per_model,
        }

    # -- Filesystem ----------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self.champion_config_path.parent.mkdir(parents=True, exist_ok=True)

    def _build_experiment_dir(self) -> Path:
        experiment_dir = self.champion_config_path.parent / "experiments" / self.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir

    # -- Main pipeline -------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        self._ensure_dirs()
        print("[pipeline] loading data...")
        train_df, val_df, test_df = self.load_data()

        x_train, y_train = self.prepare_xy(train_df)
        x_val, y_val = self.prepare_xy(val_df)
        x_test, y_test = self.prepare_xy(test_df)

        # Use capped data for screening to keep stage 1 fast.
        x_train_screen = x_train.head(self.screening_row_cap_train)
        y_train_screen = y_train.head(self.screening_row_cap_train)
        x_val_screen = x_val.head(self.screening_row_cap_val)
        y_val_screen = y_val.head(self.screening_row_cap_val)
        print(
            "[pipeline] screening row caps -> "
            f"train={len(x_train_screen):,}, val={len(x_val_screen):,}"
        )
        print(
            "[pipeline] full hpo data -> "
            f"train={len(x_train):,}, val={len(x_val):,}, test={len(x_test):,}"
        )

        schema_hash = self._schema_hash(x_train)
        experiment_dir = self._build_experiment_dir()
        mlflow.set_experiment(self.experiment_name)

        print(f"[pipeline] experiment={self.experiment_name}")
        screening_run_info = None
        hpo_candidates: List[str] = []

        # ---- Stage 1: Screening run ----
        print(f"[pipeline] stage=screening run={self.screen_run_name}")
        screening_parent = mlflow.start_run(run_name=self.screen_run_name)
        screening_status = "FAILED"
        try:
            screening_run_info = screening_parent.info
            self._safe_mlflow_call(mlflow.set_tag, "project", "NYC-YELLOW-TAXI-MLOPS")
            self._safe_mlflow_call(mlflow.set_tag, "repo", self.dagshub_repo_name)
            self._safe_mlflow_call(mlflow.set_tag, "pipeline_stage", "screening")
            self._safe_mlflow_call(mlflow.set_tag, "pipeline_run_id", self.pipeline_run_id)
            self._safe_mlflow_call(
                mlflow.log_params,
                {
                    "target_col": self.target_col,
                    "trained_on_range": self.trained_on_range,
                    "top_n_for_hpo": self.top_n_for_hpo,
                },
            )

            screening_results = self.screen_models(
                x_train_screen,
                y_train_screen,
                x_val_screen,
                y_val_screen,
            )

            for result in screening_results:
                if result["model_family"] in self.hpo_strategies and len(hpo_candidates) < self.top_n_for_hpo:
                    hpo_candidates.append(result["model_family"])

            if not hpo_candidates:
                raise ValueError(
                    "No HPO-eligible models survived screening. "
                    "Ensure models_to_train entries have search_space defined."
                )

            screening_metadata_path = experiment_dir / "metadata_screen.json"
            screening_metadata = {
                "stage": "screening",
                "pipeline_run_id": self.pipeline_run_id,
                "run_id": screening_run_info.run_id,
                "experiment_name": self.experiment_name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "data_rows": {
                    "screen_train_rows": len(x_train_screen),
                    "screen_val_rows": len(x_val_screen),
                    "full_train_rows": len(x_train),
                    "full_val_rows": len(x_val),
                    "full_test_rows": len(x_test),
                },
                "metrics": {
                    "screening_results": screening_results,
                    "hpo_candidates": hpo_candidates,
                },
                "artifacts": {
                    "metadata_path": str(screening_metadata_path),
                },
            }
            with screening_metadata_path.open("w", encoding="utf-8") as f:
                json.dump(screening_metadata, f, indent=2, sort_keys=False)
                f.write("\n")
            self._safe_mlflow_call(mlflow.log_artifact, str(screening_metadata_path))
            screening_status = "FINISHED"
        finally:
            self._close_parent_run(screening_parent.info.run_id, status=screening_status)

        # ---- Stage 2: HPO run ----
        print(f"[pipeline] stage=hpo run={self.hpo_run_name}")
        hpo_parent = mlflow.start_run(run_name=self.hpo_run_name)
        hpo_status = "FAILED"
        try:
            hpo_run_info = hpo_parent.info
            self._safe_mlflow_call(mlflow.set_tag, "project", "NYC-YELLOW-TAXI-MLOPS")
            self._safe_mlflow_call(mlflow.set_tag, "repo", self.dagshub_repo_name)
            self._safe_mlflow_call(mlflow.set_tag, "pipeline_stage", "hpo")
            self._safe_mlflow_call(mlflow.set_tag, "pipeline_run_id", self.pipeline_run_id)
            self._safe_mlflow_call(
                mlflow.log_params,
                {
                    "target_col": self.target_col,
                    "n_trials_per_model": self.n_trials_per_model,
                    "trained_on_range": self.trained_on_range,
                    "top_n_for_hpo": self.top_n_for_hpo,
                    "hpo_candidates": ", ".join(hpo_candidates),
                },
            )

            family_results: List[Dict[str, Any]] = []
            for family in hpo_candidates:
                result = self.tune_one_family(family, x_train, y_train, x_val, y_val)
                family_results.append(result)

            family_results.sort(key=lambda r: r["val_metrics"]["rmse"])
            champion = family_results[0]
            champion_model = champion["model"]
            print(f"[pipeline] champion={champion['model_family']}")

            test_pred = champion_model.predict(x_test)
            test_metrics = self._metric_dict(y_test, test_pred)

            gate = QualityGate(self.min_r2, self.max_rmse, self.max_mae)
            gate_result = gate.evaluate(test_metrics)

            self._safe_mlflow_call(mlflow.set_tag, "quality_gate_passed", str(gate_result.passed))
            if not gate_result.passed:
                self._safe_mlflow_call(
                    mlflow.set_tag,
                    "quality_gate_violations",
                    "; ".join(gate_result.violations),
                )

            created_at = datetime.now(timezone.utc).isoformat()

            champion_config: Dict[str, Any] = {
                "model_family": champion["model_family"],
                "params": champion["params"],
                "metrics": {
                    "validation": champion["val_metrics"],
                    "test": test_metrics,
                },
                "data": {
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                    "target_col": self.target_col,
                    "trained_on_range": self.trained_on_range,
                    "data_schema_hash": schema_hash,
                },
                "quality_gate": {
                    "passed": gate_result.passed,
                    "violations": gate_result.violations,
                    "thresholds": {
                        "min_r2": self.min_r2,
                        "max_rmse": self.max_rmse,
                        "max_mae": self.max_mae,
                    },
                },
                "mlflow": {
                    "pipeline_run_id": self.pipeline_run_id,
                    "experiment_name": self.experiment_name,
                    "screening_run_id": screening_run_info.run_id if screening_run_info else "",
                    "hpo_run_id": hpo_run_info.run_id,
                },
                "created_at_utc": created_at,
            }

            with self.champion_config_path.open("w", encoding="utf-8") as f:
                json.dump(champion_config, f, indent=2, sort_keys=False)
                f.write("\n")

            run_scoped_champion_path = experiment_dir / "champion.json"
            with run_scoped_champion_path.open("w", encoding="utf-8") as f:
                json.dump(champion_config, f, indent=2, sort_keys=False)
                f.write("\n")

            hpo_metadata_path = experiment_dir / "metadata_hpo.json"
            hpo_metadata = {
                "stage": "hpo",
                "pipeline_run_id": self.pipeline_run_id,
                "run_id": hpo_run_info.run_id,
                "experiment_name": self.experiment_name,
                "created_at_utc": created_at,
                "data_rows": {
                    "train_rows": len(x_train),
                    "val_rows": len(x_val),
                    "test_rows": len(x_test),
                },
                "metrics": {
                    "n_trials_per_model": self.n_trials_per_model,
                    "hpo_candidates": hpo_candidates,
                    "leaderboard": [
                        {
                            "model_family": item["model_family"],
                            "params": item["params"],
                            "val_metrics": item["val_metrics"],
                        }
                        for item in family_results
                    ],
                    "champion": {
                        "model_family": champion["model_family"],
                        "params": champion["params"],
                        "validation_metrics": champion["val_metrics"],
                        "test_metrics": test_metrics,
                        "quality_gate_passed": gate_result.passed,
                    },
                },
                "artifacts": {
                    "metadata_path": str(hpo_metadata_path),
                    "champion_path": str(self.champion_config_path),
                    "run_scoped_champion_path": str(run_scoped_champion_path),
                },
            }
            with hpo_metadata_path.open("w", encoding="utf-8") as f:
                json.dump(hpo_metadata, f, indent=2, sort_keys=False)
                f.write("\n")

            self._safe_mlflow_call(
                mlflow.log_metrics,
                {
                    "champion_val_rmse": champion["val_metrics"]["rmse"],
                    "champion_val_mae": champion["val_metrics"]["mae"],
                    "champion_val_r2": champion["val_metrics"]["r2"],
                    "champion_test_rmse": test_metrics["rmse"],
                    "champion_test_mae": test_metrics["mae"],
                    "champion_test_r2": test_metrics["r2"],
                },
            )
            self._safe_mlflow_call(
                mlflow.log_params,
                {
                    "champion_model_family": champion["model_family"],
                    "data_schema_hash": schema_hash,
                },
            )
            self._safe_mlflow_call(mlflow.log_artifact, str(self.champion_config_path))
            self._safe_mlflow_call(mlflow.log_artifact, str(run_scoped_champion_path))
            self._safe_mlflow_call(mlflow.log_artifact, str(hpo_metadata_path))

            if not gate_result.passed:
                warnings.warn(
                    f"QUALITY GATE FAILED: Champion '{champion['model_family']}' "
                    f"did not meet metric thresholds.\nViolations:\n"
                    + "\n".join(f"  - {v}" for v in gate_result.violations),
                    UserWarning,
                    stacklevel=2,
                )

            hpo_status = "FINISHED"
            return champion_config
        except Exception:  # noqa: BLE001
            print("[pipeline] HPO stage failed:\n" + traceback.format_exc())
            raise
        finally:
            self._close_parent_run(hpo_parent.info.run_id, status=hpo_status)

    def close(self) -> None:
        if self.spark:
            self.spark.stop()


def main() -> None:
    from src.config.config import PipelineConfig

    config = PipelineConfig.from_yaml(DEFAULT_CONFIG_PATH)
    tuner = HyperparameterTuner(config=config)
    try:
        result = tuner.run()
        print(json.dumps(result, indent=2))
    finally:
        tuner.close()


if __name__ == "__main__":
    main()
