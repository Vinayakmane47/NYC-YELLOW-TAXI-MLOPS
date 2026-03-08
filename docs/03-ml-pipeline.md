# ML Pipeline

## Overview

The ML pipeline handles model training, evaluation, registry management, and offline hyperparameter optimization. All experiments are tracked in MLflow hosted on DagShub.

## Hyperparameter Optimization (Offline)

**Files:** `src/hpo/mlflow.py`, `src/hpo/config_mlflow.yaml`

HPO runs offline (not in the Airflow DAG) and produces `champion.json`, which the training pipeline uses.

### Two-Stage Process

**Stage 1 - Screening:**
- Tests all 10 model families with default parameters
- Uses a small data sample (20K train, 10K val) for speed
- Selects the top 2 models by RMSE

**Stage 2 - Optuna HPO:**
- Runs 50 Optuna trials per selected model
- 3-fold cross-validation per trial
- Full hyperparameter search spaces (learning rate, depth, estimators, etc.)
- Selects the overall champion by validation RMSE

### Supported Models (10 families)

| Family | Type |
|--------|------|
| LightGBM | Gradient boosting |
| RandomForest | Ensemble |
| ExtraTrees | Ensemble |
| GradientBoosting | Gradient boosting |
| HistGradientBoosting | Gradient boosting |
| AdaBoost | Ensemble |
| Ridge | Linear |
| Lasso | Linear |
| ElasticNet | Linear |
| DecisionTree | Tree |

### Output: champion.json

```json
{
  "model_family": "random_forest",
  "params": {
    "n_estimators": 865,
    "max_depth": 25,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt"
  },
  "metrics": {
    "val": { "rmse": 4.90, "r2": 0.774, "mae": 3.12 },
    "test": { "rmse": 5.98, "r2": 0.768, "mae": 3.88 }
  },
  "quality_gate": { "passed": true, "violations": [] }
}
```

### Running HPO

```bash
python src/hpo/mlflow.py
```

Requires `DAGSHUB_TOKEN` set in environment or `.env` file.

## Model Training

**File:** `src/model_training.py`

Trains the champion model on the full training set using parameters from `champion.json`.

### Process

1. Reads `src/hpo/champion.json` for model family and hyperparameters
2. Loads train/val/test splits from MinIO via Spark
3. Converts to pandas (capped at 2M rows)
4. Builds model using `src/models/model_factory.py`
5. Fits on training data
6. Evaluates on validation set
7. Saves model as `.joblib` artifact
8. Logs to MLflow (metrics, params, model artifact, feature importances)

### MLflow Logging

- **Experiment:** `nyc-taxi-pipeline`
- **Run structure:** Nested under a shared parent run per pipeline execution
- **Logged items:**
  - Metrics: RMSE, MAE, R2, MAPE
  - Parameters: All model hyperparameters
  - Artifacts: `.joblib` model file, feature importances plot
  - Tags: model_family, pipeline_run_id, data range

### Model Artifact

The model is saved using MLflow's custom pyfunc wrapper (`src/utils/mlflow_pyfunc.py`):

```python
class SklearnJoblibPyfuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model_file"])

    def predict(self, context, model_input):
        return self.model.predict(model_input)
```

This wraps a compressed joblib file as an MLflow model, enabling registry and versioning.

## Model Evaluation

**File:** `src/model_evaluation.py`

Compares the newly trained model against the current Production model.

### Evaluation Logic

1. Load new model's metrics from training metadata
2. Pull current Production model from MLflow registry
3. Evaluate both on the test set
4. Apply quality gate thresholds
5. Compare metrics and decide on promotion

### Quality Gate

Configurable thresholds in `src/config/settings.yaml`:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| R2 | >= 0.3 | Minimum explained variance |
| RMSE | <= 30.0 | Maximum root mean squared error |
| MAE | <= 20.0 | Maximum mean absolute error |

### Promotion Decision Matrix

| Condition | Decision |
|-----------|----------|
| Quality gate passes + beats Production | Promote to Production |
| Quality gate passes + no Production model | Register as first Production |
| Quality gate fails | Archive (do not promote) |
| Worse than Production | Archive |

### Comparison Thresholds

```yaml
evaluation_policy:
  min_rmse_improvement: 0.0    # New must be at least this much better
  max_mae_regression: 0.0      # New can be at most this much worse
  max_r2_regression: 0.0       # New can be at most this much worse
```

## Model Registry

**File:** `src/model_registry.py`

Manages model versioning and stage transitions in MLflow Model Registry.

### Registry Name

`nyc-taxi-trip-duration` (single registered model, multiple versions)

### Stage Transitions

```
(new model) -> Staging -> Production
                              |
                    (old Production) -> Archived
```

### Process

1. Read evaluation decision from metadata
2. If `should_register == True`:
   - Register model version in MLflow
   - Transition to Staging
   - If quality gate passed: transition to Production
   - Archive previous Production version
3. Log all transitions to MLflow tags

## Experiment Tracking (MLflow on DagShub)

### Authentication

All MLflow operations authenticate via `DAGSHUB_TOKEN`. The shared auth utility (`src/utils/mlflow_auth.py`) handles:
- Reading token from environment variable or `.env` file
- Setting `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`
- Configuring the tracking URI

### Tracking URI

```
https://dagshub.com/Vinayakmane47/NYC-YELLOW-TAXI-MLOPS.mlflow/
```

### Run Structure

```
Experiment: nyc-taxi-pipeline
  |
  +-- Parent Run: pipeline_{RUN_ID}
       |
       +-- Child Run: model_training
       +-- Child Run: model_evaluation
       +-- Child Run: model_registry
```

## Configuration

ML pipeline settings in `src/config/settings.yaml`:

```yaml
tracking:
  tracking_uri: https://dagshub.com/Vinayakmane47/NYC-YELLOW-TAXI-MLOPS.mlflow/
  username: Vinayakmane47
  repo_name: NYC-YELLOW-TAXI-MLOPS
  token_env_key: DAGSHUB_TOKEN
  env_file_path: .env

metric_thresholds:
  min_r2: 0.3
  max_rmse: 30.0
  max_mae: 20.0

data:
  target_col: trip_duration_min
  max_rows_train: 2000000
```
