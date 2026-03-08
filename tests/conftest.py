"""Shared test fixtures."""

import json

import pytest
import yaml


@pytest.fixture
def sample_settings_yaml(tmp_path):
    """Write a minimal settings.yaml for testing."""
    settings = {
        "tracking": {
            "tracking_uri": "https://example.com/mlflow",
            "username": "testuser",
            "repo_name": "test-repo",
        },
        "minio": {
            "endpoint": "http://localhost:9000",
            "access_key": "minioadmin",
            "secret_key": "minioadmin",
            "buckets": {
                "bronze": "bronze",
                "silver": "silver",
                "gold": "gold",
                "ml_transformed": "ml-transformed",
            },
        },
        "data": {
            "train_path": "s3a://ml-transformed/train.parquet",
            "val_path": "s3a://ml-transformed/val.parquet",
            "test_path": "s3a://ml-transformed/test.parquet",
            "target_col": "trip_duration_min",
        },
        "training": {
            "random_state": 42,
            "models_dir": "artifacts/models",
            "champion_config_path": "src/hpo/champion.json",
        },
    }
    path = tmp_path / "settings.yaml"
    with open(path, "w") as f:
        yaml.dump(settings, f)
    return str(path)


@pytest.fixture
def sample_champion_json(tmp_path):
    """Write a minimal champion.json for testing."""
    champion = {
        "model_family": "lightgbm",
        "params": {"n_estimators": 100, "learning_rate": 0.1},
        "metrics": {
            "validation": {"rmse": 5.0, "mae": 3.5, "r2": 0.85},
        },
        "created_at_utc": "2025-01-01T00:00:00Z",
    }
    path = tmp_path / "champion.json"
    with open(path, "w") as f:
        json.dump(champion, f)
    return str(path)
