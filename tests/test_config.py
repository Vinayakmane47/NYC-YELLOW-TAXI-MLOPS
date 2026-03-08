"""Tests for src.config.config."""

import pytest

from src.config.config import ChampionConfig, PipelineConfig


class TestPipelineConfig:
    def test_loads_from_yaml(self, sample_settings_yaml):
        config = PipelineConfig.from_yaml(sample_settings_yaml)
        assert config.tracking.username == "testuser"
        assert config.data.target_col == "trip_duration_min"
        assert config.training.random_state == 42

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_yaml("/nonexistent/settings.yaml")

    def test_defaults_applied(self, sample_settings_yaml):
        config = PipelineConfig.from_yaml(sample_settings_yaml)
        assert config.drift.max_share_drifted == 0.3
        assert config.evaluation.min_rmse_improvement == 0.0
        assert config.metadata.base_dir == "src/metadata"

    def test_minio_config(self, sample_settings_yaml):
        config = PipelineConfig.from_yaml(sample_settings_yaml)
        assert config.minio.endpoint == "http://localhost:9000"
        assert config.minio.access_key == "minioadmin"
        assert config.minio.buckets.bronze == "bronze"
        assert config.minio.buckets.ml_transformed == "ml-transformed"


class TestChampionConfig:
    def test_loads_from_json(self, sample_champion_json):
        champion = ChampionConfig.load(sample_champion_json)
        assert champion.model_family == "lightgbm"
        assert champion.params["n_estimators"] == 100
        assert champion.metrics.validation.rmse == 5.0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ChampionConfig.load("/nonexistent/champion.json")

    def test_save_roundtrip(self, sample_champion_json, tmp_path):
        champion = ChampionConfig.load(sample_champion_json)
        save_path = str(tmp_path / "saved_champion.json")
        champion.save(save_path)
        reloaded = ChampionConfig.load(save_path)
        assert reloaded.model_family == champion.model_family
        assert reloaded.params == champion.params
