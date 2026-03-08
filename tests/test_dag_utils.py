"""Tests for airflow.dags.dag_utils."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add dags directory to path so we can import dag_utils directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "airflow" / "dags"))

from dag_utils import set_pipeline_run_id


class TestSetPipelineRunId:
    def test_uses_ds_nodash_when_no_conf(self):
        context = {
            "ds_nodash": "20260301",
            "dag_run": None,
        }
        result = set_pipeline_run_id(**context)
        assert result == "20260301"
        assert os.environ["PIPELINE_RUN_ID"] == "20260301"

    def test_uses_conf_pipeline_run_id(self):
        dag_run = MagicMock()
        dag_run.conf = {"pipeline_run_id": "custom_run_123"}
        context = {
            "ds_nodash": "20260301",
            "dag_run": dag_run,
        }
        result = set_pipeline_run_id(**context)
        assert result == "custom_run_123"
        assert os.environ["PIPELINE_RUN_ID"] == "custom_run_123"

    def test_falls_back_to_ds_nodash_when_conf_empty(self):
        dag_run = MagicMock()
        dag_run.conf = {}
        context = {
            "ds_nodash": "20260301",
            "dag_run": dag_run,
        }
        result = set_pipeline_run_id(**context)
        assert result == "20260301"

    def test_falls_back_when_conf_missing_key(self):
        dag_run = MagicMock()
        dag_run.conf = {"other_key": "value"}
        context = {
            "ds_nodash": "20260301",
            "dag_run": dag_run,
        }
        result = set_pipeline_run_id(**context)
        assert result == "20260301"
