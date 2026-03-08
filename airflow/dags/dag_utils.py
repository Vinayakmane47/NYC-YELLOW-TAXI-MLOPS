"""Shared utilities for Airflow DAGs."""

import os


def set_pipeline_run_id(**context) -> str:
    """Inject a shared PIPELINE_RUN_ID from the Airflow execution date.

    For DAGs triggered via TriggerDagRunOperator, prefers the pipeline_run_id
    from dag_run.conf if available.
    """
    conf = context.get("dag_run")
    run_id = conf.conf.get("pipeline_run_id", context["ds_nodash"]) if conf and conf.conf else context["ds_nodash"]
    os.environ["PIPELINE_RUN_ID"] = run_id
    print(f"PIPELINE_RUN_ID set to {run_id}")
    return run_id
