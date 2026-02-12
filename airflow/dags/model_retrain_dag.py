"""
DAG B: NYC Model Retrain Pipeline

Triggered conditionally by DAG A (nyc_data_refresh_dag) when data drift
is detected. Retrains the champion model, evaluates against the current
Production model, and registers/promotes if improved.

  Training -> Evaluation -> Registry

No schedule - only runs when triggered via TriggerDagRunOperator.
The PIPELINE_RUN_ID is passed via DAG conf from DAG A so all metadata
lands in the same pipeline folder.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add project root to Python path for imports
sys.path.insert(0, '/opt/airflow/src')
sys.path.insert(0, '/opt/airflow')

from src.model_training import main as model_training_main
from src.model_evaluation import main as model_evaluation_main
from src.model_registry import main as model_registry_main


def _set_pipeline_run_id(**context):
    """Read PIPELINE_RUN_ID from DAG conf (passed by DAG A) or fallback to ds_nodash."""
    run_id = context['dag_run'].conf.get('pipeline_run_id', context['ds_nodash'])
    os.environ['PIPELINE_RUN_ID'] = run_id
    print(f"PIPELINE_RUN_ID set to {run_id}")


def run_model_training(**context):
    """Train champion model from champion.json."""
    _set_pipeline_run_id(**context)
    model_training_main()


def run_model_evaluation(**context):
    """Evaluate new model against registered Production model."""
    _set_pipeline_run_id(**context)
    model_evaluation_main()


def run_model_registry(**context):
    """Register and promote model in MLflow Model Registry."""
    _set_pipeline_run_id(**context)
    model_registry_main()


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    'owner': 'ml-engineer',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=3),
}

dag = DAG(
    'nyc_model_retrain_dag',
    default_args=default_args,
    description='Conditional model retrain: training + evaluation + registry (triggered by drift)',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['nyc-taxi', 'ml', 'retrain', 'production'],
)

# -- ML stages -------------------------------------------------------------

training_task = PythonOperator(
    task_id='model_training',
    python_callable=run_model_training,
    dag=dag,
)

evaluation_task = PythonOperator(
    task_id='model_evaluation',
    python_callable=run_model_evaluation,
    dag=dag,
)

registry_task = PythonOperator(
    task_id='model_registry',
    python_callable=run_model_registry,
    dag=dag,
)

# -- Task dependencies -----------------------------------------------------

training_task >> evaluation_task >> registry_task
