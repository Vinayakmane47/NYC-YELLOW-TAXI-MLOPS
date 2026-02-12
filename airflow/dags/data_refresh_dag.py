"""
DAG A: NYC Data Refresh Pipeline

Runs monthly to keep the curated dataset fresh:
  Ingestion -> Validation -> Preprocessing -> Transformation -> ML Transformation -> Retrain Decider

The retrain_decider task runs Evidently data drift detection. If significant
drift is found, it triggers DAG B (nyc_model_retrain_dag) via TriggerDagRunOperator.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# Add project root to Python path for imports
sys.path.insert(0, '/opt/airflow/src')
sys.path.insert(0, '/opt/airflow')

from src.data_ingestion import main as data_ingestion_main
from src.data_validation import main as data_validation_main
from src.data_preprocessing import main as data_preprocessing_main
from src.data_transformation import main as data_transformation_main
from src.ml_transformed import main as ml_transformation_main
from src.drift_detection import main as drift_detection_main


def _set_pipeline_run_id(**context):
    """Inject a shared PIPELINE_RUN_ID from the Airflow execution date."""
    run_id = context["ds_nodash"]
    os.environ["PIPELINE_RUN_ID"] = run_id
    print(f"PIPELINE_RUN_ID set to {run_id}")


def run_data_ingestion(**context):
    """Download raw NYC taxi data (Bronze layer)."""
    _set_pipeline_run_id(**context)
    data_ingestion_main()


def run_data_validation(**context):
    """Validate ingested data schema."""
    _set_pipeline_run_id(**context)
    data_validation_main()


def run_data_preprocessing(**context):
    """Clean and filter data (Silver layer)."""
    _set_pipeline_run_id(**context)
    data_preprocessing_main()


def run_data_transformation(**context):
    """Feature engineering (Gold layer)."""
    _set_pipeline_run_id(**context)
    data_transformation_main()


def run_ml_transformation(**context):
    """Prepare ML-ready train/val/test splits."""
    _set_pipeline_run_id(**context)
    ml_transformation_main()


def run_retrain_decider(**context):
    """Run drift detection and push retrain decision to XCom."""
    _set_pipeline_run_id(**context)
    result = drift_detection_main()
    should_retrain = not result.drift_gate_passed
    context['ti'].xcom_push(key='should_retrain', value=should_retrain)
    context['ti'].xcom_push(key='pipeline_run_id', value=os.environ['PIPELINE_RUN_ID'])
    print(f"[retrain_decider] drift_gate_passed={result.drift_gate_passed}, should_retrain={should_retrain}")


def check_should_retrain(**context):
    """ShortCircuit: only proceed if retrain_decider says yes."""
    return context['ti'].xcom_pull(key='should_retrain', task_ids='retrain_decider')


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    'owner': 'data-engineer',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

dag = DAG(
    'nyc_data_refresh_dag',
    default_args=default_args,
    description='Monthly data refresh: ETL + drift detection + conditional retrain trigger',
    schedule_interval='@monthly',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['nyc-taxi', 'etl', 'data-refresh', 'drift', 'production'],
)

# -- Data stages -----------------------------------------------------------

ingestion_task = PythonOperator(
    task_id='data_ingestion',
    python_callable=run_data_ingestion,
    dag=dag,
)

validation_task = PythonOperator(
    task_id='data_validation',
    python_callable=run_data_validation,
    dag=dag,
)

preprocessing_task = PythonOperator(
    task_id='data_preprocessing',
    python_callable=run_data_preprocessing,
    dag=dag,
)

transformation_task = PythonOperator(
    task_id='data_transformation',
    python_callable=run_data_transformation,
    dag=dag,
)

ml_transformation_task = PythonOperator(
    task_id='ml_transformation',
    python_callable=run_ml_transformation,
    dag=dag,
)

# -- Drift detection + conditional trigger ---------------------------------

retrain_decider_task = PythonOperator(
    task_id='retrain_decider',
    python_callable=run_retrain_decider,
    dag=dag,
)

should_retrain_task = ShortCircuitOperator(
    task_id='check_should_retrain',
    python_callable=check_should_retrain,
    dag=dag,
)

trigger_retrain_task = TriggerDagRunOperator(
    task_id='trigger_retrain',
    trigger_dag_id='nyc_model_retrain_dag',
    conf={
        'pipeline_run_id': '{{ ti.xcom_pull(key="pipeline_run_id", task_ids="retrain_decider") }}',
    },
    dag=dag,
)

# -- Task dependencies -----------------------------------------------------

(
    ingestion_task
    >> validation_task
    >> preprocessing_task
    >> transformation_task
    >> ml_transformation_task
    >> retrain_decider_task
    >> should_retrain_task
    >> trigger_retrain_task
)
