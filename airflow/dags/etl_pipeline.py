"""
NYC Yellow Taxi MLOps Pipeline DAG

Orchestrates the complete pipeline:
  Data stages:  Ingestion -> Validation -> Preprocessing -> Transformation -> ML Transformation
  ML stages:    Model Training -> Model Evaluation -> Model Registry

Each DAG run shares a single PIPELINE_RUN_ID so all stage metadata is
written to the same folder: src/metadata/pipeline_<RUN_ID>/
"""

from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator
from dag_utils import set_pipeline_run_id

from airflow import DAG
from src.data_ingestion import main as data_ingestion_main
from src.data_preprocessing import main as data_preprocessing_main
from src.data_transformation import main as data_transformation_main
from src.data_validation import main as data_validation_main
from src.ml_transformed import main as ml_transformation_main
from src.model_evaluation import main as model_evaluation_main
from src.model_registry import main as model_registry_main
from src.model_training import main as model_training_main


def run_data_ingestion(**context):
    """Download raw NYC taxi data (Bronze layer)."""
    set_pipeline_run_id(**context)
    data_ingestion_main()



def run_data_validation(**context):
    """Validate ingested data schema."""
    set_pipeline_run_id(**context)
    data_validation_main()


def run_data_preprocessing(**context):
    """Clean and filter data (Silver layer)."""
    set_pipeline_run_id(**context)
    data_preprocessing_main()


def run_data_transformation(**context):
    """Feature engineering (Gold layer)."""
    set_pipeline_run_id(**context)
    data_transformation_main()


def run_ml_transformation(**context):
    """Prepare ML-ready train/val/test splits."""
    set_pipeline_run_id(**context)
    ml_transformation_main()


def run_model_training(**context):
    """Train champion model from champion.json."""
    set_pipeline_run_id(**context)
    model_training_main()


def run_model_evaluation(**context):
    """Evaluate new model against registered Production model."""
    set_pipeline_run_id(**context)
    model_evaluation_main()


def run_model_registry(**context):
    """Register and promote model in MLflow Model Registry."""
    set_pipeline_run_id(**context)
    model_registry_main()


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
    'nyc_taxi_mlops_pipeline',
    default_args=default_args,
    description='Complete MLOps pipeline: data ETL + model training/evaluation/registry',
    schedule_interval='@monthly',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['nyc-taxi', 'etl', 'ml', 'production', 'mlops'],
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
# Data: ingestion -> validation -> preprocessing -> transformation -> ml_transformation
# ML:   ml_transformation -> training -> evaluation -> registry

(
    ingestion_task
    >> validation_task
    >> preprocessing_task
    >> transformation_task
    >> ml_transformation_task
    >> training_task
    >> evaluation_task
    >> registry_task
)
