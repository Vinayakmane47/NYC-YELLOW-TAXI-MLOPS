"""
ETL Pipeline DAG for NYC Yellow Taxi MLOps
Orchestrates the complete data pipeline: Ingestion -> Preprocessing -> Transformation
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add src directory to Python path for imports
sys.path.insert(0, '/opt/airflow/src')
sys.path.insert(0, '/opt/airflow')

# Import pipeline modules
from src.data_ingestion import main as data_ingestion_main
from src.data_preprocessing import main as data_preprocessing_main
from src.data_transformation import main as data_transformation_main


def run_data_ingestion(**context):
    """
    Task to run data ingestion stage.
    Downloads raw NYC taxi data from source and stores in data/ directory.
    """
    print("=" * 80)
    print("Starting Data Ingestion Stage (Bronze Layer)")
    print("=" * 80)
    data_ingestion_main()
    print("Data Ingestion completed successfully!")
    return "ingestion_complete"


def run_data_preprocessing(**context):
    """
    Task to run data preprocessing stage.
    Cleans and validates data, stores in silver/ directory.
    """
    print("=" * 80)
    print("Starting Data Preprocessing Stage (Silver Layer)")
    print("=" * 80)
    data_preprocessing_main()
    print("Data Preprocessing completed successfully!")
    return "preprocessing_complete"


def run_data_transformation(**context):
    """
    Task to run data transformation stage.
    Applies feature engineering, stores in gold/ directory.
    """
    print("=" * 80)
    print("Starting Data Transformation Stage (Gold Layer)")
    print("=" * 80)
    data_transformation_main()
    print("Data Transformation completed successfully!")
    return "transformation_complete"


# Default arguments for the DAG
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

# Define the DAG
dag = DAG(
    'nyc_taxi_etl_pipeline',
    default_args=default_args,
    description='Complete ETL pipeline for NYC Yellow Taxi data processing',
    schedule_interval='@monthly',  # Run monthly
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['nyc-taxi', 'etl', 'production', 'mlops'],
)

# Task 1: Data Ingestion (Bronze Layer)
ingestion_task = PythonOperator(
    task_id='data_ingestion',
    python_callable=run_data_ingestion,
    provide_context=True,
    dag=dag,
)

# Task 2: Data Preprocessing (Silver Layer)
preprocessing_task = PythonOperator(
    task_id='data_preprocessing',
    python_callable=run_data_preprocessing,
    provide_context=True,
    dag=dag,
)

# Task 3: Data Transformation (Gold Layer)
transformation_task = PythonOperator(
    task_id='data_transformation',
    python_callable=run_data_transformation,
    provide_context=True,
    dag=dag,
)

# Define task dependencies: Ingestion -> Preprocessing -> Transformation
ingestion_task >> preprocessing_task >> transformation_task
