# Airflow DAGs

This directory contains Airflow DAGs for orchestrating the NYC Yellow Taxi MLOps pipeline.

## Available DAGs

### 1. `etl_pipeline.py` (Production)
**DAG ID**: `nyc_taxi_etl_pipeline`

Complete ETL pipeline that orchestrates all three stages:
- **Data Ingestion** (Bronze Layer): Downloads raw taxi data from NYC TLC
- **Data Preprocessing** (Silver Layer): Cleans and validates the data
- **Data Transformation** (Gold Layer): Applies feature engineering

**Schedule**: Monthly (`@monthly`)  
**Tags**: `nyc-taxi`, `etl`, `production`, `mlops`

#### Task Flow:
```
data_ingestion → data_preprocessing → data_transformation
```

#### How to Use:
1. Open Airflow UI at http://localhost:8080
2. Find the `nyc_taxi_etl_pipeline` DAG
3. Toggle the DAG to enable it (switch on the left)
4. Click "Trigger DAG" to run manually, or wait for scheduled run

#### Task Details:
- **data_ingestion**: Downloads parquet files for all months of 2025, stores in `data/2025/<month>/`
- **data_preprocessing**: Cleans data with filters and imputation, stores in `silver/2025/<month>/`
- **data_transformation**: Engineers features (time, distance, ratios), stores in `gold/2025/<month>/`

Each task generates a `metadata.json` file with run statistics, data profiles, and quality metrics.

### 2. `example_pipeline_dag.py` (Example)
**DAG ID**: `nyc_taxi_pipeline_example`

Simple example DAG using BashOperator to demonstrate the pipeline structure. This is for reference only.

## Configuration

### Required Environment Variables (in `.env`):
```bash
_PIP_ADDITIONAL_REQUIREMENTS=pyspark==4.1.1 numpy==2.4.2 pandas==3.0.0
```

These packages are required for the pipeline to run inside the Airflow container.

### Modifying DAGs

After modifying any DAG file:
1. Airflow will automatically detect the changes within ~30 seconds
2. Refresh the Airflow UI to see updates
3. If imports fail, check logs: `docker-compose logs airflow-scheduler`

## Troubleshooting

### Import Errors
If you see import errors in the DAG:
1. Check that `PYTHONPATH` is set in `docker-compose.yml`:
   ```yaml
   PYTHONPATH: '/opt/airflow:/opt/airflow/src'
   ```
2. Verify src/ is mounted: `docker-compose exec airflow-webserver ls /opt/airflow/src`

### Package Not Found
If PySpark or other packages are missing:
1. Check `.env` has `_PIP_ADDITIONAL_REQUIREMENTS` set
2. Restart services: `docker-compose restart`
3. Check logs: `docker-compose logs airflow-webserver`

### Task Failures
Check task logs in Airflow UI:
1. Click on the failed task in the Graph View
2. Click "Log" to see detailed error messages
3. Common issues:
   - Missing input data → Run data_ingestion first
   - Permission errors → Check file permissions in mounted volumes
   - Memory errors → Increase Docker memory limit

### Spark Connection Issues
If Spark fails to start:
1. Check driver host configuration in `src/utils/spark_utils.py`
2. Verify localhost binding: `spark.driver.host=localhost`
3. Check container logs for Java/Spark errors

## Adding New DAGs

To create a new DAG:

1. Create a new Python file in this directory
2. Import required Airflow modules:
   ```python
   from airflow import DAG
   from airflow.operators.python import PythonOperator
   from datetime import datetime, timedelta
   ```
3. Define your DAG:
   ```python
   dag = DAG(
       'my_custom_dag',
       default_args={...},
       schedule_interval='@daily',
       ...
   )
   ```
4. Create tasks and dependencies
5. Save the file - Airflow will auto-detect it

## Monitoring

### View DAG Status
- **Airflow UI**: http://localhost:8080
- **Graph View**: Shows task dependencies and status
- **Tree View**: Shows historical runs
- **Gantt Chart**: Shows task duration and parallelism

### Metrics
Each pipeline stage generates `metadata.json` with:
- Run timestamp and duration
- Input/output file paths and sizes
- Row counts and schema
- Null counts and column statistics
- Error messages (if any)

Check metadata files:
```bash
cat data/2025/metadata.json
cat silver/2025/metadata.json
cat gold/2025/metadata.json
```

## Best Practices

1. **Test DAGs locally** before deploying to production
2. **Use tags** to organize and filter DAGs
3. **Set appropriate timeouts** for long-running tasks
4. **Configure retries** for transient failures
5. **Monitor logs** regularly for warnings
6. **Version control** all DAG changes
7. **Document** complex task logic in docstrings
