# Project Overview

## What This Project Does

NYC Yellow Taxi MLOps is an end-to-end machine learning system that predicts trip durations for New York City yellow taxi rides. It covers the full MLOps lifecycle: data ingestion, feature engineering, model training, experiment tracking, model registry, real-time inference, monitoring, and automated retraining on data drift.

## Architecture

```
NYC TLC Data Source (HTTP)
        |
        v
+-------------------+     +-------------------+     +-------------------+
|   Bronze Layer    | --> |   Silver Layer    | --> |   Gold Layer      |
|   (Raw Parquet)   |     |   (Cleaned)       |     |   (Features)      |
|   MinIO/S3        |     |   MinIO/S3        |     |   MinIO/S3        |
+-------------------+     +-------------------+     +-------------------+
                                                            |
                                                            v
                                                    +-------------------+
                                                    | ML-Transformed    |
                                                    | (Train/Val/Test)  |
                                                    | Encoded + Scaled  |
                                                    +-------------------+
                                                            |
                          +---------------------------------+
                          |                                 |
                          v                                 v
                 +-------------------+           +-------------------+
                 | Offline HPO       |           | Model Training    |
                 | (Optuna, 10       |           | (Champion params) |
                 |  model families)  |           +-------------------+
                 +-------------------+                    |
                          |                               v
                          v                      +-------------------+
                   champion.json                 | Model Evaluation  |
                                                 | (vs Production)   |
                                                 +-------------------+
                                                          |
                                                          v
                                                 +-------------------+
                                                 | MLflow Registry   |
                                                 | (DagShub hosted)  |
                                                 | Staging/Production|
                                                 +-------------------+
                                                          |
                          +-------------------------------+
                          |
                          v
                 +-------------------+       +-------------------+
                 | FastAPI Inference | <---- | Locust Load Test  |
                 | (port 8000)       |       | (port 8089)       |
                 +-------------------+       +-------------------+
                          |
                          v
                 +-------------------+       +-------------------+
                 | Prometheus        | ----> | Grafana Dashboard |
                 | (port 9090)       |       | (port 3000)       |
                 +-------------------+       +-------------------+
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | Apache Airflow 2.8.2 |
| Data Processing | PySpark 3.4.1 |
| Object Storage | MinIO (S3-compatible) |
| ML Training | scikit-learn, LightGBM |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow on DagShub |
| Model Registry | MLflow Model Registry |
| Drift Detection | Evidently AI |
| Inference API | FastAPI + Uvicorn |
| Monitoring | Prometheus + Grafana |
| Load Testing | Locust |
| Configuration | Pydantic + YAML |
| CI/CD | GitHub Actions (ruff + pytest) |
| Containerization | Docker + Docker Compose |

## Directory Structure

```
NYC-YELLOW-TAXI-MLOPS/
|-- airflow/dags/               # 3 Airflow DAGs
|-- src/
|   |-- config/                 # Pydantic config + settings.yaml
|   |-- hpo/                    # Offline hyperparameter optimization
|   |-- inference/              # FastAPI server + web UI + Locust
|   |-- metadata/               # Pipeline run metadata (auto-generated)
|   |-- models/                 # Model factory
|   |-- utils/                  # Shared utilities
|   |-- data_ingestion.py       # Bronze layer
|   |-- data_validation.py      # Schema validation
|   |-- data_preprocessing.py   # Silver layer
|   |-- data_transformation.py  # Gold layer (feature engineering)
|   |-- ml_transformed.py       # Train/val/test splits + encoding
|   |-- model_training.py       # Train champion model
|   |-- model_evaluation.py     # Compare new vs Production
|   |-- model_registry.py       # Register + promote model
|   `-- drift_detection.py      # Evidently drift detection
|-- tests/                      # pytest test suite
|-- monitoring/                 # Prometheus + Grafana configs
|-- docs/                       # This documentation
|-- Dockerfile.airflow          # Airflow image (Spark + ML libs)
|-- Dockerfile.inference        # Inference API image
|-- docker-compose.yml          # Full stack (10 services)
`-- pyproject.toml              # Dependencies + ruff config
```

## Services & Ports

| Service | Port | URL |
|---------|------|-----|
| Airflow Web UI | 8080 | http://localhost:8080 (airflow/airflow) |
| MinIO Console | 9001 | http://localhost:9001 (minioadmin/minioadmin) |
| MinIO API | 9000 | http://localhost:9000 |
| Inference API + UI | 8000 | http://localhost:8000 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 (admin/admin) |
| Locust | 8089 | http://localhost:8089 |
| PostgreSQL | 5432 | localhost:5432 (airflow/airflow) |

## Quick Start

```bash
# 1. Clone and configure
git clone <repo-url>
cd NYC-YELLOW-TAXI-MLOPS
cp env.example .env
# Edit .env and set DAGSHUB_TOKEN

# 2. Build Docker images
docker compose build

# 3. Initialize Airflow
docker compose up airflow-init

# 4. Start all services
docker compose up -d

# 5. Access services
# Airflow: http://localhost:8080 - Enable and trigger DAGs
# Inference: http://localhost:8000 - Make predictions
# Grafana: http://localhost:3000 - View dashboards
```
