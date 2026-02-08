# NYC Yellow Taxi MLOps Pipeline

A complete MLOps pipeline for NYC Yellow Taxi trip data processing and analysis using PySpark and Apache Airflow.

## Project Structure

```
.
├── src/
│   ├── data_ingestion.py       # Download raw taxi data
│   ├── data_preprocessing.py    # Clean and preprocess data
│   ├── data_transformation.py   # Feature engineering
│   └── utils/
│       └── spark_utils.py       # Spark utilities and metadata helpers
├── airflow/
│   ├── dags/                    # Airflow DAG definitions
│   ├── logs/                    # Airflow logs (ignored by git)
│   ├── plugins/                 # Airflow plugins
│   └── config/                  # Airflow configuration
├── data/                        # Raw data (Bronze layer)
├── silver/                      # Cleaned data (Silver layer)
├── gold/                        # Feature-engineered data (Gold layer)
├── docker-compose.yml           # Docker services (Airflow + PostgreSQL)
└── env.example                  # Environment variables template
```

## Data Pipeline Stages

### 1. **Data Ingestion** (Bronze Layer)
- Downloads NYC Yellow Taxi parquet files from official source
- Stores raw data in `data/2025/<month>/trip_<month>.parquet`
- Generates metadata JSON with download statistics

### 2. **Data Preprocessing** (Silver Layer)
- Cleans and validates data using PySpark
- Applies filters and imputes missing values
- Stores cleaned data in `silver/2025/<month>/trip_<month>.parquet`
- Generates metadata JSON with data quality metrics

### 3. **Data Transformation** (Gold Layer)
- Feature engineering: time features, cyclical encoding, distance transforms
- Creates aggregated features and ratios
- Stores feature-engineered data in `gold/2025/<month>/trip_<month>.parquet`
- Generates metadata JSON with feature statistics

## Setup

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- uv (Python package manager)

### 1. Install Python Dependencies

```bash
# Install dependencies using uv
uv sync
```

### 2. Setup Airflow with Docker

```bash
# Create .env file from template
cp env.example .env

# Edit .env file to customize configuration (optional)
nano .env

# Initialize Airflow (first time only)
docker-compose up airflow-init

# Start Airflow services
docker-compose up -d
```

### 3. Access Airflow Web UI

- **URL**: http://localhost:8080
- **Username**: `airflow` (configurable in .env)
- **Password**: `airflow` (configurable in .env)

### 4. Access PostgreSQL

- **Host**: `localhost`
- **Port**: `5432`
- **Database**: `airflow`
- **Username**: `airflow`
- **Password**: `airflow`

## Running the Pipeline

### Option 1: Run with Airflow (Recommended)

1. Open Airflow UI at http://localhost:8080
2. Enable the `nyc_taxi_etl_pipeline` DAG
3. Trigger the DAG manually by clicking "Trigger DAG" button
4. Monitor progress in Graph View or Tree View

**Available DAGs:**
- `nyc_taxi_etl_pipeline` - **Production ETL pipeline** (uses Python imports)
- `nyc_taxi_pipeline_example` - Example pipeline (uses bash commands)

### Option 2: Run Manually (Individual Stages)

```bash
# Activate virtual environment
source .venv/bin/activate

# Run data ingestion
python src/data_ingestion.py

# Run data preprocessing
python src/data_preprocessing.py

# Run data transformation
python src/data_transformation.py
```

## Docker Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# Stop and remove volumes (clean slate)
docker-compose down -v

# Restart a specific service
docker-compose restart airflow-scheduler
```

## Configuration

### Airflow Configuration

Edit `.env` file to customize:
- `AIRFLOW_UID`: User ID for Airflow processes
- `_AIRFLOW_WWW_USER_USERNAME`: Web UI username
- `_AIRFLOW_WWW_USER_PASSWORD`: Web UI password
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`: PostgreSQL credentials

### Pipeline Configuration

Edit individual Python files to customize:
- **Data Ingestion**: Modify `BASE_URL`, `OUTPUT_BASE_DIR` in `src/data_ingestion.py`
- **Preprocessing**: Adjust filters and imputation logic in `src/data_preprocessing.py`
- **Transformation**: Customize feature engineering in `src/data_transformation.py`

## Metadata

Each pipeline stage generates a `metadata.json` file containing:
- Run information (timestamp, duration, status)
- Input/output file paths and sizes
- Data profile (row count, schema, null counts)
- Column statistics for numerical fields

## Development

### Adding New DAGs

1. Create a new Python file in `airflow/dags/`
2. Define your DAG using Airflow operators
3. Airflow will automatically detect and load the DAG

### Troubleshooting

**Airflow services not starting:**
```bash
# Check logs
docker-compose logs airflow-webserver
docker-compose logs airflow-scheduler

# Verify permissions
ls -la airflow/
```

**Import errors in DAGs:**
```bash
# Ensure src/ is mounted in docker-compose.yml
# Check PYTHONPATH in Airflow container
docker-compose exec airflow-webserver python -c "import sys; print(sys.path)"
```

**PostgreSQL connection issues:**
```bash
# Verify PostgreSQL is healthy
docker-compose ps postgres
docker-compose logs postgres
```

## Tech Stack

- **Apache Airflow 2.10.4**: Workflow orchestration
- **PostgreSQL 16**: Airflow metadata database
- **PySpark 4.1.1**: Distributed data processing
- **Docker Compose**: Container orchestration
- **Python 3.11**: Core language
- **uv**: Package management

## License

This project is for educational and demonstration purposes.

## Data Source

NYC Taxi & Limousine Commission (TLC) Trip Record Data
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
