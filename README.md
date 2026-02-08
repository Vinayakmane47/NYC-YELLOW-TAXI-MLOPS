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
├── docker-compose.yml           # Docker services configuration
├── Dockerfile.airflow           # Airflow + Spark client image
├── Dockerfile.spark             # Standalone Spark worker image
└── env.example                  # Environment variables template
```

## Docker Architecture

### Dockerfile.airflow
Custom Airflow image with Spark client capabilities:
- **Base**: Apache Airflow 2.8.2
- **Purpose**: Orchestration and DAG execution
- **Includes**: Java 17, Spark submit client, PySpark, AWS JARs
- **Use case**: Running Airflow webserver, scheduler, and triggering Spark jobs

### Dockerfile.spark
Standalone Spark image for distributed processing:
- **Base**: Python 3.9 slim
- **Purpose**: Spark worker/master for distributed computing
- **Includes**: Full Spark 3.4.1, Java 17, PySpark, data processing libraries
- **Use case**: Standalone Spark cluster or local Spark jobs

**Current Setup**: Using `Dockerfile.airflow` for all Airflow services with embedded Spark.

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

# Build custom Airflow image from Dockerfile.airflow
docker compose build

# Initialize Airflow (first time only)
docker compose up airflow-init

# Start Airflow services
docker compose up -d
```

**Note**: The custom Docker image includes:
- **Java 17** (OpenJDK) for Spark execution
- **Apache Spark 3.4.1** with Hadoop 3
- **PySpark 3.4.1** matching Spark version
- **NumPy, Pandas, PyArrow** for data processing
- **AWS/S3 JARs** (hadoop-aws, aws-java-sdk-bundle) for MinIO/S3 connectivity
- **DVC, Boto3, S3FS** for data versioning and cloud storage
- **Pydantic, PyYAML** for configuration management
- **Apache Airflow Spark Provider** for Spark job orchestration

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
# Build custom image
docker compose build

# Start all services
docker compose up -d

# Stop all services
docker compose down

# View logs
docker compose logs -f

# Stop and remove volumes (clean slate)
docker compose down -v

# Restart a specific service
docker compose restart airflow-scheduler

# Rebuild and restart services
docker compose up -d --build
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
docker compose logs airflow-webserver
docker compose logs airflow-scheduler

# Verify permissions
ls -la airflow/

# Rebuild image if needed
docker compose build --no-cache
```

**Import errors in DAGs:**
```bash
# Ensure src/ is mounted in docker-compose.yml
# Check PYTHONPATH in Airflow container
docker compose exec airflow-webserver python -c "import sys; print(sys.path)"

# Verify packages are installed
docker compose exec airflow-webserver pip list | grep pyspark
```

**PostgreSQL connection issues:**
```bash
# Verify PostgreSQL is healthy
docker compose ps postgres
docker compose logs postgres
```

## Tech Stack

- **Apache Airflow 2.8.2**: Workflow orchestration
- **Apache Spark 3.4.1**: Distributed data processing engine
- **PySpark 3.4.1**: Python API for Spark
- **PostgreSQL 16**: Airflow metadata database
- **Java 17 (OpenJDK)**: Required for Spark execution
- **Docker Compose**: Container orchestration
- **Python 3.9+**: Core language
- **AWS SDK / Hadoop AWS**: S3/MinIO connectivity
- **DVC**: Data version control
- **uv**: Package management (local development)

## License

This project is for educational and demonstration purposes.

## Data Source

NYC Taxi & Limousine Commission (TLC) Trip Record Data
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
