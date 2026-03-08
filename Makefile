.PHONY: help setup up down restart logs clean install minio-up minio-init minio-down

help:
	@echo "NYC Yellow Taxi MLOps Pipeline - Available Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install      - Install Python dependencies using uv"
	@echo "  make build        - Build custom Airflow Docker image"
	@echo "  make setup        - Setup Airflow with Docker Compose"
	@echo ""
	@echo "MinIO Commands:"
	@echo "  make minio-up     - Start MinIO container for local dev"
	@echo "  make minio-init   - Create bronze/silver/gold/ml-transformed buckets"
	@echo "  make minio-down   - Stop MinIO container"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make up           - Start all services (Airflow + MinIO)"
	@echo "  make down         - Stop all services"
	@echo "  make restart      - Restart all services"
	@echo "  make logs         - View logs (follow mode)"
	@echo "  make clean        - Stop services and remove volumes"
	@echo ""
	@echo "Pipeline Commands:"
	@echo "  make ingest       - Run data ingestion"
	@echo "  make preprocess   - Run data preprocessing"
	@echo "  make transform    - Run data transformation"
	@echo "  make pipeline     - Run full pipeline (all stages)"
	@echo ""
	@echo "MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
	@echo "Airflow Web UI: http://localhost:8080 (airflow/airflow)"
	@echo ""

install:
	@echo "Installing Python dependencies with uv..."
	uv sync

build:
	@echo "Building custom Airflow image..."
	docker compose build

setup: build
	@echo "Setting up Airflow..."
	./setup_airflow.sh

# MinIO for local development
minio-up:
	@echo "Starting MinIO..."
	@docker start minio 2>/dev/null || \
		docker run -d --name minio -p 9000:9000 -p 9001:9001 \
			-e MINIO_ROOT_USER=minioadmin \
			-e MINIO_ROOT_PASSWORD=minioadmin \
			minio/minio server /data --console-address ":9001"
	@echo "MinIO Console: http://localhost:9001"

minio-init:
	@echo "Creating MinIO buckets..."
	@docker run --rm --network host minio/mc sh -c \
		"mc alias set local http://localhost:9000 minioadmin minioadmin && \
		 mc mb --ignore-existing local/bronze && \
		 mc mb --ignore-existing local/silver && \
		 mc mb --ignore-existing local/gold && \
		 mc mb --ignore-existing local/ml-transformed"
	@echo "Buckets created: bronze, silver, gold, ml-transformed"

minio-down:
	@echo "Stopping MinIO..."
	@docker stop minio && docker rm minio || true

# Docker Compose services
up:
	@echo "Starting all services..."
	docker compose up -d
	@echo "Airflow Web UI: http://localhost:8080"
	@echo "MinIO Console: http://localhost:9001"

down:
	@echo "Stopping all services..."
	docker compose down

restart:
	@echo "Restarting all services..."
	docker compose restart

logs:
	@echo "Viewing logs (Ctrl+C to exit)..."
	docker compose logs -f

clean:
	@echo "Cleaning up (removing volumes)..."
	docker compose down -v
	@echo "Done. Run 'make setup' to reinitialize."

# Pipeline stages (local dev - requires MinIO running)
ingest:
	@echo "Running data ingestion..."
	uv run python -m src.data_ingestion

preprocess:
	@echo "Running data preprocessing..."
	uv run python -m src.data_preprocessing

transform:
	@echo "Running data transformation..."
	uv run python -m src.data_transformation

pipeline: ingest preprocess transform
	@echo "Pipeline completed!"

# Tests
test:
	@echo "Running tests..."
	uv run pytest tests/ -v
