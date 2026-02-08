.PHONY: help setup up down restart logs clean install

help:
	@echo "NYC Yellow Taxi MLOps Pipeline - Available Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install    - Install Python dependencies using uv"
	@echo "  make setup      - Setup Airflow with Docker Compose"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make up         - Start Airflow and PostgreSQL services"
	@echo "  make down       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo "  make logs       - View logs (follow mode)"
	@echo "  make clean      - Stop services and remove volumes"
	@echo ""
	@echo "Pipeline Commands:"
	@echo "  make ingest     - Run data ingestion"
	@echo "  make preprocess - Run data preprocessing"
	@echo "  make transform  - Run data transformation"
	@echo "  make pipeline   - Run full pipeline (all stages)"
	@echo ""

install:
	@echo "Installing Python dependencies with uv..."
	uv sync

setup:
	@echo "Setting up Airflow..."
	./setup_airflow.sh

up:
	@echo "Starting Airflow services..."
	docker-compose up -d
	@echo "Airflow Web UI: http://localhost:8080"

down:
	@echo "Stopping Airflow services..."
	docker-compose down

restart:
	@echo "Restarting Airflow services..."
	docker-compose restart

logs:
	@echo "Viewing logs (Ctrl+C to exit)..."
	docker-compose logs -f

clean:
	@echo "Cleaning up (removing volumes)..."
	docker-compose down -v
	@echo "Done. Run 'make setup' to reinitialize."

ingest:
	@echo "Running data ingestion..."
	uv run python src/data_ingestion.py

preprocess:
	@echo "Running data preprocessing..."
	uv run python src/data_preprocessing.py

transform:
	@echo "Running data transformation..."
	uv run python src/data_transformation.py

pipeline: ingest preprocess transform
	@echo "Pipeline completed!"
