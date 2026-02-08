#!/bin/bash
# Setup script for Airflow with Docker Compose

set -e

echo "=== NYC Yellow Taxi MLOps - Airflow Setup ==="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from env.example..."
    cp env.example .env
    echo "✓ .env file created"
else
    echo "✓ .env file already exists"
fi

# Set AIRFLOW_UID
echo ""
echo "Setting AIRFLOW_UID..."
if [ -z "$AIRFLOW_UID" ]; then
    export AIRFLOW_UID=50000
    echo "AIRFLOW_UID=$AIRFLOW_UID" >> .env
fi
echo "✓ AIRFLOW_UID set to $AIRFLOW_UID"

# Create airflow directories if they don't exist
echo ""
echo "Creating Airflow directories..."
mkdir -p airflow/dags airflow/logs airflow/plugins airflow/config
echo "✓ Airflow directories created"

# Check if Docker is running
echo ""
echo "Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker and try again."
    exit 1
fi
echo "✓ Docker is running"

# Initialize Airflow
echo ""
echo "Initializing Airflow (this may take a few minutes)..."
docker-compose up airflow-init
echo "✓ Airflow initialized"

# Start Airflow services
echo ""
echo "Starting Airflow services..."
docker-compose up -d
echo "✓ Airflow services started"

# Wait for services to be ready
echo ""
echo "Waiting for Airflow web server to be ready..."
sleep 10

# Check service status
echo ""
echo "Checking service status..."
docker-compose ps

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Airflow Web UI: http://localhost:8080"
echo "Username: airflow"
echo "Password: airflow"
echo ""
echo "PostgreSQL:"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  Database: airflow"
echo "  Username: airflow"
echo "  Password: airflow"
echo ""
echo "Useful commands:"
echo "  docker-compose logs -f          # View logs"
echo "  docker-compose down             # Stop services"
echo "  docker-compose down -v          # Stop and remove volumes"
echo "  docker-compose restart          # Restart services"
echo ""
