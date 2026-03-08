# Inference & Monitoring

## Inference API

**Files:** `src/inference/app.py`, `src/inference/feature_pipeline.py`, `src/inference/model_loader.py`

The inference server is a FastAPI application that serves real-time trip duration predictions. It downloads the Production model from MLflow Model Registry on DagShub at startup and replicates the training-time feature engineering in pure pandas (no Spark dependency).

### Starting the Server

```bash
# Docker (recommended)
docker compose up inference-api prometheus grafana locust

# Local development
export DAGSHUB_TOKEN=your_token
pip install -e ".[inference]"
uvicorn src.inference.app:app --reload --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI for making predictions |
| `/predict` | POST | Single trip prediction |
| `/predict/batch` | POST | Batch predictions (max 100) |
| `/health` | GET | Health check with model status |
| `/model/info` | GET | Detailed model metadata |
| `/model/reload` | POST | Reload model from registry |
| `/metrics` | GET | Prometheus metrics |

### Request Format

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_datetime": "2025-06-15T14:30:00",
    "PULocationID": 161,
    "DOLocationID": 237,
    "trip_distance": 3.5,
    "passenger_count": 1,
    "VendorID": 2,
    "RatecodeID": 1,
    "payment_type": 1,
    "store_and_fwd_flag": "N",
    "tolls_amount": 0.0,
    "Airport_fee": 0.0,
    "congestion_surcharge": 2.5
  }'
```

### Input Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `pickup_datetime` | string | Yes | - | ISO format datetime |
| `PULocationID` | int | Yes | - | Pickup taxi zone (1-263) |
| `DOLocationID` | int | Yes | - | Dropoff taxi zone (1-263) |
| `trip_distance` | float | Yes | - | Distance in miles |
| `passenger_count` | int | No | 1 | Passengers (0-9) |
| `VendorID` | int | No | 1 | 1=CMT, 2=VeriFone |
| `RatecodeID` | int | No | 1 | 1=Standard, 2=JFK, etc. |
| `payment_type` | int | No | 1 | 1=Credit, 2=Cash, etc. |
| `store_and_fwd_flag` | string | No | "N" | Y or N |
| `tolls_amount` | float | No | 0.0 | Toll charges |
| `Airport_fee` | float | No | 0.0 | Airport fee |
| `congestion_surcharge` | float | No | 0.0 | Congestion surcharge |

### Response Format

```json
{
  "predicted_duration_minutes": 17.23,
  "model_family": "random_forest",
  "model_version": "v9"
}
```

### Batch Predictions

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"pickup_datetime": "2025-06-15T08:00:00", "PULocationID": 161, "DOLocationID": 237, "trip_distance": 3.5},
    {"pickup_datetime": "2025-06-15T18:00:00", "PULocationID": 132, "DOLocationID": 48, "trip_distance": 8.2}
  ]'
```

## Feature Pipeline

**File:** `src/inference/feature_pipeline.py`

Replicates the training-time transformations (`data_transformation.py` + `ml_transformed.py`) in pure pandas. This ensures feature parity between training and inference.

### Transformations Applied

1. **Time features** - Hour, day, week, month, year, day of week from pickup datetime
2. **Binary time flags** - is_weekend, is_peak_hour, is_night, is_rush_hour
3. **Cyclical encoding** - sin/cos of hour (24h cycle) and day of week (7-day cycle)
4. **Distance features** - log, sqrt, is_short_trip (<1mi), is_long_trip (>10mi)
5. **Zone features** - is_same_zone, zone trip counts and avg durations (training defaults)
6. **Boolean fees** - has_tolls, has_airport_fee, has_congestion_fee
7. **Binary encoding** - store_and_fwd_flag (Y=1, N=0)
8. **One-hot encoding** - VendorID, RatecodeID, payment_type (fixed categories from training)
9. **Target encoding** - PULocationID, DOLocationID (using training-time zone means)
10. **Standard scaling** - 7 numerical features using training-time means/stds

### Training Stats

Scaling statistics and encoding mappings are loaded from `src/metadata/pipeline_*/ml_transformation.json`. The pipeline auto-discovers the latest metadata directory.

### Output

A DataFrame with exactly 41 features in the same column order as training.

## Model Loader

**File:** `src/inference/model_loader.py`

Downloads and caches the Production model from MLflow Model Registry on DagShub.

### Loading Process

1. Authenticate to DagShub using `DAGSHUB_TOKEN`
2. Query MLflow registry for the latest Production version of `nyc-taxi-trip-duration`
3. Download the raw `.joblib` artifact from the model's run artifacts
4. Cache locally in `artifacts/mlflow_cache/`
5. Load with `joblib.load()`

### Fallback

If the registry is unreachable, falls back to the most recent cached `.joblib` file.

### Reload

```bash
curl -X POST http://localhost:8000/model/reload
```

## Web UI

**File:** `src/inference/templates/index.html`

A Jinja2-rendered single-page form at `http://localhost:8000`. Two-column layout with a form on the left (grouped into Pickup, Trip Info, Fees sections) and a result panel on the right showing the predicted duration.

## Docker Container

**File:** `Dockerfile.inference`

```dockerfile
FROM python:3.11-slim
# Installs: FastAPI, uvicorn, prometheus-client, scikit-learn>=1.3.2,<1.4,
#           lightgbm, joblib, pandas, numpy<2, mlflow
# No local model files - downloads from MLflow at startup
EXPOSE 8000
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Note: scikit-learn is pinned to `>=1.3.2,<1.4` to match the version used during model training. Version mismatches cause deserialization errors.

---

## Monitoring

### Prometheus Metrics

**File:** `monitoring/prometheus/prometheus.yml`

The inference API exposes metrics at `/metrics`. Prometheus scrapes every 15 seconds.

#### Custom Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_requests_total` | Counter | Total requests (label: `endpoint`) |
| `prediction_errors_total` | Counter | Failed predictions (label: `error_type`) |
| `prediction_latency_seconds` | Histogram | End-to-end latency |
| `feature_pipeline_latency_seconds` | Histogram | Feature engineering latency |
| `model_inference_latency_seconds` | Histogram | Pure model prediction latency |
| `model_prediction_value_minutes` | Histogram | Distribution of predicted durations |
| `inference_batch_size` | Histogram | Number of trips per batch request |
| `inference_in_flight_requests` | Gauge | Currently processing requests |
| `active_model_loaded` | Gauge | Model availability (1/0) |

#### Process Metrics (auto-exposed)

| Metric | Description |
|--------|-------------|
| `process_resident_memory_bytes` | RSS memory usage |
| `process_virtual_memory_bytes` | Virtual memory usage |
| `process_cpu_seconds_total` | CPU time consumed |

### Grafana Dashboard

**Files:** `monitoring/grafana/dashboards/inference.json`, `monitoring/grafana/provisioning/`

Pre-provisioned dashboard with 18 panels organized in 5 rows:

| Row | Panels |
|-----|--------|
| **Overview** | Success Rate % (gauge), Total Predictions, Total Errors, p95 Latency, In-Flight Requests, Model Status |
| **Traffic** | Request Rate by endpoint, Error Rate by type, In-Flight over time, Throughput (predictions/sec) |
| **Latency** | E2E Latency p50/p95/p99, Feature Pipeline vs Model Inference p95 |
| **Predictions** | Duration Distribution, Avg Prediction Drift (now vs 1h ago), Batch Size Distribution, Avg Batch Size |
| **System** | Process Memory (RSS + Virtual), Process CPU Usage |

**Access:** http://localhost:3000 (admin/admin)

### Dashboard Provisioning

The Grafana datasource and dashboard are auto-provisioned via config files:

- `monitoring/grafana/provisioning/datasources/datasources.yaml` - Prometheus datasource (UID: `prometheus`)
- `monitoring/grafana/provisioning/dashboards/dashboards.yaml` - Dashboard discovery path
- `monitoring/grafana/dashboards/inference.json` - Dashboard definition

---

## Load Testing (Locust)

**File:** `src/inference/locustfile.py`

Locust runs as a Docker service for load testing the inference API.

### Test Scenarios

| Task | Weight | Description |
|------|--------|-------------|
| `predict_single` | 5 | Single prediction with randomized trip data |
| `predict_batch` | 2 | Batch of 5-20 random trips |
| `health_check` | 1 | GET /health |
| `model_info` | 1 | GET /model/info |

### Using Locust

**Web UI mode:**

```bash
docker compose up inference-api locust
# Open http://localhost:8089
# Set users, spawn rate, and start
```

**Headless mode:**

```bash
docker compose run --rm locust \
  -f /mnt/locust/locustfile.py \
  --host=http://inference-api:8000 \
  --headless -u 50 -r 5 -t 60s
```

This runs 50 simulated users, spawning 5 per second, for 60 seconds.

### Monitoring Under Load

While Locust is running, observe the Grafana dashboard to see:
- Request rate ramping up
- Latency distribution under load
- In-flight request count
- CPU/memory usage trends
- Prediction drift patterns

## Troubleshooting

**"Model not loaded" error:**
- Check that `DAGSHUB_TOKEN` is set in `.env`
- Verify a model is registered with stage `Production` in MLflow

**scikit-learn version mismatch:**
- The model must be loaded with the same sklearn version used for training
- Dockerfile pins `scikit-learn>=1.3.2,<1.4`

**Grafana shows "No data":**
- Check Prometheus targets at http://localhost:9090/targets
- Ensure inference-api is listed and UP
- Make a few predictions first to generate metrics

**Predictions seem wrong:**
- Verify `src/metadata/pipeline_*/ml_transformation.json` exists
- Check that input values are in reasonable ranges
