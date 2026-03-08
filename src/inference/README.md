# NYC Yellow Taxi - Inference System

Real-time trip duration prediction API with monitoring.

## Architecture

```
                    +------------------+
                    |    Web UI        |
                    |  (localhost:8000)|
                    +--------+---------+
                             |
                    +--------v---------+
                    |    FastAPI        |
                    |  /predict        |
                    |  /predict/batch  |
                    |  /health         |
                    |  /model/info     |
                    |  /metrics        |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
    +---------v----------+      +-----------v--------+
    | Feature Pipeline   |      | Model Loader       |
    | (pandas transforms)|      | (joblib, cached)   |
    +--------------------+      +--------------------+
                             |
                    +--------v---------+
                    |   Prometheus     |
                    | (localhost:9090) |
                    +--------+---------+
                             |
                    +--------v---------+
                    |    Grafana       |
                    | (localhost:3000) |
                    +------------------+
```

## Quick Start

### Start all services

```bash
docker-compose up inference-api prometheus grafana locust
```

### Services

| Service        | URL                    | Credentials   |
|---------------|------------------------|---------------|
| Inference API | http://localhost:8000   | -             |
| Web UI        | http://localhost:8000   | -             |
| Prometheus    | http://localhost:9090   | -             |
| Grafana       | http://localhost:3000   | admin / admin |
| Locust        | http://localhost:8089   | -             |

## API Usage

### Single Prediction

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

Response:
```json
{
  "predicted_duration_minutes": 12.45,
  "model_family": "random_forest",
  "model_version": "random_forest_20260211T121306Z"
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"pickup_datetime": "2025-06-15T08:00:00", "PULocationID": 161, "DOLocationID": 237, "trip_distance": 3.5, "passenger_count": 1, "VendorID": 2, "RatecodeID": 1, "payment_type": 1, "store_and_fwd_flag": "N", "tolls_amount": 0, "Airport_fee": 0, "congestion_surcharge": 2.5},
    {"pickup_datetime": "2025-06-15T18:00:00", "PULocationID": 132, "DOLocationID": 48, "trip_distance": 8.2, "passenger_count": 2, "VendorID": 1, "RatecodeID": 1, "payment_type": 1, "store_and_fwd_flag": "N", "tolls_amount": 6.55, "Airport_fee": 0, "congestion_surcharge": 2.5}
  ]'
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Model Info

```bash
curl http://localhost:8000/model/info
```

### Reload Model (after retraining)

```bash
curl -X POST http://localhost:8000/model/reload
```

## Input Fields

| Field                  | Type   | Required | Description                          |
|-----------------------|--------|----------|--------------------------------------|
| `pickup_datetime`      | string | Yes      | ISO format datetime                  |
| `PULocationID`         | int    | Yes      | Pickup taxi zone ID (1-265)          |
| `DOLocationID`         | int    | Yes      | Dropoff taxi zone ID (1-265)         |
| `trip_distance`        | float  | Yes      | Distance in miles                    |
| `passenger_count`      | int    | No       | Number of passengers (default: 1)    |
| `VendorID`             | int    | No       | 1=CMT, 2=VeriFone (default: 1)       |
| `RatecodeID`           | int    | No       | 1=Standard, 2=JFK, ... (default: 1)  |
| `payment_type`         | int    | No       | 1=Credit, 2=Cash, ... (default: 1)   |
| `store_and_fwd_flag`   | string | No       | Y or N (default: N)                  |
| `tolls_amount`         | float  | No       | Toll amount in $ (default: 0)        |
| `Airport_fee`          | float  | No       | Airport fee in $ (default: 0)        |
| `congestion_surcharge` | float  | No       | Congestion surcharge in $ (default: 0)|

## Monitoring

### Prometheus Metrics

The `/metrics` endpoint exposes:

- `prediction_requests_total` - Total prediction requests by endpoint
- `prediction_latency_seconds` - End-to-end latency histogram
- `prediction_errors_total` - Failed predictions by error type
- `model_prediction_value_minutes` - Distribution of predicted durations
- `feature_pipeline_latency_seconds` - Feature engineering latency
- `model_inference_latency_seconds` - Pure model prediction latency
- `active_model_loaded` - Whether a model is loaded (1/0)

### Grafana Dashboard

The pre-provisioned dashboard includes:
- Request rate over time
- Latency percentiles (p50, p95, p99)
- Error rate
- Feature pipeline vs model inference latency breakdown
- Prediction value distribution
- Total predictions and errors counters
- Model status indicator

## Feature Pipeline

The inference feature pipeline replicates the training-time transformations:

1. **Time features** - Hour, day, week, month, year, cyclical sin/cos encodings
2. **Distance features** - Log/sqrt transforms, short/long trip flags
3. **Zone features** - Same-zone flag, zone popularity (training averages)
4. **Encoding** - Binary (store_and_fwd), one-hot (Vendor, RateCode, Payment), target encoding (Location IDs)
5. **Scaling** - StandardScaler using training-time statistics from `ml_transformation.json`

## Model Loading

The inference server downloads the **Production** model from MLflow Model Registry on DagShub at startup. No local model files are needed.

**How it works:**
1. On startup, connects to DagShub MLflow using `DAGSHUB_TOKEN`
2. Downloads `models:/nyc-taxi-trip-duration/Production`
3. Caches the model in memory for fast inference
4. Falls back to a local cache if the registry is unreachable

**Requirements:**
- `DAGSHUB_TOKEN` must be set (in `.env` or as environment variable)
- A model must be registered in the `nyc-taxi-trip-duration` registry with stage `Production`

**Reload after retraining:**
```bash
curl -X POST http://localhost:8000/model/reload
```

## Load Testing (Locust)

Locust is included as a Docker service for load testing the inference API.

### Start load testing

```bash
docker-compose up inference-api locust
```

Open http://localhost:8089 to access the Locust web UI.

### Test scenarios

| Task             | Weight | Description                              |
|-----------------|--------|------------------------------------------|
| `predict_single` | 5      | Single trip prediction with random data  |
| `predict_batch`  | 2      | Batch of 5-20 random trips              |
| `health_check`   | 1      | GET /health                              |
| `model_info`     | 1      | GET /model/info                          |

### Run headless (no UI)

```bash
docker-compose run --rm locust \
  -f /mnt/locust/locustfile.py \
  --host=http://inference-api:8000 \
  --headless -u 50 -r 5 -t 60s
```

This runs 50 users, spawning 5/sec, for 60 seconds.

## Local Development

```bash
# Install inference dependencies
pip install -e ".[inference]"

# Set DagShub token
export DAGSHUB_TOKEN=your_token_here

# Run locally
uvicorn src.inference.app:app --reload --port 8000
```

## Troubleshooting

**Model not loading:**
- Ensure `DAGSHUB_TOKEN` is set in `.env` or as env variable
- Verify a model exists in MLflow registry: check https://dagshub.com/Vinayakmane47/NYC-YELLOW-TAXI-MLOPS.mlflow/
- Ensure the model has been promoted to `Production` stage

**Predictions seem wrong:**
- Verify `src/metadata/pipeline_*/ml_transformation.json` exists with scaling stats
- Check input values are in reasonable ranges (distance > 0, valid zone IDs)

**Grafana shows no data:**
- Confirm Prometheus is scraping: check http://localhost:9090/targets
- Make some predictions first to generate metrics
