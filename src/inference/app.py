"""FastAPI inference server for NYC Yellow Taxi trip duration prediction."""

import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from src.inference.feature_pipeline import InferenceFeaturePipeline
from src.inference.model_loader import ModelLoader

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["endpoint"],
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of failed predictions",
    ["error_type"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "End-to-end prediction latency",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
FEATURE_PIPELINE_LATENCY = Histogram(
    "feature_pipeline_latency_seconds",
    "Feature engineering latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)
MODEL_INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Pure model prediction latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)
PREDICTION_VALUE = Histogram(
    "model_prediction_value_minutes",
    "Distribution of predicted trip durations (minutes)",
    buckets=[1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120],
)
MODEL_LOADED = Gauge(
    "active_model_loaded",
    "Whether a model is currently loaded (1=yes, 0=no)",
)
IN_FLIGHT = Gauge(
    "inference_in_flight_requests",
    "Number of requests currently being processed",
)
BATCH_SIZE = Histogram(
    "inference_batch_size",
    "Number of trips per batch request",
    buckets=[1, 2, 5, 10, 20, 50, 100],
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TripInput(BaseModel):
    pickup_datetime: str = Field(..., description="Pickup datetime (ISO format)")
    PULocationID: int = Field(..., description="Pickup location zone ID")
    DOLocationID: int = Field(..., description="Dropoff location zone ID")
    trip_distance: float = Field(..., ge=0, description="Trip distance in miles")
    passenger_count: int = Field(1, ge=0, le=9, description="Number of passengers")
    VendorID: int = Field(1, description="Vendor ID (1 or 2)")
    RatecodeID: int = Field(1, description="Rate code ID (1-6)")
    payment_type: int = Field(1, description="Payment type (1-4)")
    store_and_fwd_flag: str = Field("N", description="Store and forward flag (Y/N)")
    tolls_amount: float = Field(0.0, ge=0, description="Toll amount")
    Airport_fee: float = Field(0.0, ge=0, description="Airport fee")
    congestion_surcharge: float = Field(0.0, ge=0, description="Congestion surcharge")


class PredictionResponse(BaseModel):
    predicted_duration_minutes: float
    model_family: str
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_family: str
    model_version: str
    timestamp: str


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

model_loader: Optional[ModelLoader] = None
feature_pipeline: Optional[InferenceFeaturePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and feature pipeline on startup."""
    global model_loader, feature_pipeline
    try:
        model_loader = ModelLoader()
        feature_pipeline = InferenceFeaturePipeline()
        MODEL_LOADED.set(1)
        print(f"[inference] Model loaded: {model_loader.model_family} ({model_loader.model_version})")
    except Exception as e:
        print(f"[inference] Failed to load model: {e}")
        MODEL_LOADED.set(0)
    yield


app = FastAPI(
    title="NYC Yellow Taxi Trip Duration Predictor",
    description="Predict NYC yellow taxi trip duration using ML models",
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the web UI."""
    model_info = model_loader.get_info() if model_loader else {}
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "model_info": model_info},
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(trip: TripInput):
    """Predict trip duration for a single trip."""
    PREDICTION_REQUESTS.labels(endpoint="/predict").inc()
    IN_FLIGHT.inc()

    if not model_loader or not model_loader.is_loaded:
        PREDICTION_ERRORS.labels(error_type="model_not_loaded").inc()
        IN_FLIGHT.dec()
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        # Feature engineering
        feat_start = time.time()
        features = feature_pipeline.transform(trip.model_dump())
        FEATURE_PIPELINE_LATENCY.observe(time.time() - feat_start)

        # Model prediction
        infer_start = time.time()
        prediction = model_loader.predict(features)
        MODEL_INFERENCE_LATENCY.observe(time.time() - infer_start)

        predicted_minutes = float(prediction[0])
        predicted_minutes = max(0.0, predicted_minutes)

        PREDICTION_VALUE.observe(predicted_minutes)
        PREDICTION_LATENCY.observe(time.time() - start)

        return PredictionResponse(
            predicted_duration_minutes=round(predicted_minutes, 2),
            model_family=model_loader.model_family,
            model_version=model_loader.model_version,
        )

    except Exception as e:
        PREDICTION_ERRORS.labels(error_type=type(e).__name__).inc()
        PREDICTION_LATENCY.observe(time.time() - start)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        IN_FLIGHT.dec()


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(trips: List[TripInput]):
    """Predict trip duration for multiple trips."""
    PREDICTION_REQUESTS.labels(endpoint="/predict/batch").inc()
    IN_FLIGHT.inc()

    if not model_loader or not model_loader.is_loaded:
        PREDICTION_ERRORS.labels(error_type="model_not_loaded").inc()
        IN_FLIGHT.dec()
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(trips) > 100:
        IN_FLIGHT.dec()
        raise HTTPException(status_code=400, detail="Maximum 100 trips per batch")

    BATCH_SIZE.observe(len(trips))
    start = time.time()
    try:
        raw_inputs = [t.model_dump() for t in trips]

        feat_start = time.time()
        features = feature_pipeline.transform_batch(raw_inputs)
        FEATURE_PIPELINE_LATENCY.observe(time.time() - feat_start)

        infer_start = time.time()
        predictions = model_loader.predict(features)
        MODEL_INFERENCE_LATENCY.observe(time.time() - infer_start)

        results = []
        for pred in predictions:
            val = max(0.0, float(pred))
            PREDICTION_VALUE.observe(val)
            results.append(
                PredictionResponse(
                    predicted_duration_minutes=round(val, 2),
                    model_family=model_loader.model_family,
                    model_version=model_loader.model_version,
                )
            )

        PREDICTION_LATENCY.observe(time.time() - start)
        return BatchPredictionResponse(predictions=results)

    except Exception as e:
        PREDICTION_ERRORS.labels(error_type=type(e).__name__).inc()
        PREDICTION_LATENCY.observe(time.time() - start)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        IN_FLIGHT.dec()


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    loaded = model_loader is not None and model_loader.is_loaded
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        model_family=model_loader.model_family if loaded else "none",
        model_version=model_loader.model_version if loaded else "none",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/model/info")
async def model_info() -> Dict[str, Any]:
    """Return detailed model metadata."""
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_loader.get_info()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post("/model/reload")
async def reload_model():
    """Reload the model (after retraining)."""
    global model_loader, feature_pipeline
    try:
        model_loader = ModelLoader()
        feature_pipeline = InferenceFeaturePipeline()
        MODEL_LOADED.set(1)
        return {"status": "reloaded", "model": model_loader.get_info()}
    except Exception as e:
        MODEL_LOADED.set(0)
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
