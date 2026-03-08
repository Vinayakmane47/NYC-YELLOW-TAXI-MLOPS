"""Locust load testing for NYC Taxi inference API."""

import random
from datetime import datetime, timedelta

from locust import HttpUser, between, task


def random_trip():
    """Generate a random trip payload matching TripInput schema."""
    now = datetime.now()
    pickup = now - timedelta(
        hours=random.randint(0, 72),
        minutes=random.randint(0, 59),
    )
    return {
        "pickup_datetime": pickup.strftime("%Y-%m-%dT%H:%M:%S"),
        "PULocationID": random.randint(1, 263),
        "DOLocationID": random.randint(1, 263),
        "trip_distance": round(random.uniform(0.5, 30.0), 2),
        "passenger_count": random.randint(1, 6),
        "VendorID": random.choice([1, 2]),
        "RatecodeID": random.randint(1, 6),
        "payment_type": random.randint(1, 4),
        "store_and_fwd_flag": random.choice(["Y", "N"]),
        "tolls_amount": round(random.choice([0.0, 0.0, 0.0, 6.55, 11.75]), 2),
        "Airport_fee": random.choice([0.0, 0.0, 1.75]),
        "congestion_surcharge": random.choice([0.0, 2.5, 2.75]),
    }


class TaxiInferenceUser(HttpUser):
    """Simulates a user making prediction requests to the inference API."""

    wait_time = between(0.5, 2)

    @task(5)
    def predict_single(self):
        """Single trip prediction."""
        self.client.post("/predict", json=random_trip())

    @task(2)
    def predict_batch(self):
        """Batch prediction with 5-20 trips."""
        batch_size = random.randint(5, 20)
        self.client.post(
            "/predict/batch",
            json=[random_trip() for _ in range(batch_size)],
        )

    @task(1)
    def health_check(self):
        """Health endpoint check."""
        self.client.get("/health")

    @task(1)
    def model_info(self):
        """Model info endpoint check."""
        self.client.get("/model/info")
