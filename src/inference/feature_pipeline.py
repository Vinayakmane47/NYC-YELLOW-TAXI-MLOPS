"""Feature pipeline for inference - replicates training transformations in pure pandas."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# Feature column order must match training exactly
FEATURE_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "store_and_fwd_flag",
    "pickup_hour",
    "pickup_day_of_month",
    "pickup_week_of_year",
    "pickup_month",
    "pickup_year",
    "pickup_day_of_week",
    "is_weekend",
    "is_peak_hour",
    "is_night",
    "is_rush_hour",
    "sin_hour",
    "cos_hour",
    "sin_day_of_week",
    "cos_day_of_week",
    "is_short_trip",
    "is_long_trip",
    "log_trip_distance",
    "sqrt_trip_distance",
    "is_same_zone",
    "pickup_zone_trip_count",
    "dropoff_zone_trip_count",
    "route_trip_count",
    "has_tolls",
    "has_airport_fee",
    "has_congestion_fee",
    "VendorID_2",
    "VendorID_6",
    "RatecodeID_2",
    "RatecodeID_3",
    "RatecodeID_4",
    "RatecodeID_5",
    "RatecodeID_6",
    "payment_type_1",
    "payment_type_2",
    "payment_type_3",
    "payment_type_4",
    "DOLocationID_encoded",
    "PULocationID_encoded",
]

# Default scaling stats (from training pipeline)
DEFAULT_SCALING_STATS = {
    "trip_distance": {"mean": 3.557, "stddev": 3.868},
    "log_trip_distance": {"mean": 1.256, "stddev": 0.680},
    "sqrt_trip_distance": {"mean": 1.673, "stddev": 0.871},
    "passenger_count": {"mean": 1.272, "stddev": 0.710},
    "pickup_zone_trip_count": {"mean": 104425.341, "stddev": 48951.576},
    "dropoff_zone_trip_count": {"mean": 73331.144, "stddev": 43604.781},
    "route_trip_count": {"mean": 2939.174, "stddev": 2662.820},
}

DEFAULT_TARGET_ENCODER_GLOBAL_MEAN = 16.975


class InferenceFeaturePipeline:
    """Transforms raw trip input into model-ready features."""

    def __init__(self, metadata_dir: Optional[str] = None):
        self.scaling_stats = dict(DEFAULT_SCALING_STATS)
        self.target_encoder_global_mean = DEFAULT_TARGET_ENCODER_GLOBAL_MEAN

        if metadata_dir:
            self._load_metadata(metadata_dir)
        else:
            self._auto_load_metadata()

    def _auto_load_metadata(self) -> None:
        """Find and load the latest ml_transformation.json."""
        metadata_base = Path("src/metadata")
        if not metadata_base.exists():
            return

        pipeline_dirs = sorted(metadata_base.glob("pipeline_*"), reverse=True)
        for pdir in pipeline_dirs:
            ml_meta = pdir / "ml_transformation.json"
            if ml_meta.exists():
                self._load_from_file(ml_meta)
                return

    def _load_metadata(self, metadata_dir: str) -> None:
        ml_meta = Path(metadata_dir) / "ml_transformation.json"
        if ml_meta.exists():
            self._load_from_file(ml_meta)

    def _load_from_file(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)

        metrics = meta.get("metrics", {})
        scaling = metrics.get("scaling_stats", {})
        if scaling:
            self.scaling_stats = scaling

        target_encoders = metrics.get("target_encoders", {})
        for col_info in target_encoders.values():
            if isinstance(col_info, dict) and "global_mean" in col_info:
                self.target_encoder_global_mean = col_info["global_mean"]
                break

    def transform(self, raw_input: Dict[str, Any]) -> pd.DataFrame:
        """Transform a single raw input dict into a model-ready DataFrame."""
        return self.transform_batch([raw_input])

    def transform_batch(self, raw_inputs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Transform a list of raw input dicts into a model-ready DataFrame."""
        df = pd.DataFrame(raw_inputs)
        df = self._add_time_features(df)
        df = self._add_distance_features(df)
        df = self._add_zone_features(df)
        df = self._add_boolean_features(df)
        df = self._encode_binary(df)
        df = self._encode_onehot(df)
        df = self._encode_target(df)
        df = self._scale_features(df)
        df = self._align_columns(df)
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dt = pd.to_datetime(df["pickup_datetime"])

        df["pickup_hour"] = dt.dt.hour
        df["pickup_day_of_month"] = dt.dt.day
        df["pickup_week_of_year"] = dt.dt.isocalendar().week.astype(int)
        df["pickup_month"] = dt.dt.month
        df["pickup_year"] = dt.dt.year

        # Monday=0 convention matching training
        df["pickup_day_of_week"] = (dt.dt.dayofweek).astype(int)

        df["is_weekend"] = df["pickup_day_of_week"].isin([5, 6]).astype(int)

        peak_hours = list(range(7, 11)) + list(range(16, 20))
        df["is_peak_hour"] = df["pickup_hour"].isin(peak_hours).astype(int)

        night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
        df["is_night"] = df["pickup_hour"].isin(night_hours).astype(int)

        df["is_rush_hour"] = df["is_peak_hour"]

        # Cyclical encodings
        df["sin_hour"] = np.sin(2.0 * math.pi * df["pickup_hour"] / 24.0)
        df["cos_hour"] = np.cos(2.0 * math.pi * df["pickup_hour"] / 24.0)
        df["sin_day_of_week"] = np.sin(2.0 * math.pi * df["pickup_day_of_week"] / 7.0)
        df["cos_day_of_week"] = np.cos(2.0 * math.pi * df["pickup_day_of_week"] / 7.0)

        return df

    def _add_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dist = df["trip_distance"].astype(float)
        df["is_short_trip"] = (dist < 1.0).astype(int)
        df["is_long_trip"] = (dist > 10.0).astype(int)
        df["log_trip_distance"] = np.log1p(dist)
        df["sqrt_trip_distance"] = np.sqrt(dist.clip(lower=0))
        return df

    def _add_zone_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["is_same_zone"] = (df["PULocationID"] == df["DOLocationID"]).astype(int)
        # Use training-time averages as defaults
        df["pickup_zone_trip_count"] = self.scaling_stats.get(
            "pickup_zone_trip_count", DEFAULT_SCALING_STATS["pickup_zone_trip_count"]
        )["mean"]
        df["dropoff_zone_trip_count"] = self.scaling_stats.get(
            "dropoff_zone_trip_count", DEFAULT_SCALING_STATS["dropoff_zone_trip_count"]
        )["mean"]
        df["route_trip_count"] = self.scaling_stats.get(
            "route_trip_count", DEFAULT_SCALING_STATS["route_trip_count"]
        )["mean"]
        return df

    def _add_boolean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["has_tolls"] = (df.get("tolls_amount", pd.Series([0.0] * len(df))).astype(float) > 0).astype(int)
        df["has_airport_fee"] = (df.get("Airport_fee", pd.Series([0.0] * len(df))).astype(float) > 0).astype(int)
        df["has_congestion_fee"] = (
            df.get("congestion_surcharge", pd.Series([0.0] * len(df))).astype(float) > 0
        ).astype(int)
        return df

    def _encode_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        if "store_and_fwd_flag" in df.columns:
            df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)
        else:
            df["store_and_fwd_flag"] = 0
        return df

    def _encode_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        # VendorID: categories 2, 6 (first category 1 is dropped)
        vendor = df.get("VendorID", pd.Series([1] * len(df)))
        df["VendorID_2"] = (vendor == 2).astype(int)
        df["VendorID_6"] = (vendor == 6).astype(int)

        # RatecodeID: categories 2-6 (first category 1 is dropped)
        ratecode = df.get("RatecodeID", pd.Series([1] * len(df)))
        for val in [2, 3, 4, 5, 6]:
            df[f"RatecodeID_{val}"] = (ratecode == val).astype(int)

        # payment_type: categories 1-4 (first category 0 is dropped)
        payment = df.get("payment_type", pd.Series([1] * len(df)))
        for val in [1, 2, 3, 4]:
            df[f"payment_type_{val}"] = (payment == val).astype(int)

        return df

    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        # Use global mean for all locations at inference time
        df["DOLocationID_encoded"] = self.target_encoder_global_mean
        df["PULocationID_encoded"] = self.target_encoder_global_mean
        return df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, stats in self.scaling_stats.items():
            if col in df.columns:
                mean = stats["mean"]
                stddev = stats["stddev"] if stats["stddev"] > 0 else 1.0
                df[col] = (df[col].astype(float) - mean) / stddev
        return df

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure columns match training order exactly."""
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        return df[FEATURE_COLUMNS]
