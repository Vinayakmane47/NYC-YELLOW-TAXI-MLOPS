import json
from datetime import datetime, timezone
from typing import Any, Dict

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.utils.io_utils import list_parquet_files, write_single_parquet
from src.utils.spark_utils import (
    SparkUtils,
    build_stage_metadata,
    get_pipeline_run_id,
    write_stage_metadata,
)


class MLDataTransformer:
    """
    Transforms gold-level data into ML-ready train/test/val splits using PySpark.

    Steps:
    1. Load parquet data (limit to 1M rows)
    2. Drop leakage features
    3. Drop non-predictive features
    4. Encode categorical features (binary, one-hot, target encoding)
    5. Scale numerical features
    6. Time-based split (70% train, 15% test, 15% val)
    7. Save to ml-transformed bucket as parquet
    """

    def __init__(
        self,
        input_dir: str = "s3a://gold",
        output_dir: str = "s3a://ml-transformed",
        app_name: str = "nyc-taxi-ml-transformation",
        max_rows: int = 1_000_000,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_rows = max_rows

        self.spark = SparkUtils(
            app_name,
            extra_conf={
                "spark.driver.memory": "4g",
                "spark.executor.memory": "4g",
                "spark.sql.shuffle.partitions": "8",
            },
        ).spark

        # Feature definitions
        self.leakage_cols = [
            "fare_amount",
            "tip_amount",
            "tolls_amount",
            "extra",
            "mta_tax",
            "improvement_surcharge",
            "congestion_surcharge",
            "Airport_fee",
            "cbd_congestion_fee",
            "fare_per_mile",
            "surcharge_ratio",
            "toll_ratio",
            "non_fare_amount",
        ]

        self.drop_cols = ["tpep_pickup_datetime", "route_id"]

        self.binary_encode_cols = ["store_and_fwd_flag"]
        self.onehot_cols = ["VendorID", "RatecodeID", "payment_type"]
        self.target_encode_cols = ["DOLocationID", "PULocationID"]

        self.scale_cols = [
            "trip_distance",
            "log_trip_distance",
            "sqrt_trip_distance",
            "passenger_count",
            "pickup_zone_trip_count",
            "dropoff_zone_trip_count",
            "route_trip_count",
        ]

        self.scaling_stats = {}
        self.target_encoders = {}

    def load_data(self) -> DataFrame:
        """Load parquet data from gold bucket (limit to max_rows)."""
        print(f"Loading data from {self.input_dir} (max {self.max_rows:,} rows)...")

        parquet_files = list_parquet_files(self.input_dir)

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.input_dir}")

        print(f"Found {len(parquet_files)} parquet files")

        print(f"Reading from first file: {parquet_files[0]}")
        df = self.spark.read.parquet(parquet_files[0])

        total_rows = df.count()
        print(f"Total rows in file: {total_rows:,}")

        if total_rows > self.max_rows:
            print(f"Limiting to {self.max_rows:,} rows...")
            df = df.limit(self.max_rows)

        df = df.withColumn("row_id", F.monotonically_increasing_id())

        row_count = df.count()
        col_count = len(df.columns)

        print(f"Loaded {row_count:,} rows with {col_count} columns")
        return df

    def drop_features(self, df: DataFrame) -> DataFrame:
        """Drop leakage and non-predictive features."""
        print("\nDropping features...")

        cols_to_drop = list(set(self.leakage_cols + self.drop_cols))
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]

        print(f"  Dropping {len(existing_cols_to_drop)} columns")
        df = df.drop(*existing_cols_to_drop)

        print(f"  Remaining columns: {len(df.columns)}")
        return df

    def encode_categorical_features(self, df: DataFrame, is_training: bool = True) -> DataFrame:
        print("\nEncoding categorical features...")

        if "store_and_fwd_flag" in df.columns:
            print("  Binary encoding: store_and_fwd_flag")
            df = df.withColumn("store_and_fwd_flag", F.when(F.col("store_and_fwd_flag") == "Y", 1).otherwise(0))

        onehot_existing = [col for col in self.onehot_cols if col in df.columns]
        if onehot_existing:
            print(f"  One-hot encoding: {onehot_existing}")

            for col in onehot_existing:
                if is_training:
                    distinct_vals = [row[0] for row in df.select(col).distinct().collect() if row[0] is not None]
                    distinct_vals = sorted(distinct_vals)[1:]
                    self.target_encoders[f"{col}_categories"] = distinct_vals
                else:
                    distinct_vals = self.target_encoders.get(f"{col}_categories", [])

                for val in distinct_vals:
                    df = df.withColumn(f"{col}_{val}", F.when(F.col(col) == val, 1).otherwise(0))

                df = df.drop(col)

        target_col = "trip_duration_min"
        if target_col in df.columns:
            for col in self.target_encode_cols:
                if col in df.columns:
                    print(f"  Target encoding: {col}")

                    if is_training:
                        target_means = df.groupBy(col).agg(F.mean(target_col).alias("mean_target"))
                        global_mean = df.select(F.mean(target_col)).collect()[0][0]

                        self.target_encoders[col] = {
                            "mapping": {row[col]: row["mean_target"] for row in target_means.collect()},
                            "global_mean": global_mean,
                        }

                    if col in self.target_encoders:
                        encoder_info = self.target_encoders[col]

                        mapping_expr = F.lit(encoder_info["global_mean"])
                        for category, mean_val in encoder_info["mapping"].items():
                            mapping_expr = F.when(F.col(col) == category, mean_val).otherwise(mapping_expr)

                        df = df.withColumn(f"{col}_encoded", mapping_expr)

                    df = df.drop(col)

        return df

    def scale_numerical_features(self, df: DataFrame, is_training: bool = True) -> DataFrame:
        print("\nScaling numerical features...")

        cols_to_scale = [col for col in self.scale_cols if col in df.columns]

        if not cols_to_scale:
            print("  No columns to scale")
            return df

        print(f"  Scaling {len(cols_to_scale)} columns: {cols_to_scale}")

        if is_training:
            for col in cols_to_scale:
                stats = df.select(F.mean(col).alias("mean"), F.stddev(col).alias("stddev")).collect()[0]

                self.scaling_stats[col] = {
                    "mean": stats["mean"] if stats["mean"] is not None else 0.0,
                    "stddev": stats["stddev"] if stats["stddev"] is not None and stats["stddev"] > 0 else 1.0,
                }

        for col in cols_to_scale:
            if col in self.scaling_stats:
                stats = self.scaling_stats[col]
                df = df.withColumn(col, (F.col(col) - F.lit(stats["mean"])) / F.lit(stats["stddev"]))

        return df

    def time_based_split(self, df: DataFrame) -> tuple:
        print("\nPerforming time-based split...")

        df.cache()
        total_count = df.count()

        max_row = df.select(F.max("row_id")).collect()[0][0]
        train_threshold = max_row * 0.70
        test_threshold = max_row * 0.85

        print(f"  Total rows: {total_count:,}")
        print(
            f"  Split thresholds - Train: 0-{train_threshold:.0f}, Test: {train_threshold:.0f}-{test_threshold:.0f}, Val: {test_threshold:.0f}-{max_row:.0f}"
        )

        train_df = df.filter(F.col("row_id") <= train_threshold).drop("row_id")
        test_df = df.filter((F.col("row_id") > train_threshold) & (F.col("row_id") <= test_threshold)).drop("row_id")
        val_df = df.filter(F.col("row_id") > test_threshold).drop("row_id")

        train_df.cache()
        test_df.cache()
        val_df.cache()

        train_actual = train_df.count()
        test_actual = test_df.count()
        val_actual = val_df.count()

        print(f"  Train: {train_actual:,} rows ({train_actual / total_count * 100:.1f}%)")
        print(f"  Test:  {test_actual:,} rows ({test_actual / total_count * 100:.1f}%)")
        print(f"  Val:   {val_actual:,} rows ({val_actual / total_count * 100:.1f}%)")

        df.unpersist()

        return train_df, test_df, val_df

    def save_splits(
        self,
        train_df: DataFrame,
        test_df: DataFrame,
        val_df: DataFrame,
        pipeline_run_id: str,
        duration_seconds: float,
    ) -> Dict[str, Any]:
        """Save train/test/val splits to ml-transformed bucket as parquet."""
        print(f"\nSaving splits to {self.output_dir}...")

        train_path = f"{self.output_dir}/train.parquet"
        test_path = f"{self.output_dir}/test.parquet"
        val_path = f"{self.output_dir}/val.parquet"

        train_count = train_df.count()
        test_count = test_df.count()
        val_count = val_df.count()

        print("  Writing train.parquet...")
        write_single_parquet(train_df, train_path, compression="snappy")

        print("  Writing test.parquet...")
        write_single_parquet(test_df, test_path, compression="snappy")

        print("  Writing val.parquet...")
        write_single_parquet(val_df, val_path, compression="snappy")

        train_df.unpersist()
        test_df.unpersist()
        val_df.unpersist()

        print(f"  Saved: {train_path}")
        print(f"  Saved: {test_path}")
        print(f"  Saved: {val_path}")

        # Build and write standard pipeline metadata
        metadata = build_stage_metadata(
            stage="ml_transformation",
            pipeline_run_id=pipeline_run_id,
            run_id=datetime.now(timezone.utc).isoformat(),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            data_rows={
                "train_rows": train_count,
                "test_rows": test_count,
                "val_rows": val_count,
                "max_rows": self.max_rows,
            },
            metrics={
                "duration_seconds": duration_seconds,
                "features": {
                    "total_columns": len(train_df.columns),
                    "column_names": train_df.columns,
                    "dropped_leakage": self.leakage_cols,
                    "dropped_other": self.drop_cols,
                    "binary_encoded": self.binary_encode_cols,
                    "onehot_encoded": self.onehot_cols,
                    "target_encoded": self.target_encode_cols,
                    "scaled": self.scale_cols,
                },
                "scaling_stats": {
                    k: {"mean": float(v["mean"]), "stddev": float(v["stddev"])}
                    for k, v in self.scaling_stats.items()
                },
                "target_encoders": {
                    k: {"global_mean": float(v["global_mean"]), "num_categories": len(v["mapping"])}
                    for k, v in self.target_encoders.items()
                    if isinstance(v, dict) and "global_mean" in v
                },
            },
            artifacts={
                "input_dir": self.input_dir,
                "output_dir": self.output_dir,
                "train_path": train_path,
                "test_path": test_path,
                "val_path": val_path,
            },
            status="success",
            error=None,
        )
        metadata_path = write_stage_metadata(
            stage_file_name="ml_transformation.json",
            metadata=metadata,
            pipeline_run_id=pipeline_run_id,
        )
        print(f"  Saved metadata: {metadata_path}")

        return metadata

    def process(self) -> Dict[str, Any]:
        """Execute the full ML transformation pipeline."""
        start_time = datetime.now(timezone.utc)
        pipeline_run_id = get_pipeline_run_id(strict=False)
        print("=" * 80)
        print("ML DATA TRANSFORMATION PIPELINE (PySpark)")
        print(f"  pipeline_run_id={pipeline_run_id}")
        print("=" * 80)

        df = self.load_data()
        df = self.drop_features(df)
        train_df, test_df, val_df = self.time_based_split(df)

        train_df = self.encode_categorical_features(train_df, is_training=True)
        test_df = self.encode_categorical_features(test_df, is_training=False)
        val_df = self.encode_categorical_features(val_df, is_training=False)

        train_df = self.scale_numerical_features(train_df, is_training=True)
        test_df = self.scale_numerical_features(test_df, is_training=False)
        val_df = self.scale_numerical_features(val_df, is_training=False)

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        metadata = self.save_splits(train_df, test_df, val_df, pipeline_run_id, duration)

        print("\n" + "=" * 80)
        print(f"Pipeline completed in {duration:.2f} seconds")
        print("=" * 80)

        return metadata

    def close(self) -> None:
        try:
            if self.spark:
                self.spark.stop()
        except Exception:
            pass


def main() -> None:
    transformer = MLDataTransformer()
    try:
        metadata = transformer.process()
        print("\n" + "=" * 80)
        print("METADATA SUMMARY")
        print("=" * 80)
        print(json.dumps(metadata, indent=2))
    finally:
        transformer.close()


if __name__ == "__main__":
    main()
