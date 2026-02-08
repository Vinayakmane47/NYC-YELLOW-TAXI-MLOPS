import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from pyspark.sql import DataFrame, functions as F

from utils.spark_utils import (
    SparkUtils,
    collect_dataframe_metadata,
    collect_file_sizes,
    write_metadata_json,
)


class DataTransformation:
    def __init__(
        self,
        input_dir: str = "silver",
        output_dir: str = "gold",
        app_name: str = "nyc-taxi-data-transformation",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.spark = SparkUtils(app_name).spark

    def _list_input_files(self) -> List[Path]:
        if not self.input_dir.exists():
            return []
        return sorted(
            [path for path in self.input_dir.rglob("trip_*.parquet") if path.is_file()]
        )

    def _apply_feature_engineering(self, df: DataFrame) -> DataFrame:
        pickup_ts = F.col("tpep_pickup_datetime")

        df = (
            df.withColumn("pickup_hour", F.hour(pickup_ts))
            .withColumn("pickup_day_of_month", F.dayofmonth(pickup_ts))
            .withColumn("pickup_week_of_year", F.weekofyear(pickup_ts))
            .withColumn("pickup_month", F.month(pickup_ts))
            .withColumn("pickup_year", F.year(pickup_ts))
        )

        day_of_week = F.pmod(F.dayofweek(pickup_ts) + F.lit(5), F.lit(7))
        df = df.withColumn("pickup_day_of_week", day_of_week)

        df = (
            df.withColumn("is_weekend", F.col("pickup_day_of_week").isin(5, 6))
            .withColumn(
                "is_peak_hour",
                F.col("pickup_hour").isin(list(range(7, 11)) + list(range(16, 20))),
            )
            .withColumn("is_night", F.col("pickup_hour").isin([22, 23, 0, 1, 2, 3, 4, 5]))
            .withColumn("is_rush_hour", F.col("is_peak_hour"))
        )

        df = (
            df.withColumn(
                "sin_hour",
                F.sin(F.lit(2.0) * F.lit(3.141592653589793) * F.col("pickup_hour") / F.lit(24.0)),
            )
            .withColumn(
                "cos_hour",
                F.cos(F.lit(2.0) * F.lit(3.141592653589793) * F.col("pickup_hour") / F.lit(24.0)),
            )
            .withColumn(
                "sin_day_of_week",
                F.sin(
                    F.lit(2.0)
                    * F.lit(3.141592653589793)
                    * F.col("pickup_day_of_week")
                    / F.lit(7.0)
                ),
            )
            .withColumn(
                "cos_day_of_week",
                F.cos(
                    F.lit(2.0)
                    * F.lit(3.141592653589793)
                    * F.col("pickup_day_of_week")
                    / F.lit(7.0)
                ),
            )
        )

        df = (
            df.withColumn("is_short_trip", F.col("trip_distance") < 1)
            .withColumn("is_long_trip", F.col("trip_distance") > 10)
            .withColumn("log_trip_distance", F.log1p(F.col("trip_distance")))
            .withColumn("sqrt_trip_distance", F.sqrt(F.col("trip_distance")))
        )

        df = df.withColumn("PULocationID", F.col("PULocationID").cast("int")).withColumn(
            "DOLocationID", F.col("DOLocationID").cast("int")
        )

        df = df.withColumn(
            "route_id",
            F.concat_ws("_", F.col("PULocationID").cast("string"), F.col("DOLocationID").cast("string")),
        ).withColumn("is_same_zone", F.col("PULocationID") == F.col("DOLocationID"))

        pickup_counts = (
            df.groupBy("PULocationID")
            .agg(F.count("*").alias("pickup_zone_trip_count"))
        )
        dropoff_counts = (
            df.groupBy("DOLocationID")
            .agg(F.count("*").alias("dropoff_zone_trip_count"))
        )
        route_counts = df.groupBy("route_id").agg(F.count("*").alias("route_trip_count"))

        df = (
            df.join(pickup_counts, on="PULocationID", how="left")
            .join(dropoff_counts, on="DOLocationID", how="left")
            .join(route_counts, on="route_id", how="left")
        )

        df = df.withColumn(
            "fare_per_mile",
            F.when(F.col("trip_distance") > 0, F.col("fare_amount") / F.col("trip_distance")).otherwise(
                F.lit(None)
            ),
        )

        surcharge_total = (
            F.coalesce(F.col("extra"), F.lit(0))
            + F.coalesce(F.col("congestion_surcharge"), F.lit(0))
            + F.coalesce(F.col("Airport_fee"), F.lit(0))
            + F.coalesce(F.col("cbd_congestion_fee"), F.lit(0))
        )

        df = df.withColumn(
            "surcharge_ratio",
            F.when(F.col("total_amount") != 0, surcharge_total / F.col("total_amount")).otherwise(
                F.lit(None)
            ),
        )

        df = df.withColumn(
            "toll_ratio",
            F.when(F.col("total_amount") != 0, F.col("tolls_amount") / F.col("total_amount")).otherwise(
                F.lit(None)
            ),
        )

        df = (
            df.withColumn("has_tolls", (F.coalesce(F.col("tolls_amount"), F.lit(0)) > 0).cast("int"))
            .withColumn("has_airport_fee", (F.coalesce(F.col("Airport_fee"), F.lit(0)) > 0).cast("int"))
            .withColumn(
                "has_congestion_fee",
                (F.coalesce(F.col("congestion_surcharge"), F.lit(0)) > 0).cast("int"),
            )
            .withColumn("non_fare_amount", F.col("total_amount") - F.col("fare_amount"))
        )

        return df.drop("tpep_dropoff_datetime", "total_amount")

    def _write_single_parquet(self, df: DataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()

        temp_dir = Path(tempfile.mkdtemp(dir=str(output_path.parent)))
        try:
            df.coalesce(1).write.mode("overwrite").option("compression", "uncompressed").parquet(
                str(temp_dir)
            )
            part_files = list(temp_dir.glob("part-*.parquet"))
            if not part_files:
                raise RuntimeError(f"No parquet part file found in {temp_dir}")
            shutil.move(str(part_files[0]), str(output_path))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def process_all(self) -> None:
        run_start = datetime.now(timezone.utc)
        input_files = self._list_input_files()
        if not input_files:
            print(f"No parquet files found under {self.input_dir}")
            run_end = datetime.now(timezone.utc)
            metadata = {
                "stage": "transformation",
                "run_id": run_start.isoformat(),
                "started_at": run_start.isoformat(),
                "finished_at": run_end.isoformat(),
                "duration_seconds": (run_end - run_start).total_seconds(),
                "inputs": {"root_dir": str(self.input_dir), "files": []},
                "outputs": {"root_dir": str(self.output_dir), "files": [], "total_bytes": 0},
                "summary": {"processed_files": 0},
            }
            write_metadata_json(metadata, self.output_dir)
            return

        print(f"Found {len(input_files)} files. Writing features to {self.output_dir}")
        output_files: List[Path] = []
        for input_path in input_files:
            rel_path = input_path.relative_to(self.input_dir)
            output_path = self.output_dir / rel_path
            print(f"Processing {input_path} -> {output_path}")
            df = self.spark.read.parquet(str(input_path))
            transformed = self._apply_feature_engineering(df)
            self._write_single_parquet(transformed, output_path)
            output_files.append(output_path)

        run_end = datetime.now(timezone.utc)
        metadata = {
            "stage": "transformation",
            "run_id": run_start.isoformat(),
            "started_at": run_start.isoformat(),
            "finished_at": run_end.isoformat(),
            "duration_seconds": (run_end - run_start).total_seconds(),
            "inputs": {"root_dir": str(self.input_dir), **collect_file_sizes(input_files)},
            "outputs": {"root_dir": str(self.output_dir), **collect_file_sizes(output_files)},
            "summary": {"processed_files": len(output_files)},
        }

        if output_files:
            df = self.spark.read.parquet(*[str(path) for path in output_files])
            metadata["data_profile"] = collect_dataframe_metadata(df)

        write_metadata_json(metadata, self.output_dir)

        print("Data transformation completed.")

    def close(self) -> None:
        if self.spark:
            self.spark.stop()


def main() -> None:
    transformer = DataTransformation()
    try:
        transformer.process_all()
    finally:
        transformer.close()


if __name__ == "__main__":
    main()










