from datetime import datetime, timezone
from typing import List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.utils.io_utils import list_parquet_files, path_exists, write_single_parquet
from src.utils.spark_utils import (
    SparkUtils,
    build_stage_metadata,
    collect_dataframe_metadata,
    get_pipeline_run_id,
    write_stage_metadata,
)


class DataPreprocessing:
    def __init__(
        self,
        data_dir: str = "s3a://bronze",
        output_dir: str = "s3a://silver",
        app_name: str = "nyc-taxi-data-preprocessing",
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.spark = SparkUtils(app_name).spark

    def _clean_data(self, df: DataFrame) -> DataFrame:
        base = (
            df.withColumn(
                "trip_duration_min",
                (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime"))
                / 60.0,
            )
            .filter(F.col("trip_duration_min").isNotNull())
            .filter((F.col("trip_duration_min") > 0) & (F.col("trip_duration_min") < 180))
            .filter(F.col("trip_distance").isNotNull())
            .filter((F.col("trip_distance") > 0) & (F.col("trip_distance") < 60))
            .filter(F.col("total_amount").isNotNull())
            .filter(F.col("total_amount") >= 0)
            .select(
                "VendorID",
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "passenger_count",
                "trip_distance",
                "RatecodeID",
                "store_and_fwd_flag",
                "PULocationID",
                "DOLocationID",
                "payment_type",
                "fare_amount",
                "extra",
                "mta_tax",
                "tip_amount",
                "tolls_amount",
                "improvement_surcharge",
                "total_amount",
                "congestion_surcharge",
                "Airport_fee",
                "cbd_congestion_fee",
                "trip_duration_min",
            )
        )

        base = base.na.fill(
            {
                "passenger_count": 1,
                "RatecodeID": 1,
                "congestion_surcharge": 0,
                "Airport_fee": 0,
                "store_and_fwd_flag": "N",
            }
        )

        base = base.withColumn(
            "RatecodeID",
            F.when(F.col("RatecodeID") < 1, 1)
            .when(F.col("RatecodeID") > 6, 6)
            .otherwise(F.col("RatecodeID")),
        )

        base = base.filter(F.col("fare_amount") > 0).filter(F.col("fare_amount") < 500)

        base = base.withColumn(
            "tip_amount",
            F.when(F.col("tip_amount") < 0, 0)
            .when(F.col("tip_amount") > F.col("fare_amount") * 2, F.col("fare_amount") * 2)
            .otherwise(F.col("tip_amount")),
        )

        return base

    def process_all(self) -> None:
        run_start = datetime.now(timezone.utc)
        pipeline_run_id = get_pipeline_run_id()
        status = "success"
        error = None
        input_files = list_parquet_files(self.data_dir)
        if not input_files:
            print(f"No parquet files found under {self.data_dir}")
            run_end = datetime.now(timezone.utc)
            metadata = build_stage_metadata(
                stage="data_preprocessing",
                pipeline_run_id=pipeline_run_id,
                run_id=run_start.isoformat(),
                created_at_utc=run_end.isoformat(),
                data_rows={"processed_files": 0},
                metrics={"duration_seconds": (run_end - run_start).total_seconds()},
                artifacts={
                    "input_root_dir": self.data_dir,
                    "output_root_dir": self.output_dir,
                    "files": [],
                    "total_bytes": 0,
                },
                status=status,
                error=error,
            )
            write_stage_metadata(
                stage_file_name="data_preprocessing.json",
                metadata=metadata,
                pipeline_run_id=pipeline_run_id,
            )
            return

        output_files: List[str] = []
        skipped = 0
        data_profile = {}
        try:
            print(f"Found {len(input_files)} files. Writing cleaned data to {self.output_dir}")
            for input_path in input_files:
                rel_path = input_path.replace(self.data_dir, "").lstrip("/")
                output_path = f"{self.output_dir}/{rel_path}"

                if path_exists(output_path):
                    print(f"Skipping {rel_path} - already exists in output")
                    skipped += 1
                    output_files.append(output_path)
                    continue

                print(f"Processing {input_path} -> {output_path}")
                df = self.spark.read.parquet(input_path)
                cleaned = self._clean_data(df)
                write_single_parquet(cleaned, output_path)
                output_files.append(output_path)

            if output_files:
                # Profile only the last file to avoid OOM from loading all files
                sample_df = self.spark.read.parquet(output_files[-1])
                data_profile = collect_dataframe_metadata(sample_df)
                data_profile["profiled_file"] = output_files[-1]
                data_profile["total_output_files"] = len(output_files)
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = str(exc)
            raise
        finally:
            run_end = datetime.now(timezone.utc)
            metadata = build_stage_metadata(
                stage="data_preprocessing",
                pipeline_run_id=pipeline_run_id,
                run_id=run_start.isoformat(),
                created_at_utc=run_end.isoformat(),
                data_rows={
                    "input_files": len(input_files),
                    "processed_files": len(output_files),
                },
                metrics={
                    "duration_seconds": (run_end - run_start).total_seconds(),
                    "data_profile": data_profile,
                },
                artifacts={
                    "input_root_dir": self.data_dir,
                    "output_root_dir": self.output_dir,
                },
                status=status,
                error=error,
            )
            metadata_path = write_stage_metadata(
                stage_file_name="data_preprocessing.json",
                metadata=metadata,
                pipeline_run_id=pipeline_run_id,
            )
            print(f"[data_preprocessing] metadata saved to {metadata_path}")

        if skipped:
            print(f"Skipped {skipped} files (already exist in output)")
        print("Data preprocessing completed.")

    def close(self) -> None:
        try:
            if self.spark:
                self.spark.stop()
        except Exception:
            pass


def main() -> None:
    processor = DataPreprocessing()
    try:
        processor.process_all()
    finally:
        processor.close()


if __name__ == "__main__":
    main()
