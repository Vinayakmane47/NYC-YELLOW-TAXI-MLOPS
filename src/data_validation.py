from datetime import datetime, timezone
from typing import List

from src.utils.io_utils import list_parquet_files
from src.utils.spark_utils import (
    SparkUtils,
    build_stage_metadata,
    get_pipeline_run_id,
    write_stage_metadata,
)


class DataValidation:
    """
    Validates raw ingested taxi parquet files before preprocessing.
    """

    REQUIRED_COLUMNS = [
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
        "total_amount",
    ]

    def __init__(
        self,
        input_dir: str = "s3a://bronze",
        app_name: str = "nyc-taxi-data-validation",
    ) -> None:
        self.input_dir = input_dir
        self.spark = SparkUtils(app_name=app_name).spark

    def run(self) -> None:
        run_start = datetime.now(timezone.utc)
        pipeline_run_id = get_pipeline_run_id()
        files = list_parquet_files(self.input_dir)

        status = "success"
        error = None
        missing_columns: List[str] = []
        sample_file = files[0] if files else ""
        row_count_sample = 0

        try:
            if not files:
                raise ValueError(f"No input parquet files found under {self.input_dir}")

            sample_df = self.spark.read.parquet(sample_file)
            row_count_sample = sample_df.count()
            missing_columns = [c for c in self.REQUIRED_COLUMNS if c not in sample_df.columns]
            if missing_columns:
                raise ValueError(
                    "Missing required columns in ingested data: "
                    + ", ".join(missing_columns)
                )
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = str(exc)
            raise
        finally:
            run_end = datetime.now(timezone.utc)
            metadata = build_stage_metadata(
                stage="data_validation",
                pipeline_run_id=pipeline_run_id,
                run_id=run_start.isoformat(),
                created_at_utc=run_end.isoformat(),
                data_rows={
                    "input_files": len(files),
                    "sample_file_rows": row_count_sample,
                },
                metrics={
                    "required_columns": self.REQUIRED_COLUMNS,
                    "missing_columns": missing_columns,
                    "duration_seconds": (run_end - run_start).total_seconds(),
                },
                artifacts={
                    "input_root_dir": self.input_dir,
                    "sample_file": sample_file,
                },
                status=status,
                error=error,
            )
            metadata_path = write_stage_metadata(
                stage_file_name="data_validation.json",
                metadata=metadata,
                pipeline_run_id=pipeline_run_id,
            )
            print(f"[data_validation] metadata saved to {metadata_path}")

    def close(self) -> None:
        try:
            if self.spark:
                self.spark.stop()
        except Exception:
            pass


def main() -> None:
    validator = DataValidation()
    try:
        validator.run()
    finally:
        validator.close()


if __name__ == "__main__":
    main()
