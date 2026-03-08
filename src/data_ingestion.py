import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from typing import List

import boto3

from src.utils.spark_utils import (
    SparkUtils,
    build_stage_metadata,
    get_pipeline_run_id,
    write_stage_metadata,
)


class NYCDataIngestion:
    """
    Class to handle data ingestion for NYC Yellow Taxi data.
    Downloads data for each month in a given year and stores in MinIO (S3A).
    """

    BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    OUTPUT_BASE_DIR = "s3a://bronze"
    MONTH_NAMES = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]

    def __init__(self, year: int = 2025, app_name: str = "nyc-taxi-data-ingestion"):
        self.year = year
        self.app_name = app_name
        self.spark = SparkUtils(app_name).spark
        self.output_dir = f"{self.OUTPUT_BASE_DIR}/{year}"
        self._s3 = boto3.client(
            "s3",
            endpoint_url=os.environ.get("MINIO_ENDPOINT", "http://localhost:9000"),
            aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        )

    def _get_month_name(self, month: int) -> str:
        return self.MONTH_NAMES[month - 1]

    def _exists_in_minio(self, month: int) -> bool:
        """Check if data for a given month already exists in MinIO."""
        month_name = self._get_month_name(month)
        prefix = f"{self.year}/{month_name}/trip_{month:02d}.parquet/"
        try:
            resp = self._s3.list_objects_v2(Bucket="bronze", Prefix=prefix, MaxKeys=1)
            return resp.get("KeyCount", 0) > 0
        except Exception:
            return False

    def _download_file(self, url: str, temp_path: str) -> bool:
        try:
            urllib.request.urlretrieve(url, temp_path)
            return True
        except Exception as e:
            print(f"Error downloading file from {url}: {str(e)}")
            return False

    def _download_and_save_month(self, month: int) -> int:
        """Download and save a month's data. Returns row count on success, -1 on failure."""
        temp_file = None
        try:
            url = self.BASE_URL.format(year=self.year, month=month)
            month_name = self._get_month_name(month)
            print(f"Downloading data for {self.year}-{month:02d} ({month_name}) from {url}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
                temp_file = tmp.name

            if not self._download_file(url, temp_file):
                return -1

            df = self.spark.read.parquet(temp_file)
            row_count = df.count()

            # Write to MinIO: s3a://bronze/2025/january/trip_01.parquet
            output_path = f"{self.output_dir}/{month_name}/trip_{month:02d}.parquet"
            df.coalesce(1).write.mode("overwrite").option("compression", "uncompressed").parquet(output_path)

            print(f"Successfully saved {row_count:,} rows to {output_path}")

            return row_count

        except Exception as e:
            print(f"Error processing data for {self.year}-{month:02d}: {str(e)}")
            return -1
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

    def ingest_all_months(self) -> None:
        run_start = datetime.now(timezone.utc)
        pipeline_run_id = get_pipeline_run_id()
        status = "success"
        error = None
        output_paths: List[str] = []
        input_urls: List[str] = []
        print(f"Starting data ingestion for year {self.year}")

        successful = 0
        failed = 0
        skipped = 0
        total_rows = 0

        for month in range(1, 13):
            input_urls.append(self.BASE_URL.format(year=self.year, month=month))
            month_name = self._get_month_name(month)
            output_path = f"{self.output_dir}/{month_name}/trip_{month:02d}.parquet"

            if self._exists_in_minio(month):
                print(f"Skipping {self.year}-{month:02d} ({month_name}) - already exists in MinIO")
                skipped += 1
                output_paths.append(output_path)
                continue

            row_count = self._download_and_save_month(month)
            if row_count >= 0:
                successful += 1
                total_rows += row_count
                output_paths.append(output_path)
            else:
                failed += 1

        run_end = datetime.now(timezone.utc)
        metrics = {
            "year": self.year,
            "successful_months": successful,
            "failed_months": failed,
            "skipped_months": skipped,
            "total_rows_ingested": total_rows,
            "duration_seconds": (run_end - run_start).total_seconds(),
            "inputs": {"urls": input_urls},
        }
        metadata = build_stage_metadata(
            stage="data_ingestion",
            pipeline_run_id=pipeline_run_id,
            run_id=run_start.isoformat(),
            created_at_utc=run_end.isoformat(),
            data_rows={
                "ingested_files": len(output_paths),
            },
            metrics=metrics,
            artifacts={
                "output_root_dir": self.output_dir,
            },
            status=status,
            error=error,
        )
        metadata_path = write_stage_metadata(
            stage_file_name="data_ingestion.json",
            metadata=metadata,
            pipeline_run_id=pipeline_run_id,
        )

        print("\nData ingestion completed!")
        print(f"Downloaded: {successful}/12 months")
        print(f"Skipped (already in MinIO): {skipped}/12 months")
        print(f"Failed: {failed}/12 months")
        print(f"Data saved to: {self.output_dir}")
        print(f"Metadata saved to: {metadata_path}")

    def close(self) -> None:
        try:
            if self.spark:
                self.spark.stop()
        except Exception:
            pass


def main():
    ingester = NYCDataIngestion(year=2025)
    try:
        ingester.ingest_all_months()
    finally:
        ingester.close()


if __name__ == "__main__":
    main()
