import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from pyspark.sql import DataFrame, functions as F

from utils.spark_utils import (
    SparkUtils,
    collect_dataframe_metadata,
    collect_file_sizes,
    write_metadata_json,
)


class DataPreprocessing:
    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "silver",
        app_name: str = "nyc-taxi-data-preprocessing",
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.spark = SparkUtils(app_name).spark

    def _list_input_files(self) -> List[Path]:
        if not self.data_dir.exists():
            return []
        return sorted(
            [path for path in self.data_dir.rglob("trip_*.parquet") if path.is_file()]
        )

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
            print(f"No parquet files found under {self.data_dir}")
            run_end = datetime.now(timezone.utc)
            metadata = {
                "stage": "preprocessing",
                "run_id": run_start.isoformat(),
                "started_at": run_start.isoformat(),
                "finished_at": run_end.isoformat(),
                "duration_seconds": (run_end - run_start).total_seconds(),
                "inputs": {"root_dir": str(self.data_dir), "files": []},
                "outputs": {"root_dir": str(self.output_dir), "files": [], "total_bytes": 0},
                "summary": {"processed_files": 0},
            }
            write_metadata_json(metadata, self.output_dir)
            return

        print(f"Found {len(input_files)} files. Writing cleaned data to {self.output_dir}")
        output_files: List[Path] = []
        for input_path in input_files:
            rel_path = input_path.relative_to(self.data_dir)
            output_path = self.output_dir / rel_path
            print(f"Processing {input_path} -> {output_path}")
            df = self.spark.read.parquet(str(input_path))
            cleaned = self._clean_data(df)
            self._write_single_parquet(cleaned, output_path)
            output_files.append(output_path)

        run_end = datetime.now(timezone.utc)
        metadata = {
            "stage": "preprocessing",
            "run_id": run_start.isoformat(),
            "started_at": run_start.isoformat(),
            "finished_at": run_end.isoformat(),
            "duration_seconds": (run_end - run_start).total_seconds(),
            "inputs": {"root_dir": str(self.data_dir), **collect_file_sizes(input_files)},
            "outputs": {"root_dir": str(self.output_dir), **collect_file_sizes(output_files)},
            "summary": {"processed_files": len(output_files)},
        }

        if output_files:
            df = self.spark.read.parquet(*[str(path) for path in output_files])
            metadata["data_profile"] = collect_dataframe_metadata(df)

        write_metadata_json(metadata, self.output_dir)

        print("Data preprocessing completed.")

    def close(self) -> None:
        if self.spark:
            self.spark.stop()


def main() -> None:
    processor = DataPreprocessing()
    try:
        processor.process_all()
    finally:
        processor.close()


if __name__ == "__main__":
    main()













