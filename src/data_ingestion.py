import os
import shutil
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from pyspark.sql import SparkSession
from utils.spark_utils import (
    build_stage_metadata,
    collect_dataframe_metadata,
    collect_file_sizes,
    get_pipeline_run_id,
    write_stage_metadata,
)


class NYCDataIngestion:
    """
    Class to handle data ingestion for NYC Yellow Taxi data.
    Downloads data for each month in a given year and stores in parquet format.
    """
    
    BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    OUTPUT_BASE_DIR = "data"
    MONTH_NAMES = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    
    def __init__(self, year: int = 2025, app_name: str = "nyc-taxi-data-ingestion"):
        """
        Initialize the data ingestion class.
        
        Args:
            year: Year for which to download data (default: 2025)
            app_name: Spark application name
        """
        self.year = year
        self.app_name = app_name
        self.spark = self._create_spark_session()
        self.output_dir = Path(self.OUTPUT_BASE_DIR) / str(year)
        
    def _create_spark_session(self) -> SparkSession:
        """
        Create and configure Spark session.
        
        Returns:
            Configured SparkSession
        """
        return (SparkSession.builder
                .appName(self.app_name)
                .master("local[*]")
                .config("spark.sql.shuffle.partitions", "16")
                .config("spark.driver.host", "localhost")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .getOrCreate())
    
    def _get_month_urls(self) -> List[str]:
        """
        Generate URLs for all months in the specified year.
        
        Returns:
            List of URLs for each month
        """
        urls = []
        for month in range(1, 13):
            url = self.BASE_URL.format(year=self.year, month=month)
            urls.append(url)
        return urls
    
    def _get_month_name(self, month: int) -> str:
        """
        Get month name from month number.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Month name in lowercase
        """
        return self.MONTH_NAMES[month - 1]
    
    def _ensure_output_directory(self, month_dir: Path) -> None:
        """
        Create output directory if it doesn't exist.
        
        Args:
            month_dir: Path to month directory
        """
        month_dir.mkdir(parents=True, exist_ok=True)
    
    def _download_file(self, url: str, temp_path: str) -> bool:
        """
        Download a file from URL to temporary path in one go.
        
        Args:
            url: URL to download from
            temp_path: Temporary file path to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Download entire file at once
            urllib.request.urlretrieve(url, temp_path)
            return True
        except Exception as e:
            print(f"Error downloading file from {url}: {str(e)}")
            return False
    
    def _download_and_save_month(self, month: int) -> bool:
        """
        Download data for a specific month and save as parquet.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            True if successful, False otherwise
        """
        temp_file = None
        try:
            url = self.BASE_URL.format(year=self.year, month=month)
            month_name = self._get_month_name(month)
            print(f"Downloading data for {self.year}-{month:02d} ({month_name}) from {url}")
            
            # Create temporary file for download
            with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp:
                temp_file = tmp.name
            
            # Download entire file at once
            if not self._download_file(url, temp_file):
                return False
            
            # Read parquet using PySpark from temporary file
            df = self.spark.read.parquet(temp_file)
            
            # Create month directory structure: data/2025/january/
            month_dir = self.output_dir / month_name
            self._ensure_output_directory(month_dir)
            
            # Create output path: data/2025/january/trip_01.parquet
            output_path = month_dir / f"trip_{month:02d}.parquet"
            
            # Create temporary directory for Spark to write to
            temp_output_dir = month_dir / f"temp_{month:02d}"
            
            # Coalesce to 1 partition and save as uncompressed parquet (no snappy compression)
            df.coalesce(1).write.mode("overwrite").option("compression", "uncompressed").parquet(str(temp_output_dir))
            
            # Find the part file and move it to the final location
            part_files = list(temp_output_dir.glob("part-*.parquet"))
            if part_files:
                # Move the part file to the final location
                shutil.move(str(part_files[0]), str(output_path))
                # Remove the temporary directory
                shutil.rmtree(temp_output_dir, ignore_errors=True)
            else:
                raise Exception(f"No part file found in {temp_output_dir}")
            
            row_count = df.count()
            print(f"Successfully saved {row_count} rows to {output_path}")
            
            return True
            
        except Exception as e:
            print(f"Error processing data for {self.year}-{month:02d}: {str(e)}")
            return False
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
    
    def ingest_all_months(self) -> None:
        """
        Download and save data for all months in the specified year.
        """
        run_start = datetime.now(timezone.utc)
        pipeline_run_id = get_pipeline_run_id()
        status = "success"
        error = None
        output_files: List[Path] = []
        input_urls: List[str] = []
        print(f"Starting data ingestion for year {self.year}")
        # Ensure base year directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        failed = 0
        
        for month in range(1, 13):
            input_urls.append(self.BASE_URL.format(year=self.year, month=month))
            if self._download_and_save_month(month):
                successful += 1
                month_name = self._get_month_name(month)
                output_files.append(self.output_dir / month_name / f"trip_{month:02d}.parquet")
            else:
                failed += 1
        
        run_end = datetime.now(timezone.utc)
        metrics = {
            "year": self.year,
            "successful_months": successful,
            "failed_months": failed,
            "duration_seconds": (run_end - run_start).total_seconds(),
            "inputs": {"urls": input_urls},
        }
        if output_files:
            df = self.spark.read.parquet(*[str(path) for path in output_files])
            metrics["data_profile"] = collect_dataframe_metadata(df)
        metadata = build_stage_metadata(
            stage="data_ingestion",
            pipeline_run_id=pipeline_run_id,
            run_id=run_start.isoformat(),
            created_at_utc=run_end.isoformat(),
            data_rows={
                "ingested_files": len(output_files),
            },
            metrics=metrics,
            artifacts={
                "output_root_dir": str(self.output_dir),
                **collect_file_sizes(output_files),
            },
            status=status,
            error=error,
        )
        metadata_path = write_stage_metadata(
            stage_file_name="data_ingestion.json",
            metadata=metadata,
            pipeline_run_id=pipeline_run_id,
        )

        print(f"\nData ingestion completed!")
        print(f"Successful: {successful}/12 months")
        print(f"Failed: {failed}/12 months")
        print(f"Data saved to: {self.output_dir}")
        print(f"Metadata saved to: {metadata_path}")
    
    def close(self) -> None:
        """
        Close the Spark session.
        """
        if self.spark:
            self.spark.stop()


def main():
    """
    Main function to run data ingestion.
    """
    ingester = NYCDataIngestion(year=2025)
    try:
        ingester.ingest_all_months()
    finally:
        ingester.close()


if __name__ == "__main__":
    main()
