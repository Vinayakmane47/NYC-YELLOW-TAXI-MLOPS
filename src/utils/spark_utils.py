import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType


## create a spark session
class SparkUtils:
    def __init__(
        self,
        app_name: str = "nyc-taxi-data-ingestion",
        extra_conf: Optional[Dict[str, str]] = None,
    ):
        self.app_name = app_name
        self.extra_conf = extra_conf or {}
        self.spark = self._create_spark_session()

    def _create_spark_session(self) -> SparkSession:
        """
        Create and configure Spark session with S3A/MinIO support.

        Returns:
            Configured SparkSession
        """
        minio_endpoint = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
        minio_access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        minio_secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")

        builder = (
            SparkSession.builder.appName(self.app_name)
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.driver.host", "localhost")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.hadoop.fs.s3a.endpoint", minio_endpoint)
            .config("spark.hadoop.fs.s3a.access.key", minio_access_key)
            .config("spark.hadoop.fs.s3a.secret.key", minio_secret_key)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config(
                "spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262",
            )
        )
        for key, value in self.extra_conf.items():
            builder = builder.config(key, value)
        return builder.getOrCreate()


def _safe_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool, str)):
        return value
    return str(value)


def collect_dataframe_metadata(df: DataFrame) -> Dict[str, Any]:
    schema = [
        {"name": field.name, "type": field.dataType.simpleString(), "nullable": field.nullable}
        for field in df.schema.fields
    ]
    row_count = df.count()

    null_exprs = [F.count(F.when(F.col(col).isNull(), col)).alias(col) for col in df.columns]
    null_counts_row = df.select(*null_exprs).collect()[0] if df.columns else None
    null_counts = {col: int(null_counts_row[col]) for col in df.columns} if null_counts_row is not None else {}

    numeric_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, NumericType)]
    stats: Dict[str, Dict[str, Any]] = {}
    if numeric_cols:
        agg_exprs = []
        for col in numeric_cols:
            agg_exprs.extend(
                [
                    F.min(F.col(col)).alias(f"{col}__min"),
                    F.max(F.col(col)).alias(f"{col}__max"),
                    F.mean(F.col(col)).alias(f"{col}__mean"),
                    F.stddev(F.col(col)).alias(f"{col}__stddev"),
                ]
            )
        stats_row = df.agg(*agg_exprs).collect()[0].asDict()
        for col in numeric_cols:
            stats[col] = {
                "min": _safe_json_value(stats_row.get(f"{col}__min")),
                "max": _safe_json_value(stats_row.get(f"{col}__max")),
                "mean": _safe_json_value(stats_row.get(f"{col}__mean")),
                "stddev": _safe_json_value(stats_row.get(f"{col}__stddev")),
            }

    return {
        "row_count": int(row_count),
        "column_count": len(df.columns),
        "schema": schema,
        "null_counts": null_counts,
        "column_stats": stats,
    }


def collect_file_sizes(paths: Iterable[Path]) -> Dict[str, Any]:
    file_entries = []
    total_bytes = 0
    for path in paths:
        if path.is_dir():
            size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        elif path.exists():
            size = path.stat().st_size
        else:
            size = 0
        total_bytes += size
        file_entries.append({"path": str(path), "bytes": int(size)})
    return {"total_bytes": int(total_bytes), "files": file_entries}


def write_metadata_json(metadata: Dict[str, Any], output_dir: Path, filename: str = "metadata.json") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / filename
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=_safe_json_value)
        handle.write("\n")


def get_pipeline_run_id(env_key: str = "PIPELINE_RUN_ID", strict: bool = True) -> str:
    """
    Read a shared pipeline run id from environment.

    In Airflow, this is injected once per DAG run (YYYYMMDD format from ds_nodash)
    and passed to every stage via ``set_pipeline_run_id``.

    For manual / CLI runs the fallback uses ``YYYYMMDD_HHMMSS`` so folder names
    stay human-readable and sort chronologically.
    """
    value = os.getenv(env_key, "").strip()
    if value:
        return value
    if strict:
        raise ValueError(
            f"Missing required environment variable '{env_key}'. "
            "Set it in Airflow so all stages write to the same metadata folder."
        )
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def get_pipeline_metadata_dir(
    pipeline_run_id: str,
    base_dir: str = "src/metadata",
) -> Path:
    metadata_dir = Path(base_dir) / f"pipeline_{pipeline_run_id}"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir


def build_stage_metadata(
    *,
    stage: str,
    pipeline_run_id: str,
    run_id: str,
    created_at_utc: str = "",
    data_rows: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    status: str = "success",
    error: Any = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "stage": stage,
        "pipeline_run_id": pipeline_run_id,
        "run_id": run_id,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "status": status,
        "data_rows": data_rows or {},
        "metrics": metrics or {},
        "artifacts": artifacts or {},
        "error": error,
    }
    return payload


def write_stage_metadata(
    *,
    stage_file_name: str,
    metadata: Dict[str, Any],
    pipeline_run_id: str,
    base_dir: str = "src/metadata",
) -> Path:
    """
    Write stage metadata atomically to:
    src/metadata/pipeline_<PIPELINE_RUN_ID>/<stage_file_name>
    """
    metadata_dir = get_pipeline_metadata_dir(pipeline_run_id=pipeline_run_id, base_dir=base_dir)
    metadata_path = metadata_dir / stage_file_name
    tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=False, default=_safe_json_value)
        handle.write("\n")
    tmp_path.replace(metadata_path)
    return metadata_path
