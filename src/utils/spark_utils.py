import json
from pathlib import Path
from typing import Dict, Iterable, List, Any

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import NumericType

## create a spark session
class SparkUtils:
    def __init__(self, app_name: str = "nyc-taxi-data-ingestion"):
        self.app_name = app_name
        self.spark = self._create_spark_session()

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
    null_counts = (
        {col: int(null_counts_row[col]) for col in df.columns} if null_counts_row is not None else {}
    )

    numeric_cols = [
        field.name for field in df.schema.fields if isinstance(field.dataType, NumericType)
    ]
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