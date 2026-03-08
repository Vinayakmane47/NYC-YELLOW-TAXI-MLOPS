"""Shared I/O utilities for pipeline stages."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

import boto3
from pyspark.sql import DataFrame


def _is_s3_path(path: Union[str, Path]) -> bool:
    """Check if a path is an S3/S3A path."""
    return str(path).startswith("s3a://") or str(path).startswith("s3://")


def _get_s3_client():
    """Create a boto3 S3 client configured for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
    )


def _parse_s3_path(path: str) -> tuple:
    """Parse s3a://bucket/key into (bucket, key)."""
    parsed = urlparse(path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def path_exists(path: Union[str, Path]) -> bool:
    """Check if a path exists (local or S3)."""
    path_str = str(path)
    if _is_s3_path(path_str):
        s3 = _get_s3_client()
        bucket, key = _parse_s3_path(path_str)
        # For S3 directories written by Spark, check for objects under the prefix
        prefix = key.rstrip("/") + "/"
        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
            return resp.get("KeyCount", 0) > 0
        except Exception:
            return False
    return Path(path_str).exists()


def write_single_parquet(
    df: DataFrame,
    output_path: Union[str, Path],
    compression: str = "uncompressed",
) -> None:
    """Write a Spark DataFrame as a single parquet file.

    Supports both local paths and s3a:// paths.
    For S3: writes directly via Spark's S3A connector.
    For local: uses temp directory + coalesce(1) + move pattern.
    """
    path_str = str(output_path)

    if _is_s3_path(path_str):
        df.coalesce(1).write.mode("overwrite").option("compression", compression).parquet(path_str)
        return

    # Local filesystem path
    local_path = Path(output_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        if local_path.is_dir():
            shutil.rmtree(local_path)
        else:
            local_path.unlink()

    temp_dir = Path(tempfile.mkdtemp(dir=str(local_path.parent)))
    try:
        df.coalesce(1).write.mode("overwrite").option("compression", compression).parquet(str(temp_dir))
        part_files = list(temp_dir.glob("part-*.parquet"))
        if not part_files:
            raise RuntimeError(f"No parquet part file found in {temp_dir}")
        shutil.move(str(part_files[0]), str(local_path))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def list_parquet_files(directory: Union[str, Path], pattern: str = "trip_*.parquet") -> List[str]:
    """List parquet files matching a pattern under directory.

    Supports both local paths and s3a:// paths.
    For S3: uses boto3 to list objects matching the pattern.
    For local: uses Path.rglob().

    Returns list of path strings (s3a:// URIs for S3, str(Path) for local).
    """
    path_str = str(directory)

    if _is_s3_path(path_str):
        s3 = _get_s3_client()
        bucket, prefix = _parse_s3_path(path_str)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        results = set()
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                key_parts = key.split("/")
                filename = key_parts[-1]
                # Match the pattern (simple glob: trip_*.parquet)
                import fnmatch

                if fnmatch.fnmatch(filename, pattern):
                    results.add(f"s3a://{bucket}/{key}")
                    continue

                # Spark writes parquet to S3 as a directory prefix like:
                # s3a://bucket/.../trip_04.parquet/part-0000-....parquet
                # In that case, return the parent parquet "directory" URI so
                # downstream Spark readers can load it directly.
                if len(key_parts) >= 2 and fnmatch.fnmatch(key_parts[-2], pattern):
                    results.add(f"s3a://{bucket}/{'/'.join(key_parts[:-1])}")
        return sorted(results)

    # Local filesystem
    local_dir = Path(directory)
    if not local_dir.exists():
        return []
    return sorted(str(p) for p in local_dir.rglob(pattern) if p.is_file())
