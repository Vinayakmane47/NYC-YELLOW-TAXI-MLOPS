"""Tests for src.utils.io_utils."""

import os

from src.utils.io_utils import _is_s3_path, _parse_s3_path, list_parquet_files, path_exists


class TestListParquetFiles:
    def test_returns_empty_for_nonexistent_dir(self, tmp_path):
        result = list_parquet_files(str(tmp_path / "nonexistent"))
        assert result == []

    def test_returns_empty_for_empty_dir(self, tmp_path):
        result = list_parquet_files(str(tmp_path))
        assert result == []

    def test_finds_matching_files(self, tmp_path):
        (tmp_path / "2025" / "jan").mkdir(parents=True)
        (tmp_path / "2025" / "jan" / "trip_01.parquet").write_text("fake")
        (tmp_path / "2025" / "jan" / "trip_02.parquet").write_text("fake")
        (tmp_path / "2025" / "jan" / "other.csv").write_text("fake")

        result = list_parquet_files(str(tmp_path))
        assert len(result) == 2
        assert all("trip_" in os.path.basename(p) for p in result)

    def test_returns_sorted_results(self, tmp_path):
        (tmp_path / "b").mkdir()
        (tmp_path / "a").mkdir()
        (tmp_path / "b" / "trip_02.parquet").write_text("fake")
        (tmp_path / "a" / "trip_01.parquet").write_text("fake")

        result = list_parquet_files(str(tmp_path))
        assert os.path.basename(result[0]) == "trip_01.parquet"
        assert os.path.basename(result[1]) == "trip_02.parquet"

    def test_custom_pattern(self, tmp_path):
        (tmp_path / "data.parquet").write_text("fake")
        (tmp_path / "trip_01.parquet").write_text("fake")

        result = list_parquet_files(str(tmp_path), pattern="data.parquet")
        assert len(result) == 1
        assert os.path.basename(result[0]) == "data.parquet"

    def test_skips_directories(self, tmp_path):
        (tmp_path / "trip_01.parquet").mkdir()
        result = list_parquet_files(str(tmp_path))
        assert result == []

    def test_s3_returns_spark_style_parquet_prefixes(self, monkeypatch):
        class FakePaginator:
            def paginate(self, Bucket, Prefix):
                assert Bucket == "bronze"
                assert Prefix == ""
                return [
                    {
                        "Contents": [
                            {
                                "Key": "2025/april/trip_04.parquet/part-00000-abc.snappy.parquet",
                            },
                            {
                                "Key": "2025/april/trip_04.parquet/_SUCCESS",
                            },
                            {
                                "Key": "2025/may/trip_05.parquet/part-00000-def.snappy.parquet",
                            },
                        ]
                    }
                ]

        class FakeS3Client:
            def get_paginator(self, name):
                assert name == "list_objects_v2"
                return FakePaginator()

        monkeypatch.setattr("src.utils.io_utils._get_s3_client", lambda: FakeS3Client())

        result = list_parquet_files("s3a://bronze")
        assert result == [
            "s3a://bronze/2025/april/trip_04.parquet",
            "s3a://bronze/2025/may/trip_05.parquet",
        ]


class TestS3PathHelpers:
    def test_is_s3_path_true(self):
        assert _is_s3_path("s3a://bucket/key")
        assert _is_s3_path("s3://bucket/key")

    def test_is_s3_path_false(self):
        assert not _is_s3_path("/local/path")
        assert not _is_s3_path("data/file.parquet")

    def test_parse_s3_path(self):
        bucket, key = _parse_s3_path("s3a://bronze/2025/january/trip_01.parquet")
        assert bucket == "bronze"
        assert key == "2025/january/trip_01.parquet"

    def test_parse_s3_path_no_key(self):
        bucket, key = _parse_s3_path("s3a://bronze")
        assert bucket == "bronze"
        assert key == ""


class TestPathExists:
    def test_local_file_exists(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_text("fake")
        assert path_exists(str(f))

    def test_local_dir_exists(self, tmp_path):
        d = tmp_path / "output"
        d.mkdir()
        assert path_exists(str(d))

    def test_local_nonexistent(self, tmp_path):
        assert not path_exists(str(tmp_path / "nope"))
