import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


class MLDataTransformer:
    """
    Transforms gold-level data into ML-ready train/test/val splits using PySpark.
    
    Steps:
    1. Load parquet data (limit to 1M rows)
    2. Drop leakage features
    3. Drop non-predictive features
    4. Encode categorical features (binary, one-hot, target encoding)
    5. Scale numerical features
    6. Time-based split (70% train, 15% test, 15% val)
    7. Save to ml_transformed folder as parquet
    """
    
    def __init__(
        self,
        input_dir: str = "gold",
        output_dir: str = "ml_transformed",
        app_name: str = "nyc-taxi-ml-transformation",
        max_rows: int = 1_000_000,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_rows = max_rows
        
        # Create Spark session with increased memory
        self.spark = (SparkSession.builder
                     .appName(app_name)
                     .master("local[*]")
                     .config("spark.driver.memory", "4g")
                     .config("spark.executor.memory", "4g")
                     .config("spark.sql.shuffle.partitions", "8")
                     .config("spark.driver.host", "localhost")
                     .config("spark.driver.bindAddress", "127.0.0.1")
                     .getOrCreate())
        
        # Feature definitions
        self.leakage_cols = [
            'fare_amount', 'tip_amount', 'tolls_amount', 'extra',
            'mta_tax', 'improvement_surcharge', 'congestion_surcharge',
            'Airport_fee', 'cbd_congestion_fee', 'fare_per_mile',
            'surcharge_ratio', 'toll_ratio', 'non_fare_amount'
        ]
        
        self.drop_cols = ['tpep_pickup_datetime', 'route_id']
        
        self.binary_encode_cols = ['store_and_fwd_flag']
        self.onehot_cols = ['VendorID', 'RatecodeID', 'payment_type']
        self.target_encode_cols = ['DOLocationID', 'PULocationID']
        
        self.scale_cols = [
            'trip_distance', 'log_trip_distance', 'sqrt_trip_distance',
            'passenger_count', 'pickup_zone_trip_count',
            'dropoff_zone_trip_count', 'route_trip_count'
        ]
        
        # Store statistics for scaling and target encoding
        self.scaling_stats = {}
        self.target_encoders = {}
        
    def load_data(self) -> DataFrame:
        """Load parquet data from gold folder (limit to max_rows)."""
        print(f"Loading data from {self.input_dir} (max {self.max_rows:,} rows)...")
        
        # Find all parquet files in gold directory
        parquet_files = sorted(self.input_dir.rglob("trip_*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.input_dir}")
        
        print(f"Found {len(parquet_files)} parquet files")
        
        # Read only the first file (already sorted by time) to limit memory usage
        # If you need data from multiple months, load first N files instead
        print(f"Reading from first file: {parquet_files[0]}")
        df = self.spark.read.parquet(str(parquet_files[0]))
        
        # Check row count before limiting
        total_rows = df.count()
        print(f"Total rows in file: {total_rows:,}")
        
        if total_rows > self.max_rows:
            print(f"Limiting to {self.max_rows:,} rows...")
            # Just take first N rows without sorting (file is already time-ordered)
            df = df.limit(self.max_rows)
        
        # Add monotonically increasing ID for splitting
        df = df.withColumn("row_id", F.monotonically_increasing_id())
        
        row_count = df.count()
        col_count = len(df.columns)
        
        print(f"Loaded {row_count:,} rows with {col_count} columns")
        return df
    
    def drop_features(self, df: DataFrame) -> DataFrame:
        """Drop leakage and non-predictive features."""
        print("\nDropping features...")
        
        # Combine all columns to drop
        cols_to_drop = list(set(self.leakage_cols + self.drop_cols))
        
        # Only drop columns that exist
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        print(f"  Dropping {len(existing_cols_to_drop)} columns")
        df = df.drop(*existing_cols_to_drop)
        
        print(f"  Remaining columns: {len(df.columns)}")
        return df
    
    def encode_categorical_features(self, df: DataFrame, is_training: bool = True) -> DataFrame:
        """
        Encode categorical features:
        - Binary encode: store_and_fwd_flag
        - One-hot: VendorID, RatecodeID, payment_type
        - Target encoding: DOLocationID, PULocationID
        
        If is_training=True, fit encoders on df (training data).
        If is_training=False, use previously fitted encoders.
        """
        
        print("\nEncoding categorical features...")
        
        # 1. Binary encoding for store_and_fwd_flag
        if 'store_and_fwd_flag' in df.columns:
            print("  Binary encoding: store_and_fwd_flag")
            df = df.withColumn(
                'store_and_fwd_flag',
                F.when(F.col('store_and_fwd_flag') == 'Y', 1).otherwise(0)
            )
        
        # 2. One-hot encoding for VendorID, RatecodeID, payment_type
        onehot_existing = [col for col in self.onehot_cols if col in df.columns]
        if onehot_existing:
            print(f"  One-hot encoding: {onehot_existing}")
            
            for col in onehot_existing:
                # Get distinct values
                if is_training:
                    # Use df (which is train_df in this case)
                    distinct_vals = [row[0] for row in df.select(col).distinct().collect() if row[0] is not None]
                    distinct_vals = sorted(distinct_vals)[1:]  # Drop first for drop_first=True
                    self.target_encoders[f'{col}_categories'] = distinct_vals
                else:
                    distinct_vals = self.target_encoders.get(f'{col}_categories', [])
                
                # Create binary columns
                for val in distinct_vals:
                    df = df.withColumn(
                        f'{col}_{val}',
                        F.when(F.col(col) == val, 1).otherwise(0)
                    )
                
                # Drop original column
                df = df.drop(col)
        
        # 3. Target encoding for location IDs (using trip_duration_min as target)
        target_col = 'trip_duration_min'
        if target_col in df.columns:
            for col in self.target_encode_cols:
                if col in df.columns:
                    print(f"  Target encoding: {col}")
                    
                    if is_training:
                        # Calculate mean target value for each category on training data (use df here)
                        target_means = df.groupBy(col).agg(
                            F.mean(target_col).alias('mean_target')
                        )
                        global_mean = df.select(F.mean(target_col)).collect()[0][0]
                        
                        # Store encoder for later use
                        self.target_encoders[col] = {
                            'mapping': {row[col]: row['mean_target'] for row in target_means.collect()},
                            'global_mean': global_mean
                        }
                    
                    # Apply encoding
                    if col in self.target_encoders:
                        encoder_info = self.target_encoders[col]
                        
                        # Create mapping expression
                        mapping_expr = F.lit(encoder_info['global_mean'])
                        for category, mean_val in encoder_info['mapping'].items():
                            mapping_expr = F.when(F.col(col) == category, mean_val).otherwise(mapping_expr)
                        
                        df = df.withColumn(f'{col}_encoded', mapping_expr)
                    
                    # Drop original column
                    df = df.drop(col)
        
        return df
    
    def scale_numerical_features(self, df: DataFrame, is_training: bool = True) -> DataFrame:
        """
        Scale numerical features using StandardScaler (Z-score normalization).
        
        If is_training=True, compute statistics on df (training data).
        If is_training=False, use previously computed statistics.
        """
        
        print("\nScaling numerical features...")
        
        # Only scale columns that exist
        cols_to_scale = [col for col in self.scale_cols if col in df.columns]
        
        if not cols_to_scale:
            print("  No columns to scale")
            return df
        
        print(f"  Scaling {len(cols_to_scale)} columns: {cols_to_scale}")
        
        if is_training:
            # Compute mean and stddev for each column on training data (use df here)
            for col in cols_to_scale:
                stats = df.select(
                    F.mean(col).alias('mean'),
                    F.stddev(col).alias('stddev')
                ).collect()[0]
                
                self.scaling_stats[col] = {
                    'mean': stats['mean'] if stats['mean'] is not None else 0.0,
                    'stddev': stats['stddev'] if stats['stddev'] is not None and stats['stddev'] > 0 else 1.0
                }
        
        # Apply scaling
        for col in cols_to_scale:
            if col in self.scaling_stats:
                stats = self.scaling_stats[col]
                df = df.withColumn(
                    col,
                    (F.col(col) - F.lit(stats['mean'])) / F.lit(stats['stddev'])
                )
        
        return df
    
    def time_based_split(self, df: DataFrame) -> tuple:
        """
        Split data based on time order (70% train, 15% test, 15% val).
        Uses row_id which was created based on time-ordered data.
        """
        print("\nPerforming time-based split...")
        
        # Cache df to avoid recomputation
        df.cache()
        
        total_count = df.count()
        
        # Get max row_id for splits
        max_row = df.select(F.max('row_id')).collect()[0][0]
        train_threshold = max_row * 0.70
        test_threshold = max_row * 0.85
        
        print(f"  Total rows: {total_count:,}")
        print(f"  Split thresholds - Train: 0-{train_threshold:.0f}, Test: {train_threshold:.0f}-{test_threshold:.0f}, Val: {test_threshold:.0f}-{max_row:.0f}")
        
        train_df = df.filter(F.col('row_id') <= train_threshold).drop('row_id')
        test_df = df.filter((F.col('row_id') > train_threshold) & (F.col('row_id') <= test_threshold)).drop('row_id')
        val_df = df.filter(F.col('row_id') > test_threshold).drop('row_id')
        
        # Cache splits
        train_df.cache()
        test_df.cache()
        val_df.cache()
        
        # Get actual counts
        train_actual = train_df.count()
        test_actual = test_df.count()
        val_actual = val_df.count()
        
        print(f"  Train: {train_actual:,} rows ({train_actual/total_count*100:.1f}%)")
        print(f"  Test:  {test_actual:,} rows ({test_actual/total_count*100:.1f}%)")
        print(f"  Val:   {val_actual:,} rows ({val_actual/total_count*100:.1f}%)")
        
        # Unpersist original df
        df.unpersist()
        
        return train_df, test_df, val_df
    
    def _write_parquet(self, df: DataFrame, output_path: Path) -> None:
        """Write DataFrame as single parquet file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        
        temp_dir = Path(tempfile.mkdtemp(dir=str(output_path.parent)))
        try:
            df.coalesce(1).write.mode("overwrite").option("compression", "snappy").parquet(str(temp_dir))
            part_files = list(temp_dir.glob("part-*.parquet"))
            if not part_files:
                raise RuntimeError(f"No parquet part file found in {temp_dir}")
            shutil.move(str(part_files[0]), str(output_path))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def save_splits(self, train_df: DataFrame, test_df: DataFrame, val_df: DataFrame) -> Dict[str, Any]:
        """Save train/test/val splits to ml_transformed folder as parquet."""
        print(f"\nSaving splits to {self.output_dir}...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define paths
        train_path = self.output_dir / "train.parquet"
        test_path = self.output_dir / "test.parquet"
        val_path = self.output_dir / "val.parquet"
        
        # Get counts before writing (they're cached so this is fast)
        train_count = train_df.count()
        test_count = test_df.count()
        val_count = val_df.count()
        
        # Write parquet files
        print("  Writing train.parquet...")
        self._write_parquet(train_df, train_path)
        
        print("  Writing test.parquet...")
        self._write_parquet(test_df, test_path)
        
        print("  Writing val.parquet...")
        self._write_parquet(val_df, val_path)
        
        # Unpersist cached dataframes to free memory
        train_df.unpersist()
        test_df.unpersist()
        val_df.unpersist()
        
        print(f"  Saved: {train_path} ({train_path.stat().st_size / 1024 / 1024:.2f} MB)")
        print(f"  Saved: {test_path} ({test_path.stat().st_size / 1024 / 1024:.2f} MB)")
        print(f"  Saved: {val_path} ({val_path.stat().st_size / 1024 / 1024:.2f} MB)")
        
        # Prepare metadata
        metadata = {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'source_dir': str(self.input_dir),
            'max_rows': self.max_rows,
            'splits': {
                'train': {
                    'rows': train_count,
                    'path': str(train_path),
                    'size_bytes': train_path.stat().st_size
                },
                'test': {
                    'rows': test_count,
                    'path': str(test_path),
                    'size_bytes': test_path.stat().st_size
                },
                'val': {
                    'rows': val_count,
                    'path': str(val_path),
                    'size_bytes': val_path.stat().st_size
                },
            },
            'features': {
                'total_columns': len(train_df.columns),
                'column_names': train_df.columns,
                'dropped_leakage': self.leakage_cols,
                'dropped_other': self.drop_cols,
                'binary_encoded': self.binary_encode_cols,
                'onehot_encoded': self.onehot_cols,
                'target_encoded': self.target_encode_cols,
                'scaled': self.scale_cols,
            },
            'scaling_stats': {
                k: {'mean': float(v['mean']), 'stddev': float(v['stddev'])}
                for k, v in self.scaling_stats.items()
            },
            'target_encoders': {
                k: {
                    'global_mean': float(v['global_mean']),
                    'num_categories': len(v['mapping'])
                }
                for k, v in self.target_encoders.items()
                if isinstance(v, dict) and 'global_mean' in v
            }
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved: {metadata_path}")
        
        return metadata
    
    def process(self) -> Dict[str, Any]:
        """Execute the full ML transformation pipeline."""
        start_time = datetime.now(timezone.utc)
        print("=" * 80)
        print("ML DATA TRANSFORMATION PIPELINE (PySpark)")
        print("=" * 80)
        
        # 1. Load data
        df = self.load_data()
        
        # 2. Drop features
        df = self.drop_features(df)
        
        # 3. Time-based split (before encoding/scaling to prevent data leakage)
        train_df, test_df, val_df = self.time_based_split(df)
        
        # 4. Encode categorical features (fit on train, transform on all)
        train_df = self.encode_categorical_features(train_df, is_training=True)
        test_df = self.encode_categorical_features(test_df, is_training=False)
        val_df = self.encode_categorical_features(val_df, is_training=False)
        
        # 5. Scale numerical features (fit on train, transform on all)
        train_df = self.scale_numerical_features(train_df, is_training=True)
        test_df = self.scale_numerical_features(test_df, is_training=False)
        val_df = self.scale_numerical_features(val_df, is_training=False)
        
        # 6. Save splits
        metadata = self.save_splits(train_df, test_df, val_df)
        
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print(f"Pipeline completed in {duration:.2f} seconds")
        print("=" * 80)
        
        return metadata
    
    def close(self) -> None:
        """Stop Spark session."""
        if self.spark:
            self.spark.stop()


def main() -> None:
    """Main entry point."""
    transformer = MLDataTransformer(
        input_dir="gold",
        output_dir="ml_transformed",
        max_rows=1_000_000
    )
    
    try:
        metadata = transformer.process()
        print("\n" + "=" * 80)
        print("METADATA SUMMARY")
        print("=" * 80)
        print(json.dumps(metadata, indent=2))
    finally:
        transformer.close()


if __name__ == "__main__":
    main()
