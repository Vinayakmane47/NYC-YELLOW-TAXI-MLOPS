import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from utils.spark_utils import SparkUtils


class MLDataTransformer:
    """
    Transforms gold-level data into ML-ready train/test/val splits.
    
    Steps:
    1. Load parquet data (limit to 1M rows)
    2. Drop leakage features
    3. Drop non-predictive features
    4. Encode categorical features (binary, one-hot, target encoding)
    5. Scale numerical features
    6. Time-based split (70% train, 15% test, 15% val)
    7. Save to ml_transformed folder
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
        self.spark = SparkUtils(app_name).spark
        
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
        
        self.scaler = StandardScaler()
        self.target_encoders = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load parquet data from gold folder (limit to max_rows)."""
        print(f"Loading data from {self.input_dir} (max {self.max_rows:,} rows)...")
        
        # Find all parquet files in gold directory
        parquet_files = sorted(self.input_dir.rglob("trip_*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.input_dir}")
        
        print(f"Found {len(parquet_files)} parquet files")
        
        # Read parquet files using Spark
        df_spark = self.spark.read.parquet(*[str(f) for f in parquet_files])
        
        # Limit rows and sort by time for proper time-based splitting
        df_spark = df_spark.orderBy('tpep_pickup_datetime').limit(self.max_rows)
        
        # Convert to pandas for easier manipulation
        df = df_spark.toPandas()
        
        print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        return df
    
    def drop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop leakage and non-predictive features."""
        print("\nDropping features...")
        
        # Combine all columns to drop
        cols_to_drop = list(set(self.leakage_cols + self.drop_cols))
        
        # Only drop columns that exist
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        print(f"  Dropping {len(existing_cols_to_drop)} columns: {existing_cols_to_drop}")
        df = df.drop(columns=existing_cols_to_drop)
        
        print(f"  Remaining columns: {len(df.columns)}")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Encode categorical features:
        - Binary encode: store_and_fwd_flag
        - One-hot: VendorID, RatecodeID, payment_type
        - Target encoding: DOLocationID, PULocationID
        """
        print("\nEncoding categorical features...")
        
        # 1. Binary encoding for store_and_fwd_flag
        if 'store_and_fwd_flag' in df.columns:
            print("  Binary encoding: store_and_fwd_flag")
            df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)
        
        # 2. One-hot encoding for VendorID, RatecodeID, payment_type
        onehot_existing = [col for col in self.onehot_cols if col in df.columns]
        if onehot_existing:
            print(f"  One-hot encoding: {onehot_existing}")
            df = pd.get_dummies(df, columns=onehot_existing, prefix=onehot_existing, drop_first=True)
        
        # 3. Target encoding for location IDs (using trip_duration_min as target)
        target_col = 'trip_duration_min'
        if target_col in df.columns:
            for col in self.target_encode_cols:
                if col in df.columns:
                    print(f"  Target encoding: {col}")
                    
                    if is_training:
                        # Calculate mean target value for each category
                        target_means = df.groupby(col)[target_col].mean()
                        global_mean = df[target_col].mean()
                        
                        # Store encoder for later use
                        self.target_encoders[col] = {
                            'mapping': target_means.to_dict(),
                            'global_mean': global_mean
                        }
                    
                    # Apply encoding
                    if col in self.target_encoders:
                        encoder_info = self.target_encoders[col]
                        df[f'{col}_encoded'] = df[col].map(encoder_info['mapping']).fillna(encoder_info['global_mean'])
                    
                    # Drop original column
                    df = df.drop(columns=[col])
        
        return df
    
    def scale_numerical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        print("\nScaling numerical features...")
        
        # Only scale columns that exist
        cols_to_scale = [col for col in self.scale_cols if col in df.columns]
        
        if not cols_to_scale:
            print("  No columns to scale")
            return df
        
        print(f"  Scaling {len(cols_to_scale)} columns: {cols_to_scale}")
        
        if is_training:
            # Fit and transform on training data
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        else:
            # Only transform on test/val data
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        return df
    
    def time_based_split(self, df: pd.DataFrame) -> tuple:
        """
        Split data based on time order (70% train, 15% test, 15% val).
        Data is already sorted by tpep_pickup_datetime.
        """
        print("\nPerforming time-based split...")
        
        n_rows = len(df)
        train_end = int(n_rows * 0.70)
        test_end = int(n_rows * 0.85)
        
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        val_df = df.iloc[test_end:].copy()
        
        print(f"  Train: {len(train_df):,} rows ({len(train_df)/n_rows*100:.1f}%)")
        print(f"  Test:  {len(test_df):,} rows ({len(test_df)/n_rows*100:.1f}%)")
        print(f"  Val:   {len(val_df):,} rows ({len(val_df)/n_rows*100:.1f}%)")
        
        return train_df, test_df, val_df
    
    def save_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
        """Save train/test/val splits to ml_transformed folder."""
        print(f"\nSaving splits to {self.output_dir}...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each split as parquet
        train_path = self.output_dir / "train.parquet"
        test_path = self.output_dir / "test.parquet"
        val_path = self.output_dir / "val.parquet"
        
        train_df.to_parquet(train_path, index=False, compression='snappy')
        test_df.to_parquet(test_path, index=False, compression='snappy')
        val_df.to_parquet(val_path, index=False, compression='snappy')
        
        print(f"  Saved: {train_path} ({train_path.stat().st_size / 1024 / 1024:.2f} MB)")
        print(f"  Saved: {test_path} ({test_path.stat().st_size / 1024 / 1024:.2f} MB)")
        print(f"  Saved: {val_path} ({val_path.stat().st_size / 1024 / 1024:.2f} MB)")
        
        # Save metadata
        metadata = {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'source_dir': str(self.input_dir),
            'max_rows': self.max_rows,
            'splits': {
                'train': {'rows': len(train_df), 'path': str(train_path), 'size_bytes': train_path.stat().st_size},
                'test': {'rows': len(test_df), 'path': str(test_path), 'size_bytes': test_path.stat().st_size},
                'val': {'rows': len(val_df), 'path': str(val_path), 'size_bytes': val_path.stat().st_size},
            },
            'features': {
                'total_columns': len(train_df.columns),
                'column_names': list(train_df.columns),
                'dropped_leakage': self.leakage_cols,
                'dropped_other': self.drop_cols,
                'binary_encoded': self.binary_encode_cols,
                'onehot_encoded': self.onehot_cols,
                'target_encoded': self.target_encode_cols,
                'scaled': self.scale_cols,
            },
            'target_encoders': {
                k: {'global_mean': float(v['global_mean']), 'num_categories': len(v['mapping'])}
                for k, v in self.target_encoders.items()
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
        print("ML DATA TRANSFORMATION PIPELINE")
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
        print("\nMetadata summary:")
        print(json.dumps(metadata, indent=2))
    finally:
        transformer.close()


if __name__ == "__main__":
    main()
