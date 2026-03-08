# Data Pipeline

The data pipeline follows a medallion architecture (Bronze -> Silver -> Gold -> ML-ready) orchestrated by Apache Airflow and processed with PySpark.

## Data Flow

```
NYC TLC Website
    | (HTTP download)
    v
Bronze (raw parquet)         --> MinIO s3a://bronze/
    | (clean, filter, impute)
    v
Silver (cleaned)             --> MinIO s3a://silver/
    | (feature engineering)
    v
Gold (features)              --> MinIO s3a://gold/
    | (encode, scale, split)
    v
ML-Transformed (model-ready) --> MinIO s3a://ml-transformed/
```

## Stage 1: Data Ingestion (Bronze)

**File:** `src/data_ingestion.py`

Downloads NYC Yellow Taxi trip data from the official TLC source.

- **Source:** `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet`
- **Output:** `s3a://bronze/{year}/{month_name}/trip_{month}.parquet`
- **Behavior:** Skips download if file already exists in MinIO. Writes metadata with row counts and download timestamps.

**Key columns in raw data:**
- `tpep_pickup_datetime`, `tpep_dropoff_datetime` - trip timestamps
- `PULocationID`, `DOLocationID` - taxi zone IDs (1-263)
- `trip_distance` - distance in miles
- `fare_amount`, `tip_amount`, `tolls_amount` - fare breakdown
- `passenger_count`, `VendorID`, `RatecodeID`, `payment_type`

## Stage 2: Data Validation

**File:** `src/data_validation.py`

Validates schema and required columns before processing. Checks that all 12 expected columns are present. Writes validation metadata with results.

## Stage 3: Data Preprocessing (Silver)

**File:** `src/data_preprocessing.py`

Cleans raw data and creates the target variable.

**Transformations:**
1. **Target variable:** `trip_duration_min` = (dropoff - pickup) in minutes
2. **Filters:**
   - 0 < trip_duration < 180 minutes
   - 0 < trip_distance < 60 miles
   - 0 < fare_amount < $500
3. **Missing value imputation:**
   - `passenger_count` -> 1
   - `RatecodeID` -> 1 (clamped to 1-6)
   - `congestion_surcharge` -> 0
   - `Airport_fee` -> 0
   - `store_and_fwd_flag` -> "N"
4. **Column selection:** 19 relevant columns retained

**Output:** `s3a://silver/{year}/{month_name}/trip_{month}.parquet`

## Stage 4: Feature Engineering (Gold)

**File:** `src/data_transformation.py`

Engineers predictive features from cleaned data.

**Feature groups:**

| Group | Features | Description |
|-------|----------|-------------|
| Time | `pickup_hour`, `day_of_month`, `week_of_year`, `month`, `year`, `day_of_week` | Extracted from pickup datetime |
| Binary time | `is_weekend`, `is_peak_hour`, `is_night`, `is_rush_hour` | Time-based flags |
| Cyclical | `sin_hour`, `cos_hour`, `sin_day_of_week`, `cos_day_of_week` | Cyclical encoding (sin/cos) |
| Distance | `log_trip_distance`, `sqrt_trip_distance`, `is_short_trip`, `is_long_trip` | Distance transformations |
| Zone stats | `pickup_zone_trip_count`, `dropoff_zone_trip_count`, `pickup_zone_avg_duration`, `dropoff_zone_avg_duration` | Aggregated zone features |
| Fare ratios | `fare_per_mile`, `surcharge_ratio`, `toll_ratio`, `non_fare_ratio` | Fare breakdown ratios |

**Output:** `s3a://gold/{year}/{month_name}/trip_{month}.parquet`

## Stage 5: ML Transformation

**File:** `src/ml_transformed.py`

Prepares model-ready train/val/test splits.

**Steps:**
1. **Drop leakage columns:** `fare_amount`, `tip_amount`, `extra`, `mta_tax`, `improvement_surcharge`, `total_amount` (these are known only after the trip)
2. **Drop non-predictive:** `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `route_id`
3. **Encoding:**
   - Binary: `store_and_fwd_flag` (Y=1, N=0)
   - One-hot: `VendorID` (2 categories), `RatecodeID` (5 categories), `payment_type` (4 categories)
   - Target encoding: `PULocationID`, `DOLocationID` (mean trip duration per zone)
4. **Scaling:** StandardScaler on numerical features (saves means/stds to metadata)
5. **Split:** 70% train, 15% validation, 15% test (chronological order)

**Outputs:**
- `s3a://ml-transformed/train.parquet`
- `s3a://ml-transformed/val.parquet`
- `s3a://ml-transformed/test.parquet`
- `src/metadata/pipeline_{RUN_ID}/ml_transformation.json` (scaling stats, used by inference)

**Final feature count:** 41 features + 1 target (`trip_duration_min`)

## Stage 6: Drift Detection

**File:** `src/drift_detection.py`

Uses Evidently AI to detect data drift between training and current data.

- **Method:** `DataDriftPreset` with per-column statistical tests
- **Threshold:** If >30% of features drift (p-value < 0.05), triggers retraining
- **Output:** `DriftResult` with `drift_detected`, `share_drifted`, `drifted_columns`

## Airflow DAGs

### ETL Pipeline (`nyc_taxi_mlops_pipeline`)
Full end-to-end pipeline. Schedule: `@monthly`.

```
data_ingestion -> data_validation -> data_preprocessing -> data_transformation
    -> ml_transformation -> model_training -> model_evaluation -> model_registry
```

### Data Refresh (`nyc_data_refresh_dag`)
Monthly data refresh with drift detection. If drift is detected, triggers model retraining.

```
data_ingestion -> data_validation -> data_preprocessing -> data_transformation
    -> ml_transformation -> retrain_decider -> [conditional] trigger_retrain
```

### Model Retrain (`nyc_model_retrain_dag`)
Triggered by data refresh when drift is detected. No schedule.

```
model_training -> model_evaluation -> model_registry
```

## Configuration

All data pipeline paths and parameters are configured in `src/config/settings.yaml`:

```yaml
minio:
  endpoint: http://minio:9000
  access_key: minioadmin
  secret_key: minioadmin
  buckets:
    bronze: bronze
    silver: silver
    gold: gold
    ml_transformed: ml-transformed

data:
  train_path: s3a://ml-transformed/train.parquet
  val_path: s3a://ml-transformed/val.parquet
  test_path: s3a://ml-transformed/test.parquet
  target_col: trip_duration_min
```

## Pipeline Metadata

Every stage writes metadata to `src/metadata/pipeline_{RUN_ID}/{stage}.json`. This includes row counts, column lists, timing, file sizes, and stage-specific stats. The `PIPELINE_RUN_ID` is shared across all stages in a single pipeline run (set by Airflow or auto-generated for CLI runs).
