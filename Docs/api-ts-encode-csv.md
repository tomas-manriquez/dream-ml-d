# Time Series API Endpoint Analysis: `/api/ts/encode-csv/`

## Feature Engineering Pipeline Table

| Algorithm | Problem Type | Input | Output |
|-----------|--------------|-------|--------|
| **ARIMA** | Single time series (univariate) | **encode-csv:** No encode-csv preprocessing<br><br>**Direct input:** Raw target variable (dtype: float64 or int64)<br>Date column (dtype: object ’ converted to datetime64 internally) | **Internal transformations:**<br>" Differenced series (dtype: float64)<br>" Fitted ARIMA model parameters (p, d, q)<br>" No explicit feature columns created<br><br>**Model output:** Forecasted values (dtype: float64) |
| **SARIMA** | Single time series (univariate seasonal) | **encode-csv:** No encode-csv preprocessing<br><br>**Direct input:** Raw target variable (dtype: float64 or int64)<br>Date column (dtype: object ’ converted to datetime64 internally) | **Internal transformations:**<br>" Differenced series with seasonal differencing (dtype: float64)<br>" Fitted SARIMA model parameters (p, d, q, P, D, Q, s)<br>" No explicit feature columns created<br><br>**Model output:** Forecasted values (dtype: float64) |
| **XGBoost** | Single time series (multivariate) | **encode-csv (optional):**<br>" `{feature}_lag_{n}` for n=1 to lag_periods (dtype: float64 or int64)<br>" NaN handling: drop/forward_fill/leave_as_is (see note 1)<br><br>**Internal `prepare_xgboost_features()`:**<br>" Lag features: `{feature}_lag_{n}` for n in [1,2,3,4,5] (dtype: float64)<br>" Rolling window features: `{feature}_rolling_mean_{w}`, `{feature}_rolling_std_{w}`, `{feature}_rolling_min_{w}`, `{feature}_rolling_max_{w}` for windows [3,7,14] (dtype: float64)<br>" Time features: `day_of_week`, `month`, `quarter`, `year` (dtype: int64), `is_weekend` (dtype: int64), `day_of_week_sin`, `day_of_week_cos`, `month_sin`, `month_cos` (dtype: float64)<br>" External features (if provided) (dtype: float64 or int64)<br>" Date column (dtype: object ’ converted to datetime64 internally) | **Feature matrix:**<br>" All lag features from encode-csv + internal (dtype: float64)<br>" All rolling window features (dtype: float64)<br>" All time features (dtype: int64 or float64)<br>" All external features (dtype: float64 or int64)<br>" Rows with NaN removed after feature engineering (dtype: preserved)<br><br>**Model output:** Forecasted values (dtype: float64) |
| **LSTM** | Single time series (multivariate) | **encode-csv (optional):**<br>" `{feature}_lag_{n}` for n=1 to lag_periods (dtype: float64 or int64)<br>" NaN handling: drop/forward_fill/leave_as_is (see note 1)<br><br>**Internal `prepare_xgboost_features()` (reused):**<br>" Lag features: `{feature}_lag_{n}` for n in [1,2,3,4,5] (dtype: float64)<br>" Rolling window features: `{feature}_rolling_mean_{w}`, `{feature}_rolling_std_{w}`, `{feature}_rolling_min_{w}`, `{feature}_rolling_max_{w}` for windows [3,7,14] (dtype: float64)<br>" Time features: `day_of_week`, `month`, `quarter`, `year` (dtype: int64), `is_weekend` (dtype: int64), `day_of_week_sin`, `day_of_week_cos`, `month_sin`, `month_cos` (dtype: float64)<br>" External features (if provided) (dtype: float64 or int64)<br>" Date column (dtype: object ’ converted to datetime64 internally) | **3D Sequence tensor:**<br>" Shape: (samples, sequence_length, n_features)<br>" dtype: float32 (after StandardScaler normalization)<br>" All features from input are reshaped into sequences<br>" Each sequence contains `sequence_length` consecutive time steps<br><br>**Model output:** Forecasted values (dtype: float32 ’ converted to original scale) |

---

## Notes

**Note 1: NaN Handling Strategies (from `/api/ts/encode-csv/`)**

| Strategy | Description | Impact on Data |
|----------|-------------|----------------|
| `drop` | Removes all rows containing any NaN values | Reduces number of rows (dtype: preserved) |
| `forward_fill` | Forward fills NaN values using pandas `ffill()` | Preserves number of rows, no NaN remain (dtype: preserved) |
| `leave_as_is` | Keeps NaN values unchanged | Preserves number of rows, NaN remain (dtype: preserved) |

**Note 2: Key Observations**

1. **ARIMA/SARIMA do NOT use `/api/ts/encode-csv/`**: They load data directly from raw CSV via `load_and_validate_ts_data()` and work exclusively with the target variable.

2. **Potential lag feature duplication**: If `/api/ts/encode-csv/` is used with `lag_periods > 0` for XGBoost/LSTM, there may be overlap with the internal lag features created by `prepare_xgboost_features()`. The internal function creates lag features [1,2,3,4,5] by default.

3. **Target variables are NOT lagged**: The `/api/ts/encode-csv/` endpoint creates lag features only for `input_features`, never for `target_variables`.

4. **series_id is explicitly excluded**: If a `series_id` column exists in the data, it is treated as metadata and excluded from feature engineering. The system does NOT support multi-time series (panel data) forecasting.

5. **Categorical encoding is disabled**: Despite legacy code existing for OneHotEncoder and LabelEncoder, these are NOT applied in the current implementation. All categorical features must be numeric or will be excluded.

---

## File References

- Endpoint definition: [apiTimeSeries/urls.py](../DREAM-ML-backend/GEML/apiTimeSeries/urls.py)
- View function: [apiTimeSeries/views.py:258-372](../DREAM-ML-backend/GEML/apiTimeSeries/views.py#L258-L372)
- Encoding utilities: [apiTimeSeries/data_encoding_utils.py](../DREAM-ML-backend/GEML/apiTimeSeries/data_encoding_utils.py)
- Training algorithms: [apiTimeSeries/train.py](../DREAM-ML-backend/GEML/apiTimeSeries/train.py)
  - ARIMA: [train.py:1222-1723](../DREAM-ML-backend/GEML/apiTimeSeries/train.py#L1222-L1723)
  - XGBoost: [train.py:1725-2250](../DREAM-ML-backend/GEML/apiTimeSeries/train.py#L1725-L2250)
  - LSTM: [train.py:2709-3279](../DREAM-ML-backend/GEML/apiTimeSeries/train.py#L2709-L3279)

---

## Endpoint Details

### `/api/ts/encode-csv/`

**HTTP Method:** POST

**Description:** Preprocesses time series data by creating lag features and handling NaN values. This is an optional preprocessing step primarily useful for custom lag configurations before training XGBoost or LSTM models.

### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | - | CSV file containing the time series data |
| `experiment_dir` | String | Yes | - | Directory path for the experiment |
| `input_features` | String | Yes | - | Comma-separated list of input feature column names |
| `target_variables` | String | Yes | - | Comma-separated list of target variable column names |
| `run_id` | String | Yes | - | MLflow run identifier for tracking |
| `encode_target_ohe` | Boolean | No | `false` | (Legacy/Unused) Apply OneHotEncoder to targets |
| `encode_target_label` | Boolean | No | `false` | (Legacy/Unused) Apply LabelEncoder to targets |
| `lag_periods` | Integer | No | `0` | Number of lag periods to create (0 means no lag features) |
| `lag_nan_handling` | String | No | `leave_as_is` | Strategy for handling NaN values: `drop`, `forward_fill`, or `leave_as_is` |
| `date_column` | String | No | First column | Name of the date/time column for ordering lag features |

### Response Format

```json
{
  "status": "Archivo CSV codificado correctamente.",
  "processed_train_path": "processed/processed_train_{filename}.csv",
  "run_id": "{mlflow_run_id}"
}
```

### Example Usage

```python
import requests

# Prepare the request
url = "http://localhost:8000/api/ts/encode-csv/"
files = {'file': open('data.csv', 'rb')}
data = {
    'experiment_dir': '/path/to/experiment',
    'input_features': 'temperature,humidity,pressure',
    'target_variables': 'energy_consumption',
    'run_id': 'mlflow-run-123',
    'lag_periods': 5,
    'lag_nan_handling': 'forward_fill',
    'date_column': 'timestamp'
}

# Send POST request
response = requests.post(url, files=files, data=data)
print(response.json())
```

### Output File Location

The processed CSV file is saved to:
```
{experiment_dir}/processed/processed_train_{original_filename}.csv
```

---

This document provides a comprehensive analysis of the `/api/ts/encode-csv/` endpoint and the complete feature engineering pipeline for all time series algorithms in the DREAM-ML backend.
