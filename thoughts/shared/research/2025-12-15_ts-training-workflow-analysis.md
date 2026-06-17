# Time Series Model Training Workflow Analysis

**Date**: 2025-12-15
**Research Objective**: Understand the TSTrainCard frontend and `/api/ts/train-model` backend workflow to align I/O assumptions, document user-facing options, and ensure experiment reproducibility for data scientist users.

---

## Executive Summary

### Scope
Analyzed the complete time series training pipeline spanning:
- **Frontend**: `TSTrainCard.jsx` (3,639 lines) - User form and submission logic
- **Backend**: Django 3-tier architecture (Views → Services → Training Logic)
  - `views.py:376` - HTTP endpoint handler
  - `services.py:946` - Orchestration layer with MLflow/DVC
  - `train.py` (3,806 lines) - ARIMA/XGBoost/LSTM implementations

### Key Findings

1. **Robust Reproducibility Mechanisms**
   - Global seed management (SEED=42) across Python/NumPy/TensorFlow
   - Explicit UTF-8 encoding for cross-platform data loading
   - SARIMAX optimizer defaults with stable start parameters (reduces variance from 5-10% → 0.5-2%)
   - Platform information logging (Python/NumPy/OS versions)

2. **Comprehensive Hyperparameter Optimization**
   - Manual, Grid Search, Random Search, Bayesian Search (LSTM only)
   - Walk-forward validation for ARIMA (prevents data leakage)
   - Expandable training window for time series integrity

3. **Complete Experiment Lineage**
   - MLflow tracking (metrics, parameters, artifacts)
   - DVC versioning (datasets, models)
   - pipeline_config.json (complete parameter history)
   - Energy tracking (carbon footprint via CodeCarbon)

4. **Data Scientist-Friendly Design**
   - Detailed validation messages with actionable guidance
   - Real-time progress updates via WebSockets
   - Automatic numeric feature validation with encoding suggestions
   - Three specialized algorithms (ARIMA, XGBoost, LSTM) for different use cases

5. **Architecture Pattern**
   - 3-tier separation: Views (HTTP) → Services (orchestration) → Train (ML logic)
   - Idempotent operations (safe pipeline re-runs)
   - Context managers for automatic cleanup
   - Metric filtering for edge cases (MAPE with zero division)

---

## Component Analysis

### 1. Frontend: TSTrainCard.jsx

**Location**: `DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx`

#### State Management

**Global Context Dependencies** (TSTrainCard.jsx:62-71):
```javascript
const {
  experimentDir,        // Absolute path to experiment directory
  flow,                 // Workflow state (encodeDone, trainDone, etc.)
  markStepDone,         // Callback to mark training complete
  runId,                // MLflow run ID from previous encoding step
  trainInProgress,      // Global spinner state
  setTrainInProgress,   // Control training UI state
  trainStatus,          // Status message displayed to user
  setTrainStatus        // Update status messages
} = useContext(AppContext);
```

**Critical Dependency**: `runId` must exist from the encoding step, enforcing proper workflow ordering.

#### Form Validation Chain

**Pre-Submission Validation** (TSTrainCard.jsx:641-673):
1. ✅ CSV file uploaded
2. ✅ At least 1 input feature selected
3. ✅ Target variable selected
4. ✅ Date column selected
5. ✅ Target ≠ Date column
6. ✅ Input features ∌ Date column
7. ✅ Experiment directory configured
8. ✅ Model name provided
9. ✅ Run ID exists (from encoding step)
10. ✅ Split ratios sum to 1.0

**Split Ratio Validation** (TSTrainCard.jsx:413-441):
- Debounced validation (500ms delay)
- Sum must equal 1.00 (rounded to 2 decimals)
- Visual feedback: ✓/⚠️ messages

#### Algorithm-Specific Parameters

**ARIMA Parameters** (TSTrainCard.jsx:104-115):
```javascript
{
  p: "1",                          // AR order
  d: "1",                          // Differencing order
  q: "1",                          // MA order
  seasonal_P: "1",                 // Seasonal AR
  seasonal_D: "1",                 // Seasonal differencing
  seasonal_Q: "1",                 // Seasonal MA
  seasonal_s: "12",                // Seasonal period
  trend: "n",                      // Trend: n/c/t/ct
  enforce_stationarity: "True",
  enforce_invertibility: "True"
}
```

**XGBoost Parameters** (manual mode):
- `n_estimators`, `max_depth`, `learning_rate`
- `subsample`, `gamma`, `min_child_weight`
- Parsed to integers/floats before submission

**LSTM Parameters** (manual mode):
- `lstm_units`: String "[64]" → parsed to array `[64]`
- `dropout_rate`, `recurrent_dropout_rate`
- `learning_rate`, `batch_size`, `epochs`
- `sequence_length`, `early_stopping_patience`

#### Payload Construction

**Critical Section** (TSTrainCard.jsx:724-790):
```javascript
const payload = {
  // Core configuration
  model_name: modelName,
  input_features: inputFeatures,        // Array of strings
  target_variable: targetVariable,       // String
  date_col_name: dateColumnName,         // String
  experiment_dir: experimentDir,         // Absolute path
  split_ratios: splitRatios,             // {train, val, test}
  run_id: runId,                         // From AppContext
  algorithm: algorithm,                  // "arima"/"xgboost"/"lstm"
  manual_params: finalParams,            // Algorithm-specific
  forecast_horizon: forecastHorizon,     // Integer
  problem_type: "ts_forecasting",        // Fixed
  hyperparameter_search_strategy: optimizationMethod,
  optimization_metric: optimizationMetric,

  // Conditional: Random Search
  n_random_iterations: useRandomSearch ? nRandomIterations : undefined,
  random_search_params: useRandomSearch ? {...} : undefined,

  // Conditional: Grid Search
  grid_search_params: useGridSearch ? {...} : undefined,

  // LSTM-specific overrides
  ...(algorithm === "lstm" && {
    sequence_length: sequenceLength,
    early_stopping_patience: earlyStoppingPatience,
    input_features: lstmSelectedFeatures,  // OVERRIDES top-level
    training_mode: lstmSelectedFeatures.length === 0 ? "univariate" : "multivariate",
    optimization_metric: "mse",            // OVERRIDES top-level
  })
};
```

**FormData Construction** (TSTrainCard.jsx:792-798):
```javascript
const formData = new FormData();
formData.append("file", csvFile);              // Binary CSV file
formData.append("data", JSON.stringify(payload)); // JSON string
```

**HTTP Request**:
```javascript
const response = await axios.post("/ts/train-model/", formData);
```

#### Response Handling

**Success Response** (TSTrainCard.jsx:802-836):
```javascript
{
  status: "success",
  model_path: "/path/to/model.pkl",
  metrics: {
    val_rmse: 0.1234,
    val_mae: 0.0567,
    val_mape: 2.34
  },
  mlflow_ui: "http://localhost:5000/#/experiments/.../runs/...",
  run_id: "abc123"
}
```

Frontend actions:
1. Format metrics to 4 decimals
2. Display success message with metrics
3. Update `trainStatus` state
4. Mark workflow step complete: `markStepDone("trainDone")`
5. Disable training spinner

**Error Response**:
```javascript
{
  status: "error",
  message: "High-level error description",
  error_details: "Technical debugging details"
}
```

---

### 2. Backend: views.py

**Location**: `DREAM-ML-backend/GEML/apiTimeSeries/views.py:376`

**Function**: `train_model(request)`

#### Request Validation

**Input Schema** (views.py:376-396):
```python
# FormData structure:
request.FILES['file']       # CSV binary
request.POST['data']        # JSON string

# Parsed data structure:
data = json.loads(request.POST['data'])
{
  "experiment_dir": str,     # Required - must exist
  "model_name": str,
  "algorithm": str,          # "arima"/"xgboost"/"lstm"
  "input_features": List[str],
  "target_variable": str,
  "date_col_name": str,
  "split_ratios": dict,
  "forecast_horizon": int,
  "manual_params": dict,
  "hyperparameter_search_strategy": str,
  # ... additional fields
}
```

**Validation Steps**:
1. Check HTTP method == POST (405 if not)
2. Check 'file' in request.FILES (ValueError if missing)
3. Check 'data' in request.POST (ValueError if missing)
4. Parse JSON: `data = json.loads(request.POST['data'])`
5. Validate experiment_dir exists: `os.path.isdir(experiment_dir)`

#### MLflow Configuration

**Centralized Tracking** (views.py:402-410):
```python
# Shared MLflow database at parent directory level
base_dir = os.path.dirname(experiment_dir)
shared_db_path = os.path.join(base_dir, "shared_mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{shared_db_path}")

# Clean up any orphaned runs
if mlflow.active_run():
    mlflow.end_run()
    logger.warning("Active MLflow run detected and closed")
```

**Design Decision**: Multiple experiments share one MLflow database for unified tracking across the project.

#### Service Delegation

**Orchestration Handoff** (views.py:412-420):
```python
result = trainModelService.train_model_logic(
    dataset_file=request.FILES['file'],
    data=data
)

# Cleanup after service completes
if mlflow.active_run():
    mlflow.end_run()
```

**Separation of Concerns**: View handles HTTP, Service handles business logic.

#### Response Construction

**Success Response** (views.py:422-437):
```python
experiment_name = os.path.basename(experiment_dir)
mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
mlflow_experiment_id = mlflow_experiment.experiment_id

return JsonResponse({
    "status": "success",
    "run_id": result.get("run_id", ""),
    "metrics": result.get("val_metrics", {}),  # Only validation metrics
    "model_path": result.get("model_path", ""),
    "mlflow_ui": f"http://{os.environ.get('MLFLOW_UI_URL', 'localhost:5000')}/#/experiments/{mlflow_experiment_id}/runs/{result.get('run_id', '')}"
}, status=200)
```

**Note**: Only `val_metrics` returned to frontend. Test metrics logged to MLflow but not exposed in API response.

#### Error Handling

**Hierarchical Error Responses** (views.py:439-479):
1. **JSON Decode Error (400)**: Invalid JSON in `data` field
2. **Validation Error (400)**: Missing required parameters
3. **File Not Found (404)**: Invalid experiment_dir
4. **Runtime Error (500)**: Training failures
5. **Unexpected Error (500)**: Catch-all with MLflow cleanup

```python
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    if mlflow.active_run():
        mlflow.end_run()  # Prevent orphaned runs
    return JsonResponse({
        "status": "error",
        "message": "Internal server error",
        "error_details": str(e)
    }, status=500)
```

---

### 3. Backend: services.py

**Location**: `DREAM-ML-backend/GEML/apiTimeSeries/services.py:946`

**Class**: `TrainModelService`
**Method**: `train_model_logic(dataset_file, data)`

#### Architecture

**Orchestration Responsibilities**:
1. ✅ Validate algorithm and experiment directory
2. ✅ Configure MLflow tracking URI
3. ✅ Start MLflow run with system metrics
4. ✅ Persist dataset with timestamped filename
5. ✅ Version dataset with DVC (add → commit → push)
6. ✅ Log dataset to MLflow for lineage tracking
7. ✅ Dispatch to algorithm-specific training function
8. ✅ Version model with DVC
9. ✅ Consolidate metrics (filter None values)
10. ✅ Prepare pipeline_config.json update
11. ✅ Return structured result

#### Dataset Persistence & Versioning

**Timestamped Storage** (services.py:991-1012):
```python
trained_dir = os.path.join(experiment_dir, "trained")
os.makedirs(trained_dir, exist_ok=True)

dataset_filename = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
dataset_path = os.path.join(trained_dir, dataset_filename)

# Idempotent save (skip if exists)
if not os.path.exists(dataset_path) or os.path.getsize(dataset_path) == 0:
    with open(dataset_path, 'wb') as f:
        for chunk in dataset_file.chunks():
            f.write(chunk)

# DVC versioning workflow
subprocess.run(["dvc", "add", dataset_path], cwd=experiment_dir, check=True)
subprocess.run(["git", "add", f"{dataset_path}.dvc"], cwd=experiment_dir, check=True)
subprocess.run(
    ["git", "commit", "-m", f"[DVC] Add training dataset {dataset_filename}"],
    cwd=experiment_dir, check=True
)
subprocess.run(["dvc", "push", dataset_path], cwd=experiment_dir, check=True)
```

**Reproducibility Benefit**: Every training gets a versioned, timestamped dataset snapshot.

#### MLflow Dataset Tracking

**New Feature** (services.py:1015-1023):
```python
# Explicit UTF-8 encoding for reproducibility
df_train = pd.read_csv(dataset_path, encoding='utf-8')
train_dataset = mlflow.data.from_pandas(
    df_train,
    source=None,  # Omit to avoid path inference warnings
    name="Dataset de Entrenamiento"
)
mlflow.log_input(train_dataset, context="train_data")
```

**MLflow 2.x Feature**: Dataset lineage tracking enables auditing which data was used for each run.

#### Algorithm Dispatch

**Unified Interface** (services.py:1025-1043):
```python
if algorithm == "arima":
    result = train_arima_model(
        dataset_path=dataset_path,
        data=data,
        experiment_dir=experiment_dir
    )
elif algorithm == "xgboost":
    result = train_xgboost_model(
        dataset_path=dataset_path,
        data=data,
        experiment_dir=experiment_dir
    )
elif algorithm == "lstm":
    result = train_lstm_model(
        dataset_path=dataset_path,
        data=data,
        experiment_dir=experiment_dir
    )
```

**Contract**: All training functions accept identical signatures and return standardized dictionaries.

#### Metrics Consolidation

**Filtering None Values** (services.py:1056-1067):
```python
combined_metrics = {}
if "val_metrics" in result:
    # Filter out None values (e.g., MAPE with division by zero)
    filtered_val_metrics = {k: v for k, v in result["val_metrics"].items() if v is not None}
    combined_metrics["val"] = filtered_val_metrics
    mlflow.log_metrics(filtered_val_metrics)

if "test_metrics" in result:
    filtered_test_metrics = {k: v for k, v in result["test_metrics"].items() if v is not None}
    combined_metrics["test"] = filtered_test_metrics
    mlflow.log_metrics(filtered_test_metrics)

mlflow.set_tag("training_phase", "completed")
```

**Edge Case Handling**: MAPE can be `None` when `y_true` contains zeros. MLflow rejects `None`, so filtering is required.

#### Return Schema

**Service Response** (services.py:1069-1088):
```python
{
  "status": str,                        # Human-readable status
  "val_metrics": {rmse, mae, mape},     # Validation metrics
  "test_metrics": {rmse, mae, mape},    # Test metrics (not returned to frontend)
  "model_path": str,                    # Absolute path
  "step_config": {                      # For pipeline_config.json
    "step": str,                        # "train_arima"
    "run_id": str,                      # MLflow run ID
    "algorithm": str,                   # "arima"
    "dataset_path": str,                # Relative path
    "model_path": str,                  # Relative path
    "metrics": {val: {...}, test: {...}},
    "timestamp": ISO8601
  },
  "run_id": str                         # MLflow run ID
}
```

---

### 4. Backend: train.py

**Location**: `DREAM-ML-backend/GEML/apiTimeSeries/train.py` (3,806 lines, 158KB)

#### Global Configuration

**Reproducibility Seeds** (train.py:70-76):
```python
SEED = int(os.environ.get('RANDOM_SEED', '42'))

SARIMAX_OPTIMIZER_DEFAULTS = {
    "method": "lbfgs",       # L-BFGS-B optimizer
    "maxiter": 500,          # Increased for better convergence
    "disp": 0,               # Quiet mode
    "ftol": 1e-8,            # Tighter function tolerance
    "gtol": 1e-6,            # Tighter gradient tolerance
}
```

**Cross-Platform Reproducibility Impact**: Reduces ARIMA variance from 5-10% → 0.5-2%.

**Global Seed Initialization** (train.py:97-131):
```python
def set_global_seeds():
    import random
    import os

    random.seed(SEED)
    np.random.seed(SEED)

    import tensorflow as tf
    tf.random.set_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = '42'

    logger.info(f"Global seeds initialized: SEED={SEED}")

# Run at module import
set_global_seeds()
```

**Coverage**: Python random, NumPy, TensorFlow (CPU/GPU ops), environment variables.

#### A. train_arima_model()

**Location**: `train.py:1415`

**Platform Logging** (train.py:1425-1448):
```python
platform_info = {
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "architecture": platform.machine(),
    "processor": platform.processor() or "unknown",
    "numpy_version": np.__version__,
    "scipy_version": scipy.__version__,
    "statsmodels_version": statsmodels.__version__,
}

mlflow.log_params({f"platform_{k}": str(v)[:250] for k, v in platform_info.items()})
```

**Purpose**: Debug cross-platform discrepancies by recording exact environment.

**Exogenous Variable Handling** (train.py:1484-1512):
```python
# Extract exogenous features (ARIMAX/SARIMAX)
if input_features:
    # Filter to ONLY numeric columns
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_features = [f for f in input_features if train_data[f].dtype in numeric_dtypes]

    if len(numeric_features) < len(input_features):
        excluded = set(input_features) - set(numeric_features)
        logger.warning(f"Excluding non-numeric features: {excluded}")

    if numeric_features:
        exog_train = train_data.loc[:, numeric_features]
        exog_val = val_data.loc[:, numeric_features]
        exog_test = test_data.loc[:, numeric_features]
```

**Critical Insight**: ARIMA/SARIMAX requires numeric exogenous variables. Categorical features are automatically filtered with warnings.

**Grid Search with Walk-Forward Validation** (train.py:1536-1677):
```python
# Walk-forward validation setup
n_folds = 5
initial_train_size = int(len(df) * split_ratios["train"])
optimization_metric = data.get("optimization_metric", "val_rmse")

# Prepare full time series
y_full = df[target_variable]
exog_full = df[numeric_features] if numeric_features else None

# Grid search
for i, params in enumerate(param_grid):
    try:
        fold_metrics = walk_forward_validate_sarimax(
            y_data=y_full,
            exog_data=exog_full,
            params=params,
            n_folds=n_folds,
            initial_train_size=initial_train_size,
            forecast_horizon=forecast_horizon
        )

        metric_value = fold_metrics.get(optimization_metric)

        if metric_value < best_score:
            best_score = metric_value
            best_params = params.copy()
            best_iteration_metrics = fold_metrics.copy()
```

**Walk-Forward Validation**: Expanding window approach ensures temporal integrity. Each fold trains on historical data, tests on future data.

**Reproducibility Fix** (train.py:1706-1716):
```python
# Compute stable start parameters
start_params = compute_stable_start_params(model_spec, y_train, exog_train)

fit_kwargs = {**SARIMAX_OPTIMIZER_DEFAULTS}
if start_params is not None:
    fit_kwargs['start_params'] = start_params
    logger.info(f"Using computed start_params for reproducibility")

model = model_spec.fit(**fit_kwargs)
```

**Impact**: Stable initialization + tight tolerances → consistent results across platforms.

#### B. train_xgboost_model()

**Location**: `train.py:2009`

**Feature Validation** (train.py:2069-2086):
```python
# Use input_features from frontend if provided
input_features = data.get("input_features", None)
if input_features:
    feature_cols = [col for col in input_features
                    if col in df.columns and col != target_variable]

    missing = set(input_features) - set(feature_cols) - {target_variable}
    if missing:
        logger.warning(f"Features not found: {missing}")

if len(feature_cols) == 0:
    raise ValueError("No valid features for XGBoost")
```

**Manual Training** (train.py:2219-2241):
```python
xgb_params = {
    "n_estimators": int(hyperparams.get("n_estimators", 100)),
    "max_depth": int(hyperparams.get("max_depth", 6)),
    "learning_rate": float(hyperparams.get("learning_rate", 0.1)),
    "subsample": float(hyperparams.get("subsample", 0.8)),
    "colsample_bytree": float(hyperparams.get("colsample_bytree", 0.8)),
    "gamma": float(hyperparams.get("gamma", 0)),
    "min_child_weight": int(hyperparams.get("min_child_weight", 1)),
    "random_state": SEED,  # Reproducibility
    "n_jobs": -1           # Use all CPU cores
}

model = xgb.XGBRegressor(**xgb_params)
model.fit(X_train, y_train)
```

**Random Search** (train.py:2176-2218):
```python
rng = np.random.default_rng(seed=SEED)  # Reproducible RNG

for i in range(n_random_iterations):
    random_params = generate_random_xgboost_params(random_search_params, rng)

    model_spec = xgb.XGBRegressor(**random_params)
    model_spec.fit(X_train, y_train)

    val_pred = model_spec.predict(X_val)
    val_score = calculate_ts_metric(y_val, val_pred, optimization_metric)

    if val_score < best_score:
        best_score = val_score
        best_model = model_spec
        best_params = random_params.copy()
```

#### C. train_lstm_model()

**Location**: `train.py:2947`

**CPU Warning** (train.py:3012-3017):
```python
logger.warning(
    "⚠️ LSTM training uses CPU only (no GPU support). "
    "Expected time: 30-60 minutes for 100 epochs. "
    "Consider reducing 'epochs' if time is excessive."
)
```

**Numeric Feature Validation** (train.py:3062-3081):
```python
# Validate LSTM inputs are numeric
for feature in input_features:
    if not pd.api.types.is_numeric_dtype(df[feature]):
        numeric_cols = [col for col in df.columns
                       if pd.api.types.is_numeric_dtype(df[col]) and col != target_variable]

        raise ValueError(
            f"Feature '{feature}' must be numeric for LSTM.\n"
            f"Current dtype: {df[feature].dtype}\n\n"
            f"LSTM requires numeric features. Categorical columns must be "
            f"encoded in 'data_encoding' step before training.\n\n"
            f"Encoding options:\n"
            f"  - One-Hot Encoding: For low cardinality categories\n"
            f"  - Label Encoding: For ordinal categories\n"
            f"  - Embedding Layers: For high cardinality (advanced)\n\n"
            f"Numeric columns available: {numeric_cols}"
        )
```

**Why Critical**: Prevents cryptic TensorFlow errors ("Can't convert string to tensor"). Provides actionable guidance.

**Training Mode Detection** (train.py:3085-3098):
```python
n_input_features = len(input_features)
if n_input_features == 1 and input_features[0] == target_variable:
    training_mode = "univariate"
    logger.info("Training LSTM in univariate mode (target only)")
elif n_input_features == 0:
    logger.warning("input_features empty - forcing univariate mode")
    input_features = [target_variable]
    training_mode = "univariate"
else:
    training_mode = "multivariate"
    logger.info(f"Training LSTM in multivariate mode with {n_input_features} features: {input_features}")
```

**Sequence Creation** (train.py:3103-3111):
```python
X, y = create_sequences_for_lstm(
    df=df,
    feature_cols=input_features,
    target_col=target_variable,
    sequence_length=sequence_length,
    forecast_horizon=forecast_horizon
)
```

**Manual Training** (train.py:3148-3200):
```python
lstm_params = data.get("lstm_params", {
    "lstm_units": [64],
    "dropout_rate": 0.2,
    "recurrent_dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
})

# Build model
input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, n_features)
model = build_lstm_model(lstm_params, input_shape)

# Create callbacks
callbacks, checkpoint_path = create_lstm_callbacks(
    experiment_dir=experiment_dir,
    early_stopping_patience=early_stopping_patience
)

# Train with EarlyStopping and ModelCheckpoint
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=lstm_params.get("epochs", 100),
    batch_size=lstm_params.get("batch_size", 32),
    callbacks=callbacks,
    verbose=1
)

best_val_loss = min(history.history["val_loss"])
mlflow.log_metric("best_val_loss", best_val_loss)
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. USER INTERACTION (TSTrainCard.jsx)                           │
│                                                                   │
│  User fills form:                                                │
│   ✅ Upload CSV                                                  │
│   ✅ Select algorithm (arima/xgboost/lstm)                       │
│   ✅ Configure hyperparameters                                   │
│   ✅ Set split ratios (train/val/test)                           │
│   ✅ Choose optimization method                                  │
│                                                                   │
│  Validation:                                                     │
│   ✅ inputFeatures.length > 0                                    │
│   ✅ targetVariable !== dateColumnName                           │
│   ✅ splitRatios sum === 1.0                                     │
│   ✅ runId exists (from encoding step)                           │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 2. PAYLOAD CONSTRUCTION (TSTrainCard.jsx:724-794)               │
│                                                                   │
│  FormData:                                                       │
│   • file: Binary CSV                                             │
│   • data: JSON.stringify({                                       │
│       model_name, input_features, target_variable,              │
│       date_col_name, experiment_dir, split_ratios,              │
│       run_id, algorithm, manual_params,                         │
│       forecast_horizon, hyperparameter_search_strategy          │
│     })                                                           │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 3. HTTP REQUEST                                                  │
│                                                                   │
│  axios.post("/ts/train-model/", formData)                        │
│   → Content-Type: multipart/form-data                            │
│   → Method: POST                                                 │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 4. DJANGO VIEW (views.py:376)                                   │
│                                                                   │
│  Validation:                                                     │
│   • Check method === POST                                        │
│   • Check 'file' in request.FILES                                │
│   • Parse: data = json.loads(request.POST['data'])               │
│   • Validate experiment_dir exists                               │
│                                                                   │
│  MLflow Setup:                                                   │
│   • Set tracking URI → shared_mlflow.db                          │
│   • Clean up orphaned runs                                       │
│                                                                   │
│  Service Delegation:                                             │
│   result = trainModelService.train_model_logic(...)              │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 5. SERVICE ORCHESTRATION (services.py:946)                      │
│                                                                   │
│  Workflow:                                                       │
│   1. Validate algorithm in ["arima", "xgboost", "lstm"]          │
│   2. Start MLflow run (with system metrics)                      │
│   3. Save dataset → trained/dataset_YYYYMMDD_HHMMSS.csv          │
│   4. DVC version: dvc add → git add → git commit → dvc push      │
│   5. Log dataset to MLflow                                       │
│   6. Dispatch to train_arima/xgboost/lstm_model()                │
│   7. DVC version model                                           │
│   8. Consolidate metrics (filter None values)                    │
│   9. Return result with step_config                              │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 6. ALGORITHM TRAINING (train.py)                                │
│                                                                   │
│  Common Preprocessing:                                           │
│   • Load: pd.read_csv(dataset_path, encoding='utf-8')            │
│   • Validate date column and target                              │
│   • Split: train/val/test (temporal ordering)                    │
│   • Log platform info (Python/NumPy/OS versions)                 │
│                                                                   │
│  Algorithm-Specific:                                             │
│                                                                   │
│  [ARIMA]                                                         │
│   • Filter numeric exogenous variables                           │
│   • Build ARIMA/SARIMA with optimizer defaults                   │
│   • Fit with stable_start_params                                 │
│   • Forecast on val/test                                         │
│   • Calculate RMSE, MAE, MAPE                                    │
│   • Save with pickle, log to MLflow                              │
│                                                                   │
│  [XGBoost]                                                       │
│   • Validate numeric features                                    │
│   • Train XGBRegressor(random_state=SEED)                        │
│   • Evaluate on val/test                                         │
│   • Save with pickle, log to MLflow                              │
│                                                                   │
│  [LSTM]                                                          │
│   • Validate features are numeric (detailed error if not)        │
│   • Determine univariate vs multivariate mode                    │
│   • Create sequences with lookback window                        │
│   • Build LSTM with TensorFlow/Keras                             │
│   • Train with EarlyStopping + ModelCheckpoint                   │
│   • Save with model.save(), log to MLflow                        │
│                                                                   │
│  Hyperparameter Optimization:                                    │
│   • Manual: Use provided params                                  │
│   • Grid: Walk-forward validation                                │
│   • Random: Sample from distributions (n_iterations)             │
│   • Bayesian: Gaussian Process (LSTM only)                       │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 7. RESPONSE PROPAGATION                                         │
│                                                                   │
│  train.py → services.py → views.py → Frontend                    │
│                                                                   │
│  Final Response:                                                 │
│   {                                                              │
│     status: "success",                                           │
│     run_id: "mlflow_run_id",                                     │
│     metrics: {val_rmse, val_mae, val_mape},                      │
│     model_path: "ModelName.pkl",                                 │
│     mlflow_ui: "http://localhost:5000/#/experiments/..."         │
│   }                                                              │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 8. FRONTEND RESPONSE HANDLING (TSTrainCard.jsx:802-836)         │
│                                                                   │
│  Success:                                                        │
│   • Format metrics to 4 decimals                                 │
│   • Display success message                                      │
│   • markStepDone("trainDone")                                    │
│   • setTrainInProgress(false)                                    │
│                                                                   │
│  Error:                                                          │
│   • Extract error.response.data.message                          │
│   • Display error to user                                        │
│   • setTrainInProgress(false)                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## I/O Interface Contracts

### Frontend → Backend (Request)

**Endpoint**: `POST /api/ts/train-model/`
**Content-Type**: `multipart/form-data`

**FormData Structure**:
```javascript
formData.append("file", csvFile)              // Binary CSV
formData.append("data", JSON.stringify({      // JSON config
  experiment_dir: "/absolute/path",           // Required
  model_name: "ModelName",                    // Required
  run_id: "abc123",                           // Required (from encoding step)
  algorithm: "arima" | "xgboost" | "lstm",    // Required
  input_features: ["feature1", "feature2"],   // Required (array)
  target_variable: "target",                  // Required
  date_col_name: "date",                      // Required
  split_ratios: {train: 0.7, val: 0.15, test: 0.15},  // Sum must be 1.0
  forecast_horizon: 12,                       // Integer >= 1
  problem_type: "ts_forecasting",             // Fixed
  hyperparameter_search_strategy: "manual" | "grid" | "random" | "bayesian",
  optimization_metric: "val_rmse" | "val_mae" | "val_mape",

  // Conditional: Manual parameters
  manual_params: {
    // ARIMA
    p: "1", d: "1", q: "1",                   // Strings (will be parsed to int)
    seasonal_P: "1", seasonal_D: "1",         // Optional seasonality
    seasonal_Q: "1", seasonal_s: "12",
    trend: "n" | "c" | "t" | "ct",
    enforce_stationarity: "True" | "False",
    enforce_invertibility: "True" | "False",

    // XGBoost
    n_estimators: 100, max_depth: 6,          // Numbers
    learning_rate: 0.1, subsample: 0.8,
    gamma: 0, min_child_weight: 1,

    // LSTM
    lstm_units: [64],                         // Array of integers
    dropout_rate: 0.2, recurrent_dropout_rate: 0.2,
    learning_rate: 0.001, batch_size: 32, epochs: 100
  },

  // Conditional: Grid search
  grid_search_params: {
    // For ARIMA: p, d, q arrays
    // For XGBoost: parameter arrays
    // For LSTM: lstm_units_options, dropout_rate_options, etc.
  },

  // Conditional: Random search
  n_random_iterations: 100,
  random_search_params: {
    // Parameter ranges for sampling
  },

  // LSTM-specific
  sequence_length: 10,                        // Lookback window
  early_stopping_patience: 20,
  training_mode: "univariate" | "multivariate"  // Inferred from input_features
}))
```

### Backend → Frontend (Response)

**Success (HTTP 200)**:
```json
{
  "status": "success",
  "run_id": "7f8a9b0c1d2e3f4g",
  "metrics": {
    "val_rmse": 0.1234,
    "val_mae": 0.0567,
    "val_mape": 2.34
  },
  "model_path": "ModelName.pkl",
  "mlflow_ui": "http://localhost:5000/#/experiments/1/runs/7f8a9b0c1d2e3f4g"
}
```

**Error (HTTP 400/404/500)**:
```json
{
  "status": "error",
  "message": "High-level error description",
  "error_details": "Technical debugging details",
  "error_code": "HTTP_400_BAD_REQUEST"
}
```

---

## User-Facing Options

### Algorithm Selection

| Algorithm | Best For | Strengths | Limitations |
|-----------|----------|-----------|-------------|
| **ARIMA** | Univariate TS with trends/seasonality | Interpretable, fast, statistical rigor | Linear patterns only, requires stationarity |
| **XGBoost** | Multivariate TS with complex patterns | Handles non-linearity, feature importance | Requires feature engineering (lags) |
| **LSTM** | Long-term dependencies, complex patterns | Learns temporal patterns, univariate/multivariate | Slow (30-60 min), CPU-only, requires numeric features |

### Hyperparameter Search Strategies

| Strategy | Description | Computational Cost | When to Use |
|----------|-------------|-------------------|-------------|
| **Manual** | User provides exact parameters | Low (single training run) | You know optimal params OR initial exploration |
| **Grid Search** | Exhaustive search over parameter grid | High (N^params combinations) | Small parameter space, budget available |
| **Random Search** | Sample random combinations | Medium (n_iterations runs) | Large parameter space, want faster results |
| **Bayesian Search** | Gaussian Process optimization | Medium (intelligent sampling) | Most efficient search, LSTM only |

### ARIMA Parameters

| Parameter | Range | Description | Default | Notes |
|-----------|-------|-------------|---------|-------|
| `p` | [0, 5] | AR order (lagged observations) | 1 | Higher = more past influence |
| `d` | [0, 2] | Differencing order (stationarity) | 1 | 1 or 2 usually sufficient |
| `q` | [0, 5] | MA order (moving average) | 1 | Higher = more smoothing |
| `seasonal_P` | [0, 2] | Seasonal AR | 1 | For seasonal patterns |
| `seasonal_D` | [0, 2] | Seasonal differencing | 1 | Removes seasonal trends |
| `seasonal_Q` | [0, 2] | Seasonal MA | 1 | Seasonal smoothing |
| `seasonal_s` | [1, ∞) | Seasonal period | 12 | 12=monthly, 7=weekly, 4=quarterly |
| `trend` | n/c/t/ct | Trend component | n | n=none, c=constant, t=linear, ct=both |

**Backend Behavior**:
- Filters exogenous variables to numeric only (warns if non-numeric excluded)
- Uses `compute_stable_start_params()` for reproducible initialization
- Fits with `SARIMAX_OPTIMIZER_DEFAULTS` for cross-platform consistency

### XGBoost Parameters

| Parameter | Range | Description | Default |
|-----------|-------|-------------|---------|
| `n_estimators` | [50, 1000] | Number of boosting rounds | 100 |
| `max_depth` | [3, 10] | Maximum tree depth | 6 |
| `learning_rate` | [0.01, 0.3] | Step size shrinkage | 0.1 |
| `subsample` | [0.5, 1.0] | Fraction of samples per tree | 0.8 |
| `gamma` | [0, 5] | Minimum loss reduction for split | 0 |
| `min_child_weight` | [1, 10] | Minimum sum of weights in child | 1 |

**Backend Behavior**:
- Validates `input_features` exist in dataset
- Trains with `random_state=SEED` for reproducibility
- Uses all CPU cores (`n_jobs=-1`)
- Logs feature importances to MLflow

### LSTM Parameters

| Parameter | Range | Description | Default | Notes |
|-----------|-------|-------------|---------|-------|
| `lstm_units` | e.g., [64], [128, 64] | LSTM units per layer | [64] | Array defines architecture |
| `dropout_rate` | [0.0, 0.8] | Dropout regularization | 0.2 | Prevents overfitting |
| `recurrent_dropout_rate` | [0.0, 0.8] | Recurrent dropout | 0.2 | Regularizes LSTM connections |
| `learning_rate` | [0.0001, 0.01] | Adam optimizer LR | 0.001 | Lower = more stable |
| `batch_size` | [16, 64] | Samples per gradient update | 32 | Higher = faster but less precise |
| `epochs` | [50, 500] | Maximum training epochs | 100 | Early stopping prevents overfitting |
| `sequence_length` | [1, ∞) | Lookback window size | 10 | How far back to look |
| `early_stopping_patience` | [1, ∞) | Epochs to wait for improvement | 20 | Stops if no val improvement |

**Backend Behavior**:
- **Validates numeric features**: Raises detailed error with encoding suggestions if non-numeric
- **Determines mode**: Univariate (target only) vs Multivariate (target + features)
- **Creates sequences**: Sliding window with `sequence_length` timesteps
- **Trains with callbacks**: EarlyStopping, ModelCheckpoint (saves best model)
- **CPU-only**: Training takes 30-60 minutes for 100 epochs

### Split Ratios

| Component | Range | Constraint | Description |
|-----------|-------|-----------|-------------|
| `train` | [0, 1] | sum = 1.0 | Training data proportion |
| `val` | [0, 1] | sum = 1.0 | Validation data (hyperparameter tuning) |
| `test` | [0, 1] | sum = 1.0 | Test data (final evaluation) |

**Backend Behavior**:
- **Temporal ordering preserved**: Always splits chronologically (no shuffling)
- **Walk-forward validation**: Grid search uses expanding training window

---

## Reproducibility Chain

### 1. Seed Management

**Global Seed**: `SEED = 42` (configurable via `RANDOM_SEED` env var)

**Initialized Libraries**:
- Python `random` module
- NumPy `np.random`
- TensorFlow `tf.random` + deterministic ops
- Environment variables: `TF_DETERMINISTIC_OPS=1`, `PYTHONHASHSEED=42`

**Usage**:
- XGBoost: `XGBRegressor(random_state=SEED)`
- Random Search: `rng = np.random.default_rng(seed=SEED)`
- LSTM: TensorFlow deterministic mode enabled

### 2. Explicit UTF-8 Encoding

**All Data Loading** (train.py:141-143):
```python
df = pd.read_csv(dataset_path, encoding='utf-8')
```

**Rationale**: Prevents encoding variance across OS locales (Windows CP-1252 vs Linux UTF-8).

### 3. SARIMAX Optimizer Defaults

**Configuration**:
```python
{
  "method": "lbfgs",       # L-BFGS-B optimizer
  "maxiter": 500,          # Increased for better convergence
  "ftol": 1e-8,            # Tighter function tolerance
  "gtol": 1e-6             # Tighter gradient tolerance
}
```

**Impact**: Reduces ARIMA cross-platform variance from 5-10% → 0.5-2%.

### 4. Stable Start Parameters (ARIMA)

**Function**: `compute_stable_start_params(model_spec, y_train, exog_train)`

**Purpose**: Provides deterministic initialization for ARIMA/SARIMAX models.

**Usage**: Applied before `model.fit()` to reduce optimizer variance.

### 5. Platform Information Logging

**Logged to MLflow**:
- Python version
- Platform (OS, architecture, processor)
- NumPy, SciPy, statsmodels versions

**Purpose**: Debug cross-platform discrepancies by recording exact environment.

### 6. MLflow Parameter Tracking

**What Gets Logged**:
- Training parameters (algorithm, hyperparameters, split_ratios)
- Search strategy (hyperparameter_search_strategy, optimization_metric)
- Data characteristics (input_features, target_variable)
- Best parameters (for grid/random/bayesian search)
- Platform info
- Energy metrics (kWh, carbon kg)

### 7. DVC Versioning

**Versioned Artifacts**:
1. Dataset: `trained/dataset_YYYYMMDD_HHMMSS.csv`
2. Model: `{model_name}.pkl`

**Workflow**:
```bash
dvc add <artifact>                  # Create .dvc metadata
git add <artifact>.dvc              # Stage metadata
git commit -m "[DVC] Add artifact"  # Commit to Git
dvc push <artifact>                 # Push to remote storage
```

**Reproducibility**: Enables exact experiment recreation:
```bash
git checkout <commit_sha>   # Restore code + metadata
dvc pull                     # Download versioned data/models
```

### 8. pipeline_config.json

**Purpose**: Single source of truth for experiment reproducibility.

**Structure**:
```json
{
  "steps": [
    {
      "step": "train_arima",
      "run_id": "mlflow_run_id",
      "algorithm": "arima",
      "dataset_path": "trained/dataset_20251215_143022.csv",
      "model_path": "ARIMA_Model.pkl",
      "metrics": {
        "val": {"val_rmse": 0.123, "val_mae": 0.045},
        "test": {"test_rmse": 0.135, "test_mae": 0.050}
      },
      "timestamp": "2025-12-15T14:35:10.123456"
    }
  ]
}
```

**Recreating Experiment**:
1. Read `pipeline_config.json` for parameters
2. Use DVC to retrieve exact dataset: `dvc get . <dataset_path>`
3. Call training function with exact parameters
4. Compare metrics with original run

---

## Architecture Insights

### 1. Three-Tier Separation of Concerns

```
Views (views.py)
  ↓ HTTP validation, error formatting
Services (services.py)
  ↓ MLflow/DVC orchestration, algorithm dispatch
Training Logic (train.py)
  ↓ Algorithm-specific implementation
```

**Benefits**:
- Views don't know ML details
- Services don't know HTTP details
- Training functions reusable (notebooks, CLI)
- Easy to test each layer independently

### 2. Idempotent Operations

**Example**: Dataset saving (services.py:998-1004)
```python
if not os.path.exists(dataset_path) or os.path.getsize(dataset_path) == 0:
    # Save file
else:
    logger.info("File exists, using without rewrite")
```

**Benefit**: Safe pipeline re-runs. If training fails, re-running doesn't duplicate files.

### 3. Context Managers for Cleanup

**MLflow Run Management**:
```python
with start_run(experiment_id=..., log_system_metrics=True) as run:
    # Training logic
    # Automatic cleanup on exception or completion
```

**Benefits**:
- No orphaned runs
- Exception-safe
- Clear scope boundaries

### 4. Metric Filtering for Edge Cases

**Implementation** (services.py:1057-1065):
```python
filtered_metrics = {k: v for k, v in metrics.items() if v is not None}
mlflow.log_metrics(filtered_metrics)
```

**Rationale**: MAPE can be `None` with division by zero. MLflow rejects `None` values.

### 5. Energy Tracking

**Pattern**:
```python
tracker = EmissionsTracker(...)
tracker.start()
# Training
tracker.stop()
mlflow.log_metric("energy_consumed_total_kWh", tracker._total_energy)
mlflow.log_metric("carbon_emission_kg", tracker.final_emissions)
```

**Purpose**: Track environmental impact for carbon footprint analysis.

---

## Open Questions

### 1. Why are test_metrics not returned to frontend?

**Current Behavior**: views.py:434 returns only `val_metrics` to frontend, but service computes both validation and test metrics.

**Hypothesis**: Intentional to prevent overfitting to test set. Test metrics logged to MLflow for auditing.

**Question**: Should frontend display test metrics after training? Or keep them hidden to enforce good ML practices?

### 2. How to handle MAPE when y_true contains zeros?

**Current Behavior**: Returns `None`, filtered before MLflow logging.

**Alternatives**:
- Use SMAPE (Symmetric MAPE) instead
- Skip MAPE calculation entirely
- Warn user when MAPE is `None`

**Question**: Should backend provide guidance when MAPE is unavailable?

### 3. What happens if grid search finds no valid models?

**Current Behavior**: Raises `RuntimeError("All iterations failed")`.

**Question**: Should backend provide more actionable guidance on fixing parameter grid (e.g., suggest looser constraints)?

### 4. How are categorical features handled in XGBoost?

**Current State**: XGBoost can handle categorical features natively with `enable_categorical=True`, but current implementation doesn't use this.

**Question**: Should data encoding step apply OHE before training, or should XGBoost use native categorical support?

### 5. Inconsistency: `lstm_params` vs `manual_params`?

**Observation**:
- ARIMA/XGBoost extract from `data.get("manual_params")`
- LSTM extracts from `data.get("lstm_params")`

**Question**: Is this a schema inconsistency or intentional design? Should LSTM use `manual_params` like other algorithms?

### 6. Bayesian Search for ARIMA/XGBoost?

**Current State**: Bayesian Search implemented for LSTM only (train.py:3202-3400).

**Question**: Is Bayesian Search planned for ARIMA/XGBoost, or intentionally LSTM-only due to long training times?

### 7. Purpose of `problem_type` field?

**Current State**: `problem_type = "ts_forecasting"` is fixed for all TS models.

**Question**: Is this field reserved for future TS classification/regression tasks, or can it be removed?

---

## Code References

**Frontend**:
- Form validation: `TSTrainCard.jsx:641-673`
- Payload construction: `TSTrainCard.jsx:724-790`
- Response handling: `TSTrainCard.jsx:802-836`

**Backend Views**:
- Request validation: `views.py:376-396`
- MLflow setup: `views.py:402-410`
- Response formatting: `views.py:422-437`

**Backend Services**:
- Dataset versioning: `services.py:991-1012`
- Algorithm dispatch: `services.py:1025-1043`
- Metrics consolidation: `services.py:1056-1067`

**Backend Training**:
- Global seeds: `train.py:97-131`
- ARIMA reproducibility fix: `train.py:1706-1716`
- XGBoost training: `train.py:2219-2241`
- LSTM numeric validation: `train.py:3062-3081`
- LSTM training: `train.py:3148-3200`

---

## Summary for Data Scientists

### Prerequisites Before Training

1. ✅ Create experiment (experimentDir in AppContext)
2. ✅ Initialize DVC/Git
3. ✅ Upload & clean data
4. ✅ Run EDA (recommended)
5. ✅ **Encode data** (lag features, OHE for categorical)
6. ✅ Verify `runId` exists (from encoding step)

### Training Workflow

1. **Upload CSV** → Click "Cargar Variables"
2. **Select Algorithm**:
   - ARIMA: Univariate TS with seasonality
   - XGBoost: Multivariate TS with non-linear patterns
   - LSTM: Complex long-term dependencies (slow, 30-60 min)
3. **Select Variables**: Input features, target, date column (no overlaps)
4. **Configure Hyperparameters**: Manual/Grid/Random/Bayesian
5. **Set Training Options**: Forecast horizon, split ratios (sum to 1.0), model name
6. **Submit**: Click "Entrenar Modelo"
7. **Review Results**: Validation metrics in frontend, full details in MLflow UI

### Common Pitfalls

❌ **LSTM with categorical features** → Apply OHE/Label Encoding first
❌ **ARIMA with categorical features** → Silently filtered (check logs)
❌ **Split ratios ≠ 1.0** → Blocked by validation
❌ **Skipping encoding step** → Blocked (requires runId)
❌ **Large grid search** → Use Random Search for LSTM (faster)
❌ **No lag features for XGBoost** → Add in encoding step (3-7 lags)

### Reproducibility Best Practices

1. ✅ Use same environment (Python/NumPy versions match)
2. ✅ Don't modify SEED (default=42 ensures reproducibility)
3. ✅ Save pipeline_config.json for documentation
4. ✅ DVC handles data/models automatically
5. ✅ Document hyperparameter choices in experiment notes

---

**End of Research Document**

This analysis provides complete understanding of the time series training workflow, enabling data scientists to effectively use the feature while ensuring experiment reproducibility through MLflow, DVC, and robust seed management.
