# LSTM Model Training Implementation Research

**Date**: 2025-11-06
**Scope**: Django Backend (apiTimeSeries) + React Frontend (TSTrainCard.jsx)
**Objective**: Implement LSTM model training following established ARIMA/XGBoost patterns
**Status**: 🔴 CRITICAL - LSTM function imported but not implemented (ImportError at runtime)

---

## Executive Summary

### Current State
- ✅ ARIMA training fully implemented with grid/random search
- ✅ XGBoost training fully implemented with grid/random search + feature engineering
- ⚠️ LSTM training **imported but not implemented** in [train.py:48](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L48)
- ⚠️ Frontend TSTrainCard.jsx already has LSTM UI (algorithm dropdown line 1087, params lines 236-292)
- ✅ Service layer routing ready for LSTM at [services.py:1036](DREAM-ML-backend/GEML/apiTimeSeries/services.py#L1036)
- ✅ Test file exists: [test_lstm_fixes.py](DREAM-ML-backend/GEML/apiTimeSeries/tests/test_lstm_fixes.py) documenting expected behavior

### Critical Issue
```python
# services.py:48 - Will cause ImportError
from apiTimeSeries.train import train_lstm_model  # Function does NOT exist
```

Git status shows `train.py` was recently modified (M DREAM-ML-backend/GEML/apiTimeSeries/train.py), and commit `55f7552` says "added TS LSTM model training" - suggesting incomplete implementation or deletion.

---

## Architecture Overview

### Three-Tier Service Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (TSTrainCard.jsx)                  │
│  - Algorithm selection (ARIMA/XGBoost/LSTM)                    │
│  - Hyperparameter configuration UI                              │
│  - Optimization method selection                                │
│  - CSV upload + variable selection                              │
└────────────────────────┬────────────────────────────────────────┘
                         │ POST /api/ts/train-model/
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 View Layer (views.py:376-480)                   │
│  - Request validation (file, data JSON)                        │
│  - MLflow run lifecycle management                              │
│  - Delegates to TrainModelService                               │
└────────────────────────┬────────────────────────────────────────┘
                         │ .train_model_logic()
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Service Layer (services.py:944-1091)               │
│  - Algorithm routing (arima/xgboost/lstm)                      │
│  - MLflow experiment initialization                             │
│  - DVC versioning (model artifacts)                             │
│  - WebSocket progress updates                                   │
│  - Pipeline config JSON updates                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │ train_lstm_model(dataset_path, data, experiment_dir)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Training Module (train.py:1529+ MISSING)           │
│  - Data loading & validation                                    │
│  - Sequence creation (3D tensors)                               │
│  - Train/val/test splitting (temporal)                          │
│  - Hyperparameter optimization (grid/random/bayesian)           │
│  - Model training with callbacks                                │
│  - Evaluation & metrics logging                                 │
│  - MLflow model registration                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Existing Training Patterns

### Function Signature Contract

All training functions follow this signature:

```python
def train_[algorithm]_model(dataset_path: str, data: Dict, experiment_dir: str) -> Dict:
    """
    Args:
        dataset_path: Path to CSV dataset
        data: Configuration dictionary from frontend
        experiment_dir: Experiment directory for artifacts

    Returns:
        {
            "status": "success" | "error",
            "val_metrics": {"rmse": float, "mae": float, "mape": float},
            "test_metrics": {"rmse": float, "mae": float, "mape": float},
            "model_path": str,  # Relative to experiment_dir
            "run_id": str,  # MLflow run ID
            "features_used": List[str]  # Optional
        }
    """
```

**Reference Implementation:**
- ARIMA: [train.py:785-1098](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L785-L1098)
- XGBoost: [train.py:1104-1459](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L1104-L1459)

### Common Parameters (from `data` dict)

| Parameter | Type | Default | Source | Description |
|-----------|------|---------|--------|-------------|
| `date_col_name` | str | - | Required | Date column for temporal ordering |
| `target_variable` | str | - | Required | Target column name |
| `input_features` | List[str] | - | Required for XGBoost/LSTM | Feature columns |
| `model_name` | str | - | Required | MLflow registered model name |
| `forecast_horizon` | int | 10 | [train.py:799](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L799) | Steps ahead to predict |
| `split_ratios` | Dict | `{"train": 0.7, "val": 0.15, "test": 0.15}` | [train.py:803](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L803) | Temporal split ratios |
| `hyperparameter_search_strategy` | str | "none" | [train.py:807](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L807) | "none" \| "grid" \| "random" \| "bayesian" |
| `n_random_iterations` | int | 100 | [train.py:824](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L824) | Random search iterations |
| `random_search_params` | Dict | - | [train.py:825](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L825) | Parameter ranges for random search |

### LSTM-Specific Parameters

Based on [test_lstm_fixes.py:116-144](DREAM-ML-backend/GEML/apiTimeSeries/tests/test_lstm_fixes.py#L116-L144):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequence_length` | int | 10 | Lookback window for LSTM sequences |
| `early_stopping_patience` | int | 5-20 | Epochs to wait for validation improvement |
| `optimization_metric` | str | "mse" | Metric for hyperparameter search validation |
| `n_bayesian_iterations` | int | 30 | Bayesian optimization iterations |
| `bayesian_search_params` | Dict | - | Parameter space definition (see below) |
| `bayesian_config` | Dict | - | Advanced Bayesian optimizer config |

### Hyperparameter Search Strategies

#### 1. Manual (No Search)

```python
# Frontend provides explicit params
params = {
    "lstm_units": [64],  # Single layer
    "dropout_rate": 0.2,
    "recurrent_dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
}
```

**Reference:** [TSTrainCard.jsx:240-247](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L240-L247)

#### 2. Grid Search

**ARIMA Implementation:** [train.py:862-926](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L862-L926)

```python
# Exhaustive combination of discrete values
param_grid = []
for p in range(0, 4):
    for d in range(0, 3):
        for q in range(0, 4):
            param_grid.append({"order": (p, d, q)})

best_aic = float('inf')
for params in param_grid:
    model = ARIMA(..., order=params["order"]).fit()
    if model.aic < best_aic:
        best_aic = model.aic
        best_model = model
```

**LSTM Grid Example** (needs implementation):
```python
param_grid = [
    {"lstm_units": [32], "dropout": 0.0, "lr": 0.001, "batch": 16},
    {"lstm_units": [64], "dropout": 0.2, "lr": 0.001, "batch": 32},
    # ... all combinations
]
```

#### 3. Random Search

**XGBoost Implementation:** [train.py:1299-1340](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L1299-L1340)

```python
# Random sampling from parameter ranges
for i in range(n_random_iterations):
    random_params = generate_random_xgboost_params(random_search_params)
    model = xgb.XGBRegressor(**random_params).fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_pred)
    if val_mse < best_val_mse:
        best_model = model
        best_params = random_params
```

**Random Parameter Generator Pattern:** [train.py:703-748](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L703-L748)

```python
def generate_random_xgboost_params(random_search_params: Dict) -> Dict:
    params = {
        "n_estimators": int(np.random.randint(*random_search_params["n_estimators_range"])),
        "max_depth": int(np.random.randint(*random_search_params["max_depth_range"])),
        # Log-uniform for learning rate
        "learning_rate": float(np.exp(np.random.uniform(
            np.log(random_search_params["learning_rate_range"][0]),
            np.log(random_search_params["learning_rate_range"][1])
        ))),
        # ... more params
    }
    return params
```

**LSTM Random Search Params:** [test_lstm_fixes.py:129-136](DREAM-ML-backend/GEML/apiTimeSeries/tests/test_lstm_fixes.py#L129-L136)

```python
"random_search_params": {
    "lstm_units_options": [[32], [64], [128], [64, 32]],  # Categorical
    "dropout_rate_range": [0.0, 0.3],
    "recurrent_dropout_rate_range": [0.0, 0.3],
    "learning_rate_range": [0.001, 0.01],
    "batch_size_options": [16, 32, 64],  # Categorical
    "epochs_range": [20, 100]
}
```

#### 4. Bayesian Search (LSTM Only)

**NOT implemented for ARIMA/XGBoost**, but test file shows expected structure for LSTM.

**Reference:** [test_lstm_fixes.py:198-244](DREAM-ML-backend/GEML/apiTimeSeries/tests/test_lstm_fixes.py#L198-L244)

```python
"bayesian_search_params": {
    "lstm_units": {
        "type": "categorical",
        "choices": [[32], [64], [128]]  # Must be tuples for hashing
    },
    "dropout_rate": {
        "type": "real",
        "distribution": "uniform",
        "low": 0.0,
        "high": 0.3
    },
    "learning_rate": {
        "type": "real",
        "distribution": "log-uniform",
        "low": 0.0001,
        "high": 0.01
    },
    "batch_size": {
        "type": "categorical",
        "choices": [16, 32]
    },
    "epochs": {
        "type": "integer",
        "low": 20,
        "high": 50
    }
}
```

**Bayesian Advanced Config:** [test_lstm_fixes.py:246-253](DREAM-ML-backend/GEML/apiTimeSeries/tests/test_lstm_fixes.py#L246-L253)

```python
"bayesian_config": {
    "n_initial_points": 5,  # Random exploration before GP optimization
    "acq_func": "EI",  # Expected Improvement (or "PI", "LCB")
    "convergence_tolerance": 0.001,
    "convergence_patience": 5,
    "save_gp_model": True
}
```

---

## Data Flow Comparison

### XGBoost (2D: Samples × Features)

```
CSV Dataset (time series)
    ↓
load_and_validate_ts_data() → pd.DataFrame with datetime index
    ↓
prepare_xgboost_features() → Validate external features
    ↓ [train.py:1184-1193]
xgboost_train_val_test_split()
    ├→ create_supervised_dataset() [train.py:400-436]
    │   ├→ Shift target by -forecast_horizon
    │   └→ Remove last N rows (no future target)
    ├→ Temporal split (70% train, 15% val, 15% test)
    └→ Returns: X_train (2D), y_train (1D), X_val, y_val, X_test, y_test
    ↓
Hyperparameter Search Loop
    ├→ xgb.XGBRegressor(**params).fit(X_train, y_train)
    ├→ val_pred = model.predict(X_val)
    └→ Evaluate metric, update best_model
    ↓
evaluate_xgboost_model() → Metrics + plots
    ↓
mlflow.sklearn.log_model() → Register in MLflow
```

### LSTM (3D: Samples × Sequence Length × Features) - NEEDS IMPLEMENTATION

```
CSV Dataset (time series)
    ↓
load_and_validate_ts_data() → pd.DataFrame with datetime index
    ↓
🔴 create_sequences_for_lstm() [MISSING]
    ├→ Sliding window over time series
    ├→ Input shape: (n_sequences, sequence_length, n_features)
    ├→ Target shape: (n_sequences,) or (n_sequences, forecast_horizon)
    └→ Returns: X_sequences (3D), y_sequences (1D or 2D)
    ↓
🔴 lstm_train_val_test_split() [MISSING]
    ├→ Temporal split maintaining 3D shape
    └→ Returns: X_train (3D), y_train, X_val, y_val, X_test, y_test
    ↓
Hyperparameter Search Loop
    ├→ 🔴 build_lstm_model(params, input_shape) [MISSING]
    ├→ model.fit(X_train, y_train,
    │            validation_data=(X_val, y_val),
    │            callbacks=[EarlyStopping, ModelCheckpoint],
    │            epochs=params["epochs"],
    │            batch_size=params["batch_size"])
    ├→ Extract best val_loss from history.history
    ├→ 🔴 Clear session: tf.keras.backend.clear_session() [MISSING]
    └→ Update best_model, best_params
    ↓
🔴 evaluate_lstm_model() [MISSING] → Metrics + plots
    ↓
mlflow.keras.log_model() → Register in MLflow (NOT sklearn)
```

---

## Key Implementation Differences

| Aspect | ARIMA | XGBoost | LSTM (Required) |
|--------|-------|---------|-----------------|
| **Data Shape** | 1D (univariate) | 2D (samples × features) | **3D (samples × seq_len × features)** |
| **Feature Engineering** | None | Lag columns, rolling stats | **Sliding window sequences** |
| **Model Type** | statsmodels.SARIMAX | xgboost.XGBRegressor | **tensorflow.keras.Sequential** |
| **Training API** | `model.fit()` returns fitted model | `model.fit(X, y)` in-place | **`model.fit(X, y, validation_data, callbacks)`** |
| **Best Model Tracking** | AIC/BIC (lower is better) | Validation MSE | **`min(history.history['val_loss'])`** |
| **Model Saving** | `pickle.dump()` | `pickle.dump()` | **`model.save()` or Keras format** |
| **MLflow Logging** | `mlflow.sklearn.log_model()` | `mlflow.sklearn.log_model()` | **`mlflow.keras.log_model()`** |
| **Memory Management** | Low (single fit) | Moderate | **🔴 HIGH - Must clear session after each iteration** |
| **Callbacks** | N/A | N/A | **EarlyStopping, ModelCheckpoint, ReduceLROnPlateau** |
| **Progress Tracking** | Per grid/random iteration | Per grid/random iteration | **Per epoch + per iteration** |

---

## Critical Code References

### Backend (Django)

#### 1. Service Layer Routing
**File:** [services.py:1013-1040](DREAM-ML-backend/GEML/apiTimeSeries/services.py#L1013-L1040)

```python
if algorithm == "arima":
    result = train_arima_model(dataset_path, data, experiment_dir)
elif algorithm == "xgboost":
    result = train_xgboost_model(dataset_path, data, experiment_dir)
elif algorithm == "lstm":
    result = train_lstm_model(dataset_path, data, experiment_dir)  # 🔴 DOES NOT EXIST
```

#### 2. Import Statement (ImportError Source)
**File:** [services.py:48](DREAM-ML-backend/GEML/apiTimeSeries/services.py#L48)

```python
from apiTimeSeries.train import train_arima_model, train_xgboost_model, train_lstm_model
```

#### 3. DVC Versioning Pattern
**File:** [services.py:1043-1051](DREAM-ML-backend/GEML/apiTimeSeries/services.py#L1043-L1051)

```python
# Add model to DVC tracking
subprocess.run(["dvc", "add", model_path], cwd=experiment_dir, check=True)
subprocess.run(["git", "add", f"{model_path}.dvc"], cwd=experiment_dir, check=True)
subprocess.run(["git", "commit", "-m", f"[DVC] Add model {os.path.basename(model_path)}"],
               cwd=experiment_dir, check=True)
subprocess.run(["dvc", "push", model_path], cwd=experiment_dir, check=True)
```

#### 4. MLflow Lifecycle Management
**File:** [views.py:409-421](DREAM-ML-backend/GEML/apiTimeSeries/views.py#L409-L421)

```python
# Cleanup before training
if mlflow.active_run():
    mlflow.end_run()
    logger.warning("Run MLflow activa detectada y finalizada")

# Delegate to service
result = trainModelService.train_model_logic(dataset_file=request.FILES['file'], data=data)

# Cleanup after training
if mlflow.active_run():
    mlflow.end_run()
```

#### 5. Request Payload Structure
**File:** [views.py:390-397](DREAM-ML-backend/GEML/apiTimeSeries/views.py#L390-L397)

```python
if 'file' not in request.FILES:
    raise ValueError("No se encontró archivo CSV")
if 'data' not in request.POST:
    raise ValueError("Datos de configuración faltantes")

data = json.loads(request.POST['data'])
```

#### 6. Temporal Data Splitting (No Shuffling)
**File:** [train.py:112-137](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L112-L137)

```python
def ts_train_val_test_split(df, target_variable, split_ratios):
    """
    Critical: Respects temporal order - NO random shuffling
    """
    n = len(df)
    train_size = int(n * split_ratios["train"])
    val_size = int(n * split_ratios["val"])

    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size + val_size]
    test_data = df.iloc[train_size + val_size:]

    return train_data, val_data, test_data
```

#### 7. Energy Tracking with CodeCarbon
**File:** [train.py:281-287](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L281-L287)

```python
def log_energy_metrics(tracker) -> Tuple[float, float]:
    energy_kwh = float(tracker._total_energy.kWh)
    emissions_kg = float(tracker.final_emissions)
    mlflow.log_metric("energy_consumed_total_kWh", energy_kwh)
    mlflow.log_metric("carbon_emission_kg", emissions_kg)
    return energy_kwh, emissions_kg
```

#### 8. Pipeline Config JSON Update
**File:** [train.py:325-353](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L325-L353)

```python
def save_pipeline_config(experiment_dir: str, step_config: Dict) -> None:
    pipeline_config_path = os.path.join(experiment_dir, "pipeline_config.json")

    # Load existing config
    if os.path.exists(pipeline_config_path):
        with open(pipeline_config_path, "r") as f:
            pipeline_config = json.load(f)
    else:
        pipeline_config = {"steps": []}

    # Append new step
    pipeline_config["steps"].append(step_config)

    # Save updated config
    with open(pipeline_config_path, "w") as f:
        json.dump(pipeline_config, f, indent=2)
```

#### 9. MLflow Model Registration Pattern
**File:** [train.py:1032-1043](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L1032-L1043)

```python
# For sklearn models (ARIMA, XGBoost)
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="arima_model",
    signature=signature,
    registered_model_name=model_name,
    metadata={
        "dataset": os.path.basename(dataset_path),
        "target": target_variable,
        "forecast_horizon": forecast_horizon
    }
)

# For LSTM (needs implementation):
# mlflow.keras.log_model(
#     model=model,
#     artifact_path="lstm_model",
#     registered_model_name=model_name,
#     ...
# )
```

### Frontend (React)

#### 1. Algorithm Selection Dropdown
**File:** [TSTrainCard.jsx:1069-1089](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L1069-L1089)

```jsx
<FormControl fullWidth sx={{ mb: 2 }}>
  <InputLabel>Selecciona un algoritmo</InputLabel>
  <Select
    value={algorithm}
    onChange={(e) => {
      setAlgorithm(e.target.value);
      // Clear params when switching algorithms
    }}
    label="Selecciona un algoritmo"
  >
    <MenuItem value="arima">ARIMA (Time Series)</MenuItem>
    <MenuItem value="xgboost">XGBoost (Time Series)</MenuItem>
    <MenuItem value="lstm">LSTM (Deep Learning)</MenuItem>
  </Select>
</FormControl>
```

#### 2. LSTM Manual Parameters State
**File:** [TSTrainCard.jsx:240-247](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L240-L247)

```jsx
const [lstmManualParams, setLstmManualParams] = useState({
  lstm_units: "[64]",  // String for display, parsed to array
  dropout_rate: "0.2",
  recurrent_dropout_rate: "0.2",
  learning_rate: "0.001",
  batch_size: "32",
  epochs: "100"
});
```

#### 3. LSTM Random Search Ranges
**File:** [TSTrainCard.jsx:249-257](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L249-L257)

```jsx
const [lstmRandomRanges, setLstmRandomRanges] = useState({
  lstm_units_options: ["[32]", "[64]", "[128]", "[64,32]", "[128,64]"],
  dropout_rate_range: [0.0, 0.5],
  recurrent_dropout_rate_range: [0.0, 0.5],
  learning_rate_range: [0.0001, 0.01],
  batch_size_options: [16, 32, 64],
  epochs_range: [50, 300]
});
```

#### 4. LSTM Bayesian Search Parameters
**File:** [TSTrainCard.jsx:259-292](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L259-L292)

```jsx
const [lstmBayesianParams, setLstmBayesianParams] = useState({
  lstm_units: {
    type: "categorical",
    choices: [[32], [64], [128], [64, 32], [128, 64]]
  },
  dropout_rate: {
    type: "real",
    distribution: "uniform",
    low: 0.0,
    high: 0.5
  },
  recurrent_dropout_rate: {
    type: "real",
    distribution: "uniform",
    low: 0.0,
    high: 0.5
  },
  learning_rate: {
    type: "real",
    distribution: "log-uniform",
    low: 0.0001,
    high: 0.01
  },
  batch_size: {
    type: "categorical",
    choices: [16, 32, 64]
  },
  epochs: {
    type: "integer",
    low: 50,
    high: 300
  }
});
```

#### 5. LSTM Sequence Configuration
**File:** [TSTrainCard.jsx:236-237](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L236-L237)

```jsx
const [sequenceLength, setSequenceLength] = useState(10);
const [earlyStoppingPatience, setEarlyStoppingPatience] = useState(20);
```

#### 6. Training Payload Construction
**File:** [TSTrainCard.jsx:743-761](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L743-L761)

```jsx
// LSTM-specific parameters
if (algorithm === "lstm") {
  payload.sequence_length = sequenceLength;
  payload.early_stopping_patience = earlyStoppingPatience;
  payload.optimization_metric = "mse"; // Default for LSTM
}

// Bayesian Search configuration
if (optimizationMethod === "bayesian") {
  payload.hyperparameter_search_strategy = "bayesian";
  payload.n_bayesian_iterations = nBayesianIterations;

  if (algorithm === "lstm") {
    payload.bayesian_search_params = lstmBayesianParams;
  }

  // Clean bayesianConfig (remove null values)
  const cleanBayesianConfig = {};
  Object.keys(bayesianConfig).forEach(key => {
    if (bayesianConfig[key] !== null) {
      cleanBayesianConfig[key] = bayesianConfig[key];
    }
  });
  payload.bayesian_config = cleanBayesianConfig;
}
```

#### 7. LSTM Parameter Parsing (Manual Mode)
**File:** [TSTrainCard.jsx:673-692](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L673-L692)

```jsx
// Parse lstm_units string to array
const parseLstmUnits = (unitsStr) => {
  try {
    return JSON.parse(unitsStr);  // "[64]" → [64]
  } catch {
    return [64]; // default
  }
};

if (optimizationMethod === "manual") {
  finalParams = {
    lstm_units: parseLstmUnits(lstmManualParams.lstm_units),
    dropout_rate: parseFloat(lstmManualParams.dropout_rate),
    recurrent_dropout_rate: parseFloat(lstmManualParams.recurrent_dropout_rate),
    learning_rate: parseFloat(lstmManualParams.learning_rate),
    batch_size: parseInt(lstmManualParams.batch_size),
    epochs: parseInt(lstmManualParams.epochs)
  };
}
```

---

## Missing Components (Implementation Checklist)

### 🔴 CRITICAL - Backend (train.py)

#### 1. Core Training Function
**Location:** [train.py](DREAM-ML-backend/GEML/apiTimeSeries/train.py) after line 1529

```python
def train_lstm_model(dataset_path: str, data: Dict, experiment_dir: str) -> Dict:
    """
    Train LSTM model for time series forecasting.

    Follows same contract as train_arima_model and train_xgboost_model.
    Must handle: manual params, grid search, random search, bayesian search.
    """
    # TODO: Implement following XGBoost pattern (lines 1104-1459)
    pass
```

**Dependencies:**
- ✅ TensorFlow/Keras already imported ([train.py:47-51](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L47-L51))
- ✅ MLflow client available
- ⚠️ Need `skopt` for Bayesian optimization (check if installed)

#### 2. Sequence Creation Utility
**Location:** [train.py](DREAM-ML-backend/GEML/apiTimeSeries/train.py) before `train_lstm_model()`

```python
def create_sequences_for_lstm(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time series DataFrame to 3D sequences for LSTM.

    Args:
        df: Time series data with datetime index
        feature_cols: Input feature columns
        target_col: Target column
        sequence_length: Number of timesteps in each sequence
        forecast_horizon: Steps ahead to predict (default 1)

    Returns:
        X: Shape (n_sequences, sequence_length, n_features)
        y: Shape (n_sequences,) or (n_sequences, forecast_horizon)
    """
    # TODO: Implement sliding window logic
    pass
```

**Reference Pattern:** Similar to [train.py:400-436](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L400-L436) (`create_supervised_dataset`) but 3D output

#### 3. LSTM-Specific Train/Val/Test Split
**Location:** [train.py](DREAM-ML-backend/GEML/apiTimeSeries/train.py)

```python
def lstm_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    split_ratios: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Temporal split for LSTM sequences (maintains 3D shape).

    Args:
        X: Shape (n_sequences, sequence_length, n_features)
        y: Shape (n_sequences,)
        split_ratios: {"train": 0.7, "val": 0.15, "test": 0.15}

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # TODO: Adapt ts_train_val_test_split for 3D arrays
    pass
```

**Reference Pattern:** [train.py:112-137](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L112-L137)

#### 4. Model Building Function
**Location:** [train.py](DREAM-ML-backend/GEML/apiTimeSeries/train.py)

```python
def build_lstm_model(params: Dict, input_shape: Tuple[int, int]) -> keras.Model:
    """
    Build Keras LSTM model from hyperparameters.

    Args:
        params: {
            "lstm_units": [64] or [64, 32],
            "dropout_rate": 0.2,
            "recurrent_dropout_rate": 0.2,
            "learning_rate": 0.001
        }
        input_shape: (sequence_length, n_features)

    Returns:
        Compiled Keras Sequential model
    """
    # TODO: Implement Sequential model with LSTM layers
    # Handle single-layer vs multi-layer architectures
    # Add Dense output layer for single-step forecast
    # Compile with Adam optimizer and MSE loss
    pass
```

**Imports:** Already available ([train.py:47-51](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L47-L51))

#### 5. Keras Callbacks Configuration
**Location:** [train.py](DREAM-ML-backend/GEML/apiTimeSeries/train.py)

```python
def create_lstm_callbacks(
    experiment_dir: str,
    early_stopping_patience: int
) -> List[keras.callbacks.Callback]:
    """
    Create Keras callbacks for LSTM training.

    Args:
        experiment_dir: Directory to save checkpoints
        early_stopping_patience: Epochs to wait for improvement

    Returns:
        List of Keras callbacks
    """
    # TODO: Implement EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    pass
```

#### 6. Random Parameter Generator
**Location:** [train.py](DREAM-ML-backend/GEML/apiTimeSeries/train.py)

```python
def generate_random_lstm_params(random_search_params: Dict) -> Dict:
    """
    Generate random LSTM hyperparameters.

    Args:
        random_search_params: {
            "lstm_units_options": [[32], [64], [128]],
            "dropout_rate_range": [0.0, 0.3],
            "learning_rate_range": [0.001, 0.01],
            "batch_size_options": [16, 32],
            "epochs_range": [20, 100]
        }

    Returns:
        Random parameter dict with Python native types (not numpy)
    """
    # TODO: Follow pattern from generate_random_xgboost_params (lines 703-748)
    # Use log-uniform for learning_rate
    # Return Python int/float (not numpy types)
    pass
```

**Critical:** Must convert numpy types to Python natives for JSON serialization ([train.py:289-323](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L289-L323))

#### 7. LSTM Evaluation Function
**Location:** [train.py](DREAM-ML-backend/GEML/apiTimeSeries/train.py)

```python
def evaluate_lstm_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    prefix: str,
    experiment_dir: str
) -> Tuple[Dict, List[str]]:
    """
    Evaluate LSTM model and generate plots.

    Args:
        model: Trained Keras model
        X_test: Test sequences (3D)
        y_test: Test targets (1D)
        prefix: "val" or "test"
        experiment_dir: Directory for plots

    Returns:
        metrics: {"val_rmse": float, "val_mae": float, "val_mape": float}
        artifacts: List of plot paths
    """
    # TODO: Follow pattern from evaluate_xgboost_model (lines 483-525)
    # Generate predictions: model.predict(X_test)
    # Calculate RMSE, MAE, MAPE
    # Create plots: forecast, residuals, training curves
    # Log artifacts to MLflow
    pass
```

**Reference Pattern:** [train.py:483-525](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L483-L525)

#### 8. Memory Management Utilities
**Location:** Throughout training loops

```python
import gc
import tensorflow as tf

# After each iteration in grid/random/bayesian search:
del model
tf.keras.backend.clear_session()
gc.collect()
```

**Reason:** [test_lstm_fixes.py:1-8](DREAM-ML-backend/GEML/apiTimeSeries/tests/test_lstm_fixes.py#L1-L8) documents memory leak without this cleanup

---

## Known Issues & Fixes

### 1. Memory Leak in Random/Bayesian Search
**Source:** [test_lstm_fixes.py:1-8](DREAM-ML-backend/GEML/apiTimeSeries/tests/test_lstm_fixes.py#L1-L8)

**Issue:** Keras models not properly cleared between iterations, causing memory accumulation.

**Fix:**
```python
for i in range(n_random_iterations):
    model = build_lstm_model(params, input_shape)
    history = model.fit(...)

    # CRITICAL: Clear memory after each iteration
    del model
    tf.keras.backend.clear_session()
    gc.collect()
```

**Test Validation:** Memory increase should be < 500MB (random) or < 1000MB (bayesian)

### 2. TypeError with List Parameters in Bayesian Search
**Source:** [test_lstm_fixes.py:1-8](DREAM-ML-backend/GEML/apiTimeSeries/tests/test_lstm_fixes.py#L1-L8)

**Issue:** Bayesian optimizer uses dict hashing, lists are unhashable.

**Fix:** Convert list parameters to tuples before search
```python
# Frontend sends: "choices": [[32], [64], [128]]
# Backend must convert to: choices: [(32,), (64,), (128,)]

def prepare_bayesian_space(bayesian_search_params: Dict) -> List:
    space = []
    for param_name, param_config in bayesian_search_params.items():
        if param_config["type"] == "categorical":
            # Convert lists to tuples for hashing
            choices = [tuple(c) if isinstance(c, list) else c for c in param_config["choices"]]
            space.append(Categorical(choices, name=param_name))
    return space
```

### 3. Numpy Type JSON Serialization
**Source:** [train.py:289-323](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L289-L323)

**Issue:** `numpy.int64`, `numpy.float64` are not JSON serializable.

**Fix:** Explicitly convert to Python natives
```python
params = {
    "lstm_units": [int(x) for x in lstm_units],  # numpy.int64 → int
    "learning_rate": float(lr),  # numpy.float64 → float
    "batch_size": int(batch_size)
}
```

---

## Open Questions for Clarification

### Backend Implementation

1. **Multi-Step Forecasting:**
   - Should LSTM output single-step (`forecast_horizon=1`) or multi-step predictions?
   - If multi-step, should we use:
     - **Recursive forecasting** (predict step 1, feed back, predict step 2, ...) like XGBoost [train.py:1461-1529](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L1461-L1529)?
     - **Direct multi-output** (Dense layer with `forecast_horizon` units)?
     - **Sequence-to-sequence** (LSTM encoder-decoder)?

2. **Feature Engineering:**
   - Should LSTM use **lag features** like XGBoost ([data_encoding_utils.py:5-34](DREAM-ML-backend/GEML/apiTimeSeries/data_encoding_utils.py#L5-L34))?
   - Or should it use **raw sequences** (sliding window on original features)?
   - What about `external_features` parameter - should LSTM support this?

3. **Sequence Length:**
   - Frontend default is `sequenceLength = 10` ([TSTrainCard.jsx:236](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L236))
   - Should this be configurable or auto-determined based on data seasonality?
   - Min/max bounds for validation?

4. **Bayesian Search Dependencies:**
   - Is `scikit-optimize` (skopt) installed in the project?
   - Should we add it to `requirements.txt`?
   - Alternative: Use Optuna or Keras Tuner?

5. **Model Checkpointing:**
   - Should we save checkpoints during training (ModelCheckpoint callback)?
   - If yes, should these be MLflow artifacts or just local files?
   - How to handle cleanup of checkpoints after training?

6. **Progress Updates:**
   - Should LSTM training send **per-epoch** WebSocket updates (verbose progress)?
   - Or just **per-iteration** updates like ARIMA/XGBoost?
   - How to balance between informativeness and message volume?

7. **Energy Tracking:**
   - CodeCarbon tracker is used for ARIMA/XGBoost ([train.py:281-287](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L281-L287))
   - Should LSTM training track energy consumption?
   - If yes, per-epoch or total?

8. **Stateful LSTM:**
   - Should we support **stateful LSTM** for very long sequences?
   - This requires batch size to be a factor of training samples
   - Additional complexity - worth it?

9. **Validation Strategy:**
   - ARIMA uses **walk-forward validation** implicitly
   - XGBoost uses **fixed validation set**
   - Should LSTM use **TimeSeriesSplit** (k-fold cross-validation for time series)?
   - Or stick with fixed val set for simplicity?

10. **Grid Search Combinations:**
    - ARIMA grid search generates ~48 combinations ([train.py:862-885](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L862-L885))
    - LSTM grid could be **hundreds** of combinations with multiple hyperparameters
    - Should we limit grid search or recommend random/bayesian instead?

### Frontend Integration

11. **LSTM Parameter Ranges:**
    - Current frontend ranges ([TSTrainCard.jsx:249-257](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L249-L257)):
      - `lstm_units_options`: `["[32]", "[64]", "[128]", "[64,32]", "[128,64]"]`
      - `dropout_rate_range`: `[0.0, 0.5]`
      - `epochs_range`: `[50, 300]`
    - Are these reasonable defaults or should we adjust based on typical time series datasets?

12. **UI for Sequence Length:**
    - Currently hardcoded as state variable ([TSTrainCard.jsx:236](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L236))
    - Should we add a UI input field for users to configure?
    - If yes, where to place it (with algorithm params or general config)?

13. **External Features for LSTM:**
    - XGBoost has external features selection UI ([TSTrainCard.jsx:1950-1987](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L1950-L1987))
    - Should LSTM also support external features?
    - If yes, how do they interact with sequence creation?

14. **Training Progress Display:**
    - Should we show LSTM training progress differently (e.g., epoch-level progress bar)?
    - Current ProgressBar component might not handle nested progress (iterations × epochs)

15. **Bayesian Config Advanced Settings:**
    - Frontend has collapsible advanced settings ([TSTrainCard.jsx:1193-1323](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L1193-L1323))
    - Are all these settings necessary for LSTM or can we simplify?

### Testing & Validation

16. **Test Data:**
    - [test_lstm_fixes.py](DREAM-ML-backend/GEML/apiTimeSeries/tests/test_lstm_fixes.py) uses synthetic sine wave data
    - Should we add tests with real-world time series datasets?
    - Where to store test datasets (fixtures)?

17. **Integration Tests:**
    - Should we add end-to-end integration tests (frontend → backend → MLflow)?
    - Or just unit tests for each component?

18. **Performance Benchmarks:**
    - Should we establish baseline performance metrics (training time, memory usage)?
    - Compare LSTM vs XGBoost vs ARIMA on standard datasets?

### Deployment & Production

19. **GPU Support:**
    - Should LSTM training use GPU if available?
    - How to handle GPU memory limits?
    - Fallback to CPU if GPU OOM?

20. **Model Versioning:**
    - DVC is used for model files ([services.py:1043-1051](DREAM-ML-backend/GEML/apiTimeSeries/services.py#L1043-L1051))
    - Keras models are directories (SavedModel format) or single .h5 files
    - Which format should we use for DVC compatibility?

---

## Recommended Implementation Order

### Phase 1: Minimal Viable LSTM (Manual Params Only)
**Goal:** Get basic LSTM training working with user-provided hyperparameters

1. ✅ Implement `create_sequences_for_lstm()` - 3D sequence conversion
2. ✅ Implement `lstm_train_val_test_split()` - Maintain 3D shape
3. ✅ Implement `build_lstm_model()` - Keras Sequential model
4. ✅ Implement `create_lstm_callbacks()` - EarlyStopping + ModelCheckpoint
5. ✅ Implement `evaluate_lstm_model()` - Metrics + plots
6. ✅ Implement `train_lstm_model()` - Manual params path only
7. ✅ Test with simple dataset (synthetic or real)
8. ✅ Verify MLflow logging works

**Estimated Effort:** 4-6 hours

### Phase 2: Grid Search Support
**Goal:** Add exhaustive hyperparameter search

1. ✅ Add grid search logic to `train_lstm_model()`
2. ✅ Test with small parameter grid (avoid combinatorial explosion)
3. ✅ Add memory cleanup between iterations
4. ✅ Verify progress tracking

**Estimated Effort:** 2-3 hours

### Phase 3: Random Search Support
**Goal:** Add efficient random sampling

1. ✅ Implement `generate_random_lstm_params()`
2. ✅ Add random search logic to `train_lstm_model()`
3. ✅ Add memory profiling tests (following test_lstm_fixes.py pattern)
4. ✅ Verify memory leak is fixed

**Estimated Effort:** 2-3 hours

### Phase 4: Bayesian Search Support
**Goal:** Add advanced optimization

1. ✅ Check/install `scikit-optimize` dependency
2. ✅ Implement list-to-tuple conversion for parameter space
3. ✅ Add Bayesian search logic to `train_lstm_model()`
4. ✅ Test with `n_bayesian_iterations = 30`
5. ✅ Verify convergence criteria work

**Estimated Effort:** 3-4 hours

### Phase 5: Frontend Integration Testing
**Goal:** Verify end-to-end workflow

1. ✅ Test LSTM selection in TSTrainCard.jsx
2. ✅ Verify all parameter combinations reach backend correctly
3. ✅ Test error handling (validation, training failures)
4. ✅ Verify MLflow UI displays LSTM runs properly

**Estimated Effort:** 2-3 hours

### Phase 6: Documentation & Deployment
**Goal:** Production readiness

1. ✅ Add docstrings to all new functions
2. ✅ Update API documentation
3. ✅ Add user guide for LSTM training
4. ✅ Performance benchmarking
5. ✅ Code review & optimization

**Estimated Effort:** 3-4 hours

**Total Estimated Effort:** 16-23 hours

---

## Dependencies Checklist

### Python Backend
- ✅ `tensorflow` (2.x) - Already imported ([train.py:47-51](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L47-L51))
- ✅ `keras` - Part of TensorFlow 2.x
- ❓ `scikit-optimize` - **Need to verify** for Bayesian search
- ✅ `mlflow` - Already used
- ✅ `pandas`, `numpy` - Already used
- ✅ `codecarbon` - Already used for energy tracking

### React Frontend
- ✅ All LSTM state variables already defined ([TSTrainCard.jsx:236-292](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L236-L292))
- ✅ UI components already exist (Material-UI)
- ✅ Payload construction already includes LSTM params ([TSTrainCard.jsx:743-761](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L743-L761))

---

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Memory leak in iterative search** | High (OOM crash) | High (documented in tests) | Add `tf.keras.backend.clear_session()` after each iteration |
| **Long training times (epochs × iterations)** | Medium (poor UX) | High | Use early stopping, lower default epochs (50-100), recommend random over grid |
| **Bayesian search crashes with list params** | High (TypeError) | Medium | Convert lists to tuples before `gp_minimize()` |
| **GPU memory overflow** | High (crash) | Medium | Add batch size limits, catch OOM exceptions, fallback to CPU |
| **Incompatible sequence shapes** | High (ValueError) | Low | Add shape validation before training, clear error messages |
| **Frontend payload mismatch** | Medium (training fails) | Low | Add backend validation for all LSTM params, return 400 with details |

---

## Success Criteria

### Functional Requirements
- ✅ LSTM training completes without ImportError
- ✅ Manual hyperparameters work (single fit)
- ✅ Grid search works (< 50 combinations recommended)
- ✅ Random search works (default 100 iterations)
- ✅ Bayesian search works (default 30 iterations)
- ✅ MLflow logs model, metrics, and artifacts correctly
- ✅ DVC versions model files
- ✅ Pipeline config JSON updates
- ✅ Frontend can trigger all optimization methods

### Performance Requirements
- ⚠️ Memory increase < 500MB for random search (per test)
- ⚠️ Memory increase < 1000MB for Bayesian search (per test)
- ⚠️ Training time reasonable (< 30 min for 100 random iterations on typical dataset)
- ⚠️ No GPU memory overflow (with batch size limits)

### Code Quality Requirements
- ✅ Follows existing code style and patterns
- ✅ Comprehensive docstrings
- ✅ Unit tests for all new functions
- ✅ Integration test from frontend to MLflow
- ✅ Error handling matches ARIMA/XGBoost patterns

---

## Next Steps

1. **Answer clarifying questions** (see Open Questions section above)
2. **Verify dependencies** (especially `scikit-optimize`)
3. **Choose implementation approach:**
   - Option A: Implement all phases sequentially (16-23 hours)
   - Option B: MVP first (Phase 1 only), iterate based on feedback (4-6 hours)
   - Option C: Parallelize frontend/backend work (requires 2 developers)
4. **Set up test environment:**
   - Synthetic dataset for unit tests
   - Real-world dataset for integration tests
   - MLflow tracking server running
5. **Begin Phase 1 implementation** (if approved)

---

## Appendix: Code Comparison

### ARIMA vs XGBoost vs LSTM Training Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│ ARIMA (Univariate)                                                   │
├──────────────────────────────────────────────────────────────────────┤
│ load_data() → pd.Series (1D)                                        │
│ ts_train_val_test_split() → y_train, y_val, y_test (1D)            │
│ ARIMA(order=(p,d,q), seasonal_order=(P,D,Q,s)).fit()               │
│ model.forecast(steps=len(y_val))                                    │
│ mlflow.sklearn.log_model(model)                                     │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ XGBoost (Supervised Learning)                                        │
├──────────────────────────────────────────────────────────────────────┤
│ load_data() → pd.DataFrame                                          │
│ prepare_xgboost_features() → Validate external features             │
│ create_supervised_dataset() → Shift target, remove NaN              │
│ xgboost_train_val_test_split() → X_train (2D), y_train (1D), ...   │
│ xgb.XGBRegressor(**params).fit(X_train, y_train)                   │
│ model.predict(X_val) → Validation predictions                       │
│ mlflow.sklearn.log_model(model)                                     │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ LSTM (Deep Learning, Sequences) - TO IMPLEMENT                      │
├──────────────────────────────────────────────────────────────────────┤
│ load_data() → pd.DataFrame                                          │
│ create_sequences_for_lstm() → X (3D), y (1D)                       │
│ lstm_train_val_test_split() → X_train (3D), y_train (1D), ...      │
│ build_lstm_model(params, input_shape) → Keras Sequential            │
│ model.fit(X_train, y_train, validation_data=(X_val, y_val),        │
│           callbacks=[EarlyStopping, ...], epochs=100, batch_size=32) │
│ history.history['val_loss'] → Best validation loss                  │
│ mlflow.keras.log_model(model)  ← DIFFERENT FROM SKLEARN             │
│ tf.keras.backend.clear_session()  ← MEMORY CLEANUP                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-06
**Authors:** Codebase Analysis Agent + Expert Technical Documentation Specialist
**Review Status:** ⏳ Awaiting User Clarification
