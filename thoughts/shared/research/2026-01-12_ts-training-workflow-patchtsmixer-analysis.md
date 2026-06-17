# Time Series Training Workflow Analysis for PatchTSMixer Implementation

**Research Date:** 2026-01-12
**Purpose:** Understand existing TS training workflow to implement PatchTSMixer model training
**Target Users:** Data Scientists (experienced and new), knowledgeable in ML/DL
**Scope:** Frontend TSTrainCard.jsx + Backend `/api/ts/train-model/` endpoint + PatchTSMixer comparison

---

## Executive Summary

This document provides a comprehensive analysis of DREAM-ML's time series forecasting model training workflow, examining both frontend user interactions and backend training pipeline. The research focuses on understanding existing patterns (LSTM, ARIMA, XGBoost) to inform the implementation of PatchTSMixer, a lightweight transformer-based model.

### Key Findings

**Frontend (TSTrainCard.jsx):**
- 4531-line React component supporting 3 algorithms with 4 optimization strategies
- Comprehensive validation pipeline preventing invalid submissions
- FormData multipart upload (CSV + JSON payload)
- Context-based state management with workflow progression tracking
- Real-time validation with debounced split ratio checks

**Backend Architecture:**
- Three-layer separation: Views (HTTP) → Services (orchestration) → Train (implementation)
- MLflow experiment tracking with nested runs and system metrics
- DVC versioning with Git integration (add → commit → push pattern)
- Energy tracking via CodeCarbon
- LSTM implementation provides direct template for PatchTSMixer

**PatchTSMixer Requirements:**
- Sequence → Patch conversion (new data preparation logic)
- Keras → Hugging Face Transformers framework change
- Single-step → Multi-step forecasting (multi-horizon evaluation)
- Different hyperparameter space (d_model, num_layers, patch_length vs lstm_units)
- PyTorch tensors vs NumPy arrays

**Reusable Components:**
- ✅ MLflow setup: Exact same pattern works
- ✅ DVC versioning: Exact same pattern works
- ✅ Energy tracking: Exact same wrapper works
- ✅ Temporal splitting: Same logic, different data shapes
- ✅ Service orchestration: Minimal changes needed
- ✅ View layer: Add algorithm name, rest unchanged

---

## Table of Contents

1. [Frontend Workflow Analysis](#frontend-workflow-analysis)
2. [Backend Training Pipeline](#backend-training-pipeline)
3. [LSTM Implementation Deep Dive](#lstm-implementation-deep-dive)
4. [PatchTSMixer Comparison](#patchtsmixer-comparison)
5. [Implementation Recommendations](#implementation-recommendations)
6. [Code References](#code-references)
7. [Open Questions](#open-questions)

---

## Frontend Workflow Analysis

### Component Architecture

**TSTrainCard.jsx** ([src/components/TSTrainCard.jsx](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx)):
- **Lines:** 4531
- **State Variables:** 40+ useState hooks
- **Context Integration:** AppContext for global workflow state
- **Algorithms Supported:** ARIMA, XGBoost, LSTM
- **Optimization Methods:** Manual, Grid Search, Random Search, Bayesian

### User Workflow (11 Phases)

```
1. File Selection & Column Loading
   ├─ User selects CSV file → csvFile state
   └─ User clicks "Cargar Variables" → POST /api/analyze-csv/ → columns state

2. Variable Selection
   ├─ Target variable (RadioGroup, single selection)
   ├─ Input features (Checkboxes, multi-selection)
   └─ Date column (RadioGroup, single selection)
   └─ Validation runs continuously via validateSelections()

3. Algorithm Selection
   ├─ Dropdown: ARIMA | XGBoost | LSTM
   └─ Conditional UI renders algorithm-specific forms

4. Optimization Method Selection
   ├─ Radio buttons: Manual | Grid | Random | Bayesian
   └─ Conditional parameter forms render

5. Hyperparameter Configuration
   ├─ Manual: Direct input fields
   ├─ Grid: Comma-separated option lists
   ├─ Random: Range inputs (min/max)
   └─ Bayesian: Trial count + optional param_ranges

6. Data Split Configuration
   ├─ Train/Val/Test split ratios (TextFields)
   ├─ Comma decimal format (0,70 → 0.70 internally)
   └─ Debounced validation (500ms)

7. Training Execution
   ├─ Pre-submission validation (16 checks)
   ├─ Payload construction (algorithm + optimization specific)
   ├─ FormData: CSV file + JSON payload
   └─ POST /api/ts/train-model/

8. Backend Processing
   └─ (See Backend Training Pipeline section)

9. Response Handling
   ├─ Extract: run_id, metrics, model_path, mlflow_ui
   └─ Update: trainStatus, trainInProgress, flow.trainDone

10. Success/Error Display
    ├─ Success: Green indicator + formatted metrics
    └─ Error: Structured error message extraction

11. Workflow Progression
    └─ flow.trainDone = true → Button disabled
```

### API Request Structure

**Endpoint:** `POST /api/ts/train-model/`

**Request Format:**
```javascript
// FormData with two parts:
formData.append("file", csvFile);           // CSV file (multipart)
formData.append("data", JSON.stringify({    // JSON payload
  // Core
  model_name: "lstm_sales_forecast",
  algorithm: "lstm",
  problem_type: "ts_forecasting",

  // Data configuration
  input_features: ["temperature", "humidity"],
  target_variable: "sales",
  date_col_name: "date",
  experiment_dir: "/workspaces/dream-ml-c/experiments/exp_20260112",
  run_id: "abc123def456",

  // Splitting
  split_ratios: { train: 0.7, val: 0.15, test: 0.15 },

  // Time series specific
  forecast_horizon: 12,
  optimization_metric: "val_rmse",

  // Hyperparameter strategy
  hyperparameter_search_strategy: "manual",

  // Manual parameters (if strategy=manual)
  manual_params: {
    lstm_units: [64, 32],
    dropout_rate: 0.2,
    recurrent_dropout_rate: 0.2,
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 100
  },

  // LSTM-specific
  sequence_length: 10,
  early_stopping_patience: 20,
  training_mode: "multivariate",  // or "univariate"
  feature_config: {}
}));
```

**Response Format:**
```json
{
  "status": "success",
  "run_id": "abc123def456",
  "metrics": {
    "val_rmse": 12.45,
    "val_mae": 9.32,
    "val_mape": 5.67
  },
  "model_path": "trained/lstm_model.keras",
  "mlflow_ui": "http://localhost:5000/#/experiments/1/runs/abc123def456"
}
```

### State Management Patterns

**AppContext** ([src/AppContext.jsx:46-139](DREAM-ML-frontend/frontend/src/AppContext.jsx#L46-L139)):
```javascript
// Global state consumed by TSTrainCard
{
  experimentDir: "/path/to/experiment",
  runId: "abc123",
  trainInProgress: false,
  trainStatus: "✅ Modelo entrenado exitosamente",
  flow: {
    experimentCreated: true,
    encodeDone: true,    // Prerequisite for training
    trainDone: false
  },
  markStepDone: (stepName) => {...}
}
```

**Component State Flow:**
```
File Upload → loadColumns() → columns state
    ↓
Variable Selection → validateSelections() → validationWarnings
    ↓
Algorithm Selection → Conditional UI Rendering
    ↓
Hyperparameter Config → Algorithm-specific state
    ↓
Split Ratios → validateSplitRatios (debounced) → splitRatiosValid
    ↓
Validation → isDisabled computed → Button enable/disable
    ↓
handleTrain() → Payload construction → API call
    ↓
Response → trainStatus update → flow.trainDone
```

### Validation Strategy

**Multi-stage validation pipeline:**

1. **Real-time Validation** (TSTrainCard.jsx:368):
   - Runs on every feature/target/date change
   - Checks overlaps, required selections
   - Updates `validationWarnings` array

2. **Debounced Validation** (500ms):
   - Split ratios validated after last input
   - Prevents excessive validation calls
   - Visual feedback (green/red borders)

3. **Pre-submission Validation** (TSTrainCard.jsx:701-751):
   - 16 comprehensive checks before API call
   - Algorithm-specific requirements (XGBoost needs ≥1 feature)
   - Bayesian config validation
   - Parameter range validation

4. **Button Disable Logic** (TSTrainCard.jsx:1234):
```javascript
const isDisabled =
  trainInProgress ||
  !experimentDir || !runId ||
  !flow.encodeDone ||
  flow.trainDone ||
  (algorithm === "xgboost" && !inputFeatures.length) ||
  !targetVariable || !dateColumnName ||
  !modelName.trim() ||
  targetVariable === dateColumnName ||
  inputFeatures.includes(dateColumnName) ||
  !isRandomSearchParamsValid() ||
  !isLSTMParamsValid() ||
  validationWarnings.length > 0 ||
  !splitRatiosValid;
```

### Key Design Patterns

1. **State Machine:** Workflow progression (experimentCreated → encodeDone → trainDone)
2. **Conditional Rendering:** Massive nested conditionals (algorithm × optimization method)
3. **FormData Multipart:** CSV + JSON in single request
4. **Context Lifting:** Global state hoisted to AppContext
5. **Validation Pipeline:** Real-time → Debounced → Pre-submission

### Critical Frontend Code References

| Function | Location | Purpose |
|----------|----------|---------|
| `handleTrain` | TSTrainCard.jsx:700 | Primary training orchestration |
| Payload construction | TSTrainCard.jsx:803-893 | Algorithm-specific parameter serialization |
| API call | TSTrainCard.jsx:903 | POST to `/ts/train-model/` |
| `validateSelections` | TSTrainCard.jsx:368 | Feature/target/date validation |
| `isDisabled` | TSTrainCard.jsx:1234 | Button enable/disable logic (16 checks) |
| AppContext provider | AppContext.jsx:46-139 | Global workflow state |

---

## Backend Training Pipeline

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        VIEWS LAYER                          │
│  (views.py - HTTP request handling, validation, response)  │
│                                                             │
│  train_model(request) [views.py:376]                       │
│  ├─ Validate file and data                                 │
│  ├─ Configure MLflow tracking URI                          │
│  ├─ Call TrainModelService.train_model_logic()            │
│  └─ Return JSON response                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      SERVICES LAYER                         │
│   (services.py - Orchestration, MLflow/DVC management)     │
│                                                             │
│   TrainModelService.train_model_logic() [services.py:962]  │
│   1. Validate experiment_dir and algorithm                  │
│   2. Configure MLflow tracking URI                          │
│   3. Start MLflow run with system metrics                   │
│   4. Version dataset with DVC (add/commit/push)             │
│   5. Route to algorithm-specific training function          │
│   6. Version trained model with DVC                         │
│   7. Log metrics and artifacts to MLflow                    │
│   8. Return step_config dict                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                       TRAIN LAYER                           │
│   (train.py - Algorithm implementations, model building)    │
│                                                             │
│   train_lstm_model() [train.py:3819]                       │
│   1. Load & validate data                                   │
│   2. Create sequences (sliding window)                      │
│   3. Split train/val/test (temporal, no shuffling)          │
│   4. Start energy tracking (CodeCarbon)                     │
│   5. Execute hyperparameter search                          │
│   6. Evaluate on val/test sets                              │
│   7. Save model and generate plots                          │
│   8. Return metrics dict and model_path                     │
└─────────────────────────────────────────────────────────────┘
```

### Request Processing Flow

**Complete flow with code references:**

1. **views.py:389-393** - Validate CSV file and JSON data presence
2. **views.py:396-405** - Configure MLflow tracking URI (`sqlite:///{base_dir}/shared_mlflow.db`)
3. **views.py:408-410** - Cleanup any active MLflow runs
4. **views.py:413-416** - Call `TrainModelService.train_model_logic()`
5. **services.py:977-984** - Validate experiment_dir and algorithm
6. **services.py:986-999** - Configure MLflow, get experiment ID
7. **services.py:1002-1004** - Start MLflow run with `log_system_metrics=True`
8. **services.py:1008-1028** - Save dataset to `trained/` directory, version with DVC
9. **services.py:1031-1060** - Route to `train_lstm_model()` based on algorithm
10. **train.py:3862-3882** - Extract and validate training parameters
11. **train.py:3896-3919** - Start nested MLflow logging
12. **train.py:3926-3955** - Load data, validate numeric features
13. **train.py:3976-3990** - Create LSTM sequences via `create_sequences_for_lstm()`
14. **train.py:3997-4002** - Split sequences temporally (no shuffling)
15. **train.py:4008-4016** - Start energy tracking (CodeCarbon)
16. **train.py:4022-4074** - Train model with selected strategy (manual/grid/random/bayesian)
17. **train.py:4882-4903** - Evaluate on val/test sets
18. **train.py:4910-4924** - Log metrics/artifacts to MLflow
19. **train.py:4946-4948** - Save model as `.keras` file
20. **services.py:1062-1070** - Version model with DVC (add → git commit → dvc push)
21. **services.py:1073-1083** - Consolidate metrics, log to MLflow
22. **services.py:1086-1104** - Return step_config dict
23. **views.py:431-436** - Format JSON response with run_id, metrics, model_path, MLflow UI URL

### MLflow Integration Patterns

**Tracking URI Configuration:**
```python
# services.py:986-989
base_dir = os.path.dirname(experiment_dir)
shared_db_path = os.path.join(base_dir, "shared_mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{shared_db_path}")
```
- **Pattern:** SQLite database per experiment base directory
- **Example:** `/app/experimentos/shared_mlflow.db` shared by all experiments

**Run Management:**
```python
# services.py:1002-1004
with start_run(experiment_id=mlflow_experiment_id,
               description=f"Entrenamiento {algorithm}",
               log_system_metrics=True) as run:
    run_id = run.info.run_id
    # Training logic here
```
- **Pattern:** Single run per training, nested runs for data steps
- **System metrics:** CPU, memory, disk logged automatically

**Parameter Logging:**
```python
# train.py:3908-3919
mlflow.log_params({
    "model_type": "LSTM",
    "date_col_name": date_col_name,
    "target_variable": target_variable,
    "input_features": str(input_features),
    "forecast_horizon": forecast_horizon,
    "sequence_length": sequence_length,
    "hyperparameter_search_strategy": hyperparameter_search_strategy,
    "cpu_only": True
})
```

**Metric Logging:**
```python
# services.py:1073-1081
combined_metrics = {}
if "val_metrics" in result:
    filtered_val_metrics = {k: v for k, v in result["val_metrics"].items() if v is not None}
    mlflow.log_metrics(filtered_val_metrics)
if "test_metrics" in result:
    filtered_test_metrics = {k: v for k, v in result["test_metrics"].items() if v is not None}
    mlflow.log_metrics(filtered_test_metrics)
```

**Dataset Lineage Tracking:**
```python
# services.py:196-202
raw_data = pd.read_csv(raw_file_path, encoding='utf-8')
raw_dataset = mlflow.data.from_pandas(
    raw_data,
    source=raw_file_path,
    name="Dataset Crudo"
)
mlflow.log_input(raw_dataset, context="raw_data")
```
- **Pattern:** Track data lineage at each pipeline step (raw, cleaned, encoded, training)

### DVC Version Control Patterns

**Initialization:**
```bash
# api/utils.py:93-152
git init                                    # Initialize Git
dvc init                                    # Initialize DVC
dvc cache dir .dvc_cache/                   # Configure cache
# Update .gitignore with DVC exclusions
git add .dvc .dvcignore .gitignore
git commit -m "Initialize DVC"
```

**Remote Configuration:**
```bash
# api/utils.py:203-224
shared_remote=/app/experimentos/dvc_remote  # Shared across experiments
dvc remote add shared_remote ${shared_remote}
dvc remote default shared_remote
```

**File Versioning Workflow:**
```bash
# services.py:317-345 (example)
dvc add processed_eda_path                  # 1. Add file to DVC
git add processed_eda_path.dvc .dvc/config  # 2. Commit .dvc pointer
git commit -m "Add processed EDA data"      # 3. Git commit
dvc push processed_eda_path                 # 4. Push actual data to remote
```

**Files Versioned:**
- Raw datasets: `raw/*.csv`
- Processed datasets: `processed/*.csv`
- Trained models: `trained/*.keras`, `trained/*.pkl`
- Pipeline config: `pipeline_config.json`

**Reproducibility via DVC Get:**
```python
# services.py:1174-1182
subprocess.run(
    ["dvc", "get", old_experiment_dir, raw_rel_path, "-o", new_raw_path],
    cwd=new_experiment_dir,
    check=True
)
```
- **Purpose:** Retrieve versioned datasets from previous experiments for reproduction

### Energy Tracking

**CodeCarbon Integration:**
```python
# train.py:4008-4016
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(
    project_name=f"train_{algorithm}",
    measure_power_secs=15,
    save_to_file=False,
    log_level="error",
)
tracker.start()

# ... training logic ...

tracker.stop()
```

**Logging Energy Metrics:**
```python
# train.py:444-461
def log_energy_metrics(tracker):
    energy_kwh = float(tracker._total_energy.kWh) if tracker._total_energy else 0.0
    emissions_kg = float(tracker.final_emissions) if tracker.final_emissions else 0.0

    mlflow.log_metric("energy_consumed_total_kWh", energy_kwh)
    mlflow.log_metric("carbon_emission_kg", emissions_kg)
    return energy_kwh, emissions_kg
```

### Error Handling Patterns

**View Layer (Granular exception catching):**
```python
# views.py:439-479
try:
    result = trainModelService.train_model_logic(...)
except json.JSONDecodeError as e:
    return JsonResponse({"status": "error", ...}, status=400)
except ValueError as ve:
    return JsonResponse({"status": "error", ...}, status=400)
except FileNotFoundError as fnfe:
    return JsonResponse({"status": "error", ...}, status=404)
except RuntimeError as re:
    return JsonResponse({"status": "error", ...}, status=500)
except Exception as e:
    if mlflow.active_run():
        mlflow.end_run()
    return JsonResponse({"status": "error", ...}, status=500)
```

**Train Layer (Fail-safe iteration):**
```python
# train.py:4201-4208 (Grid search example)
for i, params in enumerate(grid):
    try:
        model = build_lstm_model(params, input_shape)
        history = model.fit(...)
        # Update best model
    except Exception as e:
        logger.error(f"Error en iteración {i+1}: {e}")
        continue  # Don't fail entire search

if best_model is None:
    raise RuntimeError("No se pudo entrenar ningún modelo")
```

**Memory Cleanup (Critical for search loops):**
```python
# train.py (After each grid/random iteration)
tf.keras.backend.clear_session()
gc.collect()
```

### Critical Backend Code References

| Component | Location | Purpose |
|-----------|----------|---------|
| `train_model` view | views.py:376 | Main training endpoint |
| `train_model_logic` service | services.py:962 | Training orchestration |
| `train_lstm_model` | train.py:3819 | LSTM training implementation |
| `create_sequences_for_lstm` | train.py:3286 | Sequence creation (sliding window) |
| `build_lstm_model` | train.py:3460 | Model architecture builder |
| `lstm_train_val_test_split` | train.py:3390 | Temporal data splitting |
| MLflow config | services.py:986-999 | Tracking URI setup |
| DVC add workflow | services.py:317-345 | File versioning pattern |
| Energy tracking | train.py:444-461 | CodeCarbon metrics logging |

---

## LSTM Implementation Deep Dive

### Why LSTM is the Best Reference for PatchTSMixer

Both LSTM and PatchTSMixer are:
- Deep learning models for time series
- Require sequence-based data preparation
- Support univariate and multivariate modes
- Use temporal train/val/test splitting
- Require similar hyperparameter search strategies
- Generate multi-step forecasts (though LSTM currently single-step)

### Sequence Creation Pattern

**LSTM Approach** (train.py:3286):
```python
def create_sequences_for_lstm(df, feature_cols, target_col, sequence_length, forecast_horizon):
    """
    Convert DataFrame to 3D tensors via sliding window.

    Input: df with datetime index, feature_cols, target_col
    Output: X shape (n_sequences, sequence_length, n_features)
            y shape (n_sequences,)
    """
    # Univariate mode (target only)
    if len(feature_cols) == 0:
        features = df[[target_col]].values
        n_features = 1
    # Multivariate mode
    else:
        features = df[feature_cols].values
        n_features = len(feature_cols)

    target = df[target_col].values

    # Sliding window
    X_sequences = []
    y_sequences = []
    for i in range(len(df) - sequence_length - forecast_horizon + 1):
        X_seq = features[i:i + sequence_length]              # Past window
        y_seq = target[i + sequence_length + forecast_horizon - 1]  # Future value
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)

    X = np.array(X_sequences)  # Shape: (n_sequences, sequence_length, n_features)
    y = np.array(y_sequences)  # Shape: (n_sequences,)

    # Validation
    if len(X) < 50:
        raise ValueError(f"Insufficient sequences: {len(X)} < 50")

    return X, y
```

**Key Characteristics:**
- Maintains temporal order (no shuffling)
- Supports univariate (target only) and multivariate modes
- Validates minimum data requirements
- Auto-adjusts sequence_length if dataset too small
- Currently generates single-step output (y_seq is single value)

### Temporal Train/Val/Test Split

**LSTM Approach** (train.py:3390):
```python
def lstm_train_val_test_split(X, y, split_ratios):
    """
    Split sequences temporally (no shuffling).

    Pattern: Train (earliest) → Val (middle) → Test (most recent)
    """
    n = len(X)
    train_size = int(n * split_ratios["train"])  # e.g., 0.7
    val_size = int(n * split_ratios["val"])      # e.g., 0.15

    X_train = X[:train_size]                      # Earliest data
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]  # Middle data
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]            # Most recent data
    y_test = y[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test
```

**Why No Shuffling:**
- Time series requires testing on future unseen data
- Simulates real-world forecasting scenario
- Prevents data leakage from future to past

### Model Architecture Pattern

**LSTM Single-Layer Architecture** (train.py:3487-3509):
```python
model = Sequential([
    LSTM(units=64,
         kernel_initializer=GlorotUniform(seed=42),
         recurrent_initializer=Orthogonal(seed=42),
         dropout=0.2,
         recurrent_dropout=0.2,
         input_shape=(sequence_length, n_features)),
    Dense(1, kernel_initializer=GlorotUniform(seed=42))
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss="mse",
              metrics=["mae", "mse"])
```

**LSTM Multi-Layer Architecture** (train.py:3511-3532):
```python
model = Sequential([
    LSTM(64, return_sequences=True, ...),  # Pass sequences to next layer
    LSTM(32, ...),                         # Final LSTM layer
    Dense(1, ...)
])
```

**Design Choices:**
- Output: `Dense(1)` for single-step forecasting
- Loss: MSE (mean squared error) for regression
- Optimizer: Adam with configurable learning rate
- Initializers: GlorotUniform + Orthogonal with SEED=42 for reproducibility
- Dropout: Both standard dropout and recurrent dropout for regularization

### Callbacks Pattern

**LSTM Callbacks** (train.py:3558):
```python
def create_lstm_callbacks(experiment_dir, early_stopping_patience, checkpoint_filename):
    checkpoint_path = os.path.join(experiment_dir, checkpoint_filename)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]

    return callbacks, checkpoint_path
```

**Usage:**
```python
callbacks, checkpoint_path = create_lstm_callbacks(...)
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=1)
```

### Hyperparameter Search Patterns

**Manual** (train.py:4022-4074):
- Direct user-provided parameters
- Single training run
- Fast, good for quick experiments

**Grid Search** (train.py:4076-4242):
- Exhaustive search over parameter grid
- Example: 2 lstm_units × 2 dropout × 2 lr = 8 combinations
- Warning threshold: >50 combinations
- Memory cleanup critical after each iteration

**Random Search** (train.py:4244-4406):
- Random sampling for N iterations
- Better than grid for high-dimensional spaces
- Default N=100 iterations
- Memory cleanup pattern same as grid

**Bayesian Search** (train.py:4408-4856):
```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lstm_units = trial.suggest_categorical("lstm_units", [[32], [64], [128]])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    # Build model
    model = build_lstm_model({
        "lstm_units": lstm_units,
        "dropout_rate": dropout,
        "learning_rate": learning_rate,
        ...
    }, input_shape)

    # Train
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=0)

    # Return best val loss
    return min(history.history["val_loss"])

# Create study
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

# Add callbacks
study.optimize(
    objective,
    n_trials=n_trials,
    callbacks=[
        ConvergenceCallback(tolerance=0.001, patience=5),
        MemoryMonitorCallback(max_memory_mb=max_memory_mb)
    ]
)

# Retrain with best params
best_params = study.best_params
final_model = build_lstm_model(best_params, input_shape)
final_model.fit(...)
```

**Key Characteristics:**
- Intelligent search using previous trial results (TPESampler)
- Convergence detection (stops if no improvement)
- Memory monitoring (stops if exceeds limit)
- Retrains final model with best params after search completes

**Commonality Across All Strategies:**
1. Track `best_val_loss` across iterations
2. Clean up memory after each trial (`tf.keras.backend.clear_session()`, `gc.collect()`)
3. Log best params to MLflow
4. Return single best model

### Evaluation Pattern

**LSTM Evaluation** (train.py:4882-4903):
```python
# Validation evaluation
val_metrics, val_artifacts = evaluate_lstm_model(
    model=best_model,
    X=X_val,
    y=y_val,
    prefix="val",
    experiment_dir=experiment_dir
)

# Test evaluation
test_metrics, test_artifacts = evaluate_lstm_model(
    model=best_model,
    X=X_test,
    y=y_test,
    prefix="test",
    experiment_dir=experiment_dir
)
```

**Metrics Calculated:**
- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **MAPE:** Mean Absolute Percentage Error

**Artifacts Generated:**
- **Forecast plot:** Actual vs predicted values
- **Residual plots:** 4 subplots (vs time, histogram, Q-Q plot, vs predicted)
- **Training history:** Train/val loss curves

**MLflow Logging:**
```python
# train.py:4910-4924
mlflow.log_metric("val_rmse", val_rmse)
mlflow.log_metric("test_rmse", test_rmse)

for artifact_path in val_artifacts + test_artifacts:
    if os.path.exists(artifact_path):
        mlflow.log_artifact(artifact_path, "plots")
```

### Model Saving Pattern

**LSTM Saving:**
```python
# train.py:4946-4948
model_save_path = os.path.join(experiment_dir, "lstm_model.keras")
best_model.save(model_save_path)
```
- **Format:** `.keras` (Keras 3 native format)
- **Contains:** Full model architecture + weights + optimizer state

**DVC Versioning:**
```python
# services.py:1062-1070
subprocess.run(["dvc", "add", model_path], cwd=experiment_dir, check=True)
subprocess.run(["git", "add", f"{model_path}.dvc"], cwd=experiment_dir, check=True)
subprocess.run(["git", "commit", "-m", f"[DVC] Add model"], cwd=experiment_dir, check=True)
subprocess.run(["dvc", "push", model_path], cwd=experiment_dir, check=True)
```

---

## PatchTSMixer Comparison

### What is PatchTSMixer?

**PatchTSMixer** is a lightweight MLP-Mixer-based model for time series forecasting that achieves state-of-the-art performance while being 2-3X more efficient than Transformer models.

**Key Strengths:**
- 8-60% improvement over MLP models
- 1-2% improvement over Patch-Transformer with 2-3X less memory/compute
- Lightweight architecture (1-5MB model size)
- Compatible with HuggingFace Trainer API
- Supports CPU training (critical for DREAM-ML)
- Strong reproducibility features
- Transfer learning capabilities

**Architecture:**
```
Input Time Series (batch, context_length, num_channels)
    ↓
[Patching Layer] → Splits into patches
    ↓
[Patch Embedding] → Linear projection to d_model dimensions
    ↓
[MLP-Mixer Blocks] × num_layers
    ├─ [Patch Mixing] → Mix across time patches
    ├─ [Channel Mixing] → Mix across variables
    ├─ [Feature Mixing] → Mix hidden features
    └─ [Gated Attention] → Prioritize important features
    ↓
[Prediction Head] → Linear layer
    ↓
Output Forecast (batch, prediction_length, num_channels)
```

### Key Differences from LSTM

#### 1. Sequence → Patch Data Preparation

**Current (LSTM):**
```python
# Sliding window creates sequences
# Input: (sequence_length, n_features)
# Output: Single value at forecast_horizon
for i in range(len(df) - sequence_length - forecast_horizon + 1):
    X_seq = features[i:i + sequence_length]
    y_seq = target[i + sequence_length + forecast_horizon - 1]
```

**PatchTSMixer Needs:**
```python
def create_patches_for_tsmixer(df, feature_cols, target_col,
                                context_length, patch_length,
                                prediction_length):
    """
    Convert DataFrame to patches.

    context_length: Total input window (e.g., 512)
    patch_length: Size of each patch (e.g., 16)
    prediction_length: Forecast horizon (e.g., 96)

    Returns:
        past_values: shape (n_samples, context_length, n_features)
        future_values: shape (n_samples, prediction_length, n_features)
    """
    # Build sequences
    sequences = []
    targets = []

    for i in range(len(df) - context_length - prediction_length + 1):
        # Past window
        seq = df[feature_cols].iloc[i:i+context_length].values
        # Future window (multi-step)
        tgt = df[target_col].iloc[i+context_length:i+context_length+prediction_length].values

        sequences.append(seq)
        targets.append(tgt)

    past_values = torch.FloatTensor(sequences)
    future_values = torch.FloatTensor(targets)

    # Patching happens inside PatchTSMixer model
    # num_patches = context_length // patch_length

    return past_values, future_values
```

**Differences:**
- Multi-step output (prediction_length) vs single-step
- Patches created internally by model, not in data prep
- Additional parameters: `context_length`, `patch_length`, `prediction_length`
- PyTorch tensors instead of NumPy arrays

#### 2. Model Architecture

**Current (LSTM):**
```python
# Keras Sequential API
model = Sequential([
    LSTM(64, ...),
    Dense(1)  # Single output
])
```

**PatchTSMixer:**
```python
# Hugging Face Transformers
from transformers import PatchTSMixerForPrediction, PatchTSMixerConfig

config = PatchTSMixerConfig(
    context_length=512,
    patch_length=16,
    num_input_channels=n_features,
    prediction_length=96,
    d_model=32,             # Hidden dimension (vs lstm_units)
    num_layers=8,           # Number of mixer layers
    expansion_factor=2,     # MLP expansion factor
    dropout=0.2,
    head_dropout=0.2,
    mode="common_channel",  # "common_channel" or "mix_channel"
    scaling="std",          # Built-in normalization
    loss="mse"
)

model = PatchTSMixerForPrediction(config)
```

**Differences:**
- Transformer-based architecture vs RNN
- Hugging Face API vs Keras Sequential
- Different hyperparameters (d_model, num_layers, expansion_factor vs lstm_units)
- Mode parameter for channel handling
- Built-in scaling/normalization

#### 3. Training Loop

**Current (LSTM):**
```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)
```

**PatchTSMixer:**
```python
from transformers import Trainer, TrainingArguments

# Training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=100,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=0.001,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=42,
    data_seed=42,
)

# Custom dataset class
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, past_values, future_values):
        self.past_values = past_values
        self.future_values = future_values

    def __len__(self):
        return len(self.past_values)

    def __getitem__(self, idx):
        return {
            "past_values": self.past_values[idx],
            "future_values": self.future_values[idx]
        }

train_dataset = TimeSeriesDataset(train_past, train_future)
val_dataset = TimeSeriesDataset(val_past, val_future)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
)

trainer.train()
```

**Differences:**
- Hugging Face Trainer API vs Keras fit
- Custom Dataset class required
- Different checkpoint/callback system
- PyTorch tensors throughout

#### 4. Evaluation Metrics

**Current (LSTM):**
```python
# Simple metrics on single-step predictions
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

**PatchTSMixer:**
```python
# Multi-step evaluation
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions  # Shape: (n_samples, prediction_length, n_features)

# Evaluate per horizon
for h in range(prediction_length):
    horizon_rmse = np.sqrt(mean_squared_error(y_test[:, h, :], y_pred[:, h, :]))
    mlflow.log_metric(f"test_rmse_horizon_{h+1}", horizon_rmse)

# Aggregate metrics
overall_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mlflow.log_metric("test_rmse_overall", overall_rmse)
```

**Differences:**
- Per-horizon metrics vs single metric
- 3D output evaluation vs 1D
- Potential for horizon-specific plots

#### 5. Hyperparameter Space

**Current (LSTM):**
```python
{
    "lstm_units": [64, 128],
    "dropout_rate": [0.2, 0.3],
    "recurrent_dropout_rate": [0.2],
    "learning_rate": [0.001, 0.01],
    "batch_size": [32],
    "epochs": [100]
}
```

**PatchTSMixer:**
```python
{
    "context_length": [96, 192, 512],        # Input window size
    "patch_length": [8, 16, 32],             # Patch size
    "prediction_length": [24, 48, 96],       # Forecast horizon
    "d_model": [16, 32, 64],                 # Hidden dimension
    "num_layers": [4, 8, 16],                # Number of mixer layers
    "expansion_factor": [2, 4],              # MLP expansion
    "dropout": [0.2, 0.3, 0.5],              # Dropout rate
    "head_dropout": [0.1, 0.2],              # Head dropout
    "mode": ["common_channel", "mix_channel"], # Channel handling
    "scaling": ["mean", "std"],              # Normalization
    "learning_rate": [1e-4, 1e-3],
    "batch_size": [32, 64]
}
```

**Differences:**
- More architectural parameters (d_model, num_layers, expansion_factor)
- Patch-specific parameters (context_length, patch_length)
- Multi-step horizon parameter (prediction_length)
- Mode and scaling parameters specific to PatchTSMixer

#### 6. Model Saving

**Current (LSTM):**
```python
model.save(os.path.join(experiment_dir, "lstm_model.keras"))
```

**PatchTSMixer:**
```python
# Save Hugging Face model
model.save_pretrained(os.path.join(experiment_dir, "patchtsmixer_model"))
config.save_pretrained(os.path.join(experiment_dir, "patchtsmixer_model"))

# Or via trainer
trainer.save_model(os.path.join(experiment_dir, "patchtsmixer_model"))

# Loading:
from transformers import PatchTSMixerForPrediction
model = PatchTSMixerForPrediction.from_pretrained(model_path)
```

**Differences:**
- `save_pretrained` vs `save`
- Config saved separately
- PyTorch `.bin` files vs Keras `.keras` file

#### 7. Dependencies

**Current:** tensorflow, keras

**PatchTSMixer Needs:**
```bash
# Add to requirements-base.txt
torch>=2.0.0
transformers>=4.35.0
```

#### 8. Inference Pattern

**Current (LSTM):**
```python
model = keras.models.load_model("lstm_model.keras")
predictions = model.predict(X_new)
```

**PatchTSMixer:**
```python
from transformers import PatchTSMixerForPrediction
model = PatchTSMixerForPrediction.from_pretrained("patchtsmixer_model")
model.eval()

import torch
with torch.no_grad():
    inputs = torch.FloatTensor(X_new)
    outputs = model(past_values=inputs)
    predictions = outputs.prediction_outputs.cpu().numpy()
```

**Differences:**
- Hugging Face loading vs Keras loading
- PyTorch inference mode (`model.eval()`, `torch.no_grad()`)
- Named input parameter (`past_values`)

#### 9. Visualization

**Current:** Single-step forecast plots

**PatchTSMixer Needs:** Multi-horizon plots
```python
# Plot predictions for multiple horizons
fig, axes = plt.subplots(prediction_length // 12, 12, figsize=(20, 10))
for h in range(prediction_length):
    ax = axes[h // 12, h % 12]
    ax.plot(y_test[:, h], label="Actual")
    ax.plot(y_pred[:, h], label="Predicted")
    ax.set_title(f"Horizon {h+1}")
plt.tight_layout()
plt.savefig("multi_horizon_forecast.png")
```

#### 10. Reproducibility Setup

**Current (LSTM):**
```python
# train.py:112-146
SEED = 42

def set_global_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = '42'
```

**PatchTSMixer:**
```python
import random
import numpy as np
import torch
import os
from transformers import set_seed

def set_global_reproducibility(seed=42):
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # HuggingFace set_seed
    set_seed(seed)
```

**Differences:**
- PyTorch-specific seeds and flags
- `torch.use_deterministic_algorithms(True)` for reproducibility
- Additional environment variables for CUBLAS

### Summary of Changes for PatchTSMixer

| Aspect | LSTM (Current) | PatchTSMixer (New) | Change Magnitude |
|--------|---------------|-------------------|------------------|
| **Data Preparation** | Sliding window sequences | Multi-step sequences with internal patching | Medium |
| **Framework** | TensorFlow/Keras | PyTorch/Transformers | High |
| **Model API** | Sequential API | HuggingFace Config + Model | High |
| **Training API** | model.fit() | Trainer API | High |
| **Output Format** | Single-step (1D) | Multi-step (3D) | Medium |
| **Hyperparameters** | LSTM-specific | Transformer-specific | High |
| **Model Saving** | .keras file | save_pretrained() | Medium |
| **Inference** | model.predict() | model(past_values=...) | Medium |
| **Evaluation** | Single metrics | Per-horizon + aggregate | Medium |
| **Reproducibility** | TensorFlow seeds | PyTorch seeds | Low |

**Reusable Components (No Changes Needed):**
- ✅ MLflow tracking URI configuration
- ✅ MLflow run management and logging
- ✅ DVC versioning workflow (add → commit → push)
- ✅ Energy tracking (CodeCarbon wrapper)
- ✅ Temporal train/val/test split logic
- ✅ Service orchestration pattern
- ✅ View layer validation and error handling
- ✅ Frontend state management (AppContext)
- ✅ Frontend validation pipeline

---

## Implementation Recommendations

### Backend Implementation Strategy

**1. Create New Training File**

File: `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/train_patchtsmixer.py`

```python
def train_patchtsmixer_model(dataset_path: str, data: Dict, experiment_dir: str) -> Dict:
    """
    Train PatchTSMixer model for time series forecasting.

    Mirrors train_lstm_model() structure with PatchTSMixer-specific changes.
    """
    # 1. Set reproducibility (reuse set_global_seeds with PyTorch additions)
    set_global_reproducibility(seed=42)

    # 2. Extract parameters (similar to LSTM)
    date_col = data.get("date_col_name")
    target_col = data.get("target_variable")
    input_features = data.get("input_features", [])
    forecast_horizon = data.get("forecast_horizon", 96)

    # 3. Determine univariate vs multivariate (same logic as LSTM)
    if len(input_features) == 0:
        ts_columns = [target_col]
        num_input_channels = 1
    else:
        ts_columns = input_features + [target_col]
        num_input_channels = len(ts_columns)

    # 4. Load and validate data (same as LSTM)
    df = pd.read_csv(dataset_path)
    df = validate_numeric_features(df, ts_columns)

    # 5. Create sequences (NEW: multi-step with PyTorch tensors)
    past_values, future_values = create_sequences_patchtsmixer(
        df[ts_columns],
        context_length=manual_params.get("context_length", 512),
        prediction_length=forecast_horizon,
        split_ratios=data.get("split_ratios")
    )

    # 6. Split train/val/test (reuse temporal split logic)
    train_past, train_future, val_past, val_future, test_past, test_future = \
        patchtsmixer_train_val_test_split(past_values, future_values, split_ratios)

    # 7. Create PyTorch datasets (NEW)
    train_dataset = TimeSeriesDataset(train_past, train_future)
    val_dataset = TimeSeriesDataset(val_past, val_future)
    test_dataset = TimeSeriesDataset(test_past, test_future)

    # 8. Configure model (NEW: PatchTSMixerConfig)
    config = PatchTSMixerConfig(
        context_length=manual_params.get("context_length", 512),
        prediction_length=forecast_horizon,
        num_input_channels=num_input_channels,
        patch_length=manual_params.get("patch_length", 8),
        patch_stride=manual_params.get("patch_length", 8),
        d_model=manual_params.get("d_model", 32),
        num_layers=manual_params.get("num_layers", 8),
        expansion_factor=2,
        dropout=manual_params.get("dropout", 0.2),
        head_dropout=manual_params.get("dropout", 0.2),
        mode="common_channel",
        scaling="std",
        loss="mse",
    )

    # 9. Initialize model (NEW: Hugging Face API)
    model = PatchTSMixerForPrediction(config)

    # 10. Start MLflow run (same as LSTM)
    with mlflow.start_run(nested=True):
        # Log hyperparameters (same pattern as LSTM)
        mlflow.log_params({...})

        # 11. Start energy tracking (same as LSTM)
        tracker = EmissionsTracker(...)
        tracker.start()

        # 12. Execute training based on hyperparameter_search_strategy
        if hyperparameter_search_strategy == "manual":
            # Train with Trainer API (NEW)
            trainer = train_manual_patchtsmixer(...)
        elif hyperparameter_search_strategy == "grid":
            # Grid search with Trainer API (NEW)
            trainer = train_grid_search_patchtsmixer(...)
        # ... etc

        # 13. Stop energy tracking (same as LSTM)
        tracker.stop()
        log_energy_metrics(tracker)

        # 14. Evaluate on val/test (NEW: multi-horizon metrics)
        val_metrics = evaluate_patchtsmixer(trainer, val_dataset, "val")
        test_metrics = evaluate_patchtsmixer(trainer, test_dataset, "test")

        # 15. Log metrics (same pattern as LSTM)
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)

        # 16. Save model (NEW: save_pretrained)
        model_path = os.path.join(experiment_dir, "patchtsmixer_model")
        trainer.save_model(model_path)

        # 17. Log artifacts (same as LSTM)
        for artifact in val_artifacts + test_artifacts:
            mlflow.log_artifact(artifact, "plots")

    # 18. Return results (same format as LSTM)
    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path
    }
```

**2. Update Services Layer**

File: `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/services.py`

```python
# services.py:1031-1060
def train_model_logic(self, dataset_file, data: dict) -> dict:
    # ... existing code ...

    # Add PatchTSMixer case
    elif algorithm == "patchtsmixer":
        from apiTimeSeries.train_patchtsmixer import train_patchtsmixer_model
        result = train_patchtsmixer_model(
            dataset_path=dataset_path,
            data=data,
            experiment_dir=experiment_dir
        )

    # ... rest of function ...
```

**3. Update Views Layer**

File: `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/views.py`

```python
# views.py:982
supported_algorithms = ["arima", "xgboost", "lstm", "patchtsmixer"]
```

**4. Update Requirements**

File: `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/requirements-base.txt`

```
# Add PyTorch and Transformers
torch>=2.0.0
transformers>=4.35.0
```

### Frontend Implementation Strategy

**File:** `/workspaces/dream-ml-c/DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx`

**1. Add Algorithm Option (line ~1414):**
```javascript
<MenuItem value="patchtsmixer">PatchTSMixer (Transformer)</MenuItem>
```

**2. Add State Variables (after line 287):**
```javascript
// Manual parameters
const [patchTSMixerParams, setPatchTSMixerParams] = useState({
  context_length: 512,
  patch_length: 8,
  d_model: 32,
  num_layers: 8,
  dropout: 0.2,
  learning_rate: 0.001,
  batch_size: 32,
  epochs: 100,
  early_stopping_patience: 10,
});

// Grid search parameters
const [patchTSMixerGridOptions, setPatchTSMixerGridOptions] = useState({
  context_length_options: "512",
  patch_length_options: "8,16",
  d_model_options: "32,64",
  num_layers_options: "6,8",
  dropout_options: "0.2,0.3",
  learning_rate_options: "0.001,0.0001",
  batch_size_options: "32,64",
  epochs_options: "50,100",
});

// Random search parameters
const [patchTSMixerRandomRanges, setPatchTSMixerRandomRanges] = useState({
  context_length_options: [512],
  patch_length_range: [8, 32],
  d_model_options: [16, 32, 64],
  num_layers_range: [4, 12],
  dropout_range: [0.1, 0.5],
  learning_rate_range: [0.0001, 0.01],
  batch_size_options: [16, 32, 64],
  epochs_range: [50, 200],
});

// Bayesian search parameters
const [patchTSMixerBayesianRanges, setPatchTSMixerBayesianRanges] = useState({
  d_model: { choices: [16, 32, 64, 128] },
  num_layers: { min: 4, max: 12 },
  patch_length: { min: 8, max: 32 },
  dropout: { min: 0.1, max: 0.5 },
  learning_rate: { min: 0.0001, max: 0.01, log: true },
  batch_size: { choices: [16, 32, 64, 128] },
});
```

**3. Add Payload Construction (in handleTrain, after LSTM block):**
```javascript
// Line ~800
else if (algorithm === "patchtsmixer") {
  if (optimizationMethod === "manual") {
    finalParams = {
      context_length: parseInt(patchTSMixerParams.context_length),
      patch_length: parseInt(patchTSMixerParams.patch_length),
      d_model: parseInt(patchTSMixerParams.d_model),
      num_layers: parseInt(patchTSMixerParams.num_layers),
      dropout: parseFloat(patchTSMixerParams.dropout),
      learning_rate: parseFloat(patchTSMixerParams.learning_rate),
      batch_size: parseInt(patchTSMixerParams.batch_size),
      epochs: parseInt(patchTSMixerParams.epochs),
      early_stopping_patience: parseInt(patchTSMixerParams.early_stopping_patience),
    };
  }
}

// PatchTSMixer-specific payload additions
if (algorithm === "patchtsmixer") {
  payload.input_features = lstmSelectedFeatures;  // Reuse LSTM feature selection
  payload.training_mode = lstmSelectedFeatures.length === 0 ? "univariate" : "multivariate";
  payload.optimization_metric = "mse";
}
```

**4. Add Grid Search Parameters:**
```javascript
// Integrate into line 843-858 block
if (algorithm === "patchtsmixer") {
  payload.grid_search_params = {
    context_length_options: patchTSMixerGridOptions.context_length_options.split(',').map(s => parseInt(s.trim())),
    patch_length_options: patchTSMixerGridOptions.patch_length_options.split(',').map(s => parseInt(s.trim())),
    d_model_options: patchTSMixerGridOptions.d_model_options.split(',').map(s => parseInt(s.trim())),
    num_layers_options: patchTSMixerGridOptions.num_layers_options.split(',').map(s => parseInt(s.trim())),
    dropout_options: patchTSMixerGridOptions.dropout_options.split(',').map(s => parseFloat(s.trim())),
    learning_rate_options: patchTSMixerGridOptions.learning_rate_options.split(',').map(s => parseFloat(s.trim())),
    batch_size_options: patchTSMixerGridOptions.batch_size_options.split(',').map(s => parseInt(s.trim())),
    epochs_options: patchTSMixerGridOptions.epochs_options.split(',').map(s => parseInt(s.trim())),
  };
}
```

**5. Add UI Forms (around line 2800, after LSTM forms):**
- Manual parameters form (TextFields for all params)
- Grid search form (comma-separated lists)
- Random search form (range inputs)
- Bayesian search form (range inputs with categorical choices)

### Testing Strategy

**1. Unit Tests**

Create test file: `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_patchtsmixer_phase1.py`

```python
import pytest
import torch
from apiTimeSeries.train_patchtsmixer import (
    create_sequences_patchtsmixer,
    TimeSeriesDataset,
    train_patchtsmixer_model,
)

def test_sequence_creation_univariate():
    """Test sequence creation for univariate time series"""
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=1000, freq="D"),
        "target": np.random.randn(1000)
    })

    past_values, future_values = create_sequences_patchtsmixer(
        df[["target"]],
        context_length=512,
        prediction_length=96
    )

    assert past_values.shape[1] == 512
    assert past_values.shape[2] == 1
    assert future_values.shape[1] == 96
    assert future_values.shape[2] == 1

def test_sequence_creation_multivariate():
    """Test sequence creation for multivariate time series"""
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=1000, freq="D"),
        "target": np.random.randn(1000),
        "feature1": np.random.randn(1000),
        "feature2": np.random.randn(1000),
    })

    past_values, future_values = create_sequences_patchtsmixer(
        df[["target", "feature1", "feature2"]],
        context_length=512,
        prediction_length=96
    )

    assert past_values.shape[2] == 3
    assert future_values.shape[2] == 3

def test_pytorch_dataset():
    """Test PyTorch Dataset class"""
    past_values = torch.randn(100, 512, 3)
    future_values = torch.randn(100, 96, 3)

    dataset = TimeSeriesDataset(past_values, future_values)

    assert len(dataset) == 100
    item = dataset[0]
    assert "past_values" in item
    assert "future_values" in item
    assert item["past_values"].shape == (512, 3)
    assert item["future_values"].shape == (96, 3)

def test_reproducibility():
    """Test that same seed produces identical results"""
    from apiTimeSeries.train_patchtsmixer import set_global_reproducibility

    # First run
    set_global_reproducibility(42)
    model1 = PatchTSMixerForPrediction(config)
    output1 = model1(past_values=test_tensor)

    # Second run
    set_global_reproducibility(42)
    model2 = PatchTSMixerForPrediction(config)
    output2 = model2(past_values=test_tensor)

    assert torch.allclose(output1.prediction_outputs, output2.prediction_outputs)
```

**2. Integration Tests**

Create test file: `test_patchtsmixer_integration.py`

```python
def test_full_training_pipeline(tmp_path):
    """Test complete training pipeline from CSV to trained model"""
    # 1. Create test CSV
    df = create_test_timeseries_data(length=1000)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    # 2. Configure training
    data = {
        "date_col_name": "date",
        "target_variable": "target",
        "input_features": ["feature1", "feature2"],
        "forecast_horizon": 96,
        "hyperparameter_search_strategy": "manual",
        "manual_params": {
            "context_length": 512,
            "patch_length": 8,
            "d_model": 32,
            "num_layers": 8,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 2,  # Short for testing
            "early_stopping_patience": 1,
        },
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
    }

    # 3. Train model
    result = train_patchtsmixer_model(
        dataset_path=str(csv_path),
        data=data,
        experiment_dir=str(tmp_path)
    )

    # 4. Verify results
    assert "val_metrics" in result
    assert "test_metrics" in result
    assert "model_path" in result
    assert os.path.exists(result["model_path"])

    # 5. Test loading saved model
    loaded_model = PatchTSMixerForPrediction.from_pretrained(result["model_path"])
    assert loaded_model is not None
```

**3. End-to-End Tests**

```python
def test_api_endpoint(client):
    """Test /api/ts/train-model/ endpoint with PatchTSMixer"""
    # Upload CSV
    csv_file = open("test_data.csv", "rb")

    # Prepare payload
    payload = {
        "model_name": "patchtsmixer_test",
        "algorithm": "patchtsmixer",
        "date_col_name": "date",
        "target_variable": "target",
        "input_features": ["feature1"],
        "forecast_horizon": 96,
        "hyperparameter_search_strategy": "manual",
        "manual_params": {...},
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
    }

    # Make request
    response = client.post(
        "/api/ts/train-model/",
        data={
            "file": csv_file,
            "data": json.dumps(payload)
        }
    )

    # Verify response
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert "run_id" in result
    assert "metrics" in result
    assert "model_path" in result
```

### Deployment Checklist

- [ ] Install PyTorch (`pip install torch`)
- [ ] Install Transformers (`pip install transformers`)
- [ ] Add dependencies to `requirements-base.txt`
- [ ] Create `train_patchtsmixer.py` with complete implementation
- [ ] Add `TimeSeriesDataset` class
- [ ] Implement `create_sequences_patchtsmixer()` function
- [ ] Implement `train_manual_patchtsmixer()` function
- [ ] Implement `train_grid_search_patchtsmixer()` function
- [ ] Implement `train_random_search_patchtsmixer()` function
- [ ] Implement `train_bayesian_search_patchtsmixer()` function
- [ ] Implement `evaluate_patchtsmixer()` function
- [ ] Add reproducibility setup function
- [ ] Update `TrainModelService.train_model_logic()` in `services.py`
- [ ] Update `train_model` view in `views.py`
- [ ] Create frontend UI in `TSTrainCard.jsx`
- [ ] Add essential hyperparameter inputs (9 params)
- [ ] Add grid/random/Bayesian parameter forms
- [ ] Test univariate forecasting
- [ ] Test multivariate forecasting
- [ ] Verify reproducibility (3+ runs with same seed)
- [ ] Create unit tests (phase 1, 2, 3, 4 similar to LSTM)
- [ ] Create integration tests
- [ ] Create end-to-end tests
- [ ] Document in user guide
- [ ] Update API documentation

---

## Code References

### Frontend Code References

| Component | File | Line(s) | Purpose |
|-----------|------|---------|---------|
| `handleTrain` | TSTrainCard.jsx | 700 | Primary training orchestration |
| Payload construction | TSTrainCard.jsx | 803-893 | Algorithm-specific parameter serialization |
| FormData construction | TSTrainCard.jsx | 894-900 | CSV + JSON multipart request |
| API call | TSTrainCard.jsx | 903 | POST to `/ts/train-model/` |
| `validateSelections` | TSTrainCard.jsx | 368 | Feature/target/date validation |
| `validateBayesianConfig` | TSTrainCard.jsx | 643 | Bayesian optimization validation |
| `validateParamRanges` | TSTrainCard.jsx | 1149 | Hyperparameter range validation |
| `validateSplitRatios` | TSTrainCard.jsx | 416 | Train/val/test split validation |
| `isDisabled` computed | TSTrainCard.jsx | 1234 | Button enable/disable logic (16 checks) |
| `getCurrentState` | TSTrainCard.jsx | 455 | Training state indicator |
| `handleFileChange` | TSTrainCard.jsx | 292 | CSV file selection |
| `loadColumns` | TSTrainCard.jsx | 302 | Column analysis via API |
| `handleTargetChange` | TSTrainCard.jsx | 338 | Target variable selection |
| `handleFeatureChange` | TSTrainCard.jsx | 328 | Feature toggle |
| `handleDateColumnChange` | TSTrainCard.jsx | 360 | Date column selection |
| AppContext provider | AppContext.jsx | 46-139 | Global workflow state |
| Axios config | axiosConfig.js | 30-51 | API client + interceptors |
| ValidationSummary | ValidationSummary.jsx | 25 | Warning display component |
| ProgressBar | ProgressBar.jsx | 35 | Training progress display |
| Variable selection styles | variableSelectionStyles.js | 29 | Teal brutalist design system |

### Backend Code References

| Component | File | Line(s) | Purpose |
|-----------|------|---------|---------|
| **Views Layer** |
| `train_model` endpoint | views.py | 376 | Main training endpoint handler |
| Request validation | views.py | 389-393 | File and data validation |
| MLflow URI config | views.py | 396-405 | Tracking URI setup |
| Service call | views.py | 413-416 | Delegate to service layer |
| Response formatting | views.py | 431-436 | JSON response construction |
| Error handling | views.py | 439-479 | Granular exception catching |
| **Services Layer** |
| `train_model_logic` | services.py | 962 | Training orchestration |
| Parameter validation | services.py | 977-984 | Experiment dir and algorithm check |
| MLflow setup | services.py | 986-999 | Configure tracking, get experiment ID |
| Start MLflow run | services.py | 1002-1004 | Single run with system metrics |
| Dataset versioning | services.py | 1008-1028 | DVC add/commit/push |
| Algorithm routing | services.py | 1031-1060 | Route to train_lstm_model() |
| Model versioning | services.py | 1062-1070 | DVC version trained model |
| Metric consolidation | services.py | 1073-1083 | Log val/test metrics |
| Return step config | services.py | 1086-1104 | Return results dict |
| **Train Layer (LSTM)** |
| `train_lstm_model` | train.py | 3819 | Main LSTM training function |
| Parameter extraction | train.py | 3862-3882 | Parse config dict |
| MLflow logging | train.py | 3896-3919 | Log hyperparameters |
| Data loading | train.py | 3926-3955 | Load CSV, validate numeric |
| Univariate/multivariate | train.py | 3957-3970 | Detect mode |
| Sequence creation | train.py | 3976-3990 | Call create_sequences_for_lstm |
| Temporal split | train.py | 3997-4002 | Call lstm_train_val_test_split |
| Energy tracking | train.py | 4008-4016 | Start CodeCarbon tracker |
| Manual training | train.py | 4022-4074 | User-provided params |
| Grid search | train.py | 4076-4242 | Exhaustive search |
| Random search | train.py | 4244-4406 | Random sampling |
| Bayesian search | train.py | 4408-4856 | Optuna optimization |
| Validation eval | train.py | 4882-4889 | Evaluate on val set |
| Test eval | train.py | 4896-4903 | Evaluate on test set |
| Log metrics/artifacts | train.py | 4910-4924 | MLflow logging |
| Save model | train.py | 4946-4948 | .keras file |
| **Helpers** |
| `create_sequences_for_lstm` | train.py | 3286 | Sliding window sequence creation |
| `build_lstm_model` | train.py | 3460 | Model architecture builder |
| `lstm_train_val_test_split` | train.py | 3390 | Temporal data splitting |
| `create_lstm_callbacks` | train.py | 3558 | EarlyStopping, ModelCheckpoint, ReduceLR |
| `evaluate_lstm_model` | train.py | (not shown) | Generate predictions and metrics |
| `set_global_seeds` | train.py | 112-146 | Reproducibility setup |
| `log_energy_metrics` | train.py | 444-461 | CodeCarbon logging |
| **Infrastructure** |
| URL routing | urls.py | 20 | Map "train-model/" to view |
| `init_dvc_logic` | api/utils.py | 51 | DVC initialization |
| `configure_dvc_remote_logic` | api/utils.py | 177 | DVC remote setup |
| `is_mlflow_running` | api/utils.py | 266 | MLflow health check |
| `create_experiment_logic` | api/services.py | 76 | Experiment creation |
| **Preprocessing** |
| `create_lag_features` | data_encoding_utils.py | 5 | Lag feature creation |
| `handle_lag_nans` | data_encoding_utils.py | 36 | NaN strategies |
| `encode_data` | data_encoding_utils.py | 103 | Main encoding entry |
| Data cleaning functions | data_cleaning_utils.py | various | Whitespace, NaN, duplicates, outliers |

---

## Open Questions

### For User Clarification

1. **Hyperparameter Exposure:**
   - Should advanced PatchTSMixer parameters (expansion_factor, mode, gated_attn, self_attn) be:
     - Hidden with sensible defaults?
     - Available under "Advanced" toggle?
     - Always visible?

2. **Multi-Step Forecasting:**
   - Should the UI allow configuring `prediction_length` separately from `forecast_horizon`?
   - Or should `forecast_horizon` map directly to `prediction_length`?

3. **Feature Engineering:**
   - Should lag features be created for PatchTSMixer?
   - PatchTSMixer can work with raw time series (no lags needed), but lags might help performance

4. **Transfer Learning:**
   - Should the UI support loading pretrained PatchTSMixer models from HuggingFace Hub?
   - Would users benefit from zero-shot evaluation or fine-tuning?

5. **Visualization:**
   - For multi-horizon forecasts, should plots show:
     - All horizons in single plot?
     - Separate plot per horizon?
     - Aggregated metrics only?

6. **Grid Search Warning:**
   - Current LSTM implementation warns if grid combinations exceed threshold
   - What should the threshold be for PatchTSMixer? (Recommend: 30-50)

7. **Memory Profiling:**
   - LSTM has optional memory profiling for grid search
   - Should PatchTSMixer include similar feature?

### Technical Decisions

1. **Patch Length Validation:**
   - Should UI enforce `context_length % patch_length == 0`?
   - Or allow model to handle padding internally?

2. **Scaling Strategy:**
   - PatchTSMixer supports "std", "mean", or None
   - Should this be exposed to users or fixed to "std"?

3. **Channel Mode:**
   - "common_channel" (default) vs "mix_channel"
   - When should users switch to "mix_channel"?
   - Should UI provide guidance/tooltip?

4. **Bayesian Search with Optuna:**
   - Should we use Optuna for PatchTSMixer (consistent with LSTM)?
   - Or use HuggingFace `hyperparameter_search()` method?

5. **Model Saving Format:**
   - Should we save both PyTorch state dict AND HuggingFace format?
   - Or just HuggingFace `save_pretrained()`?

6. **CPU vs GPU:**
   - DREAM-ML targets CPU training
   - Should code detect GPU and use if available?
   - Or enforce CPU-only?

7. **Per-Horizon Metrics:**
   - Should all horizons be logged to MLflow individually?
   - Or only aggregate metrics + first/last/middle horizons?

### Architecture Questions

1. **Reuse LSTM Feature Selection:**
   - Frontend should reuse `lstmSelectedFeatures` state for PatchTSMixer?
   - Or create separate `patchTSMixerSelectedFeatures`?
   - **Recommendation:** Reuse to simplify UI

2. **Sequence Length Naming:**
   - LSTM uses `sequence_length`
   - PatchTSMixer uses `context_length`
   - Should UI label be "Sequence Length" or "Context Length" or both?

3. **Training Mode:**
   - LSTM explicitly sets `training_mode: "univariate" | "multivariate"`
   - PatchTSMixer infers from `num_input_channels`
   - Should backend still receive `training_mode` parameter?

4. **Early Stopping:**
   - HuggingFace Trainer uses `EarlyStoppingCallback`
   - LSTM uses Keras `EarlyStopping`
   - Should patience default be same (20) or different?

---

## Summary

This research provides a comprehensive understanding of DREAM-ML's time series training workflow, from frontend user interactions through backend processing. Key takeaways:

### Existing Architecture Strengths

1. **Well-structured three-layer backend** (views → services → train) with clear separation of concerns
2. **Comprehensive validation pipeline** preventing invalid submissions at multiple stages
3. **Robust MLflow + DVC integration** for experiment tracking and reproducibility
4. **Flexible hyperparameter search** supporting manual, grid, random, and Bayesian strategies
5. **Energy tracking** via CodeCarbon for sustainability metrics
6. **Context-based state management** enabling workflow progression tracking

### PatchTSMixer Implementation Feasibility

**High Feasibility (80% code reuse):**
- MLflow tracking setup and logging
- DVC versioning workflow
- Energy tracking wrapper
- Service orchestration pattern
- View layer validation and error handling
- Frontend state management
- Validation pipeline

**Medium Feasibility (50% code reuse with modifications):**
- Data preparation (sequences → multi-step sequences + patches)
- Temporal splitting (same logic, different shapes)
- Evaluation (single-step → multi-horizon)
- Model saving (Keras → HuggingFace)

**Low Feasibility (new implementation required):**
- Model architecture (Keras Sequential → HuggingFace Transformers)
- Training API (model.fit → Trainer API)
- Hyperparameter search (Keras-specific → PyTorch-specific)
- Reproducibility setup (TensorFlow → PyTorch seeds)

### Recommended Implementation Approach

1. **Phase 1:** Backend foundation
   - Create `train_patchtsmixer.py` mirroring LSTM structure
   - Implement sequence creation with PyTorch tensors
   - Integrate Hugging Face Trainer API
   - Add manual training strategy first

2. **Phase 2:** Hyperparameter search
   - Implement grid search
   - Implement random search
   - Implement Bayesian search (reuse Optuna patterns)

3. **Phase 3:** Frontend integration
   - Add PatchTSMixer to algorithm dropdown
   - Create parameter state variables
   - Implement payload construction
   - Add UI forms (manual → grid → random → bayesian)

4. **Phase 4:** Testing & validation
   - Unit tests for data preparation
   - Integration tests for full pipeline
   - Reproducibility verification
   - End-to-end API tests

5. **Phase 5:** Documentation & deployment
   - User guide updates
   - API documentation
   - Deployment to staging
   - User acceptance testing

### Estimated Effort

- **Backend:** 3-5 days (data prep + training + search strategies)
- **Frontend:** 2-3 days (state + payload + UI forms)
- **Testing:** 2-3 days (unit + integration + e2e)
- **Documentation:** 1 day
- **Total:** 8-12 days

### Success Criteria

1. ✅ PatchTSMixer training completes successfully for univariate and multivariate datasets
2. ✅ All four hyperparameter search strategies work correctly
3. ✅ Reproducibility verified (same seed → same results across 3 runs)
4. ✅ MLflow logging captures all parameters, metrics, and artifacts
5. ✅ DVC versioning works for datasets and models
6. ✅ Frontend UI matches existing LSTM complexity and patterns
7. ✅ Unit tests achieve >80% code coverage
8. ✅ End-to-end tests pass for all optimization strategies
9. ✅ Performance meets or exceeds LSTM baseline on test datasets
10. ✅ Documentation enables data scientists to use PatchTSMixer effectively

---

**Document Version:** 1.0
**Research Completed:** 2026-01-12
**Researcher:** Claude Sonnet 4.5
**Review Status:** Ready for Implementation Planning