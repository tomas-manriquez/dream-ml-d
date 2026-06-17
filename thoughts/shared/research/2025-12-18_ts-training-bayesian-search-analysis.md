# Time Series Training Workflow Analysis for Bayesian Search Implementation

**Date**: 2025-12-18
**Author**: Claude Code Analysis
**Purpose**: Understand TS training workflow to implement Bayesian Search hyperparameter tuning using Optuna

---

## Executive Summary

This document analyzes the complete time series model training workflow in DREAM-ML to guide the implementation of Bayesian Search hyperparameter tuning for all 6 supported algorithms (ARIMA, SARIMA, ARIMAX, SARIMAX, XGBoost, LSTM).

### Key Findings

1. **Partial Implementation Exists**: Frontend UI for Bayesian Search is already built, backend validation accepts "bayesian" strategy, but **no actual Optuna implementation exists**
2. **Optuna 4.6.0 is installed**: Modern hyperparameter optimization framework ready to use (requirements-base.txt:18)
3. **Current Architecture**: Clean 4-layer separation (React UI → Django View → Service Orchestration → Algorithm Training)
4. **Reproducibility is Paramount**: System uses fixed seeds (42), single-threaded execution, and comprehensive pipeline_config.json persistence
5. **Implementation Pattern Clear**: Grid/Random search already implemented - Bayesian Search follows identical pattern with Optuna TPESampler

---

## 1. System Architecture

### 1.1 Complete Data Flow

```
User (TSTrainCard.jsx)
    ↓ POST /api/ts/train-model/
Django View (views.py - NOT IN CURRENT IMPLEMENTATION)
    ↓ Delegates to service layer
Service Orchestration (services.py:962 train_model_logic())
    ↓ Preprocesses data, manages reproducibility
Algorithm Training (train.py:1415, 2048, 2986)
    ↓ Trains model with hyperparameter tuning
Results Persistence
    ↓ MLflow + pipeline_config.json + artifacts
Response to User
```

**Critical Note**: The current implementation (`apiTimeSeries/services.py:962`) uses a DIFFERENT pattern than the analyzer reported. It calls training functions like:

```python
train_arima_model(dataset_path, data, experiment_dir)
```

NOT the old signature with individual parameters.

### 1.2 File Structure

**Frontend Core**:
- `TSTrainCard.jsx:87-1400` - Main training UI component with Bayesian Search UI already built

**Backend Core**:
- `train.py:1415` - `train_arima_model()` - ARIMA/SARIMA/ARIMAX/SARIMAX training
- `train.py:2048` - `train_xgboost_model()` - XGBoost regression training
- `train.py:2986` - `train_lstm_model()` - LSTM deep learning training
- `services.py:962` - `train_model_logic()` - Service orchestration
- `views.py` - Django REST endpoint (NOT analyzed in detail - different implementation)

**Supporting**:
- `data_cleaning_utils.py` - Data preprocessing
- `data_encoding_utils.py` - Feature engineering
- `AppContext.jsx` - Global state management
- `axiosConfig.js` - API client

**Configuration**:
- `urls.py:20` - URL routing
- `settings.py` - Django settings, MLflow config
- `requirements-base.txt:17-18` - **Optuna 4.6.0** (modern), scikit-optimize 0.10.2 (EOL, must not use)

---

## 2. Current Hyperparameter Tuning Implementation

### 2.1 Frontend: User Selection (TSTrainCard.jsx)

**Tuning Method State** (line 118):
```javascript
const [optimizationMethod, setOptimizationMethod] = useState("manual");
```

**Available Options** (line 1181):
```javascript
["manual", "grid", "random", "bayesian"].map((method) => ...)
```

**Bayesian Search UI** (line 1228-1360):
```javascript
{optimizationMethod === "bayesian" && (
  <Box>
    <TextField
      label="n_initial_points"
      value={bayesianConfig.n_initial_points}
    />
    <Select
      label="acq_func"
      value={bayesianConfig.acq_func}
    />
    <TextField label="max_memory_mb" />
    <TextField label="timeout_seconds" />
    <TextField label="convergence_tolerance" />
    <TextField label="convergence_patience" />
  </Box>
)}
```

**Bayesian Config State** (line 199):
```javascript
const [bayesianConfig, setBayesianConfig] = useState({
  n_initial_points: 10,
  acq_func: "ei",
  max_memory_mb: null,
  timeout_seconds: null,
  convergence_tolerance: 1e-4,
  convergence_patience: 5
});
```

**Status**: ✅ Frontend UI is COMPLETE and ready

### 2.2 Backend: Strategy Validation (train.py)

**Valid Strategies** (line 1462):
```python
valid_strategies = ["manual", "grid", "random", "bayesian"]
if hyperparameter_search_strategy not in valid_strategies:
    raise ValueError(f"hyperparameter_search_strategy debe ser uno de: {valid_strategies}")
```

**Status**: ✅ Backend accepts "bayesian" but ❌ **NO IMPLEMENTATION EXISTS**

### 2.3 Grid Search Implementation Pattern (train.py:1536-1675)

**Current Grid Search for ARIMA** (lines 1536-1614):
```python
if hyperparameter_search_strategy == "grid":
    grid_config = data.get("grid_search", {})
    param_grid = generate_arima_grid(grid_config)

    # Walk-forward validation
    n_folds = 5
    initial_train_size = int(len(df) * split_ratios["train"])
    optimization_metric = data.get("optimization_metric", "val_rmse")

    best_score = float('inf')
    best_model = None
    best_params = None

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

            score = fold_metrics[optimization_metric]
            if score < best_score:
                best_score = score
                best_params = params
                # ... update best model
        except Exception as e:
            logger.warning(f"Grid iteration {i} failed: {e}")
            continue
```

**Key Pattern**: Exhaustive search over parameter grid with walk-forward cross-validation

### 2.4 Random Search Implementation Pattern (train.py:~1676+)

**Random Search** (similar structure):
```python
elif hyperparameter_search_strategy == "random":
    n_random_iterations = data.get("n_random_iterations", 100)
    random_search_params = data.get("random_search_params", {})

    # Sample random configurations
    for iteration in range(n_random_iterations):
        # Sample params from ranges
        # Evaluate with walk-forward validation
        # Track best
```

**Key Pattern**: Random sampling from parameter ranges with same validation strategy

---

## 3. Algorithm-Specific Details

### 3.1 ARIMA/SARIMA/ARIMAX/SARIMAX (train.py:1415+)

**Function Signature**:
```python
def train_arima_model(dataset_path: str, data: Dict, experiment_dir: str) -> Dict:
```

**Hyperparameters**:
- **Non-seasonal**: `p` (0-5), `d` (0-3), `q` (0-5)
- **Seasonal**: `P` (0-3), `D` (0-3), `Q` (0-3), `s` (2-52)
- **Other**: `trend`, `enforce_stationarity`, `enforce_invertibility`

**Validation Strategy**: Walk-forward cross-validation with `skforecast.model_selection.backtesting_sarimax`

**Optimization Metric**: `val_rmse` (default), also supports `val_mae`, `val_mape`, `test_*` variants

**Reproducibility**:
- SARIMAX optimizer config (line 83-91):
  ```python
  SARIMAX_OPTIMIZER_DEFAULTS = {
      "method": "lbfgs",
      "maxiter": 500,
      "ftol": 1e-8,
      "gtol": 1e-6,
      "epsilon": 1e-10
  }
  ```
- Deterministic given same data and parameters

### 3.2 XGBoost (train.py:2048+)

**Function Signature**:
```python
def train_xgboost_model(dataset_path: str, data: Dict, experiment_dir: str) -> Dict:
```

**Hyperparameters**:
- `n_estimators` (50-1000)
- `max_depth` (3-15)
- `learning_rate` (1e-3 to 0.3, log-uniform)
- `subsample` (0.5-1.0)
- `colsample_bytree` (0.5-1.0)
- `gamma` (0-1.0)
- `min_child_weight` (1-10)

**Validation Strategy**: Simple train/test split (NOT time series CV currently)

**Reproducibility**:
```python
XGBRegressor(random_state=SEED, n_jobs=1)
```

**Note**: `n_jobs=1` enforces single-threaded execution for reproducibility

### 3.3 LSTM (train.py:2986+)

**Function Signature**:
```python
def train_lstm_model(dataset_path: str, data: Dict, experiment_dir: str) -> Dict:
```

**Hyperparameters**:
- `lstm_units` (16-256)
- `dropout_rate` (0.0-0.5)
- `learning_rate` (1e-4 to 1e-2, log-uniform)
- `batch_size` (8-128)
- `epochs` (30-200)
- `time_steps` (5-50)

**Validation Strategy**: Simple train/test split with early stopping

**Reproducibility** (MOST COMPLEX):
```python
def set_global_seeds():
    """Line 97-120 in train.py"""
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
```

**Critical**: Seeds MUST be reset before each model build in objective function

---

## 4. Reproducibility Architecture

### 4.1 Global Seed Management

**Global Seed** (train.py:75):
```python
SEED = 42
```

**Function** (train.py:97):
```python
def set_global_seeds():
    """Fija semillas para todas las librerías relevantes"""
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    # ... TensorFlow config
```

### 4.2 Pipeline Config Persistence

**Location**: `{experiment_dir}/pipeline_config.json`

**Structure** (based on analyzer report):
```json
{
  "random_seed": 42,
  "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
  "preprocessing_metadata": {...},
  "algorithm": "ARIMA",
  "hyperparameter_search_strategy": "grid",
  "optimization_metric": "val_rmse",
  "best_params": {"p": 2, "d": 1, "q": 2},
  "val_metrics": {...},
  "test_metrics": {...},
  "platform_info": {
    "python_version": "3.11.14",
    "numpy_version": "1.26.4",
    "tensorflow_version": "2.19.1"
  }
}
```

### 4.3 MLflow Integration

**Configuration** (services.py:986-999):
```python
mlflow.set_tracking_uri(f"sqlite:///{shared_db_path}")
mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
```

**Logging** (train.py:1522-1532):
```python
mlflow.log_params({
    "model_type": "ARIMA",
    "hyperparameter_search_strategy": hyperparameter_search_strategy,
    "n_random_iterations": n_random_iterations,
    "platform_python_version": platform_info["python_version"]
})
```

---

## 5. Bayesian Search Implementation Plan with Optuna

### 5.1 Optuna API Pattern

**Basic Structure**:
```python
import optuna
from optuna.samplers import TPESampler

def objective(trial):
    # Suggest hyperparameters
    p = trial.suggest_int('p', 0, 5)
    d = trial.suggest_int('d', 0, 2)
    q = trial.suggest_int('q', 0, 5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    # Train model with params
    model = train_with_params(p, d, q)

    # Evaluate
    score = evaluate(model)

    # Return metric to MINIMIZE
    return score

# Create study with TPE sampler for reproducibility
sampler = TPESampler(seed=42, n_startup_trials=10)
study = optuna.create_study(
    direction='minimize',
    sampler=sampler
)

# Optimize
study.optimize(objective, n_trials=50)

# Get best params
best_params = study.best_params
best_score = study.best_value
```

### 5.2 Reproducibility with Optuna

**Requirements** (from web search):

1. **Fixed Seed**: `TPESampler(seed=42)`
2. **Sequential Execution**: No parallel trials (already satisfied - single-threaded training)
3. **Deterministic Objective**: Must return same value for same params (requires proper seeding in objective)

**For LSTM** (CRITICAL):
```python
def objective(trial):
    # Reset TF seeds INSIDE objective
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()

    # ... rest of objective
```

### 5.3 Implementation Template for ARIMA

**Location**: `train.py:1415` in `train_arima_model()`

**Add Branch**:
```python
elif hyperparameter_search_strategy == "bayesian":
    import optuna
    from optuna.samplers import TPESampler

    # Extract Bayesian config from data
    bayesian_config = data.get("bayesian_config", {})
    n_trials = bayesian_config.get("n_trials", 50)
    n_initial_points = bayesian_config.get("n_initial_points", 10)
    timeout = bayesian_config.get("timeout_seconds", None)

    # Define objective function
    def objective(trial):
        # Suggest hyperparameters
        p = trial.suggest_int('p', 0, 5)
        d = trial.suggest_int('d', 0, 2)
        q = trial.suggest_int('q', 0, 5)

        # Seasonal if enabled
        if enableSeasonalParams:
            P = trial.suggest_int('P', 0, 3)
            D = trial.suggest_int('D', 0, 2)
            Q = trial.suggest_int('Q', 0, 3)
            s = trial.suggest_int('s', 2, 52)

        # Other categorical params
        trend = trial.suggest_categorical('trend', ['n', 'c', 't', 'ct'])
        enforce_stationarity = trial.suggest_categorical('enforce_stationarity', [True, False])
        enforce_invertibility = trial.suggest_categorical('enforce_invertibility', [True, False])

        # Build params dict
        params = {
            'order': (p, d, q),
            'seasonal_order': (P, D, Q, s) if enableSeasonalParams else (0, 0, 0, 0),
            'trend': trend,
            'enforce_stationarity': enforce_stationarity,
            'enforce_invertibility': enforce_invertibility
        }

        try:
            # Walk-forward validation
            fold_metrics = walk_forward_validate_sarimax(
                y_data=y_full,
                exog_data=exog_full,
                params=params,
                n_folds=n_folds,
                initial_train_size=initial_train_size,
                forecast_horizon=forecast_horizon
            )

            # Return metric to minimize
            return fold_metrics[optimization_metric]

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            # Return high penalty for failed trials
            return float('inf')

    # Create study with TPE sampler
    sampler = TPESampler(
        seed=SEED,  # Fixed seed for reproducibility
        n_startup_trials=n_initial_points  # Random exploration before Bayesian
    )

    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name=f"arima_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Optimize with optional timeout
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False  # Quiet mode for logs
    )

    # Extract best params
    best_params = study.best_params
    best_score = study.best_value

    logger.info(f"Bayesian Search completed. Best {optimization_metric}: {best_score}")
    logger.info(f"Best params: {best_params}")

    # Train final model with best params
    final_model = SARIMAX(
        y_train,
        exog=exog_train,
        order=(best_params['p'], best_params['d'], best_params['q']),
        seasonal_order=(best_params.get('P', 0), best_params.get('D', 0),
                       best_params.get('Q', 0), best_params.get('s', 0)),
        trend=best_params['trend'],
        enforce_stationarity=best_params['enforce_stationarity'],
        enforce_invertibility=best_params['enforce_invertibility']
    )
    fitted_model = final_model.fit(**SARIMAX_OPTIMIZER_DEFAULTS)

    # Continue with evaluation and return...
```

### 5.4 XGBoost Implementation Pattern

```python
def objective(trial):
    # Suggest params with appropriate distributions
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    gamma = trial.suggest_float('gamma', 0, 1.0)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)

    # Train model
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        min_child_weight=min_child_weight,
        random_state=SEED,
        n_jobs=1
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds)
```

### 5.5 LSTM Implementation Pattern (WITH SEED RESET)

```python
def objective(trial):
    # CRITICAL: Reset seeds inside objective
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()

    # Suggest params
    lstm_units = trial.suggest_int('lstm_units', 16, 256)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 8, 128)
    epochs = trial.suggest_int('epochs', 30, 200)
    time_steps = trial.suggest_int('time_steps', 5, 50)

    # Recreate sequences with new time_steps
    X_tr, y_tr = create_sequences(train_features, time_steps)
    X_val, y_val = create_sequences(val_features, time_steps)

    # Build model
    model = Sequential([
        LSTM(lstm_units, activation='tanh', input_shape=(time_steps, n_features)),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    # Train with early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop]
    )

    # Evaluate
    preds = model.predict(X_val, verbose=0)
    return mean_squared_error(y_val, preds)
```

---

## 6. Code References (Critical Lines)

### Frontend
- **TSTrainCard.jsx:118** - `optimizationMethod` state
- **TSTrainCard.jsx:199** - `bayesianConfig` state
- **TSTrainCard.jsx:1181** - Tuning method radio buttons including "bayesian"
- **TSTrainCard.jsx:1228-1360** - Bayesian Search advanced options UI

### Backend Training Functions
- **train.py:75** - `SEED = 42` global constant
- **train.py:97-120** - `set_global_seeds()` function
- **train.py:1415** - `train_arima_model()` entry point
- **train.py:1462** - Valid strategies validation (includes "bayesian")
- **train.py:1536** - Grid search implementation start
- **train.py:2048** - `train_xgboost_model()` entry point
- **train.py:2986** - `train_lstm_model()` entry point

### Service Orchestration
- **services.py:962** - `train_model_logic()` main orchestration
- **services.py:1042-1059** - Algorithm dispatch logic

### Dependencies
- **requirements-base.txt:17** - scikit-optimize 0.10.2 (EOL, DO NOT USE)
- **requirements-base.txt:18** - **Optuna 4.6.0** (USE THIS)

---

## 7. Open Questions & Recommendations

### 7.1 Open Questions

1. **Frontend-Backend Contract**: Does TSTrainCard send `bayesian_config` as nested object or flat params?
   - Need to verify actual API request structure
   - Check if `data` dict in `train_arima_model()` includes `bayesian_config`

2. **Validation Metrics**: Should Bayesian Search optimize on validation or test metrics?
   - Current: `optimization_metric = data.get("optimization_metric", "val_rmse")`
   - Recommendation: Always use validation metrics to avoid test set leakage

3. **Cross-Validation Strategy**: Should XGBoost/LSTM use time series CV for robustness?
   - Current: Simple train/val split
   - Potential enhancement: `sklearn.model_selection.TimeSeriesSplit`

4. **Convergence Criteria**: How to implement early stopping for Bayesian Search?
   - Optuna has built-in early stopping with `optuna.pruners`
   - Frontend has `convergence_patience` field - needs implementation

5. **Memory Limits**: Frontend has `max_memory_mb` field - how to enforce?
   - Optuna doesn't have direct memory limits
   - May need custom callback to monitor `psutil.Process().memory_info()`

### 7.2 Recommendations

#### Priority 1: Core Implementation
1. **Implement Bayesian Search for ARIMA first** (simplest, most stable)
2. **Add Optuna imports** to `train.py`
3. **Create `elif hyperparameter_search_strategy == "bayesian":` branch** in each training function
4. **Test reproducibility** with fixed seeds

#### Priority 2: Frontend Integration
1. **Verify API contract** - what does TSTrainCard.jsx send?
2. **Update API request** in TSTrainCard to include `bayesian_config`
3. **Test full end-to-end** flow from UI to results

#### Priority 3: Advanced Features
1. **Implement convergence detection** using Optuna callbacks
2. **Add Optuna visualization** to results (optimization history plot)
3. **Save Optuna study** to experiment directory for reproducibility
4. **Integrate with MLflow** - log trials as nested runs

#### Priority 4: Testing
1. **Unit tests** for each algorithm's Bayesian Search
2. **Reproducibility tests** - same seed produces same results
3. **Convergence tests** - Bayesian beats random search in fewer trials
4. **Edge case tests** - failed trials, timeouts, memory limits

### 7.3 Architecture Insights

**Strengths**:
- ✅ Clean separation of concerns (UI → View → Service → Algorithm)
- ✅ Comprehensive reproducibility architecture
- ✅ Optuna already installed
- ✅ Frontend UI already built
- ✅ Validation accepts "bayesian" strategy

**Gaps**:
- ❌ No Optuna implementation in backend
- ❌ No Bayesian Search tests
- ❌ Unclear frontend-backend contract for `bayesian_config`
- ❌ No convergence detection implementation

**Risks**:
- ⚠️ LSTM reproducibility is complex - TF seed reset in objective critical
- ⚠️ Walk-forward validation for ARIMA is expensive - may need timeout
- ⚠️ Memory limits not enforceable natively in Optuna

---

## 8. Implementation Checklist

### Phase 1: ARIMA Bayesian Search (Start Here)
- [ ] Add `import optuna` to `train.py`
- [ ] Implement `elif hyperparameter_search_strategy == "bayesian":` in `train_arima_model()`
- [ ] Extract `bayesian_config` from `data` dict
- [ ] Create `objective()` function with proper param suggestions
- [ ] Create Optuna study with `TPESampler(seed=SEED)`
- [ ] Run optimization and extract best params
- [ ] Train final model with best params
- [ ] Log Optuna study to experiment directory
- [ ] Test reproducibility with fixed seed

### Phase 2: XGBoost & LSTM
- [ ] Implement Bayesian Search in `train_xgboost_model()`
- [ ] Implement Bayesian Search in `train_lstm_model()` with **seed reset in objective**
- [ ] Test all 6 algorithms (ARIMA, SARIMA, ARIMAX, SARIMAX, XGBoost, LSTM)

### Phase 3: Frontend Integration
- [ ] Verify TSTrainCard sends `bayesian_config` correctly
- [ ] Update API request if needed
- [ ] Test full end-to-end flow
- [ ] Validate results display in UI

### Phase 4: Advanced Features
- [ ] Implement convergence detection with Optuna callbacks
- [ ] Add Optuna optimization history visualization
- [ ] Implement timeout enforcement
- [ ] Implement memory limit monitoring (if feasible)
- [ ] Add nested MLflow runs for trials

### Phase 5: Testing & Documentation
- [ ] Write unit tests for Bayesian Search
- [ ] Write reproducibility tests
- [ ] Write convergence comparison tests (Bayesian vs Random)
- [ ] Update user documentation
- [ ] Create example notebooks

---

## 9. References & Resources

### Optuna Documentation
- [Optuna Official Documentation](https://optuna.readthedocs.io/en/stable/)
- [Optuna FAQ - Reproducibility](https://optuna.readthedocs.io/en/stable/faq.html)
- [TPESampler API Reference](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)

### Tutorials & Examples
- [Neptune.ai: How to Optimize Hyperparameter Search Using Bayesian Optimization and Optuna](https://neptune.ai/blog/how-to-optimize-hyperparameter-search) (2025)
- [Random Realizations: The Ultimate Guide to XGBoost Parameter Tuning](https://randomrealizations.com/posts/xgboost-parameter-tuning-with-optuna/)
- [Medium: Hyperparameter Tuning Using Optuna](https://medium.com/@taeefnajib/hyperparameter-tuning-using-optuna-c46d7b29a3e) (2025)

### Reproducibility
- [Optuna GitHub Issue #3137: Non-determinism in TPE sampler](https://github.com/optuna/optuna/issues/3137)
- [Optuna GitHub Issue #3526: Reproducible sampling](https://github.com/optuna/optuna/issues/3526)

### Time Series with Optuna
- [XGBoosting: XGBoost Hyperparameter Optimization with Optuna](https://xgboosting.com/xgboost-hyperparameter-optimization-with-optuna/)
- [GitHub Gist: Optuna Hyperparameter Tuning with XGBoost](https://gist.github.com/rohithteja/885a0d231016b24bc0c3c248e53d1692)

---

## 10. Summary of Findings

### Current State
- **Frontend**: ✅ Bayesian Search UI complete with advanced options
- **Backend Validation**: ✅ Accepts "bayesian" strategy
- **Backend Implementation**: ❌ No Optuna code exists - needs full implementation
- **Dependencies**: ✅ Optuna 4.6.0 installed (modern, maintained)
- **Architecture**: ✅ Clean 4-layer structure ready for extension

### Implementation Strategy
1. **Start with ARIMA** - simplest algorithm, most stable
2. **Follow grid/random search pattern** - same structure, different optimizer
3. **Use Optuna TPESampler** - modern Bayesian optimization with reproducibility
4. **Maintain reproducibility** - fixed seeds, single-threaded, deterministic objectives
5. **Extend to all 6 algorithms** - ARIMA, SARIMA, ARIMAX, SARIMAX, XGBoost, LSTM

### Critical Success Factors
1. **Reproducibility**: `TPESampler(seed=42)` + deterministic objectives
2. **LSTM Seed Reset**: Reset TF seeds inside objective function
3. **Walk-forward Validation**: Use existing `walk_forward_validate_sarimax()` for time series models
4. **Error Handling**: Return `float('inf')` for failed trials
5. **MLflow Integration**: Log Optuna study metadata to experiment

### Expected Benefits
- **Better Hyperparameters**: Bayesian Search finds better params than random search with fewer trials
- **User Control**: Data scientists can tune `n_trials`, `n_initial_points`, acquisition function
- **Reproducibility**: Same seed produces identical optimization trajectory
- **Efficiency**: TPE algorithm intelligently explores promising regions

---

**End of Analysis Document**
