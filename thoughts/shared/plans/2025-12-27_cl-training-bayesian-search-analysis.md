# Implementation Plan: Bayesian Search for Classification Models

**Date**: 2025-12-27
**Research Basis**: [2025-12-27_classification-training-workflow-analysis.md](../../../thoughts/shared/research/2025-12-27_classification-training-workflow-analysis.md)
**Objective**: Add Bayesian Search (via Optuna 4.6.0) as hyperparameter tuning option for classification models
**Scope**: Minimal implementation with configurable parameter ranges

---

## Executive Summary

This plan implements Bayesian Search hyperparameter optimization using Optuna for all 3 classification algorithms (Logistic Regression, XGBoost, MLP). The implementation follows a **minimal feature scope** strategy with:

- ✅ Core Bayesian search (n_trials, n_initial_points, timeout)
- ✅ Configurable parameter ranges via frontend (bayesian_search_params)
- ✅ Simple train/val split validation (fast iteration)
- ✅ All three algorithms implemented
- ❌ Advanced features deferred (convergence detection, memory monitoring)

**Implementation Strategy**: Incremental phases with testing at each stage

---

## User Decisions Summary

Based on clarifying questions answered by user:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Algorithm Priority** | All three (Logistic, XGBoost, MLP) | Frontend already has UI for all three |
| **Feature Scope** | Minimal (core Bayesian only) | Focus on essential functionality, faster to implement |
| **Validation Strategy** | Simple train/val split | Fastest iteration, matches random search pattern |
| **Parameter Ranges** | Support frontend bayesian_search_params | Flexibility for users to customize ranges |

---

## Table of Contents

1. [Phase 1: Optuna Setup and Imports](#phase-1-optuna-setup-and-imports)
2. [Phase 2: Logistic Regression Bayesian Search](#phase-2-logistic-regression-bayesian-search)
3. [Phase 3: XGBoost Bayesian Search](#phase-3-xgboost-bayesian-search)
4. [Phase 4: MLP Bayesian Search](#phase-4-mlp-bayesian-search)
5. [Phase 5: Testing and Verification](#phase-5-testing-and-verification)

---

## Phase 1: Optuna Setup and Imports

### Phase Overview
Add Optuna imports and configuration to train.py. This is a prerequisite for all subsequent phases.

### Files to Modify
- `DREAM-ML-backend/GEML/api/train.py`

### Specific Changes

#### Change 1: Add Optuna Imports (After line 20)

**Location**: Top of train.py, after existing imports

**Current Code** (lines 1-20):
```python
# Existing imports...
import logging
import os
```

**Add After Line 20**:
```python
# Optuna for Bayesian hyperparameter optimization
import optuna
from optuna.samplers import TPESampler
from optuna import Trial
import time
```

#### Change 2: Configure Optuna Logging (After line 75)

**Location**: After `SEED = 42` and `set_global_seeds()` (around line 75)

**Add**:
```python
# Configure Optuna logging
optuna.logging.set_verbosity(optuna.logging.INFO)
```

### Automated Verification Steps
```bash
# Test Python imports
cd DREAM-ML-backend
python3 -c "import optuna; from optuna.samplers import TPESampler; from optuna import Trial; print('Optuna imports successful')"

# Check if Optuna is installed
pip show optuna
```

### Manual Verification Steps
1. Run the Python import test command
2. Verify Optuna version is 4.6.0 or higher
3. Check that no import errors occur

### Success Criteria
- [x] Optuna imports added without syntax errors
- [x] Optuna logging configured
- [x] No import errors when loading train.py module

### Phase 1 Status: ✅ COMPLETED (2025-12-29)

**Implementation Summary:**
- ✅ Added Optuna imports to [train.py:42-45](../../DREAM-ML-backend/GEML/api/train.py#L42-L45)
- ✅ Configured Optuna logging at [train.py:78-79](../../DREAM-ML-backend/GEML/api/train.py#L78-L79)
- ✅ All automated verification passed (Optuna 4.6.0 installed, no import errors)
- ✅ Django backend check passed
- ✅ Fixed TrainCard.jsx parameter naming (aligned with TSTrainCard.jsx):
  - Removed separate `nBayesianIterations` state
  - Added `n_trials` to `bayesianConfig` state object
  - Updated UI to use `bayesianConfig.n_trials`
  - Updated validation and payload construction

---

## Phase 2: Logistic Regression Bayesian Search

### Pattern Consistency Checklist

Before implementing Phase 2, ensure consistency with existing patterns:

#### 1. **Code Structure Patterns** (Reference: Time Series Bayesian Implementation)
- [ ] Follow the same branch structure: `if/elif/elif/else` for hyperparameter strategies
- [ ] Place Bayesian branch after `random` search, before `manual` training
- [ ] Use identical validation logic for `n_trials` and `n_initial_points`
- [ ] Initialize all Bayesian-related variables at function start (set to `None`)

#### 2. **Optuna Configuration Patterns** (Reference: [apiTimeSeries/train.py](../../DREAM-ML-backend/GEML/apiTimeSeries/train.py))
- [ ] Use `TPESampler` with these exact parameters:
  - `seed=SEED` (for reproducibility)
  - `n_startup_trials=n_initial_points`
  - `multivariate=False` (independent TPE)
  - `consider_magic_clip=True`
  - `consider_endpoints=False`
- [ ] Create study with:
  - `direction='minimize'` (negative accuracy)
  - `sampler=sampler`
  - `study_name` with timestamp format: `f"logistic_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"`
- [ ] Use `n_jobs=1` for reproducibility

#### 3. **Objective Function Patterns**
- [ ] Define `objective(trial: Trial) -> float` function
- [ ] Return **negative** accuracy (Optuna minimizes)
- [ ] Use try-except to catch failures and return `float('inf')` for failed trials
- [ ] Log trial results with format: `f"Trial {trial.number}: accuracy={val_score:.4f}, ..."`
- [ ] Handle algorithm-specific constraints (e.g., solver-penalty compatibility for Logistic Regression)

#### 4. **Parameter Range Patterns**
- [ ] Define `default_ranges` dictionary with structure:
  ```python
  {
    "param_name": {"type": "float|int|categorical", "low": X, "high": Y, "log": True/False},
    ...
  }
  ```
- [ ] Merge with user ranges: `param_ranges = {**default_ranges, **bayesian_search_params}`
- [ ] Use `trial.suggest_float()`, `trial.suggest_int()`, `trial.suggest_categorical()` consistently

#### 5. **Logging and Metadata Patterns**
- [ ] Log configuration before optimization with banner format (`"="*60`)
- [ ] Log completion results with same banner format
- [ ] Track `optimization_start_time` and calculate `optimization_time_seconds`
- [ ] Count `completed_trials` using: `len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])`
- [ ] MLflow log params with `bayesian_` prefix: `bayesian_n_trials`, `bayesian_n_initial_points`, etc.
- [ ] MLflow log metrics: `bayesian_best_score`, `bayesian_optimization_time_seconds`, `bayesian_n_completed_trials`

#### 6. **Error Handling Patterns**
- [ ] Validate `n_trials >= 1` before optimization
- [ ] Validate `n_initial_points < n_trials`
- [ ] Check `study.best_trial is None or study.best_value == float('inf')` after optimization
- [ ] Raise `RuntimeError` with Spanish message if all trials failed

#### 7. **Final Model Training Patterns**
- [ ] Extract `best_params_dict = study.best_params`
- [ ] Convert back to positive: `best_score = -study.best_value`
- [ ] Train final model with `best_params_dict` merged with any base parameters
- [ ] Store `best_params` for pipeline_config (include SEED and other fixed params)

#### 8. **Pipeline Config Patterns**
- [ ] Add `bayesian_search` block parallel to `grid_search` and `random_search`
- [ ] Use conditional values: `value if hyperparameter_search_strategy == "bayesian" else None`
- [ ] Include all metadata: `use_bayesian_search`, `n_trials`, `n_initial_points`, `timeout_seconds`, `best_params`, `best_score`, `optimization_time_seconds`, `n_completed_trials`

#### 9. **Variable Naming Consistency**
- [ ] Backend uses: `n_trials`, `n_initial_points`, `timeout_seconds`, `bayesian_search_params`
- [ ] Frontend sends: `bayesian_config` object with `n_trials` inside
- [ ] Never use: `n_bayesian_iterations` (legacy/deprecated)

#### 10. **Documentation Patterns**
- [ ] Add Spanish docstrings for objective function
- [ ] Use Spanish log messages matching existing style
- [ ] Comment parameter ranges with units/meanings where applicable

**Reference Implementation**: See [apiTimeSeries/train.py](../../DREAM-ML-backend/GEML/apiTimeSeries/train.py) lines for ARIMA, LSTM, XGBoost Bayesian implementations for complete pattern examples.

---

### Phase Overview
Implement Bayesian Search for Logistic Regression with configurable parameter ranges. This is the simplest algorithm and serves as the foundation pattern for the others.

### Files to Modify
- `DREAM-ML-backend/GEML/api/train.py`

### Specific Changes

#### Change 1: Implement Bayesian Search Branch (Insert after line 559, before line 561)

**Location**: In `train_logistic_regression_model()`, after random search block ends (line 559), before manual training `else` clause (line 561)

**Current Structure** (lines 518-576):
```python
    elif hyperparameter_search_strategy == "random":
        # ... random search implementation ...
        # ends at line 559

    else:  # Line 561 - manual training
        # ... manual training ...
```

**Insert Between Lines 559-561**:
```python
    elif hyperparameter_search_strategy == "bayesian":
        # Extract Bayesian configuration
        bayesian_config = data.get("bayesian_config", {})
        n_trials = bayesian_config.get("n_trials", 50)
        n_initial_points = bayesian_config.get("n_initial_points", 10)
        timeout_seconds = bayesian_config.get("timeout_seconds", None)

        # Extract custom parameter ranges from frontend
        bayesian_search_params = data.get("bayesian_search_params", {})

        # Validate bayesian config
        if n_trials < 1:
            raise ValueError(f"n_trials must be at least 1, got {n_trials}")
        if n_initial_points >= n_trials:
            raise ValueError(
                f"n_initial_points ({n_initial_points}) must be less than n_trials ({n_trials})"
            )

        logger.info("="*60)
        logger.info("Configuración Búsqueda Bayesiana (Optuna):")
        logger.info(f"  n_trials: {n_trials}")
        logger.info(f"  n_initial_points: {n_initial_points}")
        logger.info(f"  timeout_seconds: {timeout_seconds}")
        logger.info("="*60)

        # Define default parameter ranges
        default_ranges = {
            "C": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
            "max_iter": {"type": "int", "low": 100, "high": 1000},
            "solver": {"type": "categorical", "choices": ["lbfgs", "liblinear", "saga"]},
            "penalty": {"type": "categorical", "choices": ["l2", "none"]}
        }

        # Merge with user-provided ranges (user ranges override defaults)
        param_ranges = {**default_ranges, **bayesian_search_params}

        # Define Optuna objective function
        def objective(trial: Trial) -> float:
            """
            Optuna objective function for Logistic Regression hyperparameter optimization.

            Returns:
                float: Negative accuracy (for minimization)
            """
            # Suggest parameters based on configured ranges
            C_config = param_ranges.get("C", default_ranges["C"])
            if C_config["type"] == "float":
                C = trial.suggest_float('C', C_config["low"], C_config["high"],
                                       log=C_config.get("log", True))

            max_iter_config = param_ranges.get("max_iter", default_ranges["max_iter"])
            max_iter = trial.suggest_int('max_iter', max_iter_config["low"],
                                        max_iter_config["high"])

            solver_config = param_ranges.get("solver", default_ranges["solver"])
            solver = trial.suggest_categorical('solver', solver_config["choices"])

            penalty_config = param_ranges.get("penalty", default_ranges["penalty"])
            penalty = trial.suggest_categorical('penalty', penalty_config["choices"])

            # Handle solver-penalty compatibility
            if solver == "liblinear" and penalty == "none":
                penalty = "l2"  # liblinear doesn't support penalty='none'
            if solver == "lbfgs" and penalty not in ["l2", "none"]:
                penalty = "l2"  # lbfgs only supports l2 and none

            try:
                # Train model with suggested parameters
                model_trial = LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    solver=solver,
                    penalty=penalty,
                    random_state=SEED,
                    n_jobs=N_JOBS
                )

                # Fit on training set
                model_trial.fit(X_train, y_train)

                # Evaluate on validation set
                val_pred = model_trial.predict(X_val)
                val_score = accuracy_score(y_val, val_pred)

                logger.info(
                    f"Trial {trial.number}: accuracy={val_score:.4f}, "
                    f"C={C:.4f}, solver={solver}, penalty={penalty}"
                )

                # Return negative accuracy (Optuna minimizes)
                return -val_score

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {str(e)}")
                return float('inf')  # Penalty for failed trials

        # Create Optuna study with TPE sampler
        sampler = TPESampler(
            seed=SEED,  # Fixed seed for reproducibility
            n_startup_trials=n_initial_points,  # Random exploration before Bayesian
            multivariate=False,  # Use independent TPE (simpler, more stable)
            consider_magic_clip=True,
            consider_endpoints=False
        )

        study = optuna.create_study(
            direction='minimize',  # Minimize negative accuracy
            sampler=sampler,
            study_name=f"logistic_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Track optimization time
        optimization_start_time = time.time()

        # Run optimization
        logger.info(f"Iniciando Búsqueda Bayesiana con Optuna TPESampler...")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=False,
            n_jobs=1  # Single-threaded for reproducibility
        )

        optimization_time_seconds = time.time() - optimization_start_time

        # Validate results
        if study.best_trial is None or study.best_value == float('inf'):
            raise RuntimeError(
                "Búsqueda Bayesiana falló: Todos los trials retornaron errores. "
                "Verifique los rangos de parámetros y calidad de datos."
            )

        # Extract best parameters
        best_params_dict = study.best_params
        best_score = -study.best_value  # Convert back to positive accuracy

        # Log optimization results
        logger.info("="*60)
        logger.info(f"Búsqueda Bayesiana Completada")
        logger.info(f"  Mejor accuracy: {best_score:.4f}")
        logger.info(f"  Mejores parámetros: {best_params_dict}")
        completed_trials = len([t for t in study.trials
                               if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"  Trials completados: {completed_trials}/{len(study.trials)}")
        logger.info(f"  Tiempo de optimización: {optimization_time_seconds:.2f} segundos")
        logger.info("="*60)

        # Train final model with best parameters
        model = LogisticRegression(
            C=best_params_dict['C'],
            max_iter=best_params_dict['max_iter'],
            solver=best_params_dict['solver'],
            penalty=best_params_dict['penalty'],
            random_state=SEED,
            n_jobs=N_JOBS
        )
        model.fit(X_train, y_train)

        # Store best params for logging
        best_params = best_params_dict.copy()
        best_params["random_state"] = SEED

        # Log Bayesian search metadata to MLflow
        mlflow.log_params({
            "bayesian_n_trials": n_trials,
            "bayesian_n_initial_points": n_initial_points,
            "bayesian_optimization_metric": "accuracy",
            **{f"best_{k}": v for k, v in best_params_dict.items()}
        })

        mlflow.log_metrics({
            "bayesian_best_score": best_score,
            "bayesian_optimization_time_seconds": optimization_time_seconds,
            "bayesian_n_completed_trials": completed_trials
        })
```

#### Change 2: Update pipeline_config (Around line 615-644)

**Location**: In the pipeline_config dictionary construction

**Current Code** (lines 615-644):
```python
pipeline_config = {
    "step": "train_logistic_regression",
    # ... existing fields ...
    "grid_search": {...},
    "random_search": {...},
    # Missing: bayesian_search
}
```

**Add Bayesian Search Block** (after random_search block, before val_metrics):
```python
    "bayesian_search": {
        "use_bayesian_search": hyperparameter_search_strategy == "bayesian",
        "n_trials": n_trials if hyperparameter_search_strategy == "bayesian" else None,
        "n_initial_points": n_initial_points if hyperparameter_search_strategy == "bayesian" else None,
        "timeout_seconds": timeout_seconds if hyperparameter_search_strategy == "bayesian" else None,
        "bayesian_search_params": bayesian_search_params if hyperparameter_search_strategy == "bayesian" else None,
        "best_params": best_params if hyperparameter_search_strategy == "bayesian" else None,
        "best_score": best_score if hyperparameter_search_strategy == "bayesian" else None,
        "optimization_time_seconds": optimization_time_seconds if hyperparameter_search_strategy == "bayesian" else None,
        "n_completed_trials": completed_trials if hyperparameter_search_strategy == "bayesian" else None
    },
```

**Note**: You'll need to initialize variables at the top of the function:
```python
# After line 459, add initialization for Bayesian variables
n_trials = None
n_initial_points = None
timeout_seconds = None
bayesian_search_params = {}
optimization_time_seconds = None
completed_trials = None
best_score = None
```

### Automated Verification Steps
```bash
# Test that train.py has no syntax errors
cd DREAM-ML-backend
python3 -m py_compile GEML/api/train.py

# Run pytest if tests exist
pytest GEML/tests/ -k "logistic" -v
```

### Manual Verification Steps
1. Start Django backend: `python manage.py runserver`
2. Navigate to frontend classification training page
3. Select Logistic Regression algorithm
4. Choose "Bayesian Search" as optimization method
5. Set n_trials=10, n_initial_points=5
6. Train model and verify:
   - ✅ Training completes without errors
   - ✅ MLflow shows bayesian_n_trials, bayesian_best_score metrics
   - ✅ pipeline_config.json has bayesian_search block
   - ✅ Best parameters are logged
   - ✅ Final model accuracy is reasonable

### Success Criteria
- [x] Bayesian search branch implemented for Logistic Regression
- [x] Configurable parameter ranges supported (bayesian_search_params)
- [x] MLflow logging includes Bayesian metadata
- [x] pipeline_config.json updated with bayesian_search section
- [x] Training completes successfully with test dataset
- [x] Best parameters are correctly extracted and logged
- [x] Reproducibility verified (same seed produces same results)

### Phase 2 Status: ✅ COMPLETED (2025-12-29)

**Implementation Summary:**
- ✅ Added helper function `convert_frontend_bayesian_params()` at [train.py:434-482](../../DREAM-ML-backend/GEML/api/train.py#L434-L482) to convert frontend parameter format to backend Optuna format
- ✅ Initialized Bayesian variables at [train.py:530-537](../../DREAM-ML-backend/GEML/api/train.py#L530-L537)
- ✅ Implemented complete Bayesian search branch with Optuna TPESampler at [train.py:631-807](../../DREAM-ML-backend/GEML/api/train.py#L631-L807)
  - Extracts and validates bayesian_config (n_trials, n_initial_points, timeout_seconds)
  - Converts frontend bayesian_search_params to backend format
  - Defines default parameter ranges for C, max_iter, solver, penalty
  - Implements Optuna objective function with solver-penalty compatibility handling
  - Creates TPESampler with seed=42 for reproducibility
  - Validates results and trains final model with best parameters
  - Logs comprehensive metadata to MLflow
- ✅ Updated pipeline_config with bayesian_search block at [train.py:883-893](../../DREAM-ML-backend/GEML/api/train.py#L883-L893)
- ✅ Created test dataset at [datasets/tests/test_binary_classification.csv](../../../datasets/tests/test_binary_classification.csv) (130 samples, Iris-based)
- ✅ All automated verification passed (syntax check, Django system check, Optuna 4.6.0 confirmed)
- ✅ All manual verification tests passed:
  - Basic Bayesian search with n_trials=10 completed successfully
  - Custom parameter ranges properly respected
  - Reproducibility verified (same seed → identical results)
  - Error handling validated (n_initial_points >= n_trials correctly rejected)
- ✅ MLflow logging verified: all bayesian metrics and best parameters logged correctly
- ✅ pipeline_config.json correctly includes bayesian_search block with all metadata

**Key Implementation Details:**
- Used TPESampler with `multivariate=False` for independent parameter optimization
- Returns negative accuracy for Optuna minimization
- Handles Logistic Regression solver-penalty compatibility constraints
- Supports configurable parameter ranges via frontend UI
- Fixed SEED=42 ensures deterministic optimization results

---

## Phase 3: XGBoost Bayesian Search

### Pattern Consistency Checklist

Before implementing Phase 3, ensure consistency with Phase 2 patterns:

#### 1. **Code Structure Patterns** (Reference: Phase 2 Implementation)
- [ ] Follow the same branch structure: `if/elif/elif/else` for hyperparameter strategies
- [ ] Place Bayesian branch after `random` search, before `manual` training
- [ ] Use identical validation logic for `n_trials` and `n_initial_points`
- [ ] Initialize all Bayesian-related variables at function start (set to `None`)

#### 2. **Optuna Configuration Patterns** (Reference: [train.py:728-741](../../DREAM-ML-backend/GEML/api/train.py#L728-L741))
- [ ] Use `TPESampler` with these exact parameters:
  - `seed=SEED` (for reproducibility)
  - `n_startup_trials=n_initial_points`
  - `multivariate=False` (independent TPE)
  - `consider_magic_clip=True`
  - `consider_endpoints=False`
- [ ] Create study with:
  - `direction='minimize'` (negative accuracy)
  - `sampler=sampler`
  - `study_name` with timestamp format: `f"xgboost_bayesian_{time.strftime('%Y%m%d_%H%M%S')}"`
- [ ] Use `n_jobs=1` for reproducibility

#### 3. **Objective Function Patterns** (Reference: [train.py:669-726](../../DREAM-ML-backend/GEML/api/train.py#L669-L726))
- [ ] Define `objective(trial: Trial) -> float` function
- [ ] Return **negative** accuracy (Optuna minimizes)
- [ ] Use try-except to catch failures and return `float('inf')` for failed trials
- [ ] Log trial results with format: `f"Trial {trial.number}: accuracy={val_score:.4f}, ..."`
- [ ] Handle algorithm-specific constraints (e.g., XGBoost early stopping with eval_set)

#### 4. **Parameter Range Patterns** (Reference: [train.py:657-666](../../DREAM-ML-backend/GEML/api/train.py#L657-L666))
- [ ] Define `default_ranges` dictionary with structure:
  ```python
  {
    "param_name": {"type": "float|int|categorical", "low": X, "high": Y, "log": True/False},
    ...
  }
  ```
- [ ] Merge with user ranges: `param_ranges = {**default_ranges, **bayesian_search_params}`
- [ ] Use `trial.suggest_float()`, `trial.suggest_int()`, `trial.suggest_categorical()` consistently
- [ ] Use `convert_frontend_bayesian_params()` for frontend parameter format conversion

#### 5. **Logging and Metadata Patterns** (Reference: [train.py:650-655](../../DREAM-ML-backend/GEML/api/train.py#L650-L655), [train.py:770-807](../../DREAM-ML-backend/GEML/api/train.py#L770-L807))
- [ ] Log configuration before optimization with banner format (`"="*60`)
- [ ] Log completion results with same banner format
- [ ] Track `optimization_start_time` and calculate `optimization_time_seconds`
- [ ] Count `completed_trials` using: `len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])`
- [ ] MLflow log params with `bayesian_` prefix: `bayesian_n_trials`, `bayesian_n_initial_points`, etc.
- [ ] MLflow log metrics: `bayesian_best_score`, `bayesian_optimization_time_seconds`, `bayesian_n_completed_trials`

#### 6. **Error Handling Patterns** (Reference: [train.py:643-648](../../DREAM-ML-backend/GEML/api/train.py#L643-L648), [train.py:759-763](../../DREAM-ML-backend/GEML/api/train.py#L759-L763))
- [ ] Validate `n_trials >= 1` before optimization
- [ ] Validate `n_initial_points < n_trials`
- [ ] Check `study.best_trial is None or study.best_value == float('inf')` after optimization
- [ ] Raise `RuntimeError` with Spanish message if all trials failed

#### 7. **Final Model Training Patterns** (Reference: [train.py:780-793](../../DREAM-ML-backend/GEML/api/train.py#L780-L793))
- [ ] Extract `best_params_dict = study.best_params`
- [ ] Convert back to positive: `best_score = -study.best_value`
- [ ] Train final model with `best_params_dict` merged with any base parameters (e.g., `base_params` for XGBoost)
- [ ] Store `best_params` for pipeline_config (include SEED and other fixed params)
- [ ] For XGBoost: include `best_iteration` if available from early stopping

#### 8. **Pipeline Config Patterns** (Reference: [train.py:883-893](../../DREAM-ML-backend/GEML/api/train.py#L883-L893))
- [ ] Add `bayesian_search` block parallel to `grid_search` and `random_search`
- [ ] Use conditional values: `value if hyperparameter_search_strategy == "bayesian" else None`
- [ ] Include all metadata: `use_bayesian_search`, `n_trials`, `n_initial_points`, `timeout_seconds`, `best_params`, `best_score`, `optimization_time_seconds`, `n_completed_trials`

#### 9. **Variable Naming Consistency** (Reference: [train.py:530-537](../../DREAM-ML-backend/GEML/api/train.py#L530-L537))
- [ ] Backend uses: `n_trials`, `n_initial_points`, `timeout_seconds`, `bayesian_search_params`
- [ ] Frontend sends: `bayesian_config` object with `n_trials` inside
- [ ] Never use: `n_bayesian_iterations` (legacy/deprecated)

#### 10. **Documentation Patterns** (Reference: [train.py:669-675](../../DREAM-ML-backend/GEML/api/train.py#L669-L675))
- [ ] Add Spanish docstrings for objective function
- [ ] Use Spanish log messages matching existing style
- [ ] Comment parameter ranges with units/meanings where applicable

#### 11. **XGBoost-Specific Patterns**
- [ ] Use `base_params` dictionary with fixed parameters (objective, eval_metric, random_state, etc.)
- [ ] Merge trial params with base_params: `model_params = {**base_params, **trial_params}`
- [ ] Use `eval_set=[(X_val, y_val)]` for early stopping
- [ ] Use `early_stopping_rounds=10`
- [ ] Use `callbacks=[EnableDeterministic()]` for reproducibility
- [ ] Store `best_iteration` from model if available: `if hasattr(model, 'best_iteration') and model.best_iteration is not None`

**Reference Implementation**: See [train.py:631-807](../../DREAM-ML-backend/GEML/api/train.py#L631-L807) for Logistic Regression Bayesian implementation as complete pattern example.

---

### Phase Overview
Implement Bayesian Search for XGBoost classifier. Similar structure to Logistic Regression but with XGBoost-specific parameters and early stopping support.

### Files to Modify
- `DREAM-ML-backend/GEML/api/train.py`

### Specific Changes

#### Change 1: Implement Bayesian Search Branch (Insert after line 1069, before line 1072)

**Location**: In `train_xgboost_model()`, after random search block ends (line 1069), before manual training `else` clause (line 1072)

**Current Structure** (lines 1018-1090):
```python
    elif hyperparameter_search_strategy == "random":
        # ... random search implementation ...
        # ends at line 1069

    else:  # Line 1072 - manual training
        # ... manual training ...
```

**Insert Between Lines 1069-1072**:
```python
    elif hyperparameter_search_strategy == "bayesian":
        # Extract Bayesian configuration
        bayesian_config = data.get("bayesian_config", {})
        n_trials = bayesian_config.get("n_trials", 50)
        n_initial_points = bayesian_config.get("n_initial_points", 10)
        timeout_seconds = bayesian_config.get("timeout_seconds", None)

        # Extract custom parameter ranges from frontend
        bayesian_search_params = data.get("bayesian_search_params", {})

        # Validate bayesian config
        if n_trials < 1:
            raise ValueError(f"n_trials must be at least 1, got {n_trials}")
        if n_initial_points >= n_trials:
            raise ValueError(
                f"n_initial_points ({n_initial_points}) must be less than n_trials ({n_trials})"
            )

        logger.info("="*60)
        logger.info("Configuración Búsqueda Bayesiana XGBoost (Optuna):")
        logger.info(f"  n_trials: {n_trials}")
        logger.info(f"  n_initial_points: {n_initial_points}")
        logger.info(f"  timeout_seconds: {timeout_seconds}")
        logger.info("="*60)

        # Define default parameter ranges for XGBoost
        default_ranges = {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            "gamma": {"type": "float", "low": 0.0, "high": 5.0},
            "min_child_weight": {"type": "int", "low": 1, "high": 10},
            "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
            "reg_lambda": {"type": "float", "low": 0.0, "high": 1.0}
        }

        # Merge with user-provided ranges
        param_ranges = {**default_ranges, **bayesian_search_params}

        # Define Optuna objective function
        def objective(trial: Trial) -> float:
            """
            Optuna objective function for XGBoost hyperparameter optimization.

            Returns:
                float: Negative accuracy (for minimization)
            """
            # Suggest parameters based on configured ranges
            n_estimators = trial.suggest_int('n_estimators',
                param_ranges["n_estimators"]["low"],
                param_ranges["n_estimators"]["high"])

            max_depth = trial.suggest_int('max_depth',
                param_ranges["max_depth"]["low"],
                param_ranges["max_depth"]["high"])

            learning_rate = trial.suggest_float('learning_rate',
                param_ranges["learning_rate"]["low"],
                param_ranges["learning_rate"]["high"],
                log=param_ranges["learning_rate"].get("log", True))

            subsample = trial.suggest_float('subsample',
                param_ranges["subsample"]["low"],
                param_ranges["subsample"]["high"])

            colsample_bytree = trial.suggest_float('colsample_bytree',
                param_ranges["colsample_bytree"]["low"],
                param_ranges["colsample_bytree"]["high"])

            gamma = trial.suggest_float('gamma',
                param_ranges["gamma"]["low"],
                param_ranges["gamma"]["high"])

            min_child_weight = trial.suggest_int('min_child_weight',
                param_ranges["min_child_weight"]["low"],
                param_ranges["min_child_weight"]["high"])

            reg_alpha = trial.suggest_float('reg_alpha',
                param_ranges["reg_alpha"]["low"],
                param_ranges["reg_alpha"]["high"])

            reg_lambda = trial.suggest_float('reg_lambda',
                param_ranges["reg_lambda"]["low"],
                param_ranges["reg_lambda"]["high"])

            try:
                # Build model parameters (merge with base_params)
                model_params = {
                    **base_params,  # Includes objective, eval_metric, random_state, etc.
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "gamma": gamma,
                    "min_child_weight": min_child_weight,
                    "reg_alpha": reg_alpha,
                    "reg_lambda": reg_lambda
                }

                # Train model with early stopping
                model_trial = XGBClassifier(**model_params)
                model_trial.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=0,
                    callbacks=[EnableDeterministic()]
                )

                # Evaluate on validation set
                val_pred = model_trial.predict(X_val)
                val_score = accuracy_score(y_val, val_pred)

                logger.info(
                    f"Trial {trial.number}: accuracy={val_score:.4f}, "
                    f"n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate:.4f}"
                )

                # Return negative accuracy (Optuna minimizes)
                return -val_score

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {str(e)}")
                return float('inf')

        # Create Optuna study with TPE sampler
        sampler = TPESampler(
            seed=SEED,
            n_startup_trials=n_initial_points,
            multivariate=False,
            consider_magic_clip=True,
            consider_endpoints=False
        )

        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            study_name=f"xgboost_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Track optimization time
        optimization_start_time = time.time()

        # Run optimization
        logger.info(f"Iniciando Búsqueda Bayesiana XGBoost con Optuna TPESampler...")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=False,
            n_jobs=1
        )

        optimization_time_seconds = time.time() - optimization_start_time

        # Validate results
        if study.best_trial is None or study.best_value == float('inf'):
            raise RuntimeError(
                "Búsqueda Bayesiana falló: Todos los trials retornaron errores. "
                "Verifique los rangos de parámetros y calidad de datos."
            )

        # Extract best parameters
        best_params_dict = study.best_params
        best_score = -study.best_value

        # Log optimization results
        logger.info("="*60)
        logger.info(f"Búsqueda Bayesiana XGBoost Completada")
        logger.info(f"  Mejor accuracy: {best_score:.4f}")
        logger.info(f"  Mejores parámetros: {best_params_dict}")
        completed_trials = len([t for t in study.trials
                               if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"  Trials completados: {completed_trials}/{len(study.trials)}")
        logger.info(f"  Tiempo de optimización: {optimization_time_seconds:.2f} segundos")
        logger.info("="*60)

        # Train final model with best parameters
        model_params = {
            **base_params,
            **best_params_dict
        }
        model = XGBClassifier(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=0,
            callbacks=[EnableDeterministic()]
        )

        # Store best params for logging (include best_iteration if available)
        best_params = best_params_dict.copy()
        if hasattr(model, 'best_iteration') and model.best_iteration is not None:
            best_params["best_iteration"] = model.best_iteration

        # Log Bayesian search metadata to MLflow
        mlflow.log_params({
            "bayesian_n_trials": n_trials,
            "bayesian_n_initial_points": n_initial_points,
            "bayesian_optimization_metric": "accuracy",
            **{f"best_{k}": v for k, v in best_params_dict.items()}
        })

        mlflow.log_metrics({
            "bayesian_best_score": best_score,
            "bayesian_optimization_time_seconds": optimization_time_seconds,
            "bayesian_n_completed_trials": completed_trials
        })
```

#### Change 2: Update pipeline_config for XGBoost (Around line 1125-1173)

**Add bayesian_search block** to pipeline_config (same structure as Logistic Regression):
```python
    "bayesian_search": {
        "use_bayesian_search": hyperparameter_search_strategy == "bayesian",
        "n_trials": n_trials if hyperparameter_search_strategy == "bayesian" else None,
        "n_initial_points": n_initial_points if hyperparameter_search_strategy == "bayesian" else None,
        "timeout_seconds": timeout_seconds if hyperparameter_search_strategy == "bayesian" else None,
        "bayesian_search_params": bayesian_search_params if hyperparameter_search_strategy == "bayesian" else None,
        "best_params": best_params if hyperparameter_search_strategy == "bayesian" else None,
        "best_score": best_score if hyperparameter_search_strategy == "bayesian" else None,
        "optimization_time_seconds": optimization_time_seconds if hyperparameter_search_strategy == "bayesian" else None,
        "n_completed_trials": completed_trials if hyperparameter_search_strategy == "bayesian" else None
    },
```

**Initialize variables** (after line 920, similar to Logistic Regression):
```python
# Bayesian search variables
n_trials = None
n_initial_points = None
timeout_seconds = None
bayesian_search_params = {}
optimization_time_seconds = None
completed_trials = None
best_score = None
```

### Automated Verification Steps
```bash
# Syntax check
python3 -m py_compile GEML/api/train.py

# Run XGBoost-specific tests if available
pytest GEML/tests/ -k "xgboost" -v
```

### Manual Verification Steps
1. Select XGBoost algorithm in frontend
2. Choose "Bayesian Search" with n_trials=10
3. Train and verify:
   - ✅ Training completes with early stopping logs
   - ✅ best_iteration is logged
   - ✅ MLflow shows bayesian metrics
   - ✅ pipeline_config.json has bayesian_search block

### Success Criteria
- [x] Bayesian search implemented for XGBoost
- [x] Early stopping integrated (eval_set with 10 rounds)
- [x] best_iteration tracked and logged
- [x] All 9 XGBoost parameters supported
- [x] Training completes successfully
- [x] Reproducibility verified

### Phase 3 Status: ✅ COMPLETED (2025-12-31)

**Implementation Summary:**
- ✅ Initialized Bayesian variables at [train.py:1215-1222](../../DREAM-ML-backend/GEML/api/train.py#L1215-L1222)
- ✅ Implemented complete Bayesian search branch at [train.py:1350-1562](../../DREAM-ML-backend/GEML/api/train.py#L1350-L1562):
  - Extracts and validates bayesian_config (n_trials, n_initial_points, timeout_seconds)
  - Converts frontend bayesian_search_params using `convert_frontend_bayesian_params()`
  - Defines default parameter ranges for 9 XGBoost parameters
  - Implements Optuna objective function with:
    - base_params merge for XGBoost configuration
    - eval_set=[(X_val, y_val)] for early stopping
    - early_stopping_rounds=10 (matching random search)
    - callbacks=[EnableDeterministic()] for reproducibility
  - Creates TPESampler with seed=SEED, n_startup_trials=n_initial_points
  - Validates results and trains final model with best parameters
  - Tracks best_iteration from XGBoost early stopping
  - Logs all best_params_dict items to MLflow (no exclusions)
- ✅ Updated pipeline_config with bayesian_search block at [train.py:1650-1660](../../DREAM-ML-backend/GEML/api/train.py#L1650-L1660)
- ✅ All automated verification passed (syntax check, Django check, import verification)
- ✅ All manual verification tests passed:
  - Basic Bayesian search with n_trials=10 completed successfully
  - Custom parameter ranges properly respected
  - Reproducibility verified (same seed → identical results)
  - Error handling validated (invalid configs correctly rejected)
- ✅ MLflow logging verified: all bayesian metrics and best parameters logged correctly
- ✅ pipeline_config.json correctly includes bayesian_search block with all metadata
- ✅ best_iteration tracking works correctly

**Key Implementation Details:**
- Used TPESampler with `multivariate=False` for independent parameter optimization
- Returns negative accuracy for Optuna minimization
- Handles XGBoost early stopping with 10 rounds
- Supports configurable parameter ranges via frontend UI
- Fixed SEED=42 ensures deterministic optimization results
- All 9 XGBoost parameters optimized: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, min_child_weight, reg_alpha, reg_lambda

---

## Phase 4: MLP Bayesian Search

### Pattern Consistency Checklist

Before implementing Phase 4, ensure consistency with Phase 2 and Phase 3 patterns:

#### 1. **Code Structure Patterns** (Reference: Phase 2 and Phase 3 Implementations)
- [ ] Follow the same branch structure: `if/elif/elif/else` for hyperparameter strategies
- [ ] Place Bayesian branch after `random` search, before `manual` training
- [ ] Use identical validation logic for `n_trials` and `n_initial_points`
- [ ] Initialize all Bayesian-related variables at function start (set to `None`)

#### 2. **Optuna Configuration Patterns** (Reference: [train.py:1479-1492](../../DREAM-ML-backend/GEML/api/train.py#L1479-L1492))
- [ ] Use `TPESampler` with these exact parameters:
  - `seed=SEED` (for reproducibility)
  - `n_startup_trials=n_initial_points`
  - `multivariate=False` (independent TPE)
  - `consider_magic_clip=True`
  - `consider_endpoints=False`
- [ ] Create study with:
  - `direction='minimize'` (negative accuracy)
  - `sampler=sampler`
  - `study_name` with timestamp format: `f"mlp_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"`
- [ ] Use `n_jobs=1` for reproducibility

#### 3. **Objective Function Patterns** (Reference: [train.py:1393-1477](../../DREAM-ML-backend/GEML/api/train.py#L1393-L1477))
- [ ] Define `objective(trial: Trial) -> float` function
- [ ] Return **negative** accuracy (Optuna minimizes)
- [ ] Use try-except to catch failures and return `float('inf')` for failed trials
- [ ] Log trial results with format: `f"Trial {trial.number}: accuracy={val_score:.4f}, hidden_layers={hidden_layer_sizes}, activation={activation}"`
- [ ] Handle MLP-specific constraints (e.g., memory-aware verbosity setting)

#### 4. **Parameter Range Patterns** (Reference: [train.py:1376-1390](../../DREAM-ML-backend/GEML/api/train.py#L1376-L1390))
- [ ] Define `default_ranges` dictionary with structure:
  ```python
  {
    "param_name": {"type": "float|int|categorical", "low": X, "high": Y, "log": True/False},
    "hidden_layer_sizes": {"type": "categorical", "choices": [(4,), (10,), (10,5), ...]},
    ...
  }
  ```
- [ ] Merge with user ranges: `param_ranges = {**default_ranges, **bayesian_search_params}`
- [ ] Use `trial.suggest_float()`, `trial.suggest_int()`, `trial.suggest_categorical()` consistently
- [ ] Use `convert_frontend_bayesian_params()` for frontend parameter format conversion

#### 5. **Logging and Metadata Patterns** (Reference: [train.py:1369-1374](../../DREAM-ML-backend/GEML/api/train.py#L1369-L1374), [train.py:1520-1562](../../DREAM-ML-backend/GEML/api/train.py#L1520-L1562))
- [ ] Log configuration before optimization with banner format (`"="*60`)
- [ ] Log completion results with same banner format
- [ ] Track `optimization_start_time` and calculate `optimization_time_seconds`
- [ ] Count `completed_trials` using: `len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])`
- [ ] MLflow log params with `bayesian_` prefix: `bayesian_n_trials`, `bayesian_n_initial_points`, etc.
- [ ] MLflow log metrics: `bayesian_best_score`, `bayesian_optimization_time_seconds`, `bayesian_n_completed_trials`

#### 6. **Error Handling Patterns** (Reference: [train.py:1362-1367](../../DREAM-ML-backend/GEML/api/train.py#L1362-L1367), [train.py:1510-1514](../../DREAM-ML-backend/GEML/api/train.py#L1510-L1514))
- [ ] Validate `n_trials >= 1` before optimization
- [ ] Validate `n_initial_points < n_trials`
- [ ] Check `study.best_trial is None or study.best_value == float('inf')` after optimization
- [ ] Raise `RuntimeError` with Spanish message if all trials failed

#### 7. **Final Model Training Patterns** (Reference: [train.py:1531-1548](../../DREAM-ML-backend/GEML/api/train.py#L1531-L1548))
- [ ] Extract `best_params_dict = study.best_params`
- [ ] Convert back to positive: `best_score = -study.best_value`
- [ ] Train final model with `best_params_dict` merged with any base parameters
- [ ] Store `best_params` for pipeline_config (include SEED and other fixed params like `verbose_setting`)

#### 8. **Pipeline Config Patterns** (Reference: [train.py:1650-1660](../../DREAM-ML-backend/GEML/api/train.py#L1650-L1660))
- [ ] Add `bayesian_search` block parallel to `grid_search` and `random_search`
- [ ] Use conditional values: `value if hyperparameter_search_strategy == "bayesian" else None`
- [ ] Include all metadata: `use_bayesian_search`, `n_trials`, `n_initial_points`, `timeout_seconds`, `best_params`, `best_score`, `optimization_time_seconds`, `n_completed_trials`

#### 9. **Variable Naming Consistency** (Reference: [train.py:1215-1222](../../DREAM-ML-backend/GEML/api/train.py#L1215-L1222))
- [ ] Backend uses: `n_trials`, `n_initial_points`, `timeout_seconds`, `bayesian_search_params`
- [ ] Frontend sends: `bayesian_config` object with `n_trials` inside
- [ ] Never use: `n_bayesian_iterations` (legacy/deprecated)

#### 10. **Documentation Patterns** (Reference: [train.py:1393-1399](../../DREAM-ML-backend/GEML/api/train.py#L1393-L1399))
- [ ] Add Spanish docstrings for objective function
- [ ] Use Spanish log messages matching existing style
- [ ] Comment parameter ranges with units/meanings where applicable

#### 11. **MLP-Specific Patterns**
- [ ] Use `verbose_setting = 1 if X_train.shape[0] <= 10000 else 0` for memory-aware verbosity
- [ ] Handle `hidden_layer_sizes` as categorical choices with tuple values: `[(4,), (10,), (10, 5), (50,), (100,), (100, 50), (100, 50, 10)]`
- [ ] Store `verbose_setting` in `best_params` but exclude from MLflow best_* logging
- [ ] Train final model with `random_state=SEED` and `verbose=verbose_setting`

**Reference Implementation**: See [train.py:1350-1562](../../DREAM-ML-backend/GEML/api/train.py#L1350-L1562) for XGBoost Bayesian implementation and [train.py:631-807](../../DREAM-ML-backend/GEML/api/train.py#L631-L807) for Logistic Regression Bayesian implementation as complete pattern examples.

---

### Phase Overview
Implement Bayesian Search for MLP (Multi-Layer Perceptron). Requires adding "bayesian" to valid_strategies first, then implementing the search logic.

### Files to Modify
- `DREAM-ML-backend/GEML/api/train.py`

### Specific Changes

#### Change 1: Add "bayesian" to valid_strategies (Line 681)

**Location**: In `train_mlp_model()` function

**Current Code** (line 681):
```python
valid_strategies = ["none", "grid", "random"]
```

**Updated Code**:
```python
valid_strategies = ["none", "grid", "random", "bayesian"]
```

#### Change 2: Implement Bayesian Search Branch (Insert after line 792, before line 794)

**Location**: In `train_mlp_model()`, after random search block ends (line 792), before manual training `else` clause (line 794)

**Insert Between Lines 792-794**:
```python
    elif hyperparameter_search_strategy == "bayesian":
        # Extract Bayesian configuration
        bayesian_config = data.get("bayesian_config", {})
        n_trials = bayesian_config.get("n_trials", 50)
        n_initial_points = bayesian_config.get("n_initial_points", 10)
        timeout_seconds = bayesian_config.get("timeout_seconds", None)

        # Extract custom parameter ranges from frontend
        bayesian_search_params = data.get("bayesian_search_params", {})

        # Validate bayesian config
        if n_trials < 1:
            raise ValueError(f"n_trials must be at least 1, got {n_trials}")
        if n_initial_points >= n_trials:
            raise ValueError(
                f"n_initial_points ({n_initial_points}) must be less than n_trials ({n_trials})"
            )

        logger.info("="*60)
        logger.info("Configuración Búsqueda Bayesiana MLP (Optuna):")
        logger.info(f"  n_trials: {n_trials}")
        logger.info(f"  n_initial_points: {n_initial_points}")
        logger.info(f"  timeout_seconds: {timeout_seconds}")
        logger.info("="*60)

        # Define default parameter ranges for MLP
        default_ranges = {
            "hidden_layer_sizes": {
                "type": "categorical",
                "choices": [(4,), (10,), (10, 5), (50,), (100,), (100, 50), (100, 50, 10)]
            },
            "activation": {"type": "categorical", "choices": ["relu", "tanh", "logistic"]},
            "learning_rate_init": {"type": "float", "low": 0.0001, "high": 0.1, "log": True},
            "solver": {"type": "categorical", "choices": ["adam", "sgd"]},
            "alpha": {"type": "float", "low": 0.0001, "high": 0.01, "log": True},
            "max_iter": {"type": "int", "low": 200, "high": 500}
        }

        # Merge with user-provided ranges
        param_ranges = {**default_ranges, **bayesian_search_params}

        # Define Optuna objective function
        def objective(trial: Trial) -> float:
            """
            Optuna objective function for MLP hyperparameter optimization.

            Returns:
                float: Negative accuracy (for minimization)
            """
            # Suggest parameters based on configured ranges
            hidden_layer_sizes_config = param_ranges["hidden_layer_sizes"]
            hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes',
                hidden_layer_sizes_config["choices"])

            activation = trial.suggest_categorical('activation',
                param_ranges["activation"]["choices"])

            learning_rate_init = trial.suggest_float('learning_rate_init',
                param_ranges["learning_rate_init"]["low"],
                param_ranges["learning_rate_init"]["high"],
                log=param_ranges["learning_rate_init"].get("log", True))

            solver = trial.suggest_categorical('solver',
                param_ranges["solver"]["choices"])

            alpha = trial.suggest_float('alpha',
                param_ranges["alpha"]["low"],
                param_ranges["alpha"]["high"],
                log=param_ranges["alpha"].get("log", True))

            max_iter = trial.suggest_int('max_iter',
                param_ranges["max_iter"]["low"],
                param_ranges["max_iter"]["high"])

            try:
                # Memory-aware verbosity
                verbose_setting = 1 if X_train.shape[0] <= 10000 else 0

                # Train model
                model_trial = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    solver=solver,
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    random_state=SEED,
                    verbose=verbose_setting
                )

                model_trial.fit(X_train, y_train)

                # Evaluate on validation set
                val_pred = model_trial.predict(X_val)
                val_score = accuracy_score(y_val, val_pred)

                logger.info(
                    f"Trial {trial.number}: accuracy={val_score:.4f}, "
                    f"hidden_layers={hidden_layer_sizes}, activation={activation}"
                )

                # Return negative accuracy (Optuna minimizes)
                return -val_score

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {str(e)}")
                return float('inf')

        # Create Optuna study with TPE sampler
        sampler = TPESampler(
            seed=SEED,
            n_startup_trials=n_initial_points,
            multivariate=False,
            consider_magic_clip=True,
            consider_endpoints=False
        )

        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            study_name=f"mlp_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Track optimization time
        optimization_start_time = time.time()

        # Run optimization
        logger.info(f"Iniciando Búsqueda Bayesiana MLP con Optuna TPESampler...")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=False,
            n_jobs=1
        )

        optimization_time_seconds = time.time() - optimization_start_time

        # Validate results
        if study.best_trial is None or study.best_value == float('inf'):
            raise RuntimeError(
                "Búsqueda Bayesiana falló: Todos los trials retornaron errores. "
                "Verifique los rangos de parámetros y calidad de datos."
            )

        # Extract best parameters
        best_params_dict = study.best_params
        best_score = -study.best_value

        # Log optimization results
        logger.info("="*60)
        logger.info(f"Búsqueda Bayesiana MLP Completada")
        logger.info(f"  Mejor accuracy: {best_score:.4f}")
        logger.info(f"  Mejores parámetros: {best_params_dict}")
        completed_trials = len([t for t in study.trials
                               if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"  Trials completados: {completed_trials}/{len(study.trials)}")
        logger.info(f"  Tiempo de optimización: {optimization_time_seconds:.2f} segundos")
        logger.info("="*60)

        # Train final model with best parameters
        verbose_setting = 1 if X_train.shape[0] <= 10000 else 0
        model = MLPClassifier(
            hidden_layer_sizes=best_params_dict['hidden_layer_sizes'],
            activation=best_params_dict['activation'],
            solver=best_params_dict['solver'],
            alpha=best_params_dict['alpha'],
            learning_rate_init=best_params_dict['learning_rate_init'],
            max_iter=best_params_dict['max_iter'],
            random_state=SEED,
            verbose=verbose_setting
        )
        model.fit(X_train, y_train)

        # Store best params for logging
        best_params = best_params_dict.copy()
        best_params["random_state"] = SEED
        best_params["verbose"] = verbose_setting

        # Log Bayesian search metadata to MLflow
        mlflow.log_params({
            "bayesian_n_trials": n_trials,
            "bayesian_n_initial_points": n_initial_points,
            "bayesian_optimization_metric": "accuracy",
            **{f"best_{k}": v for k, v in best_params_dict.items() if k != "verbose"}
        })

        mlflow.log_metrics({
            "bayesian_best_score": best_score,
            "bayesian_optimization_time_seconds": optimization_time_seconds,
            "bayesian_n_completed_trials": completed_trials
        })
```

#### Change 3: Update pipeline_config for MLP (Around line 855-903)

**Add bayesian_search block** to pipeline_config (same structure as previous algorithms):
```python
    "bayesian_search": {
        "use_bayesian_search": hyperparameter_search_strategy == "bayesian",
        "n_trials": n_trials if hyperparameter_search_strategy == "bayesian" else None,
        "n_initial_points": n_initial_points if hyperparameter_search_strategy == "bayesian" else None,
        "timeout_seconds": timeout_seconds if hyperparameter_search_strategy == "bayesian" else None,
        "bayesian_search_params": bayesian_search_params if hyperparameter_search_strategy == "bayesian" else None,
        "best_params": best_params if hyperparameter_search_strategy == "bayesian" else None,
        "best_score": best_score if hyperparameter_search_strategy == "bayesian" else None,
        "optimization_time_seconds": optimization_time_seconds if hyperparameter_search_strategy == "bayesian" else None,
        "n_completed_trials": completed_trials if hyperparameter_search_strategy == "bayesian" else None
    },
```

**Initialize variables** (after line 685):
```python
# Bayesian search variables
n_trials = None
n_initial_points = None
timeout_seconds = None
bayesian_search_params = {}
optimization_time_seconds = None
completed_trials = None
best_score = None
```

### Automated Verification Steps
```bash
# Syntax check
python3 -m py_compile GEML/api/train.py

# Run MLP-specific tests
pytest GEML/tests/ -k "mlp" -v
```

### Manual Verification Steps
1. Select MLP algorithm in frontend
2. Choose "Bayesian Search" with n_trials=10
3. Train and verify:
   - ✅ "bayesian" is accepted as valid strategy
   - ✅ hidden_layer_sizes tuples are handled correctly
   - ✅ Training completes successfully
   - ✅ MLflow shows bayesian metrics

### Success Criteria
- [x] "bayesian" added to valid_strategies for MLP
- [x] Bayesian search implemented for MLP
- [x] hidden_layer_sizes tuples handled correctly
- [x] Memory-aware verbosity setting preserved
- [x] Training completes successfully
- [x] Reproducibility verified

### Phase 4 Status: ✅ COMPLETED (2025-12-31)

**Implementation Summary:**
- ✅ Added "bayesian" to valid_strategies at [train.py:960](../../DREAM-ML-backend/GEML/api/train.py#L960)
- ✅ Initialized Bayesian variables at [train.py:968-975](../../DREAM-ML-backend/GEML/api/train.py#L968-L975)
- ✅ Implemented complete Bayesian search branch at [train.py:1073-1265](../../DREAM-ML-backend/GEML/api/train.py#L1073-L1265):
  - Extracts and validates bayesian_config (n_trials, n_initial_points, timeout_seconds)
  - Converts frontend bayesian_search_params using `convert_frontend_bayesian_params()`
  - Defines default parameter ranges matching random search defaults from line 357
  - Implements Optuna objective function with:
    - Categorical choices for hidden_layer_sizes: [(4,), (10,), (10, 5), (50,), (100,), (100, 50), (100, 50, 10)]
    - Memory-aware verbosity: `verbose_setting = 1 if X_train.shape[0] <= 10000 else 0`
  - Creates TPESampler with seed=SEED, n_startup_trials=n_initial_points
  - Validates results and trains final model with best parameters
  - Stores verbose_setting in best_params but excludes from MLflow best_* logging (line 1258)
  - Logs comprehensive metadata to MLflow
- ✅ Updated pipeline_config with bayesian_search block at [train.py:1168-1178](../../DREAM-ML-backend/GEML/api/train.py#L1168-L1178)
- ✅ All automated verification passed (syntax check, Optuna 4.6.0, Django check)
- ✅ All manual verification tests passed:
  - Basic Bayesian search with n_trials=10 completed successfully
  - Custom parameter ranges properly respected
  - Reproducibility verified (same seed → identical results)
  - Error handling validated (invalid configs correctly rejected)
- ✅ MLflow logging verified: all bayesian_* params and metrics logged correctly
- ✅ pipeline_config.json correctly includes bayesian_search block with all metadata
- ✅ verbose parameter correctly excluded from MLflow best_* logging while included in best_params

**Key Implementation Details:**
- Used TPESampler with `multivariate=False` for independent parameter optimization
- Returns negative accuracy for Optuna minimization
- Handles MLP-specific requirements: categorical hidden_layer_sizes tuples, memory-aware verbosity
- Supports configurable parameter ranges via frontend UI
- Fixed SEED=42 ensures deterministic optimization results
- Follows exact same patterns as Logistic Regression (Phase 2) and XGBoost (Phase 3)

---

## Phase 5: Testing and Verification

### Pattern Consistency Checklist

Before implementing Phase 5, ensure consistency with previous phases:

#### 1. **Test Structure Patterns** (Reference: Phase 2, 3, 4 Implementations)
- [ ] Follow unittest.TestCase structure
- [ ] Use `@classmethod setUpClass(cls)` for shared test fixtures
- [ ] Create synthetic dataset with `np.random.seed(42)` for reproducibility
- [ ] Use temporary directories with `tempfile.mkdtemp()`
- [ ] Setup isolated MLflow tracking with sqlite:/// database

#### 2. **Test Coverage Requirements**
- [ ] Test all 3 algorithms: Logistic Regression, XGBoost, MLP
- [ ] Test basic Bayesian search with default parameters
- [ ] Test custom parameter ranges
- [ ] Test validation errors (n_trials < 1, n_initial_points >= n_trials)
- [ ] Test reproducibility (same seed → same results)
- [ ] Test MLflow logging (params and metrics)
- [ ] Test pipeline_config.json generation

#### 3. **Dataset Requirements**
- [ ] Binary classification dataset (200 samples, 5 features)
- [ ] Simple linear boundary: `y = (X[:, 0] + X[:, 1] > 0).astype(int)`
- [ ] Split ratios: train=0.7, val=0.15, test=0.15
- [ ] Saved as CSV with column names: feature_0, feature_1, ..., target

#### 4. **Test Naming Conventions**
- [ ] Use descriptive test names: `test_<algorithm>_bayesian_<aspect>`
- [ ] Examples: `test_mlp_bayesian_basic`, `test_xgboost_bayesian_custom_ranges`
- [ ] Group related tests together

#### 5. **Assertion Patterns**
- [ ] Check result structure: `assertIsNotNone(result)`, `assertIn("val_metrics", result)`
- [ ] Check MLflow params: verify bayesian_n_trials, best_* parameters exist
- [ ] Check MLflow metrics: verify bayesian_best_score, bayesian_optimization_time_seconds
- [ ] Check reproducibility: `assertAlmostEqual(..., places=4)` for float comparisons

#### 6. **Error Testing Patterns**
- [ ] Use `with self.assertRaises(ValueError) as context:`
- [ ] Check error message content: `self.assertIn("expected message", str(context.exception))`
- [ ] Test boundary conditions: n_trials=0, n_initial_points >= n_trials

#### 7. **MLflow Cleanup Patterns**
- [ ] Always end MLflow runs: `if run: mlflow.end_run()`
- [ ] Use `with mlflow.start_run():` context manager
- [ ] Use isolated tracking URI per test class

#### 8. **File Verification Patterns**
- [ ] Check pipeline_config.json exists and has correct structure
- [ ] Verify bayesian_search block contains all required fields
- [ ] Check model pickle file is created
- [ ] Verify artifacts (confusion matrix, ROC curves) are generated

#### 9. **Performance Testing**
- [ ] Use small n_trials (5-10) for unit tests to keep execution time reasonable
- [ ] Test with n_trials=10 for reproducibility tests
- [ ] Avoid long-running tests (> 2 minutes per test)

#### 10. **Documentation Patterns**
- [ ] Add docstrings to test methods describing what is being tested
- [ ] Use clear variable names in test data
- [ ] Comment non-obvious test logic

**Reference Implementations**:
- Dataset creation pattern similar to existing test datasets
- MLflow testing patterns from existing classification tests
- Error handling patterns from Phase 2, 3, 4 implementations

---

### Phase Overview
Comprehensive testing across all three algorithms to ensure correctness, reproducibility, and integration with the existing system.

### Files to Create/Modify
- `DREAM-ML-backend/GEML/tests/api_tests/test_bayesian_search_classification.py` (new file)

### Test Structure

Create a new test file with the following test cases:

```python
"""
Test suite for Bayesian Search hyperparameter optimization in classification models.
"""
import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from api.train import (
    train_logistic_regression_model,
    train_xgboost_model,
    train_mlp_model
)
import mlflow


class TestBayesianSearchClassification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests"""
        # Create temporary experiment directory
        cls.temp_dir = tempfile.mkdtemp()
        cls.experiment_dir = os.path.join(cls.temp_dir, "test_experiment")
        os.makedirs(cls.experiment_dir, exist_ok=True)

        # Create synthetic binary classification dataset
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple linear boundary

        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        df["target"] = y

        # Save to CSV
        cls.test_csv_path = os.path.join(cls.temp_dir, "test_data.csv")
        df.to_csv(cls.test_csv_path, index=False)

        # Setup MLflow
        mlflow.set_tracking_uri(f"sqlite:///{cls.temp_dir}/mlflow.db")
        mlflow.set_experiment("test_bayesian_classification")

    def test_logistic_regression_bayesian_basic(self):
        """Test basic Bayesian search for Logistic Regression"""
        data = {
            "input_features": [f"feature_{i}" for i in range(5)],
            "target_variable": "target",
            "hyperparameter_search_strategy": "bayesian",
            "bayesian_config": {
                "n_trials": 5,
                "n_initial_points": 2,
                "timeout_seconds": None
            },
            "bayesian_search_params": {},  # Use defaults
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "model_name": "TestLogisticBayesian"
        }

        with mlflow.start_run():
            result = train_logistic_regression_model(
                self.test_csv_path, data, self.experiment_dir
            )

        # Assertions
        self.assertIsNotNone(result)
        self.assertIn("val_metrics", result)
        self.assertIn("test_metrics", result)
        self.assertIn("model_path", result)

        # Check MLflow logged params
        run = mlflow.active_run()
        if run:
            mlflow.end_run()

    def test_xgboost_bayesian_basic(self):
        """Test basic Bayesian search for XGBoost"""
        data = {
            "input_features": [f"feature_{i}" for i in range(5)],
            "target_variable": "target",
            "problem_type": "binary",
            "hyperparameter_search_strategy": "bayesian",
            "bayesian_config": {
                "n_trials": 5,
                "n_initial_points": 2
            },
            "bayesian_search_params": {},
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "model_name": "TestXGBoostBayesian"
        }

        with mlflow.start_run():
            result = train_xgboost_model(
                self.test_csv_path, data, self.experiment_dir
            )

        self.assertIsNotNone(result)
        self.assertIn("val_metrics", result)

    def test_mlp_bayesian_basic(self):
        """Test basic Bayesian search for MLP"""
        data = {
            "input_features": [f"feature_{i}" for i in range(5)],
            "target_variable": "target",
            "problem_type": "binary",
            "hyperparameter_search_strategy": "bayesian",
            "bayesian_config": {
                "n_trials": 5,
                "n_initial_points": 2
            },
            "bayesian_search_params": {},
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "model_name": "TestMLPBayesian"
        }

        with mlflow.start_run():
            result = train_mlp_model(
                self.test_csv_path, data, self.experiment_dir
            )

        self.assertIsNotNone(result)
        self.assertIn("val_metrics", result)

    def test_bayesian_validation_n_trials_too_small(self):
        """Test validation: n_trials must be >= 1"""
        data = {
            "input_features": [f"feature_{i}" for i in range(5)],
            "target_variable": "target",
            "hyperparameter_search_strategy": "bayesian",
            "bayesian_config": {"n_trials": 0},
            "bayesian_search_params": {},
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "model_name": "TestInvalid"
        }

        with self.assertRaises(ValueError) as context:
            with mlflow.start_run():
                train_logistic_regression_model(
                    self.test_csv_path, data, self.experiment_dir
                )

        self.assertIn("n_trials must be at least 1", str(context.exception))

    def test_bayesian_custom_param_ranges(self):
        """Test custom parameter ranges for Logistic Regression"""
        data = {
            "input_features": [f"feature_{i}" for i in range(5)],
            "target_variable": "target",
            "hyperparameter_search_strategy": "bayesian",
            "bayesian_config": {
                "n_trials": 5,
                "n_initial_points": 2
            },
            "bayesian_search_params": {
                "C": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
                "max_iter": {"type": "int", "low": 100, "high": 300}
            },
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "model_name": "TestCustomRanges"
        }

        with mlflow.start_run():
            result = train_logistic_regression_model(
                self.test_csv_path, data, self.experiment_dir
            )

        self.assertIsNotNone(result)

    def test_bayesian_reproducibility(self):
        """Test that same seed produces same results"""
        data = {
            "input_features": [f"feature_{i}" for i in range(5)],
            "target_variable": "target",
            "hyperparameter_search_strategy": "bayesian",
            "bayesian_config": {
                "n_trials": 10,
                "n_initial_points": 3
            },
            "bayesian_search_params": {},
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "model_name": "TestRepro1"
        }

        # First run
        with mlflow.start_run():
            result1 = train_logistic_regression_model(
                self.test_csv_path, data, self.experiment_dir
            )

        # Second run (same config, same seed)
        data["model_name"] = "TestRepro2"
        with mlflow.start_run():
            result2 = train_logistic_regression_model(
                self.test_csv_path, data, self.experiment_dir
            )

        # Results should be identical (within floating point tolerance)
        self.assertAlmostEqual(
            result1["val_metrics"]["val_accuracy"],
            result2["val_metrics"]["val_accuracy"],
            places=4
        )


if __name__ == "__main__":
    unittest.main()
```

### Automated Verification Steps
```bash
# Run all classification Bayesian tests
cd DREAM-ML-backend
pytest GEML/tests/api_tests/test_bayesian_search_classification.py -v

# Check code coverage
pytest GEML/tests/api_tests/test_bayesian_search_classification.py --cov=api.train --cov-report=html
```

### Manual Verification Steps

**End-to-End Integration Test**:

1. **Logistic Regression Test**:
   - Upload Iris dataset (binary: setosa vs others)
   - Select Logistic Regression, Bayesian Search
   - Set n_trials=20, n_initial_points=5
   - Verify training completes, accuracy > 0.9
   - Check MLflow UI for logged parameters and metrics

2. **XGBoost Test**:
   - Same dataset
   - Select XGBoost, Bayesian Search
   - Set n_trials=15
   - Verify early stopping logs appear
   - Check pipeline_config.json has bayesian_search section

3. **MLP Test**:
   - Same dataset
   - Select MLP, Bayesian Search
   - Set n_trials=10
   - Verify different hidden_layer_sizes are tried
   - Check results are reproducible (run twice with same config)

4. **Custom Parameter Ranges Test**:
   - Modify frontend to send custom bayesian_search_params
   - For Logistic: Restrict C to [0.1, 1.0]
   - Verify backend uses custom ranges (check logs)
   - Verify best C is within custom range

### Success Criteria
- [x] All unit tests pass
- [x] Reproducibility test passes (same seed → same results)
- [x] Custom parameter ranges test passes
- [x] Validation tests catch invalid configs
- [x] End-to-end integration tests pass for all 3 algorithms
- [x] MLflow logging verified for all algorithms
- [x] pipeline_config.json correctly updated for all algorithms
- [x] No memory leaks or hanging processes
- [x] Training time reasonable (< 5 min for 50 trials on Iris)

---

## Summary of Changes

### Files Modified
1. **DREAM-ML-backend/GEML/api/train.py**:
   - Added Optuna imports (lines ~21-24)
   - Configured Optuna logging (line ~76)
   - Implemented Bayesian search for Logistic Regression (insert after line 559)
   - Implemented Bayesian search for XGBoost (insert after line 1069)
   - Added "bayesian" to MLP valid_strategies (line 681)
   - Implemented Bayesian search for MLP (insert after line 792)
   - Updated pipeline_config for all 3 algorithms

### Files Created
2. **DREAM-ML-backend/GEML/tests/api_tests/test_bayesian_search_classification.py**:
   - Comprehensive test suite for Bayesian search
   - 6+ test cases covering basic functionality, validation, custom ranges, reproducibility

### Estimated Lines of Code
- Logistic Regression: ~150 lines
- XGBoost: ~160 lines
- MLP: ~155 lines
- Tests: ~250 lines
- **Total**: ~715 lines

### Implementation Timeline
- Phase 1 (Optuna setup): 15 minutes
- Phase 2 (Logistic Regression): 1-2 hours
- Phase 3 (XGBoost): 1-2 hours
- Phase 4 (MLP): 1-2 hours
- Phase 5 (Testing): 1-2 hours
- **Total Estimated Time**: 5-8 hours

---

## Key Design Decisions

1. **Minimal Feature Scope**: Focused on core Bayesian search only, deferred advanced features
2. **Configurable Ranges**: Support frontend bayesian_search_params for flexibility
3. **Simple Validation**: Train/val split for fast iteration (not k-fold CV)
4. **Reproducibility First**: SEED=42, n_jobs=1, single-threaded TPESampler
5. **Error Handling**: Return float('inf') for failed trials, log warnings
6. **Pattern Consistency**: Follow existing grid/random search structure
7. **MLflow Integration**: Log bayesian_n_trials, best_score, optimization_time
8. **Pipeline Config**: Add bayesian_search section parallel to grid_search and random_search

---

## Open Questions / Future Enhancements

1. **Advanced Features** (deferred):
   - Convergence detection (early stopping when optimization plateaus)
   - Memory monitoring (stop if memory exceeds threshold)
   - Multi-objective optimization (accuracy vs model complexity)

2. **Optimization Metrics**:
   - Currently hardcoded to accuracy
   - Could make configurable (F1, ROC-AUC, precision, recall)

3. **Cross-Validation**:
   - Currently uses simple train/val split
   - Could add optional k-fold CV within objective function

4. **Visualization**:
   - Optuna provides plot_optimization_history, plot_param_importances
   - Could integrate these into frontend or MLflow artifacts

5. **Study Persistence**:
   - Currently creates ephemeral in-memory studies
   - Could use RDB storage for resumable optimization

---

## References

- Research: [thoughts/shared/research/2025-12-27_classification-training-workflow-analysis.md](../../../thoughts/shared/research/2025-12-27_classification-training-workflow-analysis.md)
- Optuna Research: [thoughts/shared/research/2025-12-18_optuna_research.md](../../../thoughts/shared/research/2025-12-18_optuna_research.md)
- Time Series Plan: [thoughts/shared/plans/2025-12-18_ts-training-bayesian-search-analysis.md](../../../thoughts/shared/plans/2025-12-18_ts-training-bayesian-search-analysis.md)
- Optuna 4.6.0 Docs: https://optuna.readthedocs.io/en/stable/
