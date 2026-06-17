# Bayesian Optimization Migration Plan: scikit-optimize → Optuna

**Document Version:** 1.0
**Last Updated:** October 2025
**Status:** Phase 1 & 2 Complete - Ready for Phase 3 Implementation

---

## Executive Summary

### Current State
- **Library:** scikit-optimize 0.10.2
- **Status:** Repository archived February 28, 2024 (no longer maintained)
- **Usage:** Bayesian optimization for 6 model types across 2 major modules
- **Production Impact:** Critical - system actively serving users

### Why Migrate?
1. **No Future Support:** No bug fixes, security patches, or compatibility updates
2. **Risk of Breaking Changes:** Future numpy/scipy/scikit-learn updates may break compatibility
3. **Missing Modern Features:** No native pruning, visualization dashboard, or parallel optimization
4. **Active Alternative Available:** Optuna is actively maintained with 3000+ commits and frequent releases

### Migration Strategy
- **Approach:** Hybrid/Incremental (keep both libraries during transition)
- **Timeline:** 2-3 months (one model type per 2 weeks)
- **Risk Level:** Low (parallel implementations with A/B testing)
- **Current Phase:** Dependencies stabilized, migration guide complete

---

## Table of Contents

1. [Architecture Analysis](#architecture-analysis)
2. [Dependency Status](#dependency-status)
3. [API Comparison](#api-comparison)
4. [Migration Strategy](#migration-strategy)
5. [Code Migration Examples](#code-migration-examples)
6. [Testing & Validation](#testing-and-validation)
7. [Rollback Procedures](#rollback-procedures)
8. [Timeline & Milestones](#timeline-and-milestones)
9. [Resources & References](#resources-and-references)

---

## Architecture Analysis

### Current scikit-optimize Implementation

#### Files Affected
1. **[api/train.py](../DREAM-ML-backend/GEML/api/train.py)** - Classification models (26,118 tokens)
2. **[apiTimeSeries/train.py](../DREAM-ML-backend/GEML/apiTimeSeries/train.py)** - Time series models (36,352 tokens)

#### Model Types Using Bayesian Optimization

| Model Type | Module | Function | Lines | Complexity |
|------------|--------|----------|-------|------------|
| Logistic Regression | api/train.py | `get_default_bayesian_space_logistic()` | ~430-457 | Low |
| MLP | api/train.py | `get_default_bayesian_space_mlp()` | ~459-490 | Medium |
| XGBoost (CL) | api/train.py | `get_default_bayesian_space_xgboost()` | ~492-551 | Medium |
| ARIMA/SARIMA | apiTimeSeries/train.py | `get_default_bayesian_space_arima()` | ~818-856 | Medium |
| XGBoost (TS) | apiTimeSeries/train.py | `get_default_bayesian_space_xgboost_ts()` | ~858-894 | Medium |
| LSTM | apiTimeSeries/train.py | `get_default_bayesian_space_lstm()` | ~2550+ | High |

#### Core Functions

```python
# Search Space Definition
create_bayesian_search_space(bayesian_search_params: dict) -> tuple
# Returns: (dimensions, param_names)

# Callbacks & Early Stopping
create_bayesian_callbacks(bayesian_config: dict, experiment_dir: str) -> list
# Returns: [TimerStopper, DeltaYStopper]

# Visualization
generate_bayesian_visualizations(optimizer, experiment_dir: str, prefix: str)
# Generates: convergence plot, parameter importance, acquisition heatmap

# Artifact Management
save_bayesian_artifacts(optimizer, bayesian_config: dict, experiment_dir: str)
# Saves: optimization history, best parameters, search results
```

#### scikit-optimize Imports

```python
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.utils import expected_minimum
```

**Note:** The correct callback for timeout is `DeadlineStopper`, not `TimerStopper` (which doesn't exist in scikit-optimize 0.10.2).

#### Optimization Pattern (Ask/Tell Loop)

```python
optimizer = Optimizer(
    dimensions=dimensions,
    n_initial_points=n_initial_points,
    acq_func=acq_func,  # Usually "EI" (Expected Improvement)
    random_state=random_state
)

for i in range(n_bayesian_iterations):
    # Get next point to evaluate
    next_point = optimizer.ask()

    # Convert to parameter dictionary
    params_dict = {param_names[j]: next_point[j] for j in range(len(param_names))}

    # Train model and evaluate
    model = train_model(**params_dict)
    val_score = evaluate(model, validation_set)

    # Report result (minimize negative accuracy)
    objective_value = -val_score
    optimizer.tell(next_point, objective_value)

    # Check callbacks manually
    for cb in callbacks:
        if cb(optimizer) == True:
            break  # Early stopping
```

---

## Dependency Status

### Current Versions (as of October 2025)

| Package | Version | Status | Notes |
|---------|---------|--------|-------|
| Python | 3.12.12 | ✅ Supported | Using current stable |
| scikit-optimize | 0.10.2 | ⚠️ Archived | Last release June 2024 |
| scikit-learn | 1.6.1 | ✅ Active | Pinned to <1.7.0 |
| numpy | 1.26.4 | ✅ Active | Pinned to <2.0.0 |
| scipy | 1.13.1 | ✅ Active | Pinned to <2.0.0 |
| optuna | Not yet installed | ✅ Active | Ready to install >=3.0.0 |

### Dependency Pins (Updated in requirements-base.txt)

```txt
# ML/Data Science
numpy>=1.21.0,<2.0.0
scipy>=1.13.0,<2.0.0
scikit_learn>=1.6.0,<1.7.0
scikit-optimize==0.10.2  # Bayesian optimization (archived, plan migration to Optuna)
optuna>=3.0.0  # Modern hyperparameter optimization (for future migration)
```

### Compatibility Matrix

| scikit-optimize 0.10.2 | Compatible? | Notes |
|------------------------|-------------|-------|
| Python 3.8-3.12 | ✅ Yes | Officially supported |
| numpy <2.0 | ✅ Yes | Working with 1.26.4 |
| scipy <2.0 | ✅ Yes | Working with 1.13.1 |
| scikit-learn 1.6.x | ✅ Yes | Tested and working |
| Future versions | ⚠️ Unknown | No guarantees (archived) |

---

## API Comparison

### Search Space Definition

#### scikit-optimize (Current)

```python
from skopt.space import Real, Integer, Categorical

def get_default_bayesian_space_logistic() -> dict:
    return {
        "C": {
            "type": "real",
            "distribution": "log-uniform",
            "low": 0.001,
            "high": 100.0
        },
        "max_iter": {
            "type": "integer",
            "low": 100,
            "high": 1000
        },
        "solver": {
            "type": "categorical",
            "choices": ["lbfgs", "liblinear", "saga"]
        }
    }

def create_bayesian_search_space(bayesian_search_params: dict) -> tuple:
    dimensions = []
    param_names = []

    for param_name, param_spec in bayesian_search_params.items():
        param_names.append(param_name)
        param_type = param_spec.get("type")

        if param_type == "real":
            distribution = param_spec.get("distribution", "uniform")
            low = param_spec["low"]
            high = param_spec["high"]

            if distribution == "log-uniform":
                dimensions.append(Real(low, high, prior="log-uniform", name=param_name))
            elif distribution == "uniform":
                dimensions.append(Real(low, high, prior="uniform", name=param_name))

        elif param_type == "integer":
            low = param_spec["low"]
            high = param_spec["high"]
            dimensions.append(Integer(low, high, name=param_name))

        elif param_type == "categorical":
            choices = param_spec["choices"]
            dimensions.append(Categorical(choices, name=param_name))

    return dimensions, param_names
```

#### Optuna (Target)

```python
import optuna

def get_default_optuna_space_logistic() -> dict:
    """Same dict format - reuse existing functions!"""
    return {
        "C": {
            "type": "real",
            "distribution": "log-uniform",
            "low": 0.001,
            "high": 100.0
        },
        "max_iter": {
            "type": "integer",
            "low": 100,
            "high": 1000
        },
        "solver": {
            "type": "categorical",
            "choices": ["lbfgs", "liblinear", "saga"]
        }
    }

def create_optuna_objective(X_train, y_train, X_val, y_val, space_config: dict):
    """Create objective function for Optuna"""

    def objective(trial):
        # Suggest parameters based on space config
        params = {}

        for param_name, param_spec in space_config.items():
            param_type = param_spec.get("type")

            if param_type == "real":
                distribution = param_spec.get("distribution", "uniform")
                low = param_spec["low"]
                high = param_spec["high"]

                if distribution == "log-uniform":
                    params[param_name] = trial.suggest_float(param_name, low, high, log=True)
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high, log=False)

            elif param_type == "integer":
                params[param_name] = trial.suggest_int(param_name, param_spec["low"], param_spec["high"])

            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_spec["choices"])

        # Train and evaluate
        model = LogisticRegression(**params, random_state=SEED)
        model.fit(X_train, y_train)
        val_score = accuracy_score(y_val, model.predict(X_val))

        # Optuna maximizes by default (no need for negative)
        return val_score

    return objective
```

**Key Differences:**
- ✅ **Can reuse existing space definitions** (same dict format)
- ✅ **Simpler API:** No need to convert dict → dimensions → params
- ✅ **Direction:** Optuna maximizes by default (no negative scores needed)
- ⚠️ **Pattern change:** Define-by-run (objective function) vs Ask/Tell loop

---

### Optimizer Initialization & Execution

#### scikit-optimize (Current)

```python
from skopt import Optimizer

# Initialize optimizer
optimizer = Optimizer(
    dimensions=dimensions,
    n_initial_points=10,
    acq_func="EI",  # Expected Improvement
    random_state=SEED
)

# Manual ask/tell loop
best_score = 0.0
best_params = None

for i in range(n_bayesian_iterations):
    # Ask for next point
    next_point = optimizer.ask()
    params_dict = {param_names[j]: next_point[j] for j in range(len(param_names))}

    # Evaluate
    model = train_model(**params_dict)
    val_score = evaluate(model)

    # Tell result (minimize negative)
    optimizer.tell(next_point, -val_score)

    # Track best
    if val_score > best_score:
        best_score = val_score
        best_params = params_dict
```

#### Optuna (Target)

```python
import optuna

# Create study (optimizer)
study = optuna.create_study(
    direction="maximize",  # or "minimize"
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=10,  # Equivalent to n_initial_points
        seed=SEED
    )
)

# Optimize (runs loop internally)
study.optimize(
    objective,
    n_trials=n_bayesian_iterations,
    callbacks=[timeout_callback, convergence_callback]  # Optional
)

# Get results
best_params = study.best_params
best_score = study.best_value
best_trial = study.best_trial
```

**Key Differences:**
- ✅ **Cleaner:** No manual loop management
- ✅ **Built-in tracking:** Automatically tracks best trial
- ✅ **Better algorithm:** TPE (Tree-structured Parzen Estimator) often outperforms GP
- ⚠️ **Less control:** Can't manually modify each iteration (but rarely needed)

---

### Callbacks & Early Stopping

**IMPORTANT:** In scikit-optimize 0.10.2, the timeout callback is named `DeadlineStopper`, NOT `TimerStopper`. `TimerCallback` exists but only logs execution times without stopping optimization.

#### scikit-optimize (Current)

```python
from skopt.callbacks import DeadlineStopper, DeltaYStopper

def create_bayesian_callbacks(bayesian_config: dict, experiment_dir: str) -> list:
    callbacks = []

    # Timeout callback
    timeout_seconds = bayesian_config.get("timeout_seconds", None)
    if timeout_seconds is not None:
        callbacks.append(DeadlineStopper(total_time=timeout_seconds))

    # Convergence callback
    convergence_tolerance = bayesian_config.get("convergence_tolerance", 0.001)
    convergence_patience = bayesian_config.get("convergence_patience", 5)
    callbacks.append(DeltaYStopper(delta=convergence_tolerance, n_best=convergence_patience))

    return callbacks

# Manual callback evaluation in loop
for cb in callbacks:
    if cb(optimizer) == True:
        logger.info(f"Early stopping by {cb.__class__.__name__}")
        break
```

#### Optuna (Target)

```python
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

# Timeout callback
class TimeoutCallback:
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()

    def __call__(self, study, trial):
        if time.time() - self.start_time > self.timeout_seconds:
            study.stop()

# Convergence callback
class ConvergenceCallback:
    def __init__(self, tolerance=0.001, patience=5):
        self.tolerance = tolerance
        self.patience = patience
        self.no_improvement_count = 0
        self.best_value = None

    def __call__(self, study, trial):
        if self.best_value is None:
            self.best_value = study.best_value
            return

        improvement = abs(study.best_value - self.best_value)
        if improvement < self.tolerance:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
            self.best_value = study.best_value

        if self.no_improvement_count >= self.patience:
            study.stop()

def create_optuna_callbacks(bayesian_config: dict) -> list:
    callbacks = []

    timeout_seconds = bayesian_config.get("timeout_seconds", None)
    if timeout_seconds is not None:
        callbacks.append(TimeoutCallback(timeout_seconds))

    convergence_tolerance = bayesian_config.get("convergence_tolerance", 0.001)
    convergence_patience = bayesian_config.get("convergence_patience", 5)
    callbacks.append(ConvergenceCallback(convergence_tolerance, convergence_patience))

    return callbacks

# Use in study.optimize()
study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
```

**Key Differences:**
- ✅ **Same functionality:** Can implement equivalent callbacks
- ✅ **Built-in pruning:** Optuna has additional pruning algorithms (Median, Hyperband)
- ⚠️ **Different API:** Need to rewrite custom callbacks (but straightforward)

---

### Visualization & Artifacts

#### scikit-optimize (Current)

```python
def generate_bayesian_visualizations(optimizer, experiment_dir: str, prefix: str):
    artifacts = {}

    # 1. Convergence Plot
    n_initial = optimizer.n_initial_points_
    func_vals = optimizer.func_vals
    best_vals = np.minimum.accumulate(func_vals)

    plt.plot(range(len(func_vals)), func_vals, 'b.', label='Observed')
    plt.plot(range(len(best_vals)), best_vals, 'r-', label='Best')
    plt.axvline(x=n_initial, color='g', linestyle='--', label='Random init')
    plt.savefig(f"{experiment_dir}/{prefix}_convergence.png")

    # 2. Parameter Importance
    gp_model = optimizer.models[-1]
    param_names = [dim.name for dim in optimizer.space.dimensions]
    # ... calculate importances using GP variance ...

    # 3. Acquisition Heatmap
    # ... custom visualization ...

    return artifacts
```

#### Optuna (Target)

```python
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour
)

def generate_optuna_visualizations(study, experiment_dir: str, prefix: str):
    artifacts = {}

    # 1. Convergence Plot (built-in!)
    fig = plot_optimization_history(study)
    fig.write_image(f"{experiment_dir}/{prefix}_convergence.png")
    artifacts["convergence_plot"] = f"{experiment_dir}/{prefix}_convergence.png"

    # 2. Parameter Importance (built-in!)
    fig = plot_param_importances(study)
    fig.write_image(f"{experiment_dir}/{prefix}_importance.png")
    artifacts["parameter_importance"] = f"{experiment_dir}/{prefix}_importance.png"

    # 3. Additional plots (bonus!)
    fig = plot_parallel_coordinate(study)
    fig.write_image(f"{experiment_dir}/{prefix}_parallel.png")

    fig = plot_contour(study)
    fig.write_image(f"{experiment_dir}/{prefix}_contour.png")

    return artifacts
```

**Key Differences:**
- ✅ **Built-in visualizations:** Much less custom code needed
- ✅ **Better quality:** Interactive Plotly charts (can export to PNG)
- ✅ **More options:** Parallel coordinate, slice plots, contours
- ⚠️ **Dependency:** Requires `plotly` and `kaleido` for image export

---

### MLflow Integration

#### scikit-optimize (Current)

```python
# Manual logging in ask/tell loop
for i in range(n_bayesian_iterations):
    next_point = optimizer.ask()
    params_dict = {param_names[j]: next_point[j] for j in range(len(param_names))}

    # Train and evaluate
    val_score = train_and_evaluate(**params_dict)
    objective_value = -val_score
    optimizer.tell(next_point, objective_value)

    # Log to MLflow
    mlflow.log_metric(f"bayesian_iter_{i+1}_accuracy", val_score, step=i+1)
    mlflow.log_metric("bayesian_best_accuracy", best_score, step=i+1)

# Log final results
mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
mlflow.log_artifact(convergence_plot_path, "bayesian_plots")
```

#### Optuna (Target)

```python
# Option 1: Manual logging in objective function
def objective(trial):
    params = suggest_params(trial)

    model = train_model(**params)
    val_score = evaluate(model)

    # Log each trial
    mlflow.log_metric(f"trial_{trial.number}_accuracy", val_score, step=trial.number)

    return val_score

# After optimization
mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
mlflow.log_metric("best_accuracy", study.best_value)

# Option 2: Use MLflowCallback (recommended!)
from optuna.integration.mlflow import MLflowCallback

mlflow_callback = MLflowCallback(
    tracking_uri="your_tracking_uri",
    metric_name="accuracy"
)

study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_callback])
```

**Key Differences:**
- ✅ **Native integration:** Optuna has built-in MLflow callback
- ✅ **Automatic tracking:** Logs all trials, params, and metrics automatically
- ✅ **Better organization:** Each trial can be a nested run

---

## Migration Strategy

### Overview

**Approach:** Incremental hybrid migration
**Risk Level:** Low
**Timeline:** 2-3 months
**Effort:** Medium-High

### Phases

#### Phase 1: Stabilization ✅ COMPLETE
- [x] Add scikit-optimize to requirements-base.txt
- [x] Pin dependencies (numpy, scipy, scikit-learn)
- [x] Add Optuna to requirements
- [x] Verify production stability

#### Phase 2: Documentation ✅ COMPLETE
- [x] Create this migration guide
- [x] Document current architecture
- [x] Create API comparison
- [x] Provide code examples

#### Phase 3: Incremental Migration (Future)

**Order of Migration (Easiest → Hardest):**

1. **Logistic Regression** (Week 1-2)
   - Simplest model
   - Good learning opportunity
   - Low risk

2. **MLP** (Week 3-4)
   - Similar complexity
   - Tests neural network parameters

3. **XGBoost (Classification)** (Week 5-6)
   - More complex hyperparameters
   - Important production model

4. **ARIMA** (Week 7-8)
   - Time series introduction
   - Different evaluation metrics

5. **XGBoost (Time Series)** (Week 9-10)
   - Combines TS + complex params

6. **LSTM** (Week 11-12)
   - Most complex
   - Largest codebase

### Per-Model Migration Steps

For each model:

1. **Preparation** (Day 1)
   - Read current implementation
   - Identify all scikit-optimize usage
   - Create feature branch

2. **Implementation** (Day 2-3)
   - Create Optuna objective function
   - Migrate search space (reuse dict!)
   - Migrate callbacks
   - Update visualization calls

3. **Testing** (Day 4-5)
   - Unit tests for objective function
   - Integration tests with real data
   - Compare results with scikit-optimize

4. **A/B Testing** (Day 6-10)
   - Run both implementations in parallel
   - Compare optimization quality
   - Validate convergence

5. **Deployment** (Day 11-12)
   - Code review
   - Merge to main
   - Monitor production metrics

6. **Monitoring** (Day 13-14)
   - Watch for errors
   - Compare performance
   - Document lessons learned

---

## Code Migration Examples

### Example 1: Complete Logistic Regression Migration

#### Before (scikit-optimize)

```python
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from skopt.callbacks import DeadlineStopper, DeltaYStopper

def train_logistic_with_bayesian(
    X_train, y_train, X_val, y_val,
    bayesian_search_params: dict,
    bayesian_config: dict,
    n_bayesian_iterations: int
):
    # Create search space
    dimensions, param_names = create_bayesian_search_space(bayesian_search_params)

    # Initialize optimizer
    optimizer = Optimizer(
        dimensions=dimensions,
        n_initial_points=bayesian_config.get("n_initial_points", 10),
        acq_func=bayesian_config.get("acq_func", "EI"),
        random_state=SEED
    )

    # Create callbacks
    callbacks = create_bayesian_callbacks(bayesian_config, experiment_dir)

    best_score = 0.0
    best_model = None
    best_params = None

    # Ask/tell loop
    for i in range(n_bayesian_iterations):
        # Get next point
        next_point = optimizer.ask()
        params_dict = {param_names[j]: next_point[j] for j in range(len(param_names))}

        # Train model
        model = LogisticRegression(**params_dict, random_state=SEED)
        model.fit(X_train, y_train)

        # Evaluate
        val_score = accuracy_score(y_val, model.predict(X_val))
        objective_value = -val_score  # Minimize negative

        # Tell optimizer
        optimizer.tell(next_point, objective_value)

        # Update best
        if val_score > best_score:
            best_score = val_score
            best_model = model
            best_params = params_dict

        # Check callbacks
        for cb in callbacks:
            if cb(optimizer) == True:
                break

    # Generate visualizations
    artifacts = generate_bayesian_visualizations(optimizer, experiment_dir, "logistic")

    return best_model, best_params, best_score
```

#### After (Optuna)

```python
import optuna
from optuna.samplers import TPESampler

def train_logistic_with_optuna(
    X_train, y_train, X_val, y_val,
    bayesian_search_params: dict,  # Same format! No changes needed
    bayesian_config: dict,
    n_bayesian_iterations: int
):
    # Create objective function
    def objective(trial):
        params = {}

        # Suggest parameters (reuse existing space config!)
        for param_name, param_spec in bayesian_search_params.items():
            param_type = param_spec.get("type")

            if param_type == "real":
                distribution = param_spec.get("distribution", "uniform")
                low = param_spec["low"]
                high = param_spec["high"]
                log = (distribution == "log-uniform")
                params[param_name] = trial.suggest_float(param_name, low, high, log=log)

            elif param_type == "integer":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_spec["low"],
                    param_spec["high"]
                )

            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_spec["choices"]
                )

        # Train model
        params["random_state"] = SEED
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Evaluate (return positive score - Optuna maximizes)
        val_score = accuracy_score(y_val, model.predict(X_val))

        # Optional: Log to MLflow
        mlflow.log_metric(f"trial_{trial.number}_accuracy", val_score, step=trial.number)

        return val_score

    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(
            n_startup_trials=bayesian_config.get("n_initial_points", 10),
            seed=SEED
        )
    )

    # Create callbacks
    callbacks = create_optuna_callbacks(bayesian_config)

    # Optimize
    study.optimize(objective, n_trials=n_bayesian_iterations, callbacks=callbacks)

    # Get best results
    best_params = study.best_params
    best_score = study.best_value

    # Retrain best model
    best_params["random_state"] = SEED
    best_model = LogisticRegression(**best_params)
    best_model.fit(X_train, y_train)

    # Generate visualizations
    artifacts = generate_optuna_visualizations(study, experiment_dir, "logistic")

    return best_model, best_params, best_score
```

**Lines of Code:**
- Before: ~60 lines
- After: ~70 lines (slightly more due to explicit param conversion)

**Complexity:**
- Before: Manual loop management, callback checking
- After: Cleaner flow, built-in tracking

---

### Example 2: Reusable Helper Functions

Create these helper functions to DRY (Don't Repeat Yourself):

```python
# File: DREAM-ML-backend/GEML/optuna_utils.py

import optuna
from optuna.samplers import TPESampler
import time

def create_optuna_objective_from_space(
    space_config: dict,
    model_class,
    X_train, y_train, X_val, y_val,
    metric_fn,
    fixed_params: dict = None,
    direction: str = "maximize"
):
    """
    Generic objective function creator.
    Works with existing space config dicts!

    Args:
        space_config: Dict with param specs (same format as scikit-optimize)
        model_class: Model class to instantiate
        X_train, y_train: Training data
        X_val, y_val: Validation data
        metric_fn: Function to calculate score (e.g., accuracy_score)
        fixed_params: Dict of fixed parameters to always include
        direction: "maximize" or "minimize"

    Returns:
        Objective function for Optuna
    """
    fixed_params = fixed_params or {}

    def objective(trial):
        params = {}

        # Suggest parameters from space config
        for param_name, param_spec in space_config.items():
            param_type = param_spec.get("type")

            if param_type == "real":
                distribution = param_spec.get("distribution", "uniform")
                low = param_spec["low"]
                high = param_spec["high"]
                log = (distribution == "log-uniform")
                params[param_name] = trial.suggest_float(param_name, low, high, log=log)

            elif param_type == "integer":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_spec["low"],
                    param_spec["high"]
                )

            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_spec["choices"]
                )

        # Add fixed params
        params.update(fixed_params)

        try:
            # Train model
            model = model_class(**params)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_val)
            score = metric_fn(y_val, y_pred)

            return score

        except Exception as e:
            # Log failed trials
            logger.warning(f"Trial {trial.number} failed: {str(e)}")
            # Return worst possible value
            return float('-inf') if direction == "maximize" else float('inf')

    return objective


class TimeoutCallback:
    """Timeout callback for Optuna"""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()

    def __call__(self, study, trial):
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            logger.info(f"Timeout after {elapsed:.1f}s")
            study.stop()


class ConvergenceCallback:
    """Convergence callback for Optuna (equivalent to DeltaYStopper)"""
    def __init__(self, tolerance=0.001, patience=5):
        self.tolerance = tolerance
        self.patience = patience
        self.no_improvement_count = 0
        self.best_value = None

    def __call__(self, study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        if self.best_value is None:
            self.best_value = study.best_value
            return

        improvement = abs(study.best_value - self.best_value)

        if improvement < self.tolerance:
            self.no_improvement_count += 1
            logger.debug(f"No improvement for {self.no_improvement_count}/{self.patience} trials")
        else:
            self.no_improvement_count = 0
            self.best_value = study.best_value

        if self.no_improvement_count >= self.patience:
            logger.info(f"Converged after {self.patience} trials without improvement > {self.tolerance}")
            study.stop()


def run_optuna_optimization(
    objective_fn,
    n_trials: int,
    direction: str = "maximize",
    n_startup_trials: int = 10,
    timeout_seconds: int = None,
    convergence_tolerance: float = 0.001,
    convergence_patience: int = 5,
    random_state: int = 42
):
    """
    Run Optuna optimization with standard settings.

    Returns:
        study: Completed Optuna study object
    """
    # Create study
    study = optuna.create_study(
        direction=direction,
        sampler=TPESampler(
            n_startup_trials=n_startup_trials,
            seed=random_state
        )
    )

    # Create callbacks
    callbacks = []
    if timeout_seconds is not None:
        callbacks.append(TimeoutCallback(timeout_seconds))
    callbacks.append(ConvergenceCallback(convergence_tolerance, convergence_patience))

    # Optimize
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        callbacks=callbacks,
        show_progress_bar=False  # Set True for local development
    )

    return study
```

**Usage:**

```python
from optuna_utils import (
    create_optuna_objective_from_space,
    run_optuna_optimization
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Use existing space config (no changes!)
space_config = get_default_bayesian_space_logistic()

# Create objective
objective = create_optuna_objective_from_space(
    space_config=space_config,
    model_class=LogisticRegression,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    metric_fn=accuracy_score,
    fixed_params={"random_state": SEED},
    direction="maximize"
)

# Run optimization
study = run_optuna_optimization(
    objective_fn=objective,
    n_trials=n_bayesian_iterations,
    n_startup_trials=bayesian_config.get("n_initial_points", 10),
    timeout_seconds=bayesian_config.get("timeout_seconds"),
    convergence_tolerance=bayesian_config.get("convergence_tolerance", 0.001),
    convergence_patience=bayesian_config.get("convergence_patience", 5),
    random_state=SEED
)

# Get results
best_params = study.best_params
best_score = study.best_value
```

---

## Testing and Validation

### Test Plan for Each Migration

#### 1. Unit Tests

```python
# tests/test_optuna_migration.py

import pytest
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_optuna_objective_creation():
    """Test that Optuna objective function is created correctly"""
    # Create dummy data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Space config
    space_config = {
        "C": {"type": "real", "distribution": "log-uniform", "low": 0.001, "high": 100.0},
        "max_iter": {"type": "integer", "low": 100, "high": 1000}
    }

    # Create objective
    objective = create_optuna_objective_from_space(
        space_config=space_config,
        model_class=LogisticRegression,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        metric_fn=accuracy_score,
        fixed_params={"random_state": 42}
    )

    # Test that it runs
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    assert study.best_value > 0
    assert "C" in study.best_params
    assert "max_iter" in study.best_params


def test_optuna_vs_skopt_same_search_space():
    """Test that Optuna explores the same space as scikit-optimize"""
    space_config = get_default_bayesian_space_logistic()

    # Check all params are convertible
    for param_name, param_spec in space_config.items():
        param_type = param_spec["type"]
        assert param_type in ["real", "integer", "categorical"]

        if param_type == "real":
            assert "low" in param_spec and "high" in param_spec
        elif param_type == "integer":
            assert "low" in param_spec and "high" in param_spec
        elif param_type == "categorical":
            assert "choices" in param_spec
            assert len(param_spec["choices"]) > 0


def test_callbacks_work():
    """Test that timeout and convergence callbacks work"""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    space_config = {"C": {"type": "real", "low": 0.1, "high": 1.0}}

    objective = create_optuna_objective_from_space(
        space_config, LogisticRegression,
        X_train, y_train, X_val, y_val,
        accuracy_score, {"random_state": 42}
    )

    # Test timeout
    start = time.time()
    study = run_optuna_optimization(
        objective, n_trials=1000, timeout_seconds=2
    )
    elapsed = time.time() - start

    assert elapsed < 5  # Should stop around 2 seconds
    assert len(study.trials) < 1000  # Should not complete all trials
```

#### 2. Integration Tests

```python
def test_full_logistic_migration():
    """Test complete migration for Logistic Regression"""
    # Load test dataset
    df = pd.read_csv("test_data/classification_dataset.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Config
    space_config = get_default_bayesian_space_logistic()
    n_trials = 20

    # Run scikit-optimize
    skopt_model, skopt_params, skopt_score = train_logistic_with_bayesian_skopt(
        X_train, y_train, X_val, y_val,
        space_config, {}, n_trials
    )

    # Run Optuna
    optuna_model, optuna_params, optuna_score = train_logistic_with_bayesian_optuna(
        X_train, y_train, X_val, y_val,
        space_config, {}, n_trials
    )

    # Compare results
    # Note: Won't be identical due to different algorithms, but should be close
    assert abs(skopt_score - optuna_score) < 0.1  # Within 10%

    # Test set performance should be similar
    skopt_test_score = accuracy_score(y_test, skopt_model.predict(X_test))
    optuna_test_score = accuracy_score(y_test, optuna_model.predict(X_test))

    assert abs(skopt_test_score - optuna_test_score) < 0.1
```

#### 3. A/B Testing in Production

```python
def ab_test_optimization_methods(
    X_train, y_train, X_val, y_val,
    space_config: dict,
    n_trials: int,
    n_repetitions: int = 5
):
    """
    Run both methods multiple times and compare.

    Returns:
        DataFrame with comparison results
    """
    results = []

    for i in range(n_repetitions):
        seed = 42 + i

        # scikit-optimize
        start = time.time()
        _, _, skopt_score = train_with_skopt(
            X_train, y_train, X_val, y_val,
            space_config, {"random_state": seed}, n_trials
        )
        skopt_time = time.time() - start

        # Optuna
        start = time.time()
        _, _, optuna_score = train_with_optuna(
            X_train, y_train, X_val, y_val,
            space_config, {"random_state": seed}, n_trials
        )
        optuna_time = time.time() - start

        results.append({
            "repetition": i,
            "skopt_score": skopt_score,
            "skopt_time": skopt_time,
            "optuna_score": optuna_score,
            "optuna_time": optuna_time,
            "score_diff": optuna_score - skopt_score,
            "time_diff": optuna_time - skopt_time
        })

    df = pd.DataFrame(results)

    print("=" * 60)
    print("A/B Test Results")
    print("=" * 60)
    print(f"scikit-optimize avg score: {df['skopt_score'].mean():.4f} ± {df['skopt_score'].std():.4f}")
    print(f"Optuna avg score:          {df['optuna_score'].mean():.4f} ± {df['optuna_score'].std():.4f}")
    print(f"scikit-optimize avg time:  {df['skopt_time'].mean():.2f}s ± {df['skopt_time'].std():.2f}s")
    print(f"Optuna avg time:           {df['optuna_time'].mean():.2f}s ± {df['optuna_time'].std():.2f}s")
    print("=" * 60)

    return df
```

### Validation Checklist

Before deploying each migrated model:

- [ ] Unit tests pass (objective creation, callbacks, search space)
- [ ] Integration tests pass (full training pipeline)
- [ ] A/B test shows comparable or better performance
- [ ] Optimization converges in similar number of trials
- [ ] Visualizations are generated correctly
- [ ] MLflow logging works correctly
- [ ] No memory leaks (monitor memory usage)
- [ ] Error handling works (failed trials don't crash)
- [ ] Code review completed
- [ ] Documentation updated

---

## Rollback Procedures

### If Migration Fails

Each migration is isolated to one model type. If issues occur:

#### 1. Immediate Rollback (< 5 minutes)

```bash
# Revert to previous commit
git revert <migration_commit_hash>
git push origin main

# Or cherry-pick the working version
git checkout <previous_working_commit> -- api/train.py
git commit -m "Rollback Logistic Regression to scikit-optimize"
git push origin main
```

#### 2. Feature Flag Rollback

Add feature flags to switch between implementations:

```python
# In config or environment variables
USE_OPTUNA_LOGISTIC = os.getenv("USE_OPTUNA_LOGISTIC", "false").lower() == "true"

def train_logistic_bayesian(...):
    if USE_OPTUNA_LOGISTIC:
        return train_logistic_with_optuna(...)
    else:
        return train_logistic_with_skopt(...)
```

Then toggle without code deployment:

```bash
# Disable Optuna
export USE_OPTUNA_LOGISTIC=false

# Re-enable after fix
export USE_OPTUNA_LOGISTIC=true
```

#### 3. Gradual Rollout

Use percentage-based rollout:

```python
import random

def should_use_optuna(rollout_percentage: int = 0):
    """Return True with probability = rollout_percentage"""
    return random.randint(1, 100) <= rollout_percentage

# Start with 10% of traffic
if should_use_optuna(rollout_percentage=10):
    result = train_with_optuna(...)
else:
    result = train_with_skopt(...)
```

Gradually increase: 10% → 25% → 50% → 75% → 100%

### Monitoring During Migration

Track these metrics for each migrated model:

```python
# Log comparison metrics
mlflow.log_metric("optimization_method", 1 if using_optuna else 0)
mlflow.log_metric("optimization_time_seconds", elapsed_time)
mlflow.log_metric("trials_completed", n_trials_completed)
mlflow.log_metric("best_score_found", best_score)
mlflow.log_metric("convergence_iteration", convergence_iter)
```

Set up alerts for:
- Significantly worse scores than historical average
- Optimization taking >2x longer than usual
- High failure rate (>10% of trials failing)
- Memory usage spike

---

## Timeline and Milestones

### Recommended Schedule

| Week | Model | Status | Estimated Effort |
|------|-------|--------|------------------|
| 0 | Setup & Planning | ✅ Complete | 2 hours |
| 1-2 | Logistic Regression | 🔜 Next | 16 hours |
| 3-4 | MLP | ⏳ Pending | 16 hours |
| 5-6 | XGBoost (CL) | ⏳ Pending | 20 hours |
| 7-8 | ARIMA | ⏳ Pending | 20 hours |
| 9-10 | XGBoost (TS) | ⏳ Pending | 20 hours |
| 11-12 | LSTM | ⏳ Pending | 24 hours |
| 13-14 | Cleanup & Removal | ⏳ Pending | 8 hours |

**Total Estimated Effort:** ~126 hours (3-4 weeks of full-time work, or 3 months part-time)

### Milestones

#### Milestone 1: First Model Migrated ✅ (Week 2)
- Logistic Regression fully migrated
- All tests passing
- Documentation updated
- Team trained on Optuna basics

#### Milestone 2: Classification Models Complete (Week 6)
- All 3 classification models migrated
- Patterns established
- Reusable utilities created

#### Milestone 3: Time Series Models Complete (Week 12)
- All 6 models migrated
- Both modules updated
- Performance validated

#### Milestone 4: Cleanup Complete (Week 14)
- scikit-optimize removed from codebase
- All references updated
- Final documentation
- Retrospective completed

---

## Resources and References

### Optuna Documentation
- [Official Documentation](https://optuna.readthedocs.io/)
- [Tutorial](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
- [API Reference](https://optuna.readthedocs.io/en/stable/reference/index.html)
- [Examples Gallery](https://github.com/optuna/optuna-examples)

### Key Concepts
- [TPE Algorithm](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [Pruning Algorithms](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)
- [Visualization](https://optuna.readthedocs.io/en/stable/reference/visualization/index.html)
- [MLflow Integration](https://optuna.readthedocs.io/en/stable/reference/integration.html#mlflow)

### scikit-optimize (Legacy Reference)
- [Archived Repository](https://github.com/scikit-optimize/scikit-optimize)
- [Documentation](https://scikit-optimize.github.io/)

### Bayesian Optimization Theory
- [Practical Bayesian Optimization (Snoek et al.)](https://arxiv.org/abs/1206.2944)
- [Tree-structured Parzen Estimator (Bergstra et al.)](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization)

### Comparison Articles
- [Optuna vs Hyperopt](https://neptune.ai/blog/optuna-vs-hyperopt)
- [Top 10 Hyperparameter Optimization Tools](https://www.activestate.com/blog/top-10-tools-for-hyperparameter-optimization-in-python/)

---

## Appendix A: Quick Reference

### API Translation Table

| scikit-optimize | Optuna Equivalent | Notes |
|----------------|-------------------|-------|
| `Optimizer()` | `optuna.create_study()` | Study = Optimizer |
| `optimizer.ask()` | Inside `trial.suggest_*()` | Automatic in objective |
| `optimizer.tell(x, y)` | Return value from objective | Automatic |
| `Real(low, high, prior="log-uniform")` | `trial.suggest_float(name, low, high, log=True)` | |
| `Real(low, high, prior="uniform")` | `trial.suggest_float(name, low, high)` | |
| `Integer(low, high)` | `trial.suggest_int(name, low, high)` | |
| `Categorical(choices)` | `trial.suggest_categorical(name, choices)` | |
| `TimerStopper(total_time)` | `TimeoutCallback(timeout_seconds)` | Custom class |
| `DeltaYStopper(delta, n_best)` | `ConvergenceCallback(tolerance, patience)` | Custom class |
| `optimizer.func_vals` | `[t.value for t in study.trials]` | Trial history |
| `optimizer.Xi` | `[t.params for t in study.trials]` | Parameter history |
| `expected_minimum(optimizer)` | `study.best_value` | Best found value |
| `optimizer.models[-1]` | `study.sampler` (different interface) | Internal model |

### Common Patterns

```python
# Get all completed trials
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

# Get trial values
values = [t.value for t in completed_trials]

# Get trial parameters
params = [t.params for t in completed_trials]

# Get best trial
best_trial = study.best_trial
print(f"Best score: {best_trial.value}")
print(f"Best params: {best_trial.params}")

# Get trial number
trial_number = best_trial.number

# Get optimization history
history_df = study.trials_dataframe()
```

---

## Appendix B: Installation Instructions

### Install Optuna

```bash
# Basic installation
pip install optuna>=3.0.0

# With visualization dependencies
pip install optuna[visualization]

# Or in requirements-base.txt (already done!)
optuna>=3.0.0
plotly>=5.0.0  # For visualizations
kaleido>=0.2.0  # For saving plots as images
```

### Verify Installation

```python
import optuna
print(f"Optuna version: {optuna.__version__}")

# Test basic functionality
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")
# Should find x ≈ 2
```

---

## Appendix C: Troubleshooting

### Common Issues

#### Issue 1: "Trial pruned" errors

**Symptom:** Many trials show as PRUNED
**Cause:** Pruner is too aggressive
**Solution:** Disable pruning or use less aggressive pruner

```python
# Disable pruning
study = optuna.create_study(
    sampler=TPESampler(n_startup_trials=10),
    # Don't specify pruner
)
```

#### Issue 2: Optimization is slow

**Symptom:** Optuna takes longer than scikit-optimize
**Cause:** TPE overhead, or objective function is slow
**Solution:** Profile objective function, reduce n_startup_trials

```python
# Profile objective
import cProfile
cProfile.run('study.optimize(objective, n_trials=10)')

# Reduce startup trials for faster initial iterations
sampler = TPESampler(n_startup_trials=5)  # Default is 10
```

#### Issue 3: Memory usage increasing

**Symptom:** Memory grows with number of trials
**Cause:** Study stores all trial history
**Solution:** Use storage with automatic cleanup, or manual deletion

```python
# Use in-memory storage with limits
study = optuna.create_study(
    storage="sqlite:///optuna.db",  # Persistent storage
    load_if_exists=True
)

# Or manually prune old trials (keep last 100)
if len(study.trials) > 100:
    study.trials = study.trials[-100:]
```

#### Issue 4: Different results than scikit-optimize

**Symptom:** Optuna finds different optima
**Cause:** TPE vs GP algorithms explore differently
**Solution:** This is expected! Validate that both find good solutions

```python
# Compare both methods
skopt_score = run_with_skopt()
optuna_score = run_with_optuna()

# Both should be within acceptable range
assert optuna_score >= acceptable_threshold
assert skopt_score >= acceptable_threshold
```

---

## Document Change Log

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-10-28 | Initial creation | Claude |

---

**End of Migration Plan**

For questions or issues during migration, refer to:
- Optuna GitHub Issues: https://github.com/optuna/optuna/issues
- Optuna Discord: https://discord.gg/jgBwjFr
- Internal team documentation
