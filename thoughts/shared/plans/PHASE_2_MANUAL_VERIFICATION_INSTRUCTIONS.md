# Phase 2: Manual Verification Instructions
**Bayesian Search Reproducibility Testing for ARIMA/SARIMAX**

**Date**: 2025-12-22
**Phase**: Phase 2 - ARIMA/SARIMAX Reproducibility Testing
**Implementation Status**: Core implementation complete, manual testing required

---

## Overview

This document provides step-by-step instructions for manually verifying that Bayesian Search for ARIMA/SARIMAX produces reproducible results through the UI.

## Prerequisites

1. **Backend server running**:
   ```bash
   cd /workspaces/dream-ml-c/DREAM-ML-backend
   python manage.py runserver
   ```

2. **Frontend server running**:
   ```bash
   cd /workspaces/dream-ml-c/DREAM-ML-frontend/frontend
   npm start
   ```

3. **Test dataset**: Use any time series CSV with at least 200 rows

---

## Test 1: Basic Bayesian Search Functionality

**Objective**: Verify that Bayesian Search runs without errors and produces results

### Steps:

1. **Navigate to Time Series Training page** in the frontend

2. **Upload a time series dataset**
   - Use a CSV file with a date column and a numeric target column
   - Minimum 200 rows recommended

3. **Configure training parameters**:
   - **Algorithm**: ARIMA
   - **Target Variable**: Select your numeric column
   - **Date Column**: Select your date column
   - **Split Ratios**: Train=0.7, Val=0.15, Test=0.15
   - **Forecast Horizon**: 10

4. **Select Bayesian Search**:
   - **Optimization Method**: Bayesian Search
   - **Number of Trials (n_trials)**: 10
   - **Initial Points (n_initial_points)**: 3
   - **Optimization Metric**: val_rmse

5. **Start Training**

6. **Monitor Console Output** (backend terminal):
   - Look for "Bayesian Search Configuration" log block
   - Look for "Platform Information (for reproducibility)" log block
   - Verify trial-by-trial progress logs (e.g., "Trial 0: val_rmse=X.XXXX")
   - Look for "Bayesian Search Completed" summary

7. **Verify Success**:
   - ✅ Training completes without errors
   - ✅ Validation and test metrics are displayed
   - ✅ Model is saved successfully

### Expected Console Output:

```
============================================================
Bayesian Search Configuration:
  n_trials: 10
  n_initial_points: 3
  timeout_seconds: None
  optimization_metric: val_rmse
============================================================
============================================================
Platform Information (for reproducibility):
  Python version: 3.11.14
  NumPy version: 1.26.4
  Pandas version: 2.2.3
  Statsmodels version: 0.14.5
  Optuna version: 4.6.0
  Platform: linux
  SEED: 42
============================================================
...
Trial 0: val_rmse=2.5432, params=(1, 1, 1), seasonal=(0, 0, 0, 0)
Trial 1: val_rmse=2.1234, params=(2, 1, 1), seasonal=(0, 0, 0, 0)
...
============================================================
Bayesian Search Completed
  Best val_rmse: 1.8543
  Best parameters: {'p': 2, 'd': 1, 'q': 2, ...}
  Completed trials: 10/10
  Optimization time: 15.43 seconds
============================================================
```

---

## Test 2: Reproducibility Verification

**Objective**: Verify that running the same configuration twice produces identical results

### Steps:

1. **First Run**:
   - Follow all steps from Test 1
   - **IMPORTANT**: Note the exact configuration you used
   - After training completes, find the experiment directory (usually in `mlruns/`)
   - **Save a copy** of `pipeline_config.json` from the experiment directory
   - **Record the best parameters** from the console output

2. **Second Run** (identical configuration):
   - Use the **EXACT SAME** dataset
   - Use the **EXACT SAME** configuration as Run 1
   - **Optimization Method**: Bayesian Search
   - **n_trials**: Same as Run 1
   - **n_initial_points**: Same as Run 1
   - All other settings identical

3. **Compare Results**:
   - Compare `pipeline_config.json` from both runs
   - **Check these fields** in `bayesian_config.best_params`:
     - `order` (p, d, q) - **Must be identical**
     - `seasonal_order` - **Must be identical**
     - `trend` - **Must be identical**
     - `enforce_stationarity` - **Must be identical**
     - `enforce_invertibility` - **Must be identical**

   - **Check validation metrics**:
     - `val_rmse` - Should be **nearly identical** (within 1e-6 tolerance)

### Example Comparison:

**Run 1 - pipeline_config.json**:
```json
{
  "steps": [{
    "bayesian_config": {
      "best_params": {
        "order": [2, 1, 2],
        "seasonal_order": [0, 0, 0, 0],
        "trend": "ct",
        "enforce_stationarity": true,
        "enforce_invertibility": true
      },
      "n_trials": 10,
      "n_completed_trials": 10,
      "optimization_time_seconds": 15.43
    }
  }]
}
```

**Run 2 - pipeline_config.json**:
```json
{
  "steps": [{
    "bayesian_config": {
      "best_params": {
        "order": [2, 1, 2],  ✅ SAME
        "seasonal_order": [0, 0, 0, 0],  ✅ SAME
        "trend": "ct",  ✅ SAME
        "enforce_stationarity": true,  ✅ SAME
        "enforce_invertibility": true  ✅ SAME
      },
      "n_trials": 10,
      "n_completed_trials": 10,
      "optimization_time_seconds": 15.21  ✅ Similar (minor variance OK)
    }
  }]
}
```

### Success Criteria:
- ✅ `best_params` are **identical** between runs
- ✅ `val_rmse` differs by less than 0.000001 (1e-6)
- ✅ Both runs complete successfully
- ✅ Platform information is logged in both runs

---

## Test 3: Seasonal Parameter Optimization

**Objective**: Verify that Bayesian Search can optimize seasonal SARIMAX parameters

### Steps:

1. **Configure Training** (same as Test 1, but with manual seasonal params):
   - **Algorithm**: ARIMA
   - **Optimization Method**: Bayesian Search
   - **n_trials**: 8
   - **n_initial_points**: 3

2. **Enable Seasonal Parameters** (via manual params):
   - In the frontend, look for "Manual Parameters" section
   - Add these seasonal parameters:
     ```json
     {
       "seasonal_P": 1,
       "seasonal_D": 1,
       "seasonal_Q": 1,
       "seasonal_s": 30
     }
     ```
   - This tells the optimizer to search for seasonal parameters with period=30

3. **Start Training**

4. **Verify Console Output**:
   - Look for: "Seasonal parameters detected: P=1, D=1, Q=1, s=30"
   - Look for: "Bayesian Search will optimize seasonal parameters"
   - Verify trial logs show seasonal orders like `(P, D, Q, 30)`

5. **Check Results**:
   - Open `pipeline_config.json`
   - Verify `best_params.seasonal_order` is NOT `[0, 0, 0, 0]`
   - Should be something like `[1, 1, 1, 30]`

### Expected Seasonal Output:

```
Seasonal parameters detected: P=1, D=1, Q=1, s=30
Bayesian Search will optimize seasonal parameters
...
Trial 0: val_rmse=2.1543, params=(1, 1, 1), seasonal=(1, 1, 1, 30)
Trial 1: val_rmse=1.9234, params=(2, 1, 2), seasonal=(0, 1, 1, 30)
...
```

---

## Test 4: Pipeline Config Completeness

**Objective**: Verify that `pipeline_config.json` contains all reproducibility metadata

### Steps:

1. **After any Bayesian Search training**, locate the `pipeline_config.json` file

2. **Verify Structure**:
   ```bash
   cat <experiment_dir>/pipeline_config.json | python -m json.tool
   ```

3. **Check Required Fields**:

   **Top Level**:
   - ✅ `steps` (array)

   **Within `steps[0]`** (latest step):
   - ✅ `hyperparameter_search_strategy`: "bayesian"
   - ✅ `bayesian_config` (object)

   **Within `bayesian_config`**:
   - ✅ `n_trials`
   - ✅ `n_initial_points`
   - ✅ `timeout_seconds`
   - ✅ `optimization_metric`
   - ✅ `optimization_time_seconds`
   - ✅ `best_trial_number`
   - ✅ `n_completed_trials`
   - ✅ `best_params` (object with `order`, `seasonal_order`, `trend`, etc.)

### Example Complete Config:

```json
{
  "steps": [{
    "step": "train_arima",
    "hyperparameter_search_strategy": "bayesian",
    "bayesian_config": {
      "n_trials": 10,
      "n_initial_points": 3,
      "timeout_seconds": null,
      "optimization_metric": "val_rmse",
      "optimization_time_seconds": 15.43,
      "best_trial_number": 7,
      "n_completed_trials": 10,
      "best_params": {
        "order": [2, 1, 2],
        "seasonal_order": [0, 0, 0, 0],
        "trend": "ct",
        "enforce_stationarity": true,
        "enforce_invertibility": true
      }
    },
    "val_metrics": {
      "val_rmse": 1.8543,
      "val_mae": 1.4321,
      "val_mape": 5.67
    },
    "test_metrics": {
      "test_rmse": 1.9012,
      "test_mae": 1.5234,
      "test_mape": 6.12
    }
  }]
}
```

---

## Test 5: Platform Info Logging

**Objective**: Verify that platform information is logged for debugging reproducibility issues

### Steps:

1. **Run any Bayesian Search training**

2. **Check backend console logs** for the "Platform Information" block

3. **Verify all fields are present**:
   - ✅ Python version
   - ✅ NumPy version
   - ✅ Pandas version
   - ✅ Statsmodels version
   - ✅ Optuna version
   - ✅ Platform (linux/darwin/win32)
   - ✅ SEED value

4. **Verify values match your environment**:
   ```bash
   python --version  # Should match logged Python version
   python -c "import numpy; print(numpy.__version__)"  # Should be 1.26.4
   ```

---

## Troubleshooting

### Issue: Training fails with "n_trials must be at least 1"
**Solution**: Set `n_trials` to a value >= 1 (recommended: 10-50)

### Issue: Training fails with "n_initial_points must be less than n_trials"
**Solution**: Ensure `n_initial_points` < `n_trials`. Common: n_initial_points=10, n_trials=50

### Issue: Results are not reproducible
**Check**:
1. Same dataset (byte-for-byte identical CSV)
2. Same configuration (all parameters identical)
3. Same random seed (SEED=42, should be automatic)
4. Platform information matches between runs

### Issue: Seasonal parameters not optimized
**Solution**: Add manual seasonal parameters as shown in Test 3

---

## Success Criteria Summary

After completing all tests, you should be able to confirm:

- [ ] **Test 1**: Bayesian Search runs successfully with ARIMA
- [ ] **Test 2**: Same configuration produces identical `best_params`
- [ ] **Test 3**: Seasonal parameters are optimized when provided
- [ ] **Test 4**: `pipeline_config.json` contains all required fields
- [ ] **Test 5**: Platform information is logged in console

---

## Reporting Results

Please report your test results with:

1. **Test outcomes** (pass/fail for each test)
2. **Screenshots** of successful training completion
3. **Comparison** of `pipeline_config.json` files from Test 2
4. **Any errors or unexpected behavior**
5. **Platform information** from your system

---

**Implementation Date**: 2025-12-22
**Phase 2 Status**: Awaiting manual verification
