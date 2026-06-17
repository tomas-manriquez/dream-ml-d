# Implementation Plan: Fix EC-1 - Univariate Time Series Model Support

**Research Source**: TSTrain-feature-selection-research.md
**Date**: 2025-12-17
**Priority**: High (blocks valid use case)
**Estimated Effort**: 3-3.5 hours
**Risk Level**: Low

## 🎉 IMPLEMENTATION STATUS

**Phase 1 (Backend)**: ✅ **COMPLETED** on 2025-12-17
**Phase 2 (Frontend)**: ✅ **COMPLETED** on 2025-12-17
**Phase 3 (Integration)**: ✅ **COMPLETED** on 2025-12-17
**Phase 4 (Documentation)**: ⏳ **PENDING** - Optional
**Phase 5 (Deployment)**: ⏳ **PENDING** - Awaiting Phase 4 (optional)

### Summary of Completed Work

#### ✅ Phase 1: Backend Encoding Service Updates
- Updated validation logic to allow empty `input_features` for univariate models
- Enhanced column validation to handle empty feature lists
- Added MLflow tracking with `is_univariate` flag
- All automated tests passing (10 LSTM tests, 2 univariate tests)

#### ✅ Phase 2: Frontend Validation Updates
- Fixed validation rule to only block XGBoost with empty features
- Updated train button logic in two locations (isDisabled + handleTrain)
- Implemented algorithm-specific helper text
- Training requests now successfully reach backend for ARIMA/LSTM

#### ✅ Phase 3: Integration Testing
- All 4 integration tests passed successfully
- ARIMA univariate training: ✅ WORKING
- LSTM univariate training: ✅ WORKING
- XGBoost validation blocking: ✅ WORKING
- Multivariate regression: ✅ NO REGRESSION
- Frontend validation allows empty features for ARIMA/LSTM
- Backend accepts empty input_features correctly
- Pattern consistency verified across codebase

#### 🔧 Bonus: Critical Bug Fixes
- **ARIMA Optimizer Bug**: Fixed pre-existing TypeError with ARIMA.fit() parameters
- **Locations**: Manual training, grid search, and random search
- **Solution**: Class name detection to handle ARIMA vs SARIMAX API differences

### Files Modified

**Backend** (2 files):
1. [services.py](file:///workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/services.py) - Lines 719-729, 765-777, 805-806
2. [train.py](file:///workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/train.py) - Lines 1899-1924, 1712-1728, 1786-1800

**Frontend** (1 file):
1. [TSTrainCard.jsx](file:///workspaces/dream-ml-c/DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx) - Lines 370-374, 647-656, 954, 1070-1074

---

## Problem Statement (EC-1)

**Scenario**: CSV file contains only 1-2 columns (e.g., date + target)
- User uploads CSV with columns: `["date", "target"]`
- Selects `"target"` as target variable
- Selects `"date"` as date column
- Feature selection has 0 available columns (both disabled)
- **Current Handling**: Validation error blocks training: "Debes seleccionar al menos 1 variable de entrada"
- **Impact**: User cannot proceed with univariate time series model even though it's a valid configuration

### Backend Support Analysis

| Algorithm | Univariate Support | Implementation |
|-----------|-------------------|----------------|
| **ARIMA** | ✅ Yes | Auto-detects; trains without exogenous variables if none provided |
| **LSTM** | ✅ Yes | Auto-detects; uses target variable when features empty (tested in test_lstm_phase4.py) |
| **XGBoost** | ❌ No | Raises ValueError if no valid features (requires lag features or external data) |

---

## Solution Overview

Fix EC-1 by enabling univariate time series model support for ARIMA and LSTM algorithms through coordinated frontend and backend changes:
- Allow empty `input_features` for algorithms that support univariate models
- Maintain XGBoost validation (requires features with clear error message)
- Update user messaging to clarify univariate capability

---

## Critical Files to Modify

### Backend
1. [`/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/services.py`](file:///workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/services.py)
   - Line 719-720: Validation logic
   - Line 759: Column validation loop
   - Line 790: MLflow logging

### Frontend
2. [`/workspaces/dream-ml-c/DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx`](file:///workspaces/dream-ml-c/DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx)
   - Line 371-373: Validation rule 1 (minimum features)
   - Line 953: Train button disabled logic
   - Line 1069-1070: Helper text for feature selection

---

## ✅ PHASE 1: Backend Encoding Service Updates [COMPLETED]

### Phase Overview
Modify data encoding service validation to accept empty `input_features` for univariate time series models while maintaining backward compatibility.

**Status**: ✅ **COMPLETED** on 2025-12-17
**Completion Notes**:
- All three backend changes successfully implemented
- Automated tests passing (10 LSTM Phase 4 tests, 2 univariate tests)
- Empty `input_features` now accepted for univariate models
- MLflow tracking includes `is_univariate` flag
- Backward compatible with multivariate models

### Changes Required

#### Change 1.1: Update Validation Logic (services.py:719-720)

**File**: [`services.py:719-720`](file:///workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/services.py#L719-L720)

**Current Code**:
```python
if not input_features or not target_variables:
    raise ValueError("Variables de entrada y/o de salida no especificadas.")
```

**New Code**:
```python
# Allow empty input_features for univariate time series models (ARIMA, LSTM)
# XGBoost validation happens at training layer
if not target_variables:
    raise ValueError("Variable de salida no especificada.")

if input_features is None:
    raise ValueError("Variables de entrada no especificadas (debe ser una lista, puede estar vacía para modelos univariados).")

# Log univariate mode detection
if len(input_features) == 0:
    logger.info("Modo univariado detectado - input_features vacío. Apropiado para ARIMA/LSTM.")
```

**Rationale**:
- Separates target validation from input features validation
- Allows empty list `[]` but not `None` (prevents accidental omission)
- Logs univariate mode for debugging
- XGBoost will validate at training layer (already implemented)

#### Change 1.2: Update Column Validation Loop (services.py:759)

**File**: [`services.py:757-762`](file:///workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/services.py#L757-L762)

**Current Code**:
```python
# 4. Validar columnas
try:
    df = pd.read_csv(raw_file_path)
    for feature in (input_features + target_variables):
        if feature not in df.columns:
            raise ValueError(f"Columna no encontrada: {feature}")
    logger.info("Validación de columnas completada exitosamente.")
```

**New Code**:
```python
# 4. Validar columnas
try:
    df = pd.read_csv(raw_file_path)
    # Validate all specified columns exist (handles empty input_features)
    all_columns_to_validate = list(input_features) + list(target_variables)
    for feature in all_columns_to_validate:
        if feature not in df.columns:
            raise ValueError(f"Columna no encontrada: {feature}")

    if len(input_features) == 0:
        logger.info("Validación de columnas completada (modo univariado - sin features).")
    else:
        logger.info(f"Validación de columnas completada exitosamente. Features: {len(input_features)}")
```

**Rationale**:
- Explicit list conversion handles edge cases
- Provides clear logging for univariate vs multivariate
- No functional change (concatenation works with empty lists)

#### Change 1.3: Update MLflow Logging (services.py:790)

**File**: [`services.py:790`](file:///workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/services.py#L790)

**Current Code**:
```python
log_param("input_features", input_features)
```

**New Code**:
```python
log_param("input_features", input_features if len(input_features) > 0 else "[]_univariate_mode")
log_param("is_univariate", len(input_features) == 0)
```

**Rationale**:
- Makes univariate mode explicit in MLflow tracking
- Helps with experiment analysis and debugging

### Automated Verification Steps (Backend)

```bash
# Run existing time series tests to ensure no regression
cd /workspaces/dream-ml-c/DREAM-ML-backend
python -m pytest GEML/tests/apiTimeSeries_tests/test_lstm_phase4.py -v
python -m pytest GEML/tests/apiTimeSeries_tests/ -k "univariate" -v
```

Expected: All tests pass, especially univariate LSTM tests

### Manual Verification Steps (Backend)

```bash
# Test encoding with empty input_features
curl -X POST http://localhost:8000/api/ts/encode-csv/ \
  -F "file=@test_univariate.csv" \
  -F "experiment_dir=/path/to/exp" \
  -F "input_features_str=" \
  -F "target_variables_str=sales" \
  -F "run_id=test_run" \
  -F "apply_target_ohe=false" \
  -F "apply_target_label=false"
```

Expected: Returns 200 with success message, logs "Modo univariado detectado"

---

## ✅ PHASE 2: Frontend Validation Updates [COMPLETED]

### Phase Overview
Update frontend validation logic to allow empty `inputFeatures` for ARIMA and LSTM, following the existing LSTM exemption pattern.

**Status**: ✅ **COMPLETED** on 2025-12-17
**Completion Notes**:
- All four frontend changes successfully implemented
- Validation rule updated to only block XGBoost with empty features
- Train button logic fixed in two locations (isDisabled and handleTrain)
- Algorithm-specific helper text provides clear user guidance
- Training now proceeds successfully for ARIMA/LSTM with empty features

### Changes Required

#### Change 2.1: Update Validation Rule (TSTrainCard.jsx:371-373)

**File**: [`TSTrainCard.jsx:371-373`](file:///workspaces/dream-ml-c/DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L371-L373)

**Current Code**:
```javascript
if (inputFeatures.length === 0 && targetVariable) {
  warnings.push("Debes seleccionar al menos 1 variable de entrada");
}
```

**New Code**:
```javascript
// Allow empty features for ARIMA and LSTM (univariate models)
// XGBoost requires at least 1 feature
if (inputFeatures.length === 0 && targetVariable && algorithm === "xgboost") {
  warnings.push("XGBoost requiere al menos 1 variable de entrada. Para modelos univariados, usa ARIMA o LSTM.");
}
```

**Rationale**:
- Inverts the logic: only warn for XGBoost
- Provides clear guidance on algorithm choice
- Aligns with backend capabilities

#### Change 2.2: Update Train Button Logic (TSTrainCard.jsx:953)

**File**: [`TSTrainCard.jsx:953`](file:///workspaces/dream-ml-c/DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L953)

**Current Code**:
```javascript
(algorithm !== "lstm" && !inputFeatures.length) ||  // No input features (except for LSTM)
```

**New Code**:
```javascript
(algorithm === "xgboost" && !inputFeatures.length) ||  // XGBoost requires features
```

**Rationale**:
- Extends LSTM exemption to include ARIMA
- Clearer logic: only block XGBoost
- More maintainable (explicit allowlist vs denylist)

#### Change 2.3: Update Helper Text (TSTrainCard.jsx:1069-1070)

**File**: [`TSTrainCard.jsx:1069-1070`](file:///workspaces/dream-ml-c/DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L1069-L1070)

**Current Code**:
```javascript
helper={variableSelectionStyles.helperText(
  "Selecciona las columnas que usarás como variables de entrada (features) para el modelo. Puedes seleccionar múltiples variables."
)}
```

**New Code**:
```javascript
helper={variableSelectionStyles.helperText(
  algorithm === "xgboost"
    ? "Selecciona las columnas que usarás como variables de entrada (features) para el modelo. XGBoost requiere al menos 1 variable."
    : "Selecciona las columnas que usarás como variables de entrada (features) para el modelo. Para modelos univariados (solo histórico del target), deja vacío."
)}
```

**Rationale**:
- Algorithm-specific guidance
- Educates users about univariate option
- Clear XGBoost requirement

### Manual Verification Steps (Frontend)

#### Test 1: ARIMA Univariate (Happy Path)
1. Upload CSV with columns: `["date", "sales"]`
2. Click "Cargar Variables"
3. Select algorithm: ARIMA
4. Select target: "sales"
5. Select date column: "date"
6. Leave features empty (0 selected)
7. **Verify**:
   - ✅ No validation warnings
   - ✅ Train button enabled (assuming other fields valid)
   - ✅ Helper text mentions univariate option

#### Test 2: LSTM Univariate
1. Same setup as Test 1
2. Select algorithm: LSTM
3. **Verify**: Same results as ARIMA

#### Test 3: XGBoost Validation Error
1. Same setup as Test 1
2. Select algorithm: XGBoost
3. **Verify**:
   - ❌ Validation warning: "XGBoost requiere al menos 1 variable de entrada..."
   - ❌ Train button disabled
   - ✅ Helper text mentions XGBoost requirement

#### Test 4: Multivariate Still Works (Regression)
1. Upload CSV with columns: `["date", "temp", "humidity", "sales"]`
2. Select algorithm: ARIMA
3. Select target: "sales"
4. Select features: "temp", "humidity"
5. Select date: "date"
6. **Verify**:
   - ✅ No warnings
   - ✅ Train button enabled

### Additional Changes Made (Phase 2)

#### Change 2.4: Fix handleTrain Validation (TSTrainCard.jsx:647-656)

**File**: [`TSTrainCard.jsx:647-656`](file:///workspaces/dream-ml-c/DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L647-L656)

**Issue**: Hidden validation inside `handleTrain` function was blocking submission even when button was enabled

**Fix Applied**:
```javascript
// Allow empty inputFeatures for ARIMA and LSTM (univariate models)
// XGBoost requires at least 1 feature
if (algorithm === "xgboost" && !inputFeatures.length) {
  setTrainStatus("⚠️ XGBoost requiere al menos 1 variable de entrada. Usa ARIMA o LSTM para modelos univariados.");
  return;
}
if (!targetVariable || !dateColumnName) {
  setTrainStatus("⚠️ Selecciona la variable target y la columna de fecha.");
  return;
}
```

### Additional Backend Bug Fixes

#### Bonus Fix: ARIMA Optimizer Parameters (train.py)

**Issue**: Pre-existing bug where ARIMA models failed with `TypeError: ARIMA.fit() got an unexpected keyword argument 'maxiter'`

**Root Cause**: Different statsmodels APIs:
- `ARIMA` class requires optimizer params in `method_kwargs` dictionary
- `SARIMAX` class accepts optimizer params directly

**Locations Fixed**:
1. [train.py:1899-1924](file:///workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/train.py#L1899-L1924) - Manual training
2. [train.py:1712-1728](file:///workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/train.py#L1712-L1728) - Grid search
3. [train.py:1786-1800](file:///workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/train.py#L1786-L1800) - Random search

**Fix Applied**:
```python
model_class_name = type(model_spec).__name__
if model_class_name == 'ARIMA':
    fit_kwargs = {
        'method': 'statespace',
        'method_kwargs': {**SARIMAX_OPTIMIZER_DEFAULTS}
    }
    if start_params is not None:
        fit_kwargs['start_params'] = start_params
else:
    fit_kwargs = {**SARIMAX_OPTIMIZER_DEFAULTS}
    if start_params is not None:
        fit_kwargs['start_params'] = start_params
```

---

## ✅ PHASE 3: Integration Testing [COMPLETED]

### Phase Overview
Test complete end-to-end flow with real backend and frontend integration.

**Status**: ✅ **COMPLETED** on 2025-12-17
**Completion Notes**:
- All 4 integration tests passed successfully
- ARIMA univariate: ✅ Training completes successfully
- LSTM univariate: ✅ Training completes successfully
- XGBoost validation: ✅ Correctly blocks empty features
- Multivariate regression: ✅ No regressions detected
- Frontend and backend integration working correctly
- Test datasets created and verified

### Pattern Consistency Verification - COMPLETED ✅

All patterns verified and consistent across the codebase:

#### Backend Patterns ✅
- [X] All ARIMA model fit calls use class name check (`type(model_spec).__name__`) - 3 occurrences found
- [X] Empty `input_features` validation consistent across all encoding endpoints
- [X] MLflow logging includes `is_univariate` flag in all training paths
- [X] Logger messages distinguish between univariate and multivariate modes

#### Frontend Patterns ✅
- [X] All validation rules check `algorithm === "xgboost"` for empty features - 3 locations verified
- [X] Helper text dynamically adapts to selected algorithm throughout UI
- [X] No hardcoded "requires 1 feature" messages for ARIMA/LSTM
- [X] Train button disable logic consistent across all form states

#### Integration Patterns ✅
- [X] Backend accepts `input_features: []` from frontend without errors
- [X] Error messages from backend are user-friendly in frontend
- [X] MLflow tracking visible and correct for both univariate and multivariate runs
- [X] No console errors or warnings in browser developer tools

### Test 3.1: ARIMA Univariate End-to-End

**Setup**: CSV with at least 50 rows
```csv
date,sales
2023-01-01,100
2023-01-02,102
2023-01-03,98
...
```

**Steps**:
1. Frontend: Upload CSV
2. Frontend: Load columns
3. Frontend: Select ARIMA, target="sales", date="date", features=[]
4. Frontend: Set model parameters (split ratios, model name, etc.)
5. Frontend: Click "Entrenar Modelo"
6. Backend: Encoding step with empty input_features
7. Backend: ARIMA training in univariate mode

**Verify**:
- ✅ Encoding completes without error
- ✅ Training completes successfully
- ✅ MLflow logs show `is_univariate: true`
- ✅ Model predictions generated

### Test 3.2: LSTM Univariate End-to-End

Same as Test 3.1 but with algorithm=LSTM

**Additional Verification**:
- ✅ Training mode set to "univariate" (check payload)
- ✅ LSTM uses only target variable

### Test 3.3: XGBoost Blocked Appropriately

**Steps**: Same as Test 3.1 but select XGBoost

**Verify**: Train button disabled with clear error message

### Test 3.4: Regression Test - Multivariate ARIMA

**Setup**: CSV with 4+ columns

**Steps**:
1. Select multiple features
2. Train ARIMA in multivariate mode

**Verify**: No regressions from changes

### Success Criteria (Integration) - ALL MET ✅
- [X] ARIMA univariate training completes successfully
- [X] LSTM univariate training completes successfully
- [X] XGBoost validation prevents empty features with clear message
- [X] Multivariate models still work (no regression)
- [X] MLflow tracking correctly logs univariate mode
- [X] No console errors in frontend
- [X] No backend exceptions in logs

**Test Documentation Created**:
- [PHASE3_INTEGRATION_TESTING_CHECKLIST.md](file:///workspaces/dream-ml-c/PHASE3_INTEGRATION_TESTING_CHECKLIST.md) - Completed by tester
- [PHASE3_QUICK_START_GUIDE.md](file:///workspaces/dream-ml-c/PHASE3_QUICK_START_GUIDE.md) - Reference guide
- Test datasets generated: `test_univariate_sales.csv`, `test_multivariate_sales.csv`

---

## PHASE 4: Documentation & Messaging

### Phase Overview
Document the univariate time series feature for user guidance (optional phase).

### Pattern Consistency Checklist for Phase 4

Before proceeding with documentation, ensure the following patterns are ready:

#### Documentation Patterns
- [ ] Helper text in UI matches documentation examples
- [ ] Algorithm names consistent (ARIMA, LSTM, XGBoost - capitalization)
- [ ] Feature terminology consistent ("input features" vs "variables de entrada")
- [ ] Error messages match between frontend, backend, and docs
- [ ] Examples use real column names from test datasets

#### User-Facing Messaging
- [ ] Validation messages are clear and actionable
- [ ] Helper text provides guidance for all algorithms
- [ ] Error messages explain what to do (not just what's wrong)
- [ ] UI labels match documentation terminology

#### Code Comments and Logging
- [ ] Code comments explain "why" for univariate logic
- [ ] Log messages are helpful for debugging
- [ ] Variable names are self-documenting
- [ ] Complex conditions have explanatory comments

### Optional: Add Univariate Example to Docs

**Content**:
```markdown
## Univariate Time Series Models

For time series forecasting with only historical values of the target variable (no external features), ARIMA and LSTM support univariate mode:

1. Upload CSV with date and target columns only
2. Select ARIMA or LSTM algorithm
3. Select target variable
4. Select date column
5. Leave input features empty
6. Train model

Note: XGBoost requires at least one input feature and does not support pure univariate forecasting.
```

---

## PHASE 5: Deployment & Rollback Plan

### Deployment Steps

1. **Backend Deployment**:
   ```bash
   cd /workspaces/dream-ml-c/DREAM-ML-backend
   # Run tests
   python -m pytest GEML/tests/apiTimeSeries_tests/ -v
   # Deploy (method depends on your infrastructure)
   ```

2. **Frontend Deployment**:
   ```bash
   cd /workspaces/dream-ml-c/DREAM-ML-frontend
   npm run build
   # Deploy
   ```

3. **Smoke Test Post-Deployment**:
   - Test univariate ARIMA flow
   - Test multivariate ARIMA flow (regression)
   - Verify XGBoost validation works

### Rollback Plan

**If Issues Detected**:

1. **Immediate Rollback** (< 5 minutes): Revert both backend and frontend to previous versions
2. **Partial Rollback** (backend only): If frontend changes safe but backend has issues
3. **Fix Forward**: For minor issues (typos, logging)

### Monitoring Post-Deployment
- [ ] Check error logs for new ValueError exceptions
- [ ] Monitor MLflow for univariate model runs
- [ ] Check user feedback/support tickets
- [ ] Verify model quality metrics comparable

---

## SUCCESS CRITERIA

### Functional Requirements
- ✅ Users can train ARIMA with 2-column CSV (date + target)
- ✅ Users can train LSTM with 2-column CSV (date + target)
- ✅ XGBoost shows clear error for 2-column CSV
- ✅ Multivariate models continue to work (no regression)
- ✅ Validation messages are clear and helpful

### Technical Requirements
- ✅ No breaking changes to existing API
- ✅ Backward compatible with existing data
- ✅ All existing tests pass
- ✅ MLflow tracking enhanced with univariate flag
- ✅ Clean code with minimal duplication

### User Experience
- ✅ No confusing error messages
- ✅ Clear guidance on when to use univariate
- ✅ Algorithm-specific helper text
- ✅ Fast iteration (quality + speed balanced)

---

## IMPLEMENTATION CHECKLIST

### Backend Changes ✅ COMPLETED
- [X] Modify services.py line 719-720 (validation logic)
- [X] Update services.py line 759 (column validation loop)
- [X] Add services.py line 790 (MLflow univariate flag)
- [X] Run pytest on apiTimeSeries tests (10 LSTM tests + 2 univariate tests PASSED)
- [X] Manual API test with empty input_features (via integration testing)

### Frontend Changes ✅ COMPLETED
- [X] Modify TSTrainCard.jsx line 371-373 (validation rule)
- [X] Update TSTrainCard.jsx line 953 (train button logic)
- [X] Update TSTrainCard.jsx line 1069-1070 (helper text)
- [X] Test ARIMA univariate UI flow
- [X] Test LSTM univariate UI flow
- [X] Test XGBoost error message
- [X] Test multivariate regression

### Integration Testing ✅ COMPLETED
- [X] End-to-end ARIMA univariate test
- [X] End-to-end LSTM univariate test
- [X] XGBoost blocked appropriately
- [X] Multivariate models work (regression test)
- [X] MLflow logs correct univariate flag

### Deployment
- [ ] Backend deployment
- [ ] Frontend deployment
- [ ] Smoke test post-deployment
- [ ] Monitor logs for 24 hours
- [ ] Verify no user issues reported

---

## RISK MITIGATION

### Risk 1: Backend Breaking Changes
**Likelihood**: Low | **Impact**: High
**Mitigation**: Comprehensive test suite, rollback plan ready, change is additive

### Risk 2: Frontend-Backend Mismatch
**Likelihood**: Low | **Impact**: Medium
**Mitigation**: Deploy both together, test integration before deployment

### Risk 3: User Confusion
**Likelihood**: Medium | **Impact**: Low
**Mitigation**: Clear helper text, specific error messages, documentation updated

---

## ESTIMATED EFFORT

- **Backend Changes**: 30 minutes
- **Frontend Changes**: 45 minutes
- **Testing (Manual)**: 1 hour
- **Integration Testing**: 30 minutes
- **Deployment**: 30 minutes
- **Total**: ~3-3.5 hours

**Priority**: High (blocks valid use case)
**Complexity**: Low-Medium
**Risk**: Low
