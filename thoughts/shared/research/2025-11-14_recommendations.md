# Recommendations and Implementation Roadmap - DREAM-ML LSTM Training Fixes

**Date**: 2025-11-14
**Author**: Claude Code Research Agent
**Purpose**: Comprehensive recommendations for fixing identified bugs, closing validation gaps, and improving UX clarity

---

## Executive Summary

This document synthesizes findings from backend, frontend, and validation analyses to provide actionable recommendations for resolving the three critical LSTM bugs and improving the overall training pipeline. Recommendations are prioritized by impact and organized into implementation phases.

**Critical Bugs to Fix**:
1. **Bug #1**: Lag-of-lag feature creation (backend) - HIGHEST PRIORITY
2. **Bug #2**: LSTM univariate mode blocked (frontend validation) - HIGH PRIORITY
3. **Bug #3**: Incomplete partial fix (frontend button logic) - MEDIUM PRIORITY

**Additional Improvements**:
- Close post-encoding validation gap
- Improve error messaging clarity
- Enhance LSTM UX for feature selection
- Add reproducibility fields to pipeline config

---

## System Architecture Requirements (CRITICAL)

Before implementing any fixes, understand these fundamental constraints:

### 1. Feature Generation Workflow (2-Step Process)

**Step 1: Data Encoding (Preprocessing)**
- User uploads original CSV → configures lag_periods, rolling_windows
- Backend generates lag features, rolling windows → saves NEW encoded CSV
- Encoded CSV contains: original columns + engineered columns (Sales_lag_1, Sales_rolling_7, etc.)

**Step 2: Training (TSTrainCard.jsx)**
- User uploads encoded CSV (or original CSV with warning)
- User selects target, date, input features from EXISTING columns only
- Backend trains using selected columns AS-IS
- ❌ **NO feature generation happens in training step**

### 2. Training Payload Constraints

**Parameters that MUST BE REMOVED from training payload**:
- ❌ `feature_config` object (belongs in Data Encoding step)
- ❌ `lag_periods` (feature generation parameter)
- ❌ `rolling_windows` (feature generation parameter)
- ❌ `external_features` (deprecated concept - all features are "input features")

**Parameters that SHOULD EXIST in training payload**:
- ✅ `target` - selected column name from CSV
- ✅ `input_features` - array of column names from encoded CSV
- ✅ `date` - selected column name from CSV
- ✅ Algorithm-specific hyperparameters (lstm_units, dropout, etc.)

### 3. Variable Types (Univariate Time Series System)

**System Constraint**: ALL models predict exactly 1 target variable

**Variable Categories**:
- **Target** (1 required): Variable to predict (e.g., Sales)
- **Input Features** (0+ for ARIMA, 1+ for XGBoost/LSTM): Variables used for prediction
  - Can include original features (Temperature, Humidity)
  - Can include engineered features (Sales_lag_1, created in Data Encoding)
  - CANNOT include the target variable itself
- **Date** (1 required): Temporal ordering column

**Deprecated Concepts**:
- ❌ `external_features` - No longer used; all features are "input features"
- ❌ Multivariate time series (multiple targets) - NOT SUPPORTED

### 4. Minimum Features Requirements

| Algorithm | Minimum Input Features | Rationale |
|-----------|------------------------|-----------|
| **ARIMA** | 0+ (can be pure ARIMA) | ARIMA can predict from autoregressive terms only |
| **XGBoost** | 1+ (at least one feature) | XGBoost requires features to build trees |
| **LSTM** | 1+ (at least one feature) | LSTM requires sequence inputs |

**"Univariate Mode" Definition** (for LSTM/XGBoost):
- ❌ NOT: 0 input features
- ✅ CORRECT: Select lag features of target only (e.g., Sales_lag_1, Sales_lag_2)
- This uses only target history, no external features (Temperature, Humidity)

### 5. Auto-Selection Rules (Project Requirement)

**ARIMA/XGBoost**:
- System auto-selects ALL columns from CSV except target and date
- Includes pre-engineered lag features from Data Encoding
- User can deselect unwanted features

**LSTM**:
- NO auto-selection (manual checkbox selection required)
- User must explicitly select ≥1 feature
- Rationale: Explicit control over sequence composition

### 6. CSV Validation Behavior

**If user uploads CSV without lag features**:
- ✅ Accept the CSV
- ⚠️ Show warning: "No lag features detected. Consider running Data Encoding step first."
- Allow user to proceed (maybe training on original features only is intentional)

### 7. Validation Rules by Algorithm

| Rule | ARIMA | XGBoost | LSTM |
|------|-------|---------|------|
| **Target** | 1 (required, radio) | 1 (required, radio) | 1 (required, radio) |
| **Input Features** | 0+ (auto-select all) | 1+ (auto-select all) | 1+ (manual checkboxes) |
| **Date** | 1 (required, radio) | 1 (required, radio) | 1 (required, radio) |
| **Auto-selection** | ✅ Yes | ✅ Yes | ❌ No (project rule) |
| **Target in features** | ❌ Never | ❌ Never | ❌ Never |
| **Date in features** | ❌ Never | ❌ Never | ❌ Never |
| **Empty features valid** | ✅ Yes (pure ARIMA) | ❌ No (needs ≥1) | ❌ No (needs ≥1) |

**Notes**:
- All algorithms work with univariate time series (1 target only)
- "Input features" are selected from columns in uploaded CSV (including lag features from Data Encoding)
- LSTM's "univariate mode" means selecting lag features only (e.g., Sales_lag_1), not 0 features

---

## Bug Fix Recommendations

### Bug #1: Lag-of-Lag Feature Creation

**Root Cause** (CORRECTED): User selects pre-engineered lag features (e.g., Sales_lag_1) from encoded CSV in LSTM training UI. Due to **INCORRECT ARCHITECTURE** where training step attempts to generate features, backend creates lag-of-lag features (Sales_lag_1_lag_1), causing NaN values.

**Impact**: LSTM training fails with `loss=nan` when user selects lagged features from encoded CSV.

**CRITICAL Understanding**:
- Bug occurs because training step should NOT generate features at all
- Training step should use CSV columns AS-IS (no lag creation)
- Feature generation belongs ONLY in Data Encoding step

**Recommended Solutions**:

---

#### Option 1A: Remove Feature Generation from Training Step (REQUIRED)

**Approach**: Remove all feature generation logic from training backend. Training step only uses columns from uploaded CSV.

**Backend Changes** ([train.py](DREAM-ML-backend/GEML/apiTimeSeries/train.py)):

```python
def train_lstm_model(...):
    """
    Train LSTM model using pre-encoded CSV from Data Encoding step.

    CRITICAL: This function does NOT generate features. It expects:
    - Encoded CSV with lag features already created
    - input_features containing column names from that CSV
    """
    # Load CSV (no encoding, no feature generation)
    df = pd.read_csv(dataset_path)

    # Validate columns exist in CSV
    missing_cols = set(input_features + [target_variable, date_column]) - set(df.columns)
    if missing_cols:
        raise ValidationError(
            f"Columns not found in CSV: {missing_cols}\n\n" +
            "Please ensure you uploaded the encoded CSV from Data Encoding step.\n" +
            "If you haven't run Data Encoding, please complete that step first."
        )

    # Check if lag features exist (warn if not)
    lag_features = [col for col in df.columns if '_lag_' in col or '_rolling_' in col]
    if not lag_features:
        logger.warning(
            "No lag features detected in CSV. " +
            "Consider running Data Encoding step to create lag features."
        )

    # Proceed with training using columns AS-IS (no feature generation)
    X = df[input_features] if input_features else pd.DataFrame()
    y = df[target_variable]

    # Create sequences for LSTM
    X_sequences, y_sequences = create_sequences_for_lstm(
        df=df,
        target_col=target_variable,
        sequence_length=sequence_length,
        feature_cols=input_features  # Use selected columns only
    )

    # Train model
    # ...
```

**Payload Validation** (reject feature generation parameters):

```python
def validate_training_payload(payload, algorithm):
    """Validate training payload contains only allowed parameters."""

    # Parameters that should NOT exist in training payload
    forbidden_params = ['feature_config', 'lag_periods', 'rolling_windows', 'external_features']

    found_forbidden = [p for p in forbidden_params if p in payload]
    if found_forbidden:
        raise ValidationError(
            f"Training payload contains forbidden parameters: {found_forbidden}\n\n" +
            "These parameters belong in the Data Encoding step, not Training:\n" +
            "- lag_periods, rolling_windows: Create features in Data Encoding\n" +
            "- feature_config: Not used in training\n" +
            "- external_features: Use 'input_features' instead\n\n" +
            "Please remove these parameters from your request."
        )

    # Algorithm-specific validation
    if algorithm in ['xgboost', 'lstm']:
        if 'input_features' not in payload or not payload['input_features']:
            raise ValidationError(
                f"{algorithm.upper()} requires at least 1 input feature.\n\n" +
                "Please select features from your encoded CSV."
            )
```

**Frontend Changes** ([TSTrainCard.jsx](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx)):

Remove feature_config from training payload (lines 739-743, 777-781, 869-870):

```javascript
// REMOVE these lines from handleTrain():
// if (algorithm === "xgboost" || algorithm === "lstm") {
//   payload.feature_config = featureConfig;  // ← DELETE THIS
// }
```

**Pros**:
- ✅ Aligns with correct 2-step architecture
- ✅ Simplifies training logic (no feature generation)
- ✅ Clear separation of concerns (encoding vs training)
- ✅ Prevents lag-of-lag bugs entirely
- ✅ Validates user uploaded correct CSV

**Cons**:
- ❌ Requires backend refactoring
- ❌ Breaking change from current implementation

**Testing**:
1. Run Data Encoding with lag_periods=[1,2,3] → creates encoded CSV
2. Upload encoded CSV to Training step
3. Select lag features (Sales_lag_1) in LSTM UI
4. Verify training uses columns AS-IS (no re-generation)
5. Verify training succeeds without NaN errors

---

#### Option 1B: Frontend Prevents Selecting Lag Features (WORKAROUND)

**Approach**: Filter lag features from LSTM UI to prevent user from selecting them. This is a **workaround** for the incorrect architecture, not a proper fix.

**Frontend Changes** ([TSTrainCard.jsx](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx)):

```javascript
// LSTM feature selection - filter out lag features
const availableFeaturesForLSTM = columns.filter((col) => {
  const isLaggedFeature = col.includes('_lag_') || col.includes('_rolling_');
  return (
    col !== targetVariable &&
    col !== dateColumnName &&
    !isLaggedFeature  // Hide lag features (user shouldn't select these)
  );
});

{algorithm === "lstm" && (
  <div className="lstm-feature-selection">
    <h3>Seleccionar Características</h3>
    <p className="warning-text">
      ⚠️ Nota: Las características con lag/rolling window no se muestran aquí.
      El backend las procesará automáticamente.
    </p>
    {availableFeaturesForLSTM.map((feature) => (
      <label key={feature}>
        <input
          type="checkbox"
          checked={lstmSelectedFeatures.includes(feature)}
          onChange={() => handleLstmFeatureToggle(feature)}
        />
        {feature}
      </label>
    ))}
  </div>
)}
```

**Pros**:
- ✅ Quick workaround to prevent bug
- ✅ Frontend-only change

**Cons**:
- ❌ Does NOT fix root cause (training still generates features)
- ❌ Misleading to users (hides available columns)
- ❌ Still incorrect architecture

**Note**: This is a temporary workaround. Option 1A is the proper fix.

---

**RECOMMENDED APPROACH**: **Option 1A (Remove Feature Generation from Training)**

This is the correct architectural fix. It requires backend refactoring but aligns with the intended 2-step workflow:
1. Data Encoding: Generate features → save encoded CSV
2. Training: Load encoded CSV → use columns AS-IS → train model

**Implementation Steps**:
1. **Refactor backend training functions**:
   - Remove `encode_csv` parameter (training doesn't encode)
   - Remove feature generation logic
   - Add CSV column validation
   - Add warning if no lag features detected
2. **Update payload validation**:
   - Reject `feature_config`, `lag_periods`, `rolling_windows`, `external_features`
   - Require `input_features` for XGBoost/LSTM (≥1 feature)
3. **Update frontend**:
   - Remove feature_config from training payload (lines 739-743, 777-781, 869-870 in TSTrainCard.jsx)
   - Show clear message: "Upload encoded CSV from Data Encoding step"
4. **Test thoroughly**:
   - Encoded CSV → Training works
   - Original CSV → Training shows warning but proceeds
   - Missing columns → Clear error message

---

### Bug #2: LSTM Validation Checks Wrong State Variable

**Root Cause** (CORRECTED): [TSTrainCard.jsx:681](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L681) validates `!inputFeatures.length` for all algorithms, but LSTM uses separate `lstmSelectedFeatures` state.

**Impact**: LSTM validation checks wrong variable → users see confusing errors

**CRITICAL Understanding**:
- Validation rule is CORRECT: LSTM requires ≥1 feature
- Bug is checking WRONG variable: `inputFeatures` instead of `lstmSelectedFeatures`
- LSTM uses separate state for feature selection (Phase 4 implementation)

**Recommended Solution**:

---

#### Fix Validation to Check Correct State Variable (REQUIRED)

**Frontend Changes** ([TSTrainCard.jsx:680-685](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L680-L685)):

```javascript
// Current (CHECKS WRONG VARIABLE):
if (!inputFeatures.length || !targetVariable || !dateColumnName) {
  setTrainStatus("⚠️ Selecciona las variables de entrada, target y fecha.");
  return;
}

// Recommended (CHECKS CORRECT VARIABLE):
// Algorithm-specific feature validation
let hasRequiredFeatures;
if (algorithm === "lstm") {
  hasRequiredFeatures = lstmSelectedFeatures.length >= 1;  // Check LSTM state
} else if (algorithm === "arima") {
  hasRequiredFeatures = true;  // ARIMA: 0+ features allowed
} else {
  hasRequiredFeatures = inputFeatures.length >= 1;  // XGBoost: 1+ required
}

if (!hasRequiredFeatures || !targetVariable || !dateColumnName) {
  if (!targetVariable) {
    setTrainStatus("⚠️ Debes seleccionar la variable target.");
  } else if (!dateColumnName) {
    setTrainStatus("⚠️ Debes seleccionar la columna de fecha.");
  } else if (!hasRequiredFeatures) {
    if (algorithm === "lstm") {
      setTrainStatus("⚠️ LSTM requiere al menos 1 característica. Selecciona características de tu CSV codificado.");
    } else if (algorithm === "xgboost") {
      setTrainStatus("⚠️ XGBoost requiere al menos 1 característica.");
    } else {
      setTrainStatus("⚠️ Debes seleccionar al menos una variable de entrada.");
    }
  }
  return;
}
```

**Improved Validation**:
- ARIMA: 0+ features (can be pure ARIMA)
- XGBoost: 1+ features required
- LSTM: 1+ features required (checks `lstmSelectedFeatures` state)
- Algorithm-specific error messages

**"Univariate Mode" Clarification**:
- ❌ NOT: 0 features (empty selection)
- ✅ CORRECT: Select lag features only (e.g., Sales_lag_1, Sales_lag_2 from encoded CSV)
- This uses target history without external features (Temperature, Humidity)

**Pros**:
- ✅ Fixes bug (checks correct state variable)
- ✅ Maintains minimum feature requirements
- ✅ Algorithm-specific validation
- ✅ Clear error messages

**Cons**:
- None (straightforward bug fix)

**Testing**:
1. Select LSTM, target, date, 0 features → Should show "LSTM requiere al menos 1 característica"
2. Select LSTM, target, date, 1+ features → Should submit successfully
3. Select ARIMA, target, date, 0 features → Should submit successfully (pure ARIMA)
4. Select XGBoost, no features → Should show "XGBoost requiere al menos 1 característica"
5. Verify all algorithms work correctly

---

### Bug #3: Incomplete Partial Fix

**Root Cause**: [TSTrainCard.jsx:1094](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L1094) partially fixed button disable logic, but validation at line 681 still blocks submission.

**Impact**: Confusing UX (button enabled but validation blocks submission).

**Recommended Solution**:

This bug is **automatically fixed** by implementing Bug #2 fix above. No additional changes needed.

**Verification**:
- After fixing line 681 validation, button logic at line 1094 works correctly
- LSTM button enabled when `inputFeatures` empty ✅
- Validation no longer blocks submission ✅

---

## Validation Gap Recommendations

### Critical Gap: Post-Encoding NaN Validation

**Location**: After [data_encoding_utils.py:139](DREAM-ML-backend/GEML/apiTimeSeries/data_encoding_utils.py#L139) encoding step

**Recommended Solution**:

---

#### Add Post-Encoding Validation Function

**Backend Changes** ([train.py:88](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L88)):

```python
def validate_dataframe_quality(df, stage="unknown"):
    """
    Validate DataFrame has no NaN, Inf, or other invalid values.

    Args:
        df: DataFrame to validate
        stage: Description of processing stage (for error messages)

    Raises:
        ValidationError: If DataFrame contains invalid values
    """
    # Check for NaN
    if df.isnull().any().any():
        nan_columns = df.columns[df.isnull().any()].tolist()
        nan_counts = {col: df[col].isnull().sum() for col in nan_columns}
        raise ValidationError(
            f"Data contains NaN values after {stage}:\n" +
            "\n".join([f"  - {col}: {count} NaN values" for col, count in nan_counts.items()]) +
            "\n\nThis may be caused by:\n" +
            "  1. Lag feature creation from already-lagged columns (lag-of-lag)\n" +
            "  2. Rolling window periods exceeding data length\n" +
            "  3. Insufficient data at the beginning of the time series\n\n" +
            "Please check your feature selection and encoding settings."
        )

    # Check for infinity
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            raise ValidationError(
                f"Column '{col}' contains infinite values after {stage}. " +
                "This may indicate division by zero or overflow."
            )

    return True
```

**Call After Encoding** ([train.py:2188](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L2188)):

```python
def train_lstm_model(...):
    # Load and encode data
    df_encoded = load_and_validate_ts_data(
        dataset=dataset,
        encode_csv='yes',
        lag_periods=lag_periods,
        ...
    )

    # NEW: Validate no NaN after encoding
    validate_dataframe_quality(df_encoded, stage="data encoding (lag features, rolling windows)")

    # Proceed to sequence creation
    X, y = create_sequences_for_lstm(df_encoded, target_col, sequence_length, input_features)

    # NEW: Validate sequences
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValidationError(
            "LSTM sequences contain NaN values. This should not happen after data validation. " +
            "Please report this issue with your dataset and configuration."
        )

    # Proceed to training
    # ...
```

**Pros**:
- ✅ Catches NaN before training starts
- ✅ Provides detailed error message
- ✅ Helps users debug configuration issues
- ✅ Reusable for other algorithms

**Cons**:
- None (essential validation)

**Testing**:
1. Manually create scenario with lag-of-lag features
2. Verify ValidationError raised with clear message
3. Verify error displayed to user in frontend
4. Test with valid configuration → should pass validation

---

## UX Improvement Recommendations

### Recommendation 1: Add LSTM Training Mode Indicator

**Goal**: Make it clear to users whether they're in univariate or multivariate mode.

**Frontend Changes** ([TSTrainCard.jsx](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx)):

```javascript
{algorithm === "lstm" && (
  <div className="lstm-feature-selection">
    <h3>Seleccionar Características (Opcional)</h3>
    <div className="mode-indicator">
      <strong>Modo de Entrenamiento:</strong>{" "}
      {lstmSelectedFeatures.length === 0 ? (
        <span className="badge badge-info">
          Univariado (solo target: {targetVariable || "no seleccionado"})
        </span>
      ) : (
        <span className="badge badge-success">
          Multivariado (target + {lstmSelectedFeatures.length} características)
        </span>
      )}
    </div>
    <p className="info-text">
      <strong>Univariado:</strong> Predice usando solo valores históricos del target.
      <br />
      <strong>Multivariado:</strong> Predice usando el target y características adicionales.
    </p>
    {availableFeaturesForLSTM.map((feature) => (
      // ... checkboxes
    ))}
  </div>
)}
```

**CSS Styling**:
```css
.mode-indicator {
  margin-bottom: 15px;
  padding: 10px;
  background-color: #f0f0f0;
  border-radius: 5px;
}

.badge {
  padding: 5px 10px;
  border-radius: 3px;
  font-weight: bold;
}

.badge-info {
  background-color: #17a2b8;
  color: white;
}

.badge-success {
  background-color: #28a745;
  color: white;
}

.info-text {
  font-size: 0.9em;
  color: #666;
  margin-top: 10px;
}
```

**Pros**:
- ✅ Clear visual feedback
- ✅ Educational (explains modes)
- ✅ Reduces user confusion

---

### Recommendation 2: Improve Error Message Specificity

**Goal**: Tell users exactly what's wrong and how to fix it.

**Frontend Changes** ([TSTrainCard.jsx:392-422](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx#L392-L422)):

```javascript
const validateSelections = () => {
  // LSTM: Check target not in features
  if (algorithm === "lstm") {
    if (lstmSelectedFeatures.includes(targetVariable)) {
      setTrainStatus(
        `⚠️ Error: La variable target "${targetVariable}" no puede estar en las características seleccionadas.\n\n` +
        `El target es lo que el modelo intenta predecir, no una entrada.`
      );
      return false;
    }
  }

  // All algorithms: Check target and date are selected
  if (!targetVariable) {
    setTrainStatus("⚠️ Error: Debes seleccionar la variable target.");
    return false;
  }

  if (!dateColumnName) {
    setTrainStatus("⚠️ Error: Debes seleccionar la columna de fecha.");
    return false;
  }

  return true;
};
```

**Backend Changes** (already good, keep current detailed error messages)

---

### Recommendation 3: Add Tooltips for LSTM Features

**Goal**: Help users understand implications of selecting features.

**Frontend Changes** ([TSTrainCard.jsx](DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx)):

```javascript
{availableFeaturesForLSTM.map((feature) => (
  <label key={feature} className="feature-checkbox-label">
    <input
      type="checkbox"
      checked={lstmSelectedFeatures.includes(feature)}
      onChange={() => handleLstmFeatureToggle(feature)}
    />
    {feature}
    <span
      className="tooltip-icon"
      title={`Incluir "${feature}" como característica adicional en las secuencias LSTM.
              El modelo usará los valores históricos de esta variable para predecir el target.`}
    >
      ℹ️
    </span>
  </label>
))}
```

**CSS**:
```css
.feature-checkbox-label {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}

.tooltip-icon {
  margin-left: 8px;
  cursor: help;
  font-size: 0.9em;
}
```

---

## Pipeline Config Recommendations

### Recommendation: Add Missing LSTM Reproducibility Fields

**Goal**: Enable full experiment reconstruction from pipeline_config.json

**Backend Changes** ([train.py:331](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L331)):

```python
def save_pipeline_config(algorithm, model_instance, hyperparameters, metrics, mlflow_run_id, **kwargs):
    """
    Save pipeline configuration for reproducibility.

    Additional kwargs for LSTM:
        - features_used: List of features included in sequences
        - input_features_raw: Original input_features sent from frontend
        - model_name: User-defined model name
        - target_variable: Target column
        - date_col_name: Date column
        - forecast_horizon: Steps ahead to predict
        - split_ratios: Dict with train/validation/test ratios
        - training_mode: 'univariate' or 'multivariate'
        - encoded_csv_path: Path to encoded CSV used for training
    """
    config = {
        "algorithm": algorithm,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "mlflow_run_id": mlflow_run_id,
        "training_date": datetime.now().isoformat(),
        "dataset_id": model_instance.dataset.id,
        "dataset_name": model_instance.dataset.name,
    }

    # Algorithm-specific fields
    if algorithm == "lstm":
        config.update({
            "model_name": kwargs.get("model_name"),
            "target_variable": kwargs.get("target_variable"),
            "date_col_name": kwargs.get("date_col_name"),
            "forecast_horizon": kwargs.get("forecast_horizon"),
            "split_ratios": {
                "train": kwargs.get("train_ratio"),
                "validation": kwargs.get("validation_ratio"),
                "test": kwargs.get("test_ratio"),
            },
            "training_mode": kwargs.get("training_mode"),  # "lag_features_only" or "with_external_features"
            "features_used": kwargs.get("features_used", []),  # ← NEW: Features in sequences
            "input_features_raw": kwargs.get("input_features_raw", []),  # ← NEW: Original from frontend
            "sequence_length": hyperparameters.get("sequence_length"),
            "normalization": kwargs.get("normalization", "MinMaxScaler"),
            "encoded_csv_path": kwargs.get("encoded_csv_path"),  # ← NEW: Data provenance
            "preprocessing_steps": ["train_test_split", "normalization", "sequence_creation"],
        })
    elif algorithm == "xgboost":
        config.update({
            "features_used": kwargs.get("features_used", []),
            # NOTE: lag_periods belongs in Data Encoding, not training config
            # Only include if tracking which encoding was used
            "encoded_csv_path": kwargs.get("encoded_csv_path"),  # ← Track data source
            # ... existing fields
        })
    # ... ARIMA fields

    # Save to file
    config_path = f"pipeline_configs/{model_instance.id}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path
```

**Call with Additional Parameters** ([train.py:2188](DREAM-ML-backend/GEML/apiTimeSeries/train.py#L2188)):

```python
def train_lstm_model(..., input_features, target_variable, date_column_name, ...):
    # ... training logic

    # Determine features actually used in sequences (REQUIRED: ≥1 feature for LSTM)
    features_used = [target_variable] + input_features

    # Determine training mode based on feature types
    has_lag_features = any('_lag_' in f or '_rolling_' in f for f in input_features)
    has_external_features = any('_lag_' not in f and '_rolling_' not in f for f in input_features)

    if has_external_features:
        training_mode = "with_external_features"  # Lag features + external (Temperature, etc.)
    else:
        training_mode = "lag_features_only"  # Only lag features (univariate in spirit)

    # Save pipeline config with all reproducibility fields
    config_path = save_pipeline_config(
        algorithm="lstm",
        model_instance=trained_model,
        hyperparameters={
            "sequence_length": sequence_length,
            "lstm_units": lstm_units,
            # ... other hyperparameters
        },
        metrics={"mae": mae, "rmse": rmse, "mape": mape},
        mlflow_run_id=mlflow.active_run().info.run_id,
        # NEW: Additional fields
        features_used=features_used,
        input_features_raw=input_features,
        model_name=model_name,
        target_variable=target_variable,
        date_col_name=date_column_name,
        forecast_horizon=forecast_horizon,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        training_mode=training_mode,  # "lag_features_only" or "with_external_features"
        encoded_csv_path=df_encoded_path,  # Save encoded CSV path
        normalization="MinMaxScaler",
    )
```

**Pros**:
- ✅ Full reproducibility
- ✅ Matches XGBoost config structure
- ✅ Enables experiment comparison
- ✅ Documents data provenance

---

## Implementation Roadmap

### Phase 1: Critical Bug Fixes (Week 1)

**Priority**: CRITICAL
**Goal**: Unblock LSTM training

**Tasks**:
1. **Bug #1 Fix** (4 hours):
   - [ ] Implement Option 1A: Filter lagged features in frontend
   - [ ] Add explanatory text in LSTM UI
   - [ ] Test with lagged dataset
   - [ ] Verify training succeeds

2. **Bug #2 Fix** (2 hours):
   - [ ] Update validation logic at line 681
   - [ ] Improve error messages
   - [ ] Test univariate mode submission
   - [ ] Verify Bug #3 auto-fixed

3. **Post-Encoding Validation** (3 hours):
   - [ ] Implement `validate_dataframe_quality()` function
   - [ ] Add calls after encoding and sequence creation
   - [ ] Test with invalid data
   - [ ] Verify clear error messages

**Testing**:
- [ ] LSTM univariate training (empty features) → Success
- [ ] LSTM multivariate training (raw features) → Success
- [ ] LSTM with lagged features → Filtered from UI
- [ ] XGBoost training (unchanged) → Success
- [ ] ARIMA training (unchanged) → Success

**Deliverables**:
- Working LSTM univariate and multivariate training
- Clear error messages for data issues
- No lag-of-lag bugs

---

### Phase 2: UX Improvements (Week 2)

**Priority**: HIGH
**Goal**: Improve user experience and clarity

**Tasks**:
1. **Training Mode Indicator** (2 hours):
   - [ ] Add univariate/multivariate badge
   - [ ] Add explanatory text
   - [ ] Style with CSS
   - [ ] Test visibility

2. **Improved Error Messages** (3 hours):
   - [ ] Update all validation error messages
   - [ ] Add field-specific errors
   - [ ] Add tooltips for features
   - [ ] Test all error scenarios

3. **LSTM Parameter Validation Messages** (2 hours):
   - [ ] Display which parameter is invalid
   - [ ] Add inline validation feedback
   - [ ] Test all hyperparameter modes

**Testing**:
- [ ] User can understand which mode they're in
- [ ] Error messages are specific and actionable
- [ ] Tooltips provide helpful information

**Deliverables**:
- Clear UX for LSTM training modes
- Specific, actionable error messages
- Educational tooltips

---

### Phase 3: Reproducibility Enhancements (Week 3)

**Priority**: MEDIUM
**Goal**: Enable full experiment reconstruction

**Tasks**:
1. **Pipeline Config Fields** (3 hours):
   - [ ] Add `features_used` to LSTM config
   - [ ] Add root-level metadata fields
   - [ ] Add `encoded_csv_path` for data provenance
   - [ ] Update `save_pipeline_config()` function

2. **Config Validation** (2 hours):
   - [ ] Add unit tests for config generation
   - [ ] Verify all fields present
   - [ ] Test config loading and reconstruction

3. **Documentation** (2 hours):
   - [ ] Document pipeline config schema
   - [ ] Add examples for each algorithm
   - [ ] Document reconstruction procedure

**Testing**:
- [ ] Generate LSTM config, verify all fields present
- [ ] Compare LSTM vs XGBoost config structure
- [ ] Test loading config and reconstructing experiment

**Deliverables**:
- Complete pipeline config for all algorithms
- Documentation for reproducibility
- Unit tests for config generation

---

### Phase 4: Defense-in-Depth (Week 4)

**Priority**: LOW
**Goal**: Harden backend validation

**Tasks**:
1. **Backend Validation Mirroring** (4 hours):
   - [ ] Add `input_features` non-empty check for ARIMA/XGBoost
   - [ ] Add split ratio validation
   - [ ] Add feature overlap validation (XGBoost)
   - [ ] Add LSTM hyperparameter validation

2. **Auto-Detect Lagged Features** (2 hours):
   - [ ] Implement Option 1C in `create_lag_features()`
   - [ ] Add logging for skipped features
   - [ ] Test edge cases

3. **Comprehensive Testing** (4 hours):
   - [ ] Add integration tests for all algorithms
   - [ ] Add edge case tests (empty data, all NaN, etc.)
   - [ ] Add performance tests (large datasets)

**Testing**:
- [ ] Backend rejects invalid requests (bypassing frontend)
- [ ] Auto-detect prevents lag-of-lag bugs
- [ ] All tests pass

**Deliverables**:
- Hardened backend validation
- Comprehensive test suite
- Defensive coding against bugs

---

## Testing Strategy

### Unit Tests

**Frontend** ([TSTrainCard.test.jsx](DREAM-ML-frontend/frontend/src/components/TSTrainCard.test.jsx)):

```javascript
describe('TSTrainCard LSTM Feature Selection', () => {
  test('should filter lagged features from LSTM checkboxes', () => {
    const columns = ['Temperature', 'Temperature_lag_1', 'Humidity', 'Humidity_rolling_7'];
    const availableFeatures = filterLaggedFeatures(columns, 'Temperature', 'Date');
    expect(availableFeatures).toEqual(['Humidity']);
  });

  test('should allow LSTM submission with empty features (univariate)', () => {
    const isValid = validateLSTMSubmission({
      targetVariable: 'Temperature',
      dateColumnName: 'Date',
      lstmSelectedFeatures: [],
      algorithm: 'lstm',
    });
    expect(isValid).toBe(true);
  });

  test('should block LSTM submission with target in features', () => {
    const isValid = validateLSTMSubmission({
      targetVariable: 'Temperature',
      dateColumnName: 'Date',
      lstmSelectedFeatures: ['Temperature', 'Humidity'],
      algorithm: 'lstm',
    });
    expect(isValid).toBe(false);
  });
});
```

**Backend** ([test_data_encoding_utils.py](DREAM-ML-backend/GEML/tests/test_data_encoding_utils.py)):

```python
class TestLagFeatureCreation(unittest.TestCase):
    def test_skip_already_lagged_features(self):
        """Test that lagged features are not lagged again."""
        df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Temperature': np.random.randn(100),
            'Temperature_lag_1': np.random.randn(100),
        })

        result = create_lag_features(
            df=df,
            input_features=['Temperature', 'Temperature_lag_1'],
            lag_periods=2,
            date_column='Date'
        )

        # Should create Temperature_lag_1, Temperature_lag_2
        # Should NOT create Temperature_lag_1_lag_1
        self.assertIn('Temperature_lag_1', result.columns)
        self.assertIn('Temperature_lag_2', result.columns)
        self.assertNotIn('Temperature_lag_1_lag_1', result.columns)

    def test_post_encoding_validation_catches_nan(self):
        """Test that validation detects NaN after encoding."""
        df_with_nan = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Temperature': [np.nan] * 10 + list(np.random.randn(90)),
        })

        with self.assertRaises(ValidationError) as context:
            validate_dataframe_quality(df_with_nan, stage="test")

        self.assertIn("NaN values", str(context.exception))
        self.assertIn("Temperature", str(context.exception))
```

---

### Integration Tests

**Backend** ([test_lstm_training.py](DREAM-ML-backend/GEML/tests/test_lstm_training.py)):

```python
class TestLSTMTrainingIntegration(TestCase):
    def test_lstm_univariate_training_succeeds(self):
        """Test LSTM training with no additional features (univariate mode)."""
        response = self.client.post('/api/time-series/train/', {
            'dataset': self.dataset.id,
            'algorithm': 'lstm',
            'target_variable': 'Temperature',
            'date_column_name': 'Date',
            'input_features': [],  # Empty for univariate
            'model_name': 'LSTM_Univariate_Test',
            'sequence_length': 10,
            'lstm_units': 50,
            'dropout_rate': 0.2,
            'batch_size': 32,
            'epochs': 10,
            'hyperparameter_search': 'manual',
            'train_ratio': 70,
            'validation_ratio': 15,
            'test_ratio': 15,
        })

        self.assertEqual(response.status_code, 200)
        self.assertIn('trained_model', response.json())

    def test_lstm_multivariate_training_succeeds(self):
        """Test LSTM training with additional features (multivariate mode)."""
        response = self.client.post('/api/time-series/train/', {
            'dataset': self.dataset.id,
            'algorithm': 'lstm',
            'target_variable': 'Temperature',
            'date_column_name': 'Date',
            'input_features': ['Humidity', 'Pressure'],  # Raw features only
            'model_name': 'LSTM_Multivariate_Test',
            # ... other params
        })

        self.assertEqual(response.status_code, 200)

    def test_lstm_rejects_lagged_features(self):
        """Test that LSTM with lagged features is handled correctly."""
        # If Option 1A implemented: frontend filters, this shouldn't happen
        # If Option 1C implemented: backend auto-skips lagged features
        response = self.client.post('/api/time-series/train/', {
            'dataset': self.dataset_with_lags.id,
            'algorithm': 'lstm',
            'input_features': ['Temperature', 'Temperature_lag_1'],  # Including lagged
            # ... other params
        })

        # Should succeed (backend skips lag creation for Temperature_lag_1)
        self.assertEqual(response.status_code, 200)
```

---

### Manual Testing Checklist

**LSTM Univariate Mode**:
- [ ] Load dataset (e.g., Air Quality)
- [ ] Select LSTM algorithm
- [ ] Select target variable (e.g., Temperature)
- [ ] Select date column
- [ ] Leave features unchecked (empty)
- [ ] Verify mode indicator shows "Univariado"
- [ ] Fill hyperparameters
- [ ] Submit training
- [ ] Verify training succeeds
- [ ] Check pipeline_config.json has `features_used: ['Temperature']`
- [ ] Check MLflow run logged correctly

**LSTM Multivariate Mode**:
- [ ] Same setup as above
- [ ] Check 2 features (e.g., Humidity, Pressure)
- [ ] Verify mode indicator shows "Multivariado (target + 2 características)"
- [ ] Submit training
- [ ] Verify training succeeds
- [ ] Check pipeline_config.json has `features_used: ['Temperature', 'Humidity', 'Pressure']`

**Error Handling**:
- [ ] Submit without target → See "Debes seleccionar la variable target."
- [ ] Submit without date → See "Debes seleccionar la columna de fecha."
- [ ] Submit ARIMA without features → See "Debes seleccionar al menos una variable de entrada."
- [ ] Submit LSTM with target in features → See "La variable target no puede estar..."
- [ ] Submit with invalid split ratios → Button disabled, see error

**Lagged Feature Handling**:
- [ ] Load dataset with pre-existing lagged columns
- [ ] Select LSTM algorithm
- [ ] Verify feature checkboxes only show raw features (no Temperature_lag_1)
- [ ] Submit training
- [ ] Verify training succeeds

---

## Risk Assessment

### Implementation Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Bug fixes break existing ARIMA/XGBoost | Low | High | Comprehensive regression testing |
| Frontend validation too restrictive | Medium | Medium | Add override for advanced users |
| Backend validation performance impact | Low | Low | Validation is O(n), acceptable overhead |
| Missing edge cases in lagged feature detection | Medium | Medium | Add unit tests for edge cases |
| Pipeline config changes break old experiments | Low | High | Version config schema, backward compatible |

---

### Deployment Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Users confused by new UI | Low | Low | Add tooltips and explanatory text |
| Existing workflows disrupted | Low | Medium | Deploy in stages, monitor user feedback |
| Database migration issues | Low | High | Test migrations on staging environment |

---

## Success Metrics

### Functional Metrics

- [ ] LSTM univariate training succeeds (0% → 100% success rate)
- [ ] LSTM multivariate training succeeds (0% → 100% success rate)
- [ ] Zero lag-of-lag bugs reported
- [ ] All validation errors provide specific, actionable messages

### UX Metrics

- [ ] Users report understanding LSTM modes (survey)
- [ ] Error resolution time decreases (time from error to successful submission)
- [ ] Fewer support requests about LSTM training

### Technical Metrics

- [ ] 100% unit test coverage for new validation functions
- [ ] Integration tests pass for all algorithms
- [ ] Pipeline configs include all required reproducibility fields
- [ ] Backend validation rejects ≥95% of invalid requests

---

## Open Questions for Stakeholders

### Question 1: LSTM Lagged Feature Support

**Question**: Should LSTM support selecting pre-existing lagged features, or only raw features?

**Context**: Current Phase 4 UI allows selecting any column. Bug #1 occurs when lagged features selected.

**Options**:
- **A**: Filter lagged features (recommended in this document)
- **B**: Support lagged features with backend skip flag
- **C**: Support lagged features with auto-detect

**Decision Needed**: Which option aligns with user needs and intended use cases?

---

### Question 2: Error Message Language

**Question**: Should error messages be internationalized (i18n)?

**Current State**: Frontend Spanish, Backend English

**Options**:
- Keep mixed (current)
- Standardize on English
- Implement full i18n

**Decision Needed**: What is the target user base language preference?

---

### Question 3: Univariate Mode Default

**Question**: Should LSTM default to univariate or multivariate mode in UI?

**Current State**: No explicit default (features empty by default = univariate)

**Options**:
- Default to univariate (current implicit behavior)
- Default to multivariate (pre-select all features)
- Add radio button for explicit mode selection

**Decision Needed**: What is the most common use case for users?

---

### Question 4: Pipeline Config Versioning

**Question**: Should pipeline configs be versioned to support backward compatibility?

**Context**: Adding new fields to LSTM configs may break old experiment loading

**Options**:
- Add `config_version` field
- Keep unversioned (breaking changes acceptable)
- Implement migration scripts

**Decision Needed**: Are old experiments critical to preserve?

---

## Summary

### Immediate Actions Required

**Week 1** (Critical):
1. Implement Bug #1 fix (filter lagged features in frontend)
2. Implement Bug #2 fix (LSTM validation logic)
3. Add post-encoding NaN validation
4. Deploy and test

**Expected Outcome**:
- ✅ LSTM training works for univariate and multivariate modes
- ✅ No lag-of-lag bugs
- ✅ Clear error messages for data issues

---

### Long-Term Improvements

**Weeks 2-4** (High Priority):
- UX improvements (mode indicator, tooltips, better errors)
- Reproducibility enhancements (pipeline config fields)
- Backend validation hardening (defense-in-depth)

**Expected Outcome**:
- ✅ Clear, user-friendly training interface
- ✅ Full experiment reproducibility
- ✅ Robust error handling

---

**Document Status**: ✅ Complete
**Review Date**: 2025-11-14
**Related Documents**:
- [Backend Analysis](2025-11-14_backend-analysis.md)
- [Frontend Analysis](2025-11-14_frontend-analysis.md)
- [Validation Logic](2025-11-14_validation-logic.md)
