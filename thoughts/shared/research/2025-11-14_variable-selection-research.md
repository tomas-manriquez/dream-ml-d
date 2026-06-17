# Variable Selection Research for Time Series Models
**Date:** 2025-11-14
**Objective:** Determine correct/intended business logic for variable selection in ARIMA, XGBoost, and LSTM models

## Research Status
- [x] ARIMA variable selection patterns - COMPLETE
- [x] XGBoost time series feature engineering - COMPLETE
- [x] LSTM univariate vs multivariate modes - COMPLETE
- [x] Industry platform patterns (AWS, Azure, Google) - COMPLETE
- [x] Feature engineering best practices - COMPLETE

**Research Completed:** 2025-11-14
**Total Sources Reviewed:** 15+ (academic papers, documentation, tutorials, platform docs)
**Confidence Level:** HIGH (85-95% across all areas)

## Competing Hypotheses (UPDATED AFTER RESEARCH)

### Hypothesis Tree - Final Confidence Levels
```
H1: Target Variable Selection
├─ H1a: Target should NEVER appear in raw input features list → 95% CONFIRMED ✓
│       Exception: In LSTM, target history can be a sequence feature (conceptually lagged)
└─ H1b: Target can appear for autoregressive models → 5% (REJECTED - use LAGS instead)

H2: ARIMA Feature Selection
├─ H2a: ARIMA is strictly univariate, no external features → 5% (REJECTED)
└─ H2b: ARIMAX supports external regressors → 95% CONFIRMED ✓

H3: Lag Feature Generation
├─ H3a: Lag features should be auto-generated → 70% (UPDATED - for enterprise platforms)
├─ H3b: Users should manually configure lags → 30% (for custom/research implementations)
└─ H3c: HYBRID: Auto-generation with manual override → 90% BEST PRACTICE ✓

H4: LSTM Input Modes
├─ H4a: LSTM requires target history + optional features → 40% (PARTIAL - common but not required)
├─ H4b: LSTM can operate in pure exogenous mode → 60% CONFIRMED ✓ (valid but less common)
├─ H4c: LSTM supports both univariate and multivariate → 95% CONFIRMED ✓
└─ H4d: User should EXPLICITLY choose which mode → 90% CONFIRMED ✓
```

### Key Hypothesis Revisions

**Major Change: Lag Feature Generation**
- Initial hypothesis was binary (auto vs manual)
- Research reveals HYBRID approach is industry standard
- Enterprise platforms (Azure, AWS) auto-generate with override capability
- Custom implementations still require manual engineering
- **Conclusion**: Platform should auto-generate, allow manual override

**Major Change: LSTM External Features**
- Initial hypothesis underestimated flexibility
- LSTM can operate in 3 modes: univariate, multivariate with target, multivariate without target
- Pure exogenous mode (external features only) is VALID but uncommon
- **Conclusion**: Support all modes, make selection EXPLICIT

**Validated: Target Variable Separation**
- Strong consensus across all platforms and documentation
- Target should NEVER auto-appear in feature selection
- Lags of target are features; current target is prediction target
- **Conclusion**: Maintain strict separation in UI and backend

## Research Findings

### ARIMA Models

#### Core Findings
1. **ARIMA is univariate by default** - operates on a single time series
2. **ARIMAX extends ARIMA** with exogenous regressors/external features
3. **External regressors** must be known for both historical and future dates
4. **Feature selection**: Users manually specify which external variables to include via `xreg` parameter

#### Technical Implementation
- R's `auto.arima()`: `auto.arima(y, xreg=X)` where X is a matrix of regressors
- Python's statsmodels: `SARIMAX(endog, exog=None, order=(p,d,q))`
- External features go into the LINEAR component of the model
- Coefficients accessible via `model.params[model.k_trend:model.k_trend + model.k_exog]`

#### When to Use
- **ARIMA**: No external predictors, stationary data, no external influences
- **ARIMAX**: Time series depends on external/control variables (weather, prices, promotions)

#### Key Constraint
- **Major limitation**: Requires reliable forecasts or known future values of external predictors
- Cannot forecast if external feature future values are unknown

**Confidence Level: 95%** - Well-documented standard practice

---

### XGBoost for Time Series

#### Core Findings
1. **XGBoost is NOT natively a time series model** - it's a gradient boosting framework for tabular data
2. **Lag features are MANUALLY created** through feature engineering
3. **Target variable lags** are the PRIMARY mechanism for capturing temporal dependencies
4. **External features** can be added alongside lag features

#### Feature Engineering Patterns
```
Common Features for XGBoost Time Series:
- Lag features: y_lag_1, y_lag_2, ..., y_lag_n (past values of target)
- Rolling statistics: rolling_mean_7d, rolling_std_14d
- External features: temperature, promotions, holidays
- Time-based features: day_of_week, month, season
```

#### Implementation Best Practices
- **Lag selection**: Use ACF/PACF plots to determine optimal lag periods
- **Data loss**: Taking a 14-day lag removes the first 14 rows
- **Collinearity**: Check for collinearity when adding multiple lagged versions
- **Validation**: Use time series cross-validation (NOT random splits)

#### Target Variable as Feature
- **Valid approach**: Using lagged target values (y_t-1, y_t-2) is standard autoregressive modeling
- **NOT data leakage**: As long as lags respect the forecast horizon
- **Warning**: Minimum lag must match forecast horizon to avoid leakage

#### Performance Impact
- Research shows lag features reduce MAE by 35% and MSE by 32% in energy forecasting
- Lag features have "large influence on predictions"

**Confidence Level: 90%** - Industry standard, well-documented

---

### LSTM for Time Series

#### Core Findings
1. **Input Structure**: 3D tensor `[samples, timesteps, features]`
2. **Univariate Mode**: Uses only target variable history (shape: `[n, timesteps, 1]`)
3. **Multivariate Mode**: Can include multiple features (shape: `[n, timesteps, k]`)
4. **Target History**: Typically included in multivariate mode, but NOT required

#### Modes of Operation

**Univariate LSTM:**
- Input: Only historical values of the target variable
- Shape: `(n_samples, sequence_length, 1)`
- Use case: When target has strong autocorrelation
- Example: Predict Sales(t) using Sales(t-1), Sales(t-2), ..., Sales(t-10)

**Multivariate LSTM (with target history):**
- Input: Target history + external features
- Shape: `(n_samples, sequence_length, n_features)` where n_features includes target
- Use case: Combining target autocorrelation with external influences
- Example: Predict Sales(t) using [Sales(t-1), Temp(t-1), Humidity(t-1)], ..., [Sales(t-10), Temp(t-10), Humidity(t-10)]

**Multivariate LSTM (external features only):**
- Input: External features WITHOUT target history
- Shape: `(n_samples, sequence_length, n_features)` where n_features excludes target
- Use case: Predicting from exogenous variables alone
- Example: Predict Sales(t) using [Temp(t-1), Humidity(t-1)], ..., [Temp(t-10), Humidity(t-10)]
- **Note**: Less common, may have lower performance without target history

#### Key Insight
- The feature vector at time t-k includes ALL selected features at that timestep
- At prediction time, must provide ALL features for upcoming timesteps
- If using external features, those features must be known/forecasted for future periods

#### Target Variable Inclusion
- **Common practice**: Include target history in multivariate mode
- **Optional**: Can exclude target and use only external features
- **Best performance**: Usually achieved when target history is included
- **Selection**: Should be EXPLICIT user choice, not automatic

**Confidence Level: 85%** - Standard practice with some implementation variability

---

### Industry Platforms

#### AWS SageMaker (DeepAR, AutoML)

**Feature Types Supported:**
1. **Static categorical features** (`cat` field) - time-independent attributes
2. **Dynamic features** (`dynamic_feat` field) - time-dependent covariates
3. **Holiday calendars** - automatic featurization via `HolidayConfigAttributes`

**Feature Selection Approach:**
- Users explicitly provide features via configuration
- AutoML provides **feature importance** and **column impact** metrics
- Automated feature engineering and selection during training
- Explainability reports show which features contribute most

**Key Pattern**: Explicit feature specification + automated importance ranking

#### Azure AutoML

**Feature Types Supported:**
1. **Target lags** - automatically generated lag features
2. **Rolling window features** - min, max, sum over sliding windows
3. **External regressors/predictors** - user-provided features

**Lag Strategy:**
- **Decoupled from forecast horizon** - e.g., can use lag_order=1 even with horizon=7
- AutoML generates lags with respect to horizon automatically
- Users specify `target_lag` and `target_rolling_window_size`

**Key Constraint:**
- "All features used in training must be available at prediction time"
- Features must be known to the forecast horizon

**Minimum Data Requirement:**
```
min_data = (2 × forecast_horizon) + n_cross_validations + max(max_lags, rolling_window_size)
```

**Key Pattern**: Automatic lag generation + explicit external features

#### Google Cloud Vertex AI (AutoML Forecast)

**Feature Types Supported:**
1. **Available covariates** - known at prediction time (promotions, holidays)
2. **Historical covariates** - only known historically (past weather)

**Feature Selection:**
- AutoML automatically performs feature selection
- Feature importance ranking identifies relevant variables
- Users provide covariates; platform determines which to use

**Key Pattern**: User-provided covariates + automated selection

#### Facebook Prophet

**External Regressors:**
- Added via `add_regressor()` method
- Must be known for both historical AND future dates
- Regressors go into the LINEAR component of the model
- Optional arguments: prior scale, standardization

**Types of Regressors:**
- Marketing investments, weather, external events
- Must have known future values OR be separately forecasted

**Important Note:**
- "Multivariate" in Prophet means "multiple known external variables"
- Does NOT address true multivariate prediction (multiple targets)

**Key Pattern**: Explicit regressor specification with known future values

---

### Feature Engineering Best Practices

#### Automatic vs Manual Lag Features

**Automatic Generation (Preferred for production platforms):**
- **Pros**: Reduces manual effort, handles domain complexity, consistent methodology
- **Cons**: Black box, may generate unnecessary features, requires feature selection
- **When to use**: Production platforms (Azure, AWS), AutoML systems, exploratory analysis
- **Examples**: DataRobot, Azure AutoML, Feature-engine library

**Manual Configuration (Preferred for research/custom solutions):**
- **Pros**: Full control, domain expertise incorporated, interpretable
- **Cons**: Time-consuming, requires expertise, error-prone
- **When to use**: Research, domain-specific optimization, custom pipelines
- **Tools**: Pandas, scikit-learn pipelines, custom functions

#### Industry Pattern
- **Enterprise platforms**: Automatic lag generation with manual override option
- **Custom implementations**: Manual feature engineering with helper libraries
- **Best practice**: Start automatic, refine manually based on domain knowledge

#### Rolling Window Features
- **Automatic generation**: Feature-engine, tsfresh libraries
- **Common windows**: 3, 7, 14, 30 days (weekly/monthly patterns)
- **Statistics**: mean, std, min, max, sum
- **Selection**: Use feature importance to prune

#### Target Variable in Feature List

**Strong Consensus: Target should NOT appear in the raw input feature list**

**Correct Approaches:**
1. **For autoregressive models**: Use LAGGED target (y_t-1, y_t-2) as features, NOT current target
2. **Separation of concerns**: Input features ≠ Target variable
3. **LSTM/XGBoost**: Create explicit lag features; target stays separate
4. **ARIMA**: Target is `endog` parameter; external features are `exog` parameter

**Why This Matters:**
- **Data leakage**: Using y_t to predict y_t is circular
- **Clarity**: Separating target from features prevents confusion
- **Standard practice**: All major ML libraries maintain this separation

**Exception:**
- In LSTM implementations where "features" means "sequence columns", target history can be ONE of the sequence features
- But this is really "lagged target" conceptually, represented as a sequence dimension

---

### UX/Interface Patterns

#### Common Patterns Across Platforms

**1. Separate Target Selection**
- Target variable selected independently from input features
- Usually a dropdown or radio button
- Never auto-included in feature list

**2. Feature Type Classification**
- Static vs Dynamic (AWS)
- Available vs Historical (Google)
- Lags vs External (Azure)
- Clear visual separation in UI

**3. Mode Indicators**
- Show univariate vs multivariate mode
- Display expected input shapes (especially for LSTM)
- Real-time validation and warnings

**4. Auto-Selection Behavior**
- **ARIMA/XGBoost**: Often auto-select all numeric features EXCEPT target and date
- **LSTM**: Varies; some platforms disable auto-selection for explicit control
- **Override**: Users can always modify auto-selected features

**5. Validation Warnings**
- Warn if features won't be available at prediction time
- Indicate minimum data requirements
- Show feature importance after training

#### UX Best Practices
1. **Clear labeling**: "Input Features" vs "Target Variable" vs "Date Column"
2. **Visual hierarchy**: Most important selections at top
3. **Progressive disclosure**: Advanced options (lags, windows) in expandable sections
4. **Real-time feedback**: Immediate validation, shape indicators
5. **Smart defaults**: Sensible auto-selection with easy override

---

## References

### Academic/Technical Documentation
1. Duke University - "ARIMA models with regressors" (https://people.duke.edu/~rnau/arimreg.htm)
2. Forecasting: Principles and Practice (3rd ed) - "Regression with ARIMA errors" (https://otexts.com/fpp3/regarima.html)
3. statsmodels documentation - SARIMAX (https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
4. scikit-learn - "Lagged features for time series forecasting" (https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html)

### Industry Platforms
5. AWS SageMaker - DeepAR and AutoML Time Series (https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html)
6. Azure Machine Learning - AutoML Forecasting (https://learn.microsoft.com/en-us/azure/machine-learning/concept-automl-forecasting-methods)
7. Azure - "Lag features for time-series forecasting" (https://learn.microsoft.com/en-us/azure/machine-learning/concept-automl-forecasting-lags)
8. Facebook Prophet - Documentation (https://facebook.github.io/prophet/)

### Tutorials and Best Practices
9. Machine Learning Mastery - "XGBoost for Time Series Forecasting" (https://machinelearningmastery.com/xgboost-for-time-series-forecasting/)
10. Machine Learning Mastery - "Multivariate Time Series Forecasting with LSTMs in Keras" (https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)
11. Analytics Vidhya - "6 Powerful Feature Engineering Techniques For Time Series Data" (https://www.analyticsvidhya.com/blog/2019/12/6-powerful-feature-engineering-techniques-time-series/)
12. Medium - "How I Trained a Time-Series Model with XGBoost and Lag Features" (https://medium.com/@connect.hashblock/how-i-trained-a-time-series-model-with-xgboost-and-lag-features-8c17439c81e4)

### Stack Overflow / Community
13. Cross Validated - "Identify appropriate exogenous variables for ARIMA" (https://stats.stackexchange.com/questions/270795)
14. Stack Overflow - "XGBoost for Time series - using lag of target variables" (https://stats.stackexchange.com/questions/400897)
15. Data Science SE - "Recommended model for univariate or multivariate time series forecasting" (https://datascience.stackexchange.com/questions/42872)

---

## RECOMMENDATIONS: Variable Selection Logic by Model Type

### 1. ARIMA/ARIMAX

#### Recommended Variable Selection Logic

```python
# Input Configuration
target_variable: str          # Required - single target
external_features: list[str]  # Optional - empty for ARIMA, populated for ARIMAX
date_column: str             # Required

# Backend Behavior
if external_features is empty:
    → Use standard ARIMA (univariate)
    → endog = target_variable only
    → No xreg parameter
else:
    → Use ARIMAX (with external regressors)
    → endog = target_variable
    → exog = external_features
    → VALIDATE: All external features must be known for forecast horizon
```

#### UX Recommendations
- **Auto-selection**: Auto-select all numeric columns EXCEPT target and date
- **User override**: Allow users to deselect features
- **Validation warning**: "ARIMAX requires known future values for: [feature_list]"
- **Label**: "External Features (Optional)" - make it clear they're optional
- **Separate sections**: "Target Variable" | "External Features" | "Date Column"

#### Implementation Priority
- **Must have**: Basic ARIMA with optional external features
- **Nice to have**: Validation for feature availability at prediction time
- **Advanced**: Automatic feature importance analysis post-training

---

### 2. XGBoost for Time Series

#### Recommended Variable Selection Logic

```python
# Input Configuration
target_variable: str                    # Required
external_features: list[str]           # Optional - user-selected base features
lag_config: dict                       # CRITICAL - defines autoregressive behavior
  - lag_periods: list[int]             # e.g., [1, 2, 3, 7, 14]
  - rolling_windows: list[int]         # e.g., [7, 14, 30]
  - include_target_lags: bool          # Default: True
date_column: str

# Backend Feature Engineering
generated_features = []

if lag_config.include_target_lags:
    for lag in lag_config.lag_periods:
        generated_features.append(f"{target_variable}_lag_{lag}")
    for window in lag_config.rolling_windows:
        generated_features.append(f"{target_variable}_rolling_mean_{window}")
        generated_features.append(f"{target_variable}_rolling_std_{window}")

for feature in external_features:
    generated_features.append(feature)  # Original external feature
    for lag in lag_config.lag_periods:
        generated_features.append(f"{feature}_lag_{lag}")

final_features = generated_features
# Remove target from final_features (target is y, not X)
```

#### UX Recommendations
- **Two-stage selection**:
  1. **Base features**: User selects external features (temperature, promotion, etc.)
  2. **Lag configuration**: User configures which lags to generate
- **Smart defaults**:
  - lag_periods: [1, 2, 3, 7, 14] (daily data)
  - rolling_windows: [7, 14] (weekly, biweekly)
  - include_target_lags: True
- **Advanced mode**: Toggle to show/hide lag configuration
- **Visual indicator**: Show "18 features will be generated" based on configuration
- **Auto-selection for base features**: All numeric EXCEPT target and date
- **Target lag toggle**: "Include target history (autoregressive mode)" - default ON

#### Implementation Priority
- **Must have**:
  - Target lag generation (autoregressive)
  - Manual lag period specification
- **Nice to have**:
  - Automatic lag detection using ACF/PACF
  - External feature lags
  - Rolling window statistics
- **Advanced**:
  - Automatic feature selection post-generation
  - Lag importance visualization

---

### 3. LSTM for Time Series

#### Recommended Variable Selection Logic

```python
# Input Configuration
target_variable: str                    # Required
sequence_features: list[str]           # EXPLICIT selection - can be empty
sequence_length: int                   # Required - lookback window
date_column: str

# Backend Sequence Creation
if sequence_features is empty:
    # UNIVARIATE MODE
    features_for_sequences = [target_variable]
    training_mode = "univariate"
    input_shape = (n_samples, sequence_length, 1)

elif target_variable in sequence_features:
    # MULTIVARIATE MODE WITH TARGET HISTORY
    features_for_sequences = sequence_features  # Includes target
    training_mode = "multivariate_with_target_history"
    input_shape = (n_samples, sequence_length, len(sequence_features))

else:
    # MULTIVARIATE MODE - EXTERNAL FEATURES ONLY
    features_for_sequences = sequence_features  # Excludes target
    training_mode = "multivariate_external_only"
    input_shape = (n_samples, sequence_length, len(sequence_features))
    # WARNING: Performance may be lower without target history

# IMPORTANT: At prediction time, must provide sequence_length timesteps
# of ALL features in sequence_features
```

#### UX Recommendations
- **NO auto-selection** - require explicit user choice
- **Three modes with clear descriptions**:

  **Mode A: Univariate (Target History Only)**
  ```
  ○ Univariate Mode
    "Predict using only past values of the target"
    Sequence features: (empty)
    Input shape: (n, 10, 1)
  ```

  **Mode B: Multivariate with Target History**
  ```
  ○ Multivariate Mode - With Target History
    "Predict using target history + external features"
    Sequence features: [✓] Sales (Target History)
                       [✓] Temperature
                       [✓] Humidity
    Input shape: (n, 10, 3)
  ```

  **Mode C: Multivariate - External Only**
  ```
  ○ Multivariate Mode - External Features Only
    "Predict using only external features (no target history)"
    Sequence features: [ ] Sales (Target History)
                       [✓] Temperature
                       [✓] Humidity
    Input shape: (n, 10, 2)
    ⚠️ Warning: May have lower accuracy without target history
  ```

- **Real-time indicators**:
  - Show current mode based on selection
  - Display expected input shape dynamically
  - Update immediately when features change

- **Feature labeling**:
  - Mark target with "(Target History)" when it appears in feature list
  - Show external features without special marking

- **Validation**:
  - Allow empty features (univariate mode)
  - Don't force feature selection
  - Warn if external features might not be available at prediction time

#### Implementation Priority
- **Must have**:
  - Empty features = univariate fallback ✓ (partially implemented)
  - Explicit feature selection (no auto-selection) ✓ (partially implemented)
  - All three modes supported ✓ (implemented)
  - Real-time mode indicators ✓ (implemented)

- **Bugs to fix** (from error report):
  - **Error 1**: NaN values in training - need data validation before sequence creation
  - **Error 2-3**: Frontend validation incorrectly blocking empty features
  - Lag features being auto-generated when they shouldn't be

- **Nice to have**:
  - Mode selector radio buttons (instead of inferring from selection)
  - Sequence preview visualization
  - Automatic sequence_length suggestion based on data seasonality

---

## CRITICAL FINDINGS: Current Implementation Issues

Based on the error report (`2025-11-14_lstm_bugs_report.md`) and implementation plan review:

### Issue 1: Lag Features Auto-Generation for LSTM
**Problem**: The formData shows `"input_features":["Temperature","Temperature_lag_1"]` but user only selected Temperature
**Root cause**: Lag features are being auto-generated for LSTM when they shouldn't be
**Expected behavior**: LSTM should use RAW features in sequences, not pre-generated lags
**Fix required**: Disable lag/rolling window feature generation for LSTM algorithm

### Issue 2: Frontend Validation Blocking Empty Features
**Problem**: Cannot submit training even with Temperature selected (Errors 2-3)
**Root cause**: Validation logic `!inputFeatures.length` blocks submission
**Expected behavior**: LSTM should allow empty features (univariate mode)
**Fix required**: Task 13 in phase4-continuation.md - modify validation to:
```javascript
(algorithm !== "lstm" && !inputFeatures.length)
```

### Issue 3: NaN Values in Training
**Problem**: Model training produces NaN losses (Error 1)
**Root cause**: Likely data validation issue - NaN values in input data not caught
**Expected behavior**: Validate data before sequence creation
**Fix required**: Add data validation:
```python
# Before sequence creation
if df[features_for_sequences].isnull().any().any():
    raise ValueError(f"NaN values detected in features: {features_for_sequences}")
```

### Issue 4: Confusion Between Feature Types
**Current state**: Frontend has multiple feature-related variables causing confusion:
- `inputFeatures` (global, used by ARIMA/XGBoost)
- `lstmSelectedFeatures` (LSTM-specific)
- `externalFeatures` (mentioned in error report)

**Recommended clarity**:
```javascript
// For ARIMA/XGBoost - auto-selection enabled
inputFeatures: []         // Auto-populated, user can modify

// For LSTM - explicit selection, no auto-fill
lstmSequenceFeatures: []  // Rename for clarity - these go into sequences
                          // Empty = univariate mode (backend adds target)
                          // Can include target explicitly
```

---

## FINAL RECOMMENDATIONS SUMMARY

### Universal Principles (All Models)
1. **Target separation**: Target variable NEVER auto-included in feature selection
2. **Explicit is better than implicit**: Users should understand what features are being used
3. **Validation at every layer**: Frontend validation + Backend validation
4. **Clear mode indicators**: Show users what mode they're in (univariate/multivariate)

### Model-Specific Patterns

| Aspect | ARIMA/ARIMAX | XGBoost | LSTM |
|--------|--------------|---------|------|
| **Auto-select features** | Yes (all numeric) | Yes (all numeric) | NO - explicit only |
| **Lag generation** | N/A (not applicable) | YES - automatic | NO - uses sequences |
| **Empty features allowed** | Yes (→ ARIMA) | No (need features) | YES (→ univariate) |
| **Target in features** | Never | Never (use lags) | Optional (as sequence feature) |
| **External features** | Optional (ARIMAX) | Optional | Optional |

### Implementation Priorities

**Immediate fixes (for current bugs)**:
1. Fix validation to allow empty LSTM features ✓ (Task 13)
2. Add NaN data validation before LSTM training
3. Prevent lag auto-generation for LSTM algorithm
4. Complete Task 12 (payload construction with lstmSelectedFeatures)

**Short-term improvements**:
1. Implement lag configuration UI for XGBoost
2. Add mode indicators for all algorithms
3. Improve feature labeling (especially for LSTM)
4. Add prediction-time feature availability warnings

**Long-term enhancements**:
1. Automatic lag detection using ACF/PACF for XGBoost
2. Feature importance visualization post-training
3. LSTM sequence preview
4. Automatic optimal sequence_length suggestion
5. Advanced mode with manual feature engineering controls

---

## CONCLUSION

This research confirms that **variable selection logic differs significantly between model types**:

- **ARIMA/ARIMAX**: Simple external regressors model with manual feature specification
- **XGBoost**: Requires extensive feature engineering (lags, rolling windows) which should be automated
- **LSTM**: Flexible sequence-based approach requiring explicit user control over what goes into sequences

The current implementation (Phase 4) is on the right track but has critical bugs that need fixing before it can be considered complete. The core architecture - separating `lstmSelectedFeatures` from `inputFeatures` - is correct and aligns with industry best practices.
