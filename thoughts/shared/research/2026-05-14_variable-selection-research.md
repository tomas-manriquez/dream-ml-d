# Research: Time Series Model Training and Variable Selection

**Date:** 2026-05-14
**Scope:** DREAM-ML Backend (`apiTimeSeries`) and Frontend (`TSTrainCard.jsx`)
**Objective:** Document the training dataflow to support LSTM training bug fixes and model architecture understanding.

---

## 1. Summary of Findings
The DREAM-ML system employs a robust, three-tiered architecture for time series model training. Variable selection for model training (output variables, input features, and date column) begins in the React `TSTrainCard.jsx` frontend, is sent as a `multipart/form-data` payload to the backend, processed by Django views, orchestrated by a service layer with MLflow/DVC versioning, and ultimately executed in the training layer (`train.py`). 

The current system enforces strict temporal ordering and supports univariate/multivariate modes based on user selections.

---

## 2. Code References

* `DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx`: Frontend entry point and payload construction (`handleTrain`).
* `DREAM-ML-backend/GEML/apiTimeSeries/views.py`: Backend API endpoint `train_model`.
* `DREAM-ML-backend/GEML/apiTimeSeries/services.py`: Orchestration layer using `TrainModelService.train_model_logic`.
* `DREAM-ML-backend/GEML/apiTimeSeries/train.py`: Implementation of training models (ARIMA, XGBoost, LSTM, PatchTSMixer).

---

## 3. Data Model

### Core Entities (Payload)
The training process uses a consolidated payload sent from the frontend:

| Field | Type | Description |
|---|---|---|
| `model_name` | `str` | Name for MLflow tracking |
| `input_features` | `List[str]` | Columns used as input features (X) |
| `target_variable` | `str` | The variable to forecast (Y) |
| `date_col_name` | `str` | Column used for temporal ordering |
| `algorithm` | `str` | Algorithm selected ("arima", "xgboost", "lstm", "patchtsmixer") |
| `forecast_horizon`| `int` | Prediction steps ahead |
| `split_ratios` | `dict`| Temporal split ratios (train/val/test) |
| `manual_params` | `dict`| Hyperparameters for the selected model |

### Transformations Across Layers
1. **Frontend:** User selects variables via UI checkboxes/radios → Aggregated into `inputFeatures`, `targetVariable`, `dateColumnName`.
2. **Payload Construction:** Parameters serialized to JSON strings → appended to `FormData` along with CSV file.
3. **Backend API:** View parses JSON and extracts ML configurations.
4. **Service Layer:** Data is loaded and indexed by the date column. Training data versioned via DVC.
5. **Training Layer:** The data is transformed into algorithm-specific shapes (ARIMA: `exog`, XGBoost: `supervised dataset`, LSTM: `3D tensors`).

---

## 4. Architecture Insights

* **Reproducibility:** All algorithms use fixed random seeds (SEED=42) and enforce reproducibility (DVC/MLflow) to ensure consistency.
* **Separation of Concerns:** A clean division exists between HTTP handling (Views), business logic (Services), and model training implementation (Train Layer).
* **Data Integrity:** The pipeline uses strict temporal ordering to prevent data leakage.
* **Algorithm Flexibility:** Algorithms can auto-select or manually select features (with LSTM/PatchTSMixer being strictly manual to prevent data leakage).

---

## 5. Implementation Patterns

* **ARIMA/SARIMA**: Auto-selects numeric columns as `exog` variables.
* **XGBoost**: Uses `input_features` to generate supervised datasets where inputs are shifted by the `forecast_horizon`.
* **LSTM/PatchTSMixer**: Requires explicit numeric feature selection. Defaults to univariate mode (using `[target_variable]` only) if no features are provided.

---

## 6. Open Questions
* Are there edge cases for PatchTSMixer regarding univariate vs multivariate input channels that conflict with current LSTM expectations?
* Can we standardize the feature selection mechanism across all algorithms to remove the conditional logic currently present in the frontend?
* Should we tighten validation of "numeric-only" features to happen at the frontend level instead of the backend layer?
