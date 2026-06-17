# Classification Model Training Workflow Analysis
**Date:** 2025-12-27
**Purpose:** Understand tabular classification training workflow to implement Bayesian Search hyperparameter tuning
**Scope:** Frontend TrainCard.jsx, Backend API (/api/train-model), Time Series Bayesian patterns

---

## Executive Summary

This document provides a comprehensive analysis of DREAM-ML's classification model training workflow, from frontend user interaction through backend execution. The analysis reveals a **well-architected, three-layer system** with DVC versioning, MLflow tracking, and energy monitoring. The frontend is **already fully prepared** for Bayesian search with complete UI components and configuration structures. The backend supports three classification algorithms (Logistic Regression, MLP, XGBoost) with manual, grid, and random search strategies, but **Bayesian search is not yet implemented** despite being declared as valid.

The time series implementation provides a **production-ready blueprint** using **Optuna 4.6.0 with TPE sampling**, including advanced features like convergence detection, memory monitoring, and configurable parameter ranges.

**Key Findings:**
- **Frontend Status:** 100% ready for Bayesian search with parameter space definition UI
- **Backend Status:** Infrastructure ready, but Bayesian objective functions need implementation
- **Reference Implementation:** Time series Bayesian search (ARIMA, XGBoost, LSTM) fully functional
- **Architecture:** Clean 3-layer separation (API → Service → Training)
- **Gap:** Backend train.py validates "bayesian" strategy but returns NotImplemented

---

## Table of Contents

1. [Frontend Workflow Analysis](#1-frontend-workflow-analysis)
2. [Backend Architecture](#2-backend-architecture)
3. [Complete Data Flow](#3-complete-data-flow)
4. [Time Series Bayesian Patterns](#4-time-series-bayesian-patterns)
5. [Implementation Blueprint](#5-implementation-blueprint)
6. [Open Questions](#6-open-questions)

---

## 1. Frontend Workflow Analysis

### 1.1 User Interaction Flow (TrainCard.jsx)

**File:** `DREAM-ML-frontend/frontend/src/components/TrainCard.jsx`

#### Step 1: File Selection
```javascript
// Lines 339-345
const handleFileChange = (event) => {
  setCsvFile(event.target.files[0]);
  setColumns([]);
  setInputFeatures([]);
  setTargetVariable("");
  setTrainStatus("📂 Archivo CSV seleccionado.");
};
```

#### Step 2: Load Columns from CSV
```javascript
// Lines 348-371: POST /analyze-csv/
const loadColumns = async () => {
  const formData = new FormData();
  formData.append("file", csvFile);
  const response = await axios.post("/analyze-csv/", formData);
  setColumns(response.data.columns);
};
```

#### Step 3: Variable Selection
- **Input Features:** Multi-select checkboxes (Lines 374-395)
- **Target Variable:** Single-select radio buttons (Lines 374-395)
- **Auto-selection:** When target selected, remaining columns auto-added as features

#### Step 4: Algorithm & Optimization Configuration
```javascript
// Lines 72-74
const [algorithm, setAlgorithm] = useState("logistic"); // "logistic" | "mlp" | "xgboost"
const [problemType, setProblemType] = useState("binary"); // For MLP/XGBoost
const [modelName, setModelName] = useState("");

// Lines 117-126: Optimization method
const [optimizationMethod, setOptimizationMethod] = useState("manual");
// "manual" | "grid" | "random" | "bayesian"
```

### 1.2 Bayesian Search UI (Already Implemented!)

**Bayesian Parameter Space Definition:**

```javascript
// Lines 157-271: Bayesian search parameter configurations

// Logistic Regression Bayesian Params
const [logisticBayesianParams, setLogisticBayesianParams] = useState({
  C: {
    type: "real",
    distribution: "log-uniform",
    low: 0.001,
    high: 100.0
  },
  max_iter: {
    type: "integer",
    low: 100,
    high: 1000
  },
  solver: {
    type: "categorical",
    choices: ["lbfgs", "liblinear", "saga"]
  },
  penalty: {
    type: "categorical",
    choices: ["l2", "none"]
  }
});

// MLP Bayesian Params
const [mlpBayesianParams, setMlpBayesianParams] = useState({
  hidden_layer_sizes: {
    type: "categorical",
    choices: [[4], [10], [10, 5], [50], [100], [100, 50], [100, 50, 10]]
  },
  activation: {
    type: "categorical",
    choices: ["relu", "tanh", "logistic"]
  },
  learning_rate_init: {
    type: "real",
    distribution: "log-uniform",
    low: 0.0001,
    high: 0.1
  }
});

// XGBoost Bayesian Params
const [xgboostBayesianParams, setXgboostBayesianParams] = useState({
  n_estimators: { type: "integer", low: 50, high: 500 },
  max_depth: { type: "integer", low: 3, high: 10 },
  learning_rate: { type: "real", distribution: "log-uniform", low: 0.01, high: 0.3 },
  subsample: { type: "real", distribution: "uniform", low: 0.5, high: 1.0 },
  colsample_bytree: { type: "real", distribution: "uniform", low: 0.5, high: 1.0 },
  gamma: { type: "real", distribution: "uniform", low: 0.0, high: 5.0 },
  min_child_weight: { type: "integer", low: 1, high: 10 },
  reg_alpha: { type: "real", distribution: "uniform", low: 0.0, high: 1.0 },
  reg_lambda: { type: "real", distribution: "uniform", low: 0.0, high: 1.0 }
});
```

**Advanced Bayesian Configuration:**

```javascript
// Lines 262-271
const [bayesianAdvancedConfig, setBayesianAdvancedConfig] = useState({
  n_initial_points: 10,           // Random trials before TPE starts
  acq_func: "EI",                 // Acquisition function: "EI" | "PI" | "LCB"
  random_state: null,             // Optional seed
  max_memory_mb: null,            // Memory limit
  timeout_seconds: null,          // Time limit
  convergence_tolerance: 0.001,   // Minimum improvement threshold
  convergence_patience: 5,        // Trials without improvement before stopping
  save_gp_model: true             // Save Gaussian Process model
});
```

### 1.3 API Request Construction

**File:** `TrainCard.jsx:632-749`

```javascript
const handleTrain = async () => {
  // Validation (Lines 633-656)
  if (!csvFile || !inputFeatures.length || !targetVariable || !modelName) {
    setTrainStatus("❌ Complete todos los campos requeridos");
    return;
  }

  // Build base payload (Lines 661-685)
  const payload = {
    model_name: modelName,
    input_features: inputFeatures,
    target_variable: targetVariable,
    experiment_dir: experimentDir,
    split_ratios: splitRatios,  // {train: 0.7, val: 0.15, test: 0.15}
    run_id: runId,
    algorithm: algorithm,  // "logistic" | "mlp" | "xgboost"
    params: finalParams,  // Algorithm-specific manual params
    hyperparameter_search_strategy: optimizationMethod === "manual" ? "none" : optimizationMethod
  };

  // Add Bayesian configuration (Lines 700-720)
  if (optimizationMethod === "bayesian") {
    payload.n_bayesian_iterations = nBayesianIterations;  // Default: 50
    payload.bayesian_search_params = {
      // Algorithm-specific Bayesian parameter space
      // (logisticBayesianParams | mlpBayesianParams | xgboostBayesianParams)
    };
    payload.bayesian_config = {
      // Filter null values from bayesianAdvancedConfig
      n_initial_points: bayesianAdvancedConfig.n_initial_points,
      acq_func: bayesianAdvancedConfig.acq_func,
      convergence_tolerance: bayesianAdvancedConfig.convergence_tolerance,
      convergence_patience: bayesianAdvancedConfig.convergence_patience,
      max_memory_mb: bayesianAdvancedConfig.max_memory_mb,
      timeout_seconds: bayesianAdvancedConfig.timeout_seconds,
      save_gp_model: bayesianAdvancedConfig.save_gp_model
    };
  }

  // Create FormData (Lines 727-733)
  const formData = new FormData();
  formData.append("file", csvFile);
  formData.append("data", JSON.stringify(payload));

  // API Call (Line 736)
  const response = await axios.post("/train-model/", formData);
};
```

**Complete Payload Structure:**

```json
{
  "model_name": "LogisticRegression_Bayesian",
  "input_features": ["feature1", "feature2", "feature3"],
  "target_variable": "target",
  "experiment_dir": "/path/to/experiment",
  "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
  "run_id": "mlflow_run_id",
  "algorithm": "logistic",
  "hyperparameter_search_strategy": "bayesian",
  "n_bayesian_iterations": 50,
  "bayesian_search_params": {
    "C": {"type": "real", "distribution": "log-uniform", "low": 0.001, "high": 100.0},
    "max_iter": {"type": "integer", "low": 100, "high": 1000},
    "solver": {"type": "categorical", "choices": ["lbfgs", "liblinear", "saga"]},
    "penalty": {"type": "categorical", "choices": ["l2", "none"]}
  },
  "bayesian_config": {
    "n_initial_points": 10,
    "acq_func": "EI",
    "convergence_tolerance": 0.001,
    "convergence_patience": 5,
    "max_memory_mb": null,
    "timeout_seconds": null,
    "save_gp_model": true
  },
  "params": {
    "regularization": "1.0",
    "maxIter": "100",
    "solver": "lbfgs"
  }
}
```

---

## 2. Backend Architecture

### 2.1 Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: API Entry Point (views.py)                            │
│  - Request validation                                           │
│  - MLflow tracking URI setup                                    │
│  - Error handling with HTTP status codes                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Service Orchestration (services.py)                   │
│  - Single MLflow run management                                 │
│  - Dataset saving and DVC versioning                            │
│  - Algorithm dispatch                                           │
│  - Model DVC versioning                                         │
│  - Pipeline config updates                                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Training Implementation (train.py)                    │
│  - Data loading and validation                                  │
│  - Stratified train/val/test split                              │
│  - Hyperparameter search (manual, grid, random, [bayesian])    │
│  - Model training and evaluation                                │
│  - Metrics calculation and visualization                        │
│  - Energy tracking (CodeCarbon)                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 API Layer (views.py)

**File:** `DREAM-ML-backend/GEML/api/views.py:782-886`

```python
@csrf_exempt
def train_model(request):
    """
    Endpoint: POST /api/train-model/
    Handles classification model training requests.
    """
    if request.method != 'POST':
        return JsonResponse({"status": "error", "message": "Solo POST"}, status=405)

    # Validation (Lines 794-807)
    if 'file' not in request.FILES:
        return JsonResponse({"status": "error", "message": "No CSV file"}, status=400)

    if 'data' not in request.POST:
        return JsonResponse({"status": "error", "message": "No config data"}, status=400)

    data = json.loads(request.POST['data'])

    # MLflow configuration (Lines 809-817)
    experiment_dir = data.get("experiment_dir")
    base_dir = os.path.dirname(experiment_dir)
    shared_db_path = os.path.join(base_dir, "shared_mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{shared_db_path}")

    # Preventive cleanup
    if mlflow.active_run():
        mlflow.end_run()

    # Execute training (Lines 819-828)
    try:
        result = train_model_logic(dataset_file=request.FILES['file'], data=data)

        # Final cleanup
        if mlflow.active_run():
            mlflow.end_run()

        # Success response (Lines 838-844)
        return JsonResponse({
            "status": "success",
            "run_id": result.get("run_id"),
            "metrics": result.get("val_metrics"),
            "model_path": result.get("model_path"),
            "mlflow_ui": f"http://localhost:5000/#/experiments/{mlflow_experiment_id}"
        }, status=200)

    except ValueError as ve:
        return JsonResponse({"status": "error", "error_details": str(ve)}, status=400)
    except Exception as e:
        return JsonResponse({"status": "error", "error_details": str(e)}, status=500)
```

### 2.3 Service Layer (services.py)

**File:** `DREAM-ML-backend/GEML/api/services.py:1078-1224`

```python
def train_model_logic(dataset_file, data: dict) -> dict:
    """
    Orchestrates training workflow with DVC versioning and MLflow tracking.
    Executes in a SINGLE MLflow run.
    """
    # Validation (Lines 1090-1098)
    experiment_dir = data.get("experiment_dir")
    if not experiment_dir or not os.path.exists(experiment_dir):
        raise FileNotFoundError("Directorio de experimento no encontrado")

    algorithm = data.get("algorithm", "logistic").lower()
    supported_algorithms = ["logistic", "mlp", "xgboost"]
    if algorithm not in supported_algorithms:
        raise ValueError(f"Algoritmo no soportado: {algorithm}")

    # MLflow setup (Lines 1101-1114)
    base_dir = os.path.dirname(experiment_dir)
    shared_db_path = os.path.join(base_dir, "shared_mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{shared_db_path}")

    experiment_name = os.path.basename(experiment_dir)
    mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
    mlflow_experiment_id = mlflow_experiment.experiment_id

    # Single run context (Lines 1116-1223)
    with start_run(experiment_id=mlflow_experiment_id,
                   description=f"Entrenamiento {algorithm}",
                   log_system_metrics=True) as run:
        run_id = run.info.run_id

        # 1. Save dataset and version with DVC (Lines 1122-1142)
        trained_dir = os.path.join(experiment_dir, "trained")
        os.makedirs(trained_dir, exist_ok=True)

        dataset_filename = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        dataset_path = os.path.join(trained_dir, dataset_filename)

        with open(dataset_path, 'wb') as f:
            for chunk in dataset_file.chunks():
                f.write(chunk)

        subprocess.run(["dvc", "add", dataset_path], cwd=experiment_dir, check=True)
        subprocess.run(["git", "add", f"{dataset_path}.dvc"], cwd=experiment_dir, check=True)
        subprocess.run(["git", "commit", "-m", f"[DVC] Add training dataset {dataset_filename}"],
                       cwd=experiment_dir, check=True)
        subprocess.run(["dvc", "push", dataset_path], cwd=experiment_dir, check=True)

        # 2. Register dataset in MLflow (Lines 1145-1152)
        df_train = pd.read_csv(dataset_path)
        train_dataset = mlflow.data.from_pandas(df_train, source=None,
                                                name="Dataset de Entrenamiento")
        mlflow.log_input(train_dataset, context="train_data")

        # 3. Algorithm dispatch (Lines 1155-1173)
        if algorithm == "logistic":
            result = train_logistic_regression_model(dataset_path, data, experiment_dir)
        elif algorithm == "mlp":
            result = train_mlp_model(dataset_path, data, experiment_dir)
        elif algorithm == "xgboost":
            result = train_xgboost_model(dataset_path, data, experiment_dir)

        # 4. Version model with DVC (Lines 1175-1184)
        model_path = result.get("model_path")
        subprocess.run(["dvc", "add", model_path], cwd=experiment_dir, check=True)
        subprocess.run(["git", "commit", "-m", f"[DVC] Add model"], cwd=experiment_dir, check=True)
        subprocess.run(["dvc", "push", model_path], cwd=experiment_dir, check=True)

        # 5. Log metrics (Lines 1186-1196)
        mlflow.log_metrics(result["val_metrics"])
        mlflow.log_metrics(result["test_metrics"])

        return {
            "status": "success",
            "run_id": run_id,
            "val_metrics": result["val_metrics"],
            "test_metrics": result["test_metrics"],
            "model_path": model_path
        }
```

### 2.4 Training Layer (train.py)

**File:** `DREAM-ML-backend/GEML/api/train.py`

#### Logistic Regression Training

```python
# Lines 426-652
def train_logistic_regression_model(dataset_path, data, experiment_dir):
    """
    Trains logistic regression classifier with hyperparameter search support.
    """
    # Extract parameters (Lines 437-442)
    input_features = data["input_features"]
    target_variable = data["target_variable"]
    hyperparams = data.get("params", {})
    split_ratios = data.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    model_name = data.get("model_name", "LogisticRegression_Model")

    # Strategy validation (Lines 444-455)
    hyperparameter_search_strategy = data.get("hyperparameter_search_strategy", "none")
    valid_strategies = ["none", "grid", "random", "bayesian"]
    if hyperparameter_search_strategy not in valid_strategies:
        raise ValueError(f"Strategy must be one of: {valid_strategies}")

    # Load and split data (Lines 469-472)
    df = load_and_validate_data(dataset_path, input_features, target_variable)
    X = df[input_features]
    y = df[target_variable]
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, split_ratios)

    # Grid Search (Lines 494-517)
    if hyperparameter_search_strategy == "grid":
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "max_iter": [100, 200],
            "solver": ["lbfgs", "liblinear"]
        }
        grid_search = GridSearchCV(LogisticRegression(random_state=SEED),
                                   param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

    # Random Search (Lines 518-559)
    elif hyperparameter_search_strategy == "random":
        n_random_iterations = data.get("n_random_iterations", 50)
        random_search_params = data.get("random_search_params", {})

        best_score = 0.0
        best_model = None

        for i in range(n_random_iterations):
            random_params = generate_random_logistic_params(random_search_params)
            model_trial = LogisticRegression(**random_params)
            model_trial.fit(X_train, y_train)
            val_score = accuracy_score(y_val, model_trial.predict(X_val))

            if val_score > best_score:
                best_score = val_score
                best_model = model_trial

        model = best_model

    # Manual Training (Lines 561-576)
    else:  # strategy == "none"
        model_params = {
            "C": float(hyperparams.get("regularization", 1.0)),
            "max_iter": int(hyperparams.get("maxIter", 100)),
            "solver": hyperparams.get("solver", "lbfgs")
        }
        model = LogisticRegression(random_state=SEED, **model_params)
        model.fit(X_train, y_train)

    # Evaluation (Lines 578-584)
    val_metrics, val_artifacts = evaluate_model(model, X_val, y_val, "val", "binary", experiment_dir)
    test_metrics, test_artifacts = evaluate_model(model, X_test, y_test, "test", "binary", experiment_dir)

    # MLflow logging (Lines 586-613)
    signature = infer_signature(X_val, model.predict(X_val))
    mlflow.sklearn.log_model(model, artifact_path="logistic_regression_model",
                             signature=signature, registered_model_name=model_name)

    # Save model locally (Lines 617-621)
    model_path = os.path.join(experiment_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)

    return {
        "model_path": model_path,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    }
```

**KEY FINDING:** Lines 452-455 declare "bayesian" as valid strategy, but there's **NO implementation** for it. The code only handles "grid", "random", and "none" (manual).

#### XGBoost Training

**File:** `train.py:909-1176`

Similar structure to Logistic Regression:
- Validates "bayesian" in valid_strategies (Lines 936-938)
- Implements grid search (Lines 992-1016)
- Implements random search (Lines 1018-1069)
- **NO Bayesian implementation** despite validation allowing it

#### MLP Training

**File:** `train.py:654-907`

- **Does NOT support Bayesian:** valid_strategies = ["none", "grid", "random"] (Lines 681-683)
- Implements grid search (Lines 723-746)
- Implements random search (Lines 748-792)

---

## 3. Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND: TrainCard.jsx                                         │
│                                                                  │
│ User Actions:                                                    │
│  1. Select CSV file                                              │
│  2. Load columns → POST /analyze-csv/                           │
│  3. Select input features (checkboxes)                           │
│  4. Select target variable (radio)                               │
│  5. Choose algorithm (logistic/mlp/xgboost)                     │
│  6. Choose optimization (manual/grid/random/bayesian)           │
│  7. Configure Bayesian params (if bayesian selected)            │
│     - Parameter space (type, distribution, low/high/choices)    │
│     - Advanced config (n_initial_points, convergence, etc.)     │
│  8. Click "Entrenar Modelo"                                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       │ FormData: {file: CSV, data: JSON_payload}
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ BACKEND: views.py:train_model()                                 │
│                                                                  │
│  1. Validate request (file exists, data exists)                 │
│  2. Parse JSON configuration                                    │
│  3. Validate experiment_dir                                     │
│  4. Setup MLflow tracking URI (shared_mlflow.db)                │
│  5. Cleanup any active runs                                     │
│  6. Call train_model_logic()                                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ BACKEND: services.py:train_model_logic()                        │
│                                                                  │
│  1. Validate algorithm in ["logistic", "mlp", "xgboost"]        │
│  2. Get MLflow experiment by name                               │
│  3. Start SINGLE MLflow run with log_system_metrics=True        │
│  4. Save CSV dataset with timestamp                             │
│  5. DVC versioning:                                             │
│     - dvc add dataset.csv                                       │
│     - git add dataset.csv.dvc                                   │
│     - git commit -m "[DVC] Add training dataset"                │
│     - dvc push dataset.csv                                      │
│  6. Register dataset in MLflow (mlflow.log_input)               │
│  7. Dispatch to algorithm-specific training function:           │
│     - train_logistic_regression_model()                         │
│     - train_mlp_model()                                         │
│     - train_xgboost_model()                                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ BACKEND: train.py:train_[algorithm]_model()                     │
│                                                                  │
│  1. Extract configuration from data dict                        │
│  2. Validate hyperparameter_search_strategy                     │
│     - Logistic/XGBoost: ["none", "grid", "random", "bayesian"]  │
│     - MLP: ["none", "grid", "random"]                           │
│  3. Load and validate data (missing columns, nulls)             │
│  4. Stratified train/val/test split                             │
│  5. Start CodeCarbon energy tracker                             │
│  6. Hyperparameter Search:                                      │
│     ┌────────────────────────────────────────────┐             │
│     │ IF strategy == "none":                     │             │
│     │   - Use manual params from data["params"]  │             │
│     │   - Train single model                     │             │
│     └────────────────────────────────────────────┘             │
│     ┌────────────────────────────────────────────┐             │
│     │ IF strategy == "grid":                     │             │
│     │   - Define param_grid                      │             │
│     │   - GridSearchCV with 5-fold CV            │             │
│     │   - Extract best_estimator_                │             │
│     └────────────────────────────────────────────┘             │
│     ┌────────────────────────────────────────────┐             │
│     │ IF strategy == "random":                   │             │
│     │   - Extract random_search_params           │             │
│     │   - Loop n_random_iterations times         │             │
│     │   - Generate random params                 │             │
│     │   - Train and validate                     │             │
│     │   - Keep best model                        │             │
│     └────────────────────────────────────────────┘             │
│     ┌────────────────────────────────────────────┐             │
│     │ IF strategy == "bayesian":                 │             │
│     │   ❌ NOT IMPLEMENTED                       │             │
│     │   (Strategy validated but no code path)    │             │
│     └────────────────────────────────────────────┘             │
│  7. Evaluate on validation set                                  │
│  8. Evaluate on test set                                        │
│  9. Generate visualizations (confusion matrix, ROC curves)      │
│ 10. Stop energy tracker and log metrics                         │
│ 11. Register model in MLflow with signature                     │
│ 12. Save model locally (.pkl)                                   │
│ 13. Return {model_path, val_metrics, test_metrics, artifacts}   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ BACKEND: services.py (continued)                                │
│                                                                  │
│  8. Version model with DVC:                                     │
│     - dvc add model.pkl                                         │
│     - git commit -m "[DVC] Add model"                           │
│     - dvc push model.pkl                                        │
│  9. Log validation metrics to MLflow                            │
│ 10. Log test metrics to MLflow                                  │
│ 11. Set MLflow tag: training_phase = "completed"                │
│ 12. Update pipeline_config.json with step details               │
│ 13. Return result to views.py                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ BACKEND: views.py (continued)                                   │
│                                                                  │
│  7. Format success response:                                    │
│     {                                                            │
│       "status": "success",                                       │
│       "run_id": mlflow_run_id,                                  │
│       "metrics": {val_accuracy, val_f1, ...},                   │
│       "model_path": "/path/to/model.pkl"                        │
│     }                                                            │
│  8. Return HTTP 200 with JSON response                          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ FRONTEND: TrainCard.jsx (response handling)                     │
│                                                                  │
│  - Lines 737-740: Success handling                              │
│    setTrainStatus("✅ Modelo entrenado correctamente")          │
│    markStepDone("trainDone")                                    │
│                                                                  │
│  - Lines 741-745: Error handling                                │
│    setTrainStatus("❌ Error durante el entrenamiento")          │
│                                                                  │
│  - Lines 746-748: Cleanup                                       │
│    setTrainInProgress(false)                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Time Series Bayesian Patterns

### 4.1 Optuna 4.6.0 Implementation

**File:** `DREAM-ML-backend/GEML/apiTimeSeries/train.py`

**Library Choice:**
- **Optuna 4.6.0** (NOT scikit-optimize which is EOL)
- **TPE (Tree-structured Parzen Estimator)** sampling
- Production-ready with convergence detection and memory monitoring

**Import Section (Lines 74-77):**
```python
import optuna
from optuna.samplers import TPESampler
from optuna import Trial

# Configure Optuna logging
optuna.logging.set_verbosity(optuna.logging.INFO)
```

### 4.2 Architectural Pattern: Nested Objective Function

**ARIMA Example (Lines 1876-2270):**

```python
def train_arima_model(dataset_path: str, data: Dict, experiment_dir: str) -> Dict:
    # 1. Extract Bayesian configuration (Lines 1878-1891)
    bayesian_config = data.get("bayesian_config", {})
    n_trials = bayesian_config.get("n_trials", 50)
    n_initial_points = bayesian_config.get("n_initial_points", 10)
    timeout_seconds = bayesian_config.get("timeout_seconds", None)

    # Validation
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if n_initial_points >= n_trials:
        raise ValueError("n_initial_points must be less than n_trials")

    # 2. Extract custom parameter ranges (Lines 1946-2007)
    param_ranges = bayesian_config.get("param_ranges", {})

    if param_ranges:
        # Validate param_ranges structure
        known_params = {'p', 'd', 'q', 'P', 'D', 'Q', 's',
                        'trend', 'enforce_stationarity', 'enforce_invertibility'}

        for param_name, config in param_ranges.items():
            if param_name in ['trend', 'enforce_stationarity']:
                # Categorical validation
                if "choices" not in config:
                    raise ValueError(f"{param_name} must have 'choices' key")
            else:
                # Numeric validation
                if "min" not in config or "max" not in config:
                    raise ValueError(f"{param_name} must have 'min' and 'max'")
                if config["min"] >= config["max"]:
                    raise ValueError(f"{param_name} min must be < max")

    # 3. Define objective function INSIDE training function (Lines 2009-2093)
    def objective(trial: Trial) -> float:
        """
        Optuna objective function for ARIMA hyperparameter optimization.
        Returns metric to minimize (RMSE).
        """
        # Suggest parameters with custom ranges support
        p_config = param_ranges.get("p", {"min": 0, "max": 3})
        p = trial.suggest_int('p', p_config["min"], p_config["max"])

        d_config = param_ranges.get("d", {"min": 0, "max": 1})
        d = trial.suggest_int('d', d_config["min"], d_config["max"])

        q_config = param_ranges.get("q", {"min": 0, "max": 3})
        q = trial.suggest_int('q', q_config["min"], q_config["max"])

        # Categorical parameters
        trend_config = param_ranges.get("trend", {"choices": ['n', 'c', 't', 'ct']})
        trend = trial.suggest_categorical('trend', trend_config["choices"])

        # Build ARIMA params
        params = {
            'order': (p, d, q),
            'seasonal_order': seasonal_order,
            'trend': trend,
            'enforce_stationarity': enforce_stationarity,
            'enforce_invertibility': enforce_invertibility
        }

        try:
            # Walk-forward validation (5 folds)
            fold_metrics = walk_forward_validate_sarimax(
                y_data=y_full,
                exog_data=exog_full,
                params=params,
                n_folds=5,
                initial_train_size=initial_train_size,
                forecast_horizon=forecast_horizon
            )

            score = fold_metrics[optimization_metric]
            logger.info(f"Trial {trial.number}: {optimization_metric}={score:.4f}, params={params}")
            return score

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {str(e)}")
            return float('inf')  # Penalty for minimization

    # 4. Create TPESampler with fixed seed (Lines 2096-2102)
    sampler = TPESampler(
        seed=SEED,  # 42
        n_startup_trials=n_initial_points,
        multivariate=False,  # Independent TPE (simpler, more stable)
        consider_magic_clip=True,
        consider_endpoints=False
    )

    # 5. Create study (Lines 2104-2108)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name=f"arima_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # 6. Setup convergence callback (Lines 2118-2141)
    convergence_tolerance = bayesian_config.get("convergence_tolerance", 0.001)
    convergence_patience = bayesian_config.get("convergence_patience", 5)

    def convergence_callback(study, trial):
        """Stop if improvement < tolerance for patience consecutive trials."""
        completed_trials = [t for t in study.trials
                           if t.state == optuna.trial.TrialState.COMPLETE]

        if len(completed_trials) < convergence_patience:
            return

        recent_values = [t.value for t in completed_trials[-convergence_patience:]]
        improvements = [abs(recent_values[i] - recent_values[i+1])
                       for i in range(len(recent_values)-1)]

        if all(imp < convergence_tolerance for imp in improvements):
            logger.info(f"Convergence detected at trial {trial.number}")
            study.stop()

    # 7. Setup memory monitoring callback (Lines 2146-2171)
    max_memory_mb = bayesian_config.get("max_memory_mb", None)
    peak_memory_mb = 0.0
    memory_exceeded = False

    def memory_callback(study, trial):
        """Monitor memory and stop if limit exceeded."""
        nonlocal peak_memory_mb, memory_exceeded

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        if memory_mb > peak_memory_mb:
            peak_memory_mb = memory_mb

        if max_memory_mb is not None and memory_mb > max_memory_mb:
            logger.warning(f"Memory limit exceeded: {memory_mb:.2f} MB")
            memory_exceeded = True
            study.stop()

    # 8. Execute optimization (Lines 2173-2185)
    callbacks = []
    if convergence_tolerance and convergence_patience:
        callbacks.append(convergence_callback)
    if max_memory_mb is not None:
        callbacks.append(memory_callback)

    optimization_start = time.time()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        callbacks=callbacks,
        show_progress_bar=False,
        n_jobs=1  # Single-threaded for reproducibility
    )
    optimization_time_seconds = time.time() - optimization_start

    # 9. Validate results (Lines 2195-2199)
    if study.best_trial is None or study.best_value == float('inf'):
        raise RuntimeError("Bayesian Search failed: All trials returned errors")

    # 10. Extract best parameters (Lines 2201-2211)
    best_params_dict = study.best_params
    best_score = study.best_value

    logger.info(f"Bayesian Search Completed")
    logger.info(f"  Best {optimization_metric}: {best_score:.4f}")
    logger.info(f"  Best parameters: {best_params_dict}")
    logger.info(f"  Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    # 11. Train final model with best params (Lines 2213-2230)
    final_model = SARIMAX(
        y_train,
        exog=exog_train,
        order=(best_params_dict['p'], best_params_dict['d'], best_params_dict['q']),
        seasonal_order=seasonal_order,
        trend=best_params_dict['trend'],
        enforce_stationarity=best_params_dict['enforce_stationarity'],
        enforce_invertibility=best_params_dict['enforce_invertibility']
    )
    fitted_model = final_model.fit(disp=False)

    # 12. Evaluate final model
    val_metrics = evaluate_model(fitted_model, X_val, y_val, "val")
    test_metrics = evaluate_model(fitted_model, X_test, y_test, "test")

    # 13. MLflow logging (Lines 2264-2278)
    mlflow.log_params({
        "bayesian_n_trials": n_trials,
        "bayesian_n_initial_points": n_initial_points,
        "bayesian_optimization_metric": optimization_metric,
        **{f"best_{k}": v for k, v in best_params_dict.items()}
    })

    mlflow.log_metrics({
        "bayesian_best_score": best_score,
        "bayesian_optimization_time_seconds": optimization_time_seconds,
        "bayesian_n_completed_trials": len([t for t in study.trials
                                            if t.state == optuna.trial.TrialState.COMPLETE])
    })

    # 14. Save to pipeline_config (Lines 2486-2506)
    bayesian_config_metadata = {
        "n_trials": n_trials,
        "n_initial_points": n_initial_points,
        "optimization_metric": optimization_metric,
        "optimization_time_seconds": optimization_time_seconds,
        "best_trial_number": study.best_trial.number,
        "n_completed_trials": len([t for t in study.trials
                                   if t.state == optuna.trial.TrialState.COMPLETE]),
        "best_params": best_params_dict,
        "seed": SEED
    }

    if param_ranges:
        bayesian_config_metadata["param_ranges"] = param_ranges

    pipeline_config["bayesian_config"] = bayesian_config_metadata

    return {
        "model_path": model_path,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    }
```

### 4.3 Key Patterns to Replicate

#### Pattern 1: TPESampler Configuration
```python
sampler = TPESampler(
    seed=42,
    n_startup_trials=n_initial_points,  # Random trials before TPE
    multivariate=False  # Independent parameter sampling
)
```

#### Pattern 2: Error Handling in Objective
```python
def objective(trial: Trial) -> float:
    try:
        # Training logic
        return score
    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {e}")
        return float('inf')  # Penalty for minimization
```

#### Pattern 3: Convergence Detection
```python
def convergence_callback(study, trial):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < patience:
        return

    recent_values = [t.value for t in completed[-patience:]]
    improvements = [abs(recent_values[i] - recent_values[i+1])
                   for i in range(len(recent_values)-1)]

    if all(imp < tolerance for imp in improvements):
        study.stop()
```

#### Pattern 4: Memory Monitoring
```python
def memory_callback(study, trial):
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    if max_memory_mb and memory_mb > max_memory_mb:
        study.stop()
```

#### Pattern 5: Configurable Parameter Ranges
```python
param_ranges = bayesian_config.get("param_ranges", {})

# With fallback defaults
C_config = param_ranges.get("C", {"min": 1e-3, "max": 1e3, "log": True})
C = trial.suggest_float('C', C_config["min"], C_config["max"],
                        log=C_config.get("log", False))
```

### 4.4 Testing Patterns

**File:** `DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_bayesian_search_arima.py`

```python
class TestBayesianSearchARIMA(unittest.TestCase):

    def test_bayesian_search_basic(self):
        """Test basic Bayesian search functionality."""
        data = {
            "input_features": ["feature1"],
            "target_variable": "target",
            "bayesian_config": {
                "n_trials": 5,
                "n_initial_points": 2
            }
        }

        result = train_arima_model(self.test_csv_path, data, self.experiment_dir)

        assert result is not None
        assert "val_metrics" in result
        assert "test_metrics" in result

    def test_bayesian_validation_n_trials_too_small(self):
        """Test validation: n_trials must be >= 1."""
        data = {
            "bayesian_config": {"n_trials": 0}
        }

        with pytest.raises(ValueError, match="n_trials must be at least 1"):
            train_arima_model(self.test_csv_path, data, self.experiment_dir)

    def test_custom_param_ranges_basic(self):
        """Test custom parameter ranges are used."""
        data = {
            "bayesian_config": {
                "n_trials": 5,
                "param_ranges": {
                    "p": {"min": 1, "max": 2},
                    "d": {"min": 0, "max": 1}
                }
            }
        }

        result = train_arima_model(self.test_csv_path, data, self.experiment_dir)

        # Verify best params are within custom ranges
        assert 1 <= result["best_params"]["p"] <= 2
```

---

## 5. Implementation Blueprint

### 5.1 Logistic Regression Bayesian Search

**File to Modify:** `DREAM-ML-backend/GEML/api/train.py`

**Insert at:** After line 559 (after random search), before line 561 (manual training)

```python
elif hyperparameter_search_strategy == "bayesian":
    # Extract Bayesian configuration
    bayesian_config = data.get("bayesian_config", {})
    n_trials = bayesian_config.get("n_trials", 50)
    n_initial_points = bayesian_config.get("n_initial_points", 10)
    timeout_seconds = bayesian_config.get("timeout_seconds", None)
    convergence_tolerance = bayesian_config.get("convergence_tolerance", 0.001)
    convergence_patience = bayesian_config.get("convergence_patience", 5)
    max_memory_mb = bayesian_config.get("max_memory_mb", None)

    # Validation
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if n_initial_points >= n_trials:
        raise ValueError("n_initial_points must be less than n_trials")

    # Extract custom param_ranges if provided
    param_ranges = bayesian_config.get("param_ranges", {})

    # Import Optuna
    import optuna
    from optuna.samplers import TPESampler
    from optuna import Trial

    # Define objective function
    def objective(trial: Trial) -> float:
        """
        Optuna objective for Logistic Regression hyperparameter optimization.
        Returns negative accuracy (for minimization).
        """
        # Suggest parameters with custom ranges support
        C_config = param_ranges.get("C", {"min": 1e-3, "max": 1e3, "log": True})
        C = trial.suggest_float('C', C_config["min"], C_config["max"],
                                log=C_config.get("log", True))

        max_iter_config = param_ranges.get("max_iter", {"min": 100, "max": 1000})
        max_iter = trial.suggest_int('max_iter', max_iter_config["min"],
                                     max_iter_config["max"])

        solver_config = param_ranges.get("solver",
                                        {"choices": ["lbfgs", "liblinear", "saga"]})
        solver = trial.suggest_categorical('solver', solver_config["choices"])

        penalty_config = param_ranges.get("penalty", {"choices": ["l2", "none"]})
        penalty = trial.suggest_categorical('penalty', penalty_config["choices"])

        # Handle solver-penalty compatibility
        if solver == "liblinear" and penalty == "none":
            penalty = "l2"  # liblinear doesn't support penalty=none

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

            # Stratified K-Fold Cross-Validation on training set
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            scores = []

            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model_trial.fit(X_tr, y_tr)
                y_pred = model_trial.predict(X_v)
                score = accuracy_score(y_v, y_pred)
                scores.append(score)

            avg_score = np.mean(scores)
            logger.info(f"Trial {trial.number}: accuracy={avg_score:.4f}, C={C:.4f}, solver={solver}")

            # Return negative (Optuna minimizes)
            return -avg_score

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {str(e)}")
            return float('inf')

    # Create TPESampler
    sampler = TPESampler(
        seed=SEED,
        n_startup_trials=n_initial_points,
        multivariate=False
    )

    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name=f"logistic_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Setup callbacks
    callbacks = []

    # Convergence callback
    if convergence_tolerance and convergence_patience:
        def convergence_callback(study, trial):
            completed = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed) < convergence_patience:
                return

            recent_values = [t.value for t in completed[-convergence_patience:]]
            improvements = [abs(recent_values[i] - recent_values[i+1])
                           for i in range(len(recent_values)-1)]

            if all(imp < convergence_tolerance for imp in improvements):
                logger.info(f"Convergence detected at trial {trial.number}")
                study.stop()

        callbacks.append(convergence_callback)

    # Memory monitoring callback
    if max_memory_mb is not None:
        import psutil
        peak_memory_mb = 0.0

        def memory_callback(study, trial):
            nonlocal peak_memory_mb
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > peak_memory_mb:
                peak_memory_mb = memory_mb

            if memory_mb > max_memory_mb:
                logger.warning(f"Memory limit exceeded: {memory_mb:.2f} MB")
                study.stop()

        callbacks.append(memory_callback)

    # Execute optimization
    logger.info(f"Starting Bayesian Search with {n_trials} trials...")
    optimization_start = time.time()

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        callbacks=callbacks,
        show_progress_bar=False,
        n_jobs=1
    )

    optimization_time_seconds = time.time() - optimization_start

    # Validate results
    if study.best_trial is None or study.best_value == float('inf'):
        raise RuntimeError("Bayesian Search failed: All trials returned errors")

    # Extract best parameters
    best_params_dict = study.best_params
    best_score = -study.best_value  # Convert back to positive accuracy

    logger.info(f"Bayesian Search Completed")
    logger.info(f"  Best accuracy: {best_score:.4f}")
    logger.info(f"  Best parameters: {best_params_dict}")
    logger.info(f"  Optimization time: {optimization_time_seconds:.2f}s")

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

    # Log to MLflow
    mlflow.log_params({
        "bayesian_n_trials": n_trials,
        "bayesian_n_initial_points": n_initial_points,
        "bayesian_optimization_metric": "accuracy",
        **{f"best_{k}": v for k, v in best_params_dict.items()}
    })

    mlflow.log_metrics({
        "bayesian_best_score": best_score,
        "bayesian_optimization_time_seconds": optimization_time_seconds,
        "bayesian_n_completed_trials": len([t for t in study.trials
                                            if t.state == optuna.trial.TrialState.COMPLETE])
    })
```

### 5.2 XGBoost Bayesian Search

**Similar pattern, with XGBoost-specific parameters:**

```python
# In objective function
n_estimators = trial.suggest_int('n_estimators', 50, 500)
max_depth = trial.suggest_int('max_depth', 3, 10)
learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
subsample = trial.suggest_float('subsample', 0.5, 1.0)
colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
gamma = trial.suggest_float('gamma', 0, 1.0)
min_child_weight = trial.suggest_int('min_child_weight', 1, 10)

model_trial = XGBClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    gamma=gamma,
    min_child_weight=min_child_weight,
    random_state=SEED,
    n_jobs=N_JOBS
)
```

### 5.3 MLP Bayesian Search (Optional)

**MLP currently does NOT support Bayesian.** To add:

1. Update `valid_strategies` in `train_mlp_model()` (line 681):
   ```python
   valid_strategies = ["none", "grid", "random", "bayesian"]
   ```

2. Implement Bayesian search with MLP-specific parameters:
   ```python
   hidden_layer_sizes_options = [[4], [10], [10, 5], [50], [100], [100, 50], [100, 50, 10]]
   hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes',
                                                   hidden_layer_sizes_options)

   activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
   solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
   learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 0.1, log=True)
   ```

---

## 6. Open Questions

### 6.1 Classification-Specific Questions

1. **Cross-Validation Strategy:**
   - Should we use StratifiedKFold CV (like time series walk-forward) or simple train/val split?
   - How many folds? (Suggestion: 5-fold like grid search)

2. **Optimization Metric:**
   - Time series uses RMSE/MAE/MAPE
   - Classification should use accuracy/F1/ROC-AUC
   - Should this be configurable from frontend? (Currently optimization_metric not in frontend)

3. **Solver-Penalty Compatibility:**
   - Logistic Regression: liblinear doesn't support penalty="none"
   - Should we validate compatibility in frontend or backend?

4. **Problem Type Handling:**
   - Binary vs Multiclass affects metrics (macro/micro averaging)
   - How should Bayesian search adapt to problem_type?

5. **Memory Management:**
   - LSTM requires `tf.keras.backend.clear_session()` after each trial
   - Do scikit-learn models need similar cleanup?

### 6.2 Architecture Questions

1. **Frontend-Backend Contract:**
   - Frontend sends `bayesian_search_params` with `type`, `distribution`, `low`, `high`, `choices`
   - Backend needs to translate this to Optuna's `suggest_float()`, `suggest_int()`, `suggest_categorical()`
   - Should we create a mapping function?

2. **Pipeline Config Schema:**
   - Time series saves `bayesian_config` with metadata (n_trials, best_params, etc.)
   - Should classification use the same structure?
   - Validation exists in `train.py:530-629`

3. **Progress Tracking:**
   - Time series doesn't have real-time progress for Bayesian search
   - Should classification add WebSocket progress updates per trial?

4. **Testing Strategy:**
   - Time series has comprehensive test suite (test_bayesian_search_*.py)
   - Should we replicate the same test structure for classification?

### 6.3 Implementation Priority Questions

1. **Which algorithms first?**
   - Logistic Regression (simplest, good starting point) ✓
   - XGBoost (reference implementation exists in time series) ✓
   - MLP (more complex, neural network specifics) ?

2. **Feature completeness:**
   - Convergence detection (replicate from time series) ✓
   - Memory monitoring (replicate from time series) ✓
   - Custom param_ranges (replicate from time series) ✓
   - Timeout support (already in time series) ✓

3. **Backward compatibility:**
   - Current code declares "bayesian" as valid but doesn't implement
   - Should we raise NotImplementedError or implement fully?

---

## Architecture Insights

### Key Strengths

1. **Clean Layer Separation:**
   - Views.py: HTTP handling, validation, MLflow setup
   - Services.py: Workflow orchestration, DVC versioning
   - Train.py: Algorithm implementation, training logic

2. **Reproducibility First:**
   - Fixed SEED = 42 everywhere
   - Single-threaded execution (n_jobs=1)
   - Deterministic config for XGBoost
   - Platform logging for debugging

3. **Comprehensive Tracking:**
   - MLflow: Experiments, runs, metrics, artifacts, model registry
   - DVC: Dataset and model versioning
   - CodeCarbon: Energy consumption tracking
   - Pipeline config: JSON-based reproducibility

4. **Frontend Maturity:**
   - Complete Bayesian UI already implemented
   - Parameter space definition with type/distribution
   - Advanced configuration (convergence, memory, timeout)
   - Validation with debounced input

### Implementation Gaps

1. **Backend Not Implemented:**
   - Logistic Regression: Strategy validated, no implementation
   - XGBoost: Strategy validated, no implementation
   - MLP: Strategy NOT even in valid_strategies list

2. **Missing Translation Layer:**
   - Frontend sends rich parameter space definition
   - Backend needs to translate to Optuna trial.suggest_*() calls

3. **No Progress Feedback:**
   - Synchronous HTTP request blocks until completion
   - No WebSocket updates during Bayesian optimization
   - User sees "🚀 Entrenando modelo..." until finished

### Reference Implementation Quality

The time series Bayesian implementation is **production-ready** with:
- ✅ Optuna 4.6.0 with TPE sampling
- ✅ Convergence detection with configurable tolerance/patience
- ✅ Memory monitoring with psutil
- ✅ Configurable parameter ranges with validation
- ✅ Comprehensive test coverage
- ✅ MLflow integration
- ✅ Pipeline config persistence
- ✅ Error handling with graceful degradation

This can be **directly replicated** for classification with minimal adaptation.

---

## Code References

### Frontend Files
- `DREAM-ML-frontend/frontend/src/components/TrainCard.jsx`: Complete training UI
  - Lines 157-271: Bayesian parameter definitions
  - Lines 632-749: API request construction
  - Lines 700-720: Bayesian payload building

### Backend Files
- `DREAM-ML-backend/GEML/api/views.py:782-886`: `/api/train-model` endpoint
- `DREAM-ML-backend/GEML/api/services.py:1078-1224`: Training orchestration
- `DREAM-ML-backend/GEML/api/train.py`:
  - Lines 426-652: Logistic Regression training
  - Lines 452-455: Bayesian strategy validation (NOT IMPLEMENTED)
  - Lines 654-907: MLP training
  - Lines 909-1176: XGBoost training
  - Lines 76-93: Data validation utilities
  - Lines 95-116: Dataset splitting
  - Lines 118-192: Model evaluation

### Time Series Reference
- `DREAM-ML-backend/GEML/apiTimeSeries/train.py`:
  - Lines 74-86: Optuna imports and setup
  - Lines 1876-2270: ARIMA Bayesian implementation ⭐
  - Lines 2731-3052: XGBoost Bayesian implementation ⭐
  - Lines 4398-4750: LSTM Bayesian implementation ⭐

### Tests
- `DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/`:
  - `test_bayesian_search_arima.py` (417 lines)
  - `test_bayesian_search_xgboost.py` (382 lines)
  - `test_bayesian_search_lstm.py` (267 lines)

---

## Summary

This analysis reveals a **well-architected system** with a clear implementation path:

1. **Frontend is 100% ready** - Complete UI, parameter space definition, validation
2. **Backend infrastructure exists** - MLflow, DVC, energy tracking all in place
3. **Reference implementation available** - Time series Bayesian search is production-ready
4. **Gap is clear** - Need to implement Bayesian objective functions for classification algorithms
5. **Pattern is proven** - Nested objective, TPESampler, callbacks, error handling all battle-tested

**Recommended Implementation Order:**
1. Logistic Regression (simplest, good starting point)
2. XGBoost (reference exists in time series)
3. MLP (requires neural network considerations)

**Estimated Effort:**
- Logistic Regression: ~200 lines of code (following ARIMA pattern)
- XGBoost: ~250 lines (following TS XGBoost pattern)
- MLP: ~300 lines (requires seed reset like LSTM)
- Testing: ~400 lines per algorithm (following existing test patterns)

**Total:** ~1500 lines of well-structured, tested code to complete Bayesian search for all classification algorithms.

---

*This research enables data scientists to have full control over classification model training with automated hyperparameter optimization, complete reproducibility via `pipeline_config.json`, and comprehensive experiment tracking.*
