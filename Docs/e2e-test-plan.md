# E2E Test Plan for DREAM-ML Thesis Project

**Thesis Title:** "Academic Experiment Management System for Machine Learning for Classification and Time Series"

**Author:** Leonardo Espinoza Ortiz
**Testing Framework:** Playwright (Python)
**Focus:** User Stories #4 and #5 (pipeline_config.json tracking & reproducibility)
**Date:** October 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Test Environment Setup](#test-environment-setup)
3. [Test Data Preparation](#test-data-preparation)
4. [User Story #4: pipeline_config.json Tracking Tests](#user-story-4-tests)
5. [User Story #5: Reproducibility Tests](#user-story-5-tests)
6. [Edge Case Tests](#edge-case-tests)
7. [Metrics Verification Guide](#metrics-verification-guide)
8. [Test Execution Checklist](#test-execution-checklist)
9. [CI/CD Integration Guide](#cicd-integration-guide)

---

## Overview

### Thesis Context

This test plan validates the core MLOps capabilities of the DREAM-ML system:
- **Automated machine learning** workflows for Classification and Time Series
- **Reproducibility of results** through DVC versioning and MLflow tracking
- **Documentation** via pipeline_config.json tracking of all transformations

### User Stories Covered

**User Story #4:** Given the user has initialized the experiment correctly, when he performs each of the steps, then the user has to be able to see the applied changes/steps in the experiments `pipeline_config.json` file

**User Story #5:** Given the user has entered the "Ejecución de Pipeline" feature AND has selected the appropriate experiment type, when he enters an experiment's `pipeline_config.json` file, then he should be able to see a completely new experiment, that has the same `pipeline_config.json` results (same transformations, same metrics)

### Test Approach

- **Testing Framework:** Playwright with Python (modern, fast, reliable)
- **Execution Mode:** Tests run against Docker Compose services
- **Scope:** End-to-end browser automation testing all workflow steps
- **Validation:** JSON structure validation, metrics comparison, file existence checks
- **Total Test Cases:** 29 comprehensive tests

---

## Test Environment Setup

### Prerequisites

- **Docker & Docker Compose:** Version 20.10+
- **Python:** 3.9 or higher
- **Node.js:** 18+ (for frontend)
- **System Requirements:**
  - 10GB free disk space
  - 8GB RAM minimum
  - Linux/MacOS/Windows with WSL2

### Step 1: Environment Configuration

1. **Navigate to project root:**
   ```bash
   cd /workspaces/dream-ml-c
   ```

2. **Verify environment file exists:**
   ```bash
   ls -la .env
   ```

   If not, copy from example:
   ```bash
   cp .env.example .env
   ```

3. **Key environment variables to verify in `.env`:**
   ```bash
   EXPERIMENTS_DIR=/app/experimentos
   MLFLOW_TRACKING_URI=http://mlflow:5000
   DVC_REMOTE_PATH=/app/dvc-storage
   VITE_API_URL=http://localhost:8000
   VITE_WS_URL=ws://localhost:8000
   ```

### Step 2: Start All Services

1. **Build and start containers:**
   ```bash
   docker-compose up -d --build
   ```

2. **Verify all services are running:**
   ```bash
   docker-compose ps
   ```

   Expected output:
   ```
   NAME                STATUS              PORTS
   dream-ml-backend    Up                  0.0.0.0:8000->8000/tcp
   dream-ml-frontend   Up                  0.0.0.0:3000->3000/tcp
   mlflow              Up                  0.0.0.0:5000->5000/tcp
   ```

3. **Check service health:**
   - Frontend: http://localhost:3000 (should show Dashboard)
   - Backend API: http://localhost:8000/api/ (should return 200)
   - MLflow UI: http://localhost:5000 (should show MLflow interface)

### Step 3: Initialize Clean Test Environment

Before each test session, reset the experiments directory:

```bash
docker exec -it dream-ml-backend bash -c "rm -rf /app/experimentos/* && mkdir -p /app/experimentos"
```

**Verify cleanup:**
```bash
docker exec -it dream-ml-backend ls /app/experimentos
# Should return empty or only .gitkeep
```

### Step 4: Browser Configuration

**For manual testing:**
- **Browser:** Chrome or Chromium (recommended for consistency)
- **Window Size:** 1920x1080 (full HD)
- **Extensions:** Disable all browser extensions
- **Cache:** Clear browser cache before test session
- **DevTools:** Keep open on Network tab to monitor API calls

**For automated testing (Playwright):**
- Playwright will handle browser configuration automatically
- Tests run in headless mode by default (use `--headed` flag to watch)

---

## Test Data Preparation

### Valid Test Datasets

The following datasets are pre-loaded in the repository under `/datasets/air+quality/`:

#### 1. Classification Dataset

**File:** `arrhythmia_testing.csv`

**Characteristics:**
- **Size:** < 1MB (well within 10MB limit)
- **Rows:** 452
- **Target Variable:** `binaryClass` (categorical: "P" or "N")
- **Features (numeric):**
  - age
  - sex
  - height
  - weight
  - QRSduration
  - PRinterval
  - Q-Tinterval
  - Tinterval
  - Pinterval
  - QRS

**Use Case:** Perfect for testing Classification workflows with Logistic Regression, MLP, XGBoost

#### 2. Time Series Dataset

**File:** `demo_single_id_1000.csv`

**Characteristics:**
- **Size:** < 1MB
- **Rows:** 366 (one year of daily data)
- **Target Variable:** `wind_speed` (numeric, continuous)
- **Features:**
  - `series_id` (identifier)
  - `timestamp` (datetime)
  - `sin_day_of_week` (numeric, -1 to 1)
  - `cos_day_of_week` (numeric, -1 to 1)
  - `air_temperature` (numeric)

**Use Case:** Perfect for testing Time Series workflows with ARIMA, LSTM

### Creating Invalid/Edge Case Test Datasets

Create a directory for test datasets:

```bash
mkdir -p /workspaces/dream-ml-c/datasets/test_data
cd /workspaces/dream-ml-c/datasets/test_data
```

#### Test Case 1: File Size Limit (>10MB)

**Objective:** Verify system rejects files exceeding 10MB limit

**Create large CSV:**
```bash
python3 << 'EOF'
import pandas as pd
import numpy as np

# Calculate rows needed for >10MB
# Assuming ~50 bytes per row with 20 columns
target_size_mb = 11
bytes_per_row = 50
rows_needed = int((target_size_mb * 1024 * 1024) / bytes_per_row)

print(f"Generating {rows_needed} rows to create {target_size_mb}MB file...")

# Generate data
np.random.seed(42)
cols = 20
data = np.random.randn(rows_needed, cols)
columns = [f'feature_{i}' for i in range(cols-1)] + ['target']
df = pd.DataFrame(data, columns=columns)

# Save
output_file = 'large_file_11mb.csv'
df.to_csv(output_file, index=False)

# Verify size
import os
size_mb = os.path.getsize(output_file) / 1024 / 1024
print(f"✓ Created {output_file}: {size_mb:.2f} MB")
EOF
```

**Expected result:** File should be >10MB (verify with `ls -lh large_file_11mb.csv`)

#### Test Case 2: Invalid CSV Format - No Headers

**Objective:** Test system handles CSV without header row

**Create CSV:**
```bash
cat > invalid_no_headers.csv << 'EOF'
1,2,3,4,5
6,7,8,9,10
11,12,13,14,15
20,21,22,23,24
25,26,27,28,29
EOF
```

**Expected system behavior:** Error message "El archivo no contiene columnas válidas"

#### Test Case 3: Invalid CSV Format - Inconsistent Columns

**Objective:** Test handling of malformed CSV with varying column counts

**Create CSV:**
```bash
cat > invalid_inconsistent_cols.csv << 'EOF'
col1,col2,col3,col4
1,2,3,4
5,6,7
8,9,10,11,12
13,14
EOF
```

**Expected system behavior:** Parsing error or data corruption warning

#### Test Case 4: Invalid Data - All Missing Values

**Objective:** Test handling when all cells are NaN/empty

**Create CSV:**
```bash
cat > invalid_all_nan.csv << 'EOF'
col1,col2,col3,target
NaN,NaN,NaN,NaN
NaN,NaN,NaN,NaN
NaN,NaN,NaN,NaN
NaN,NaN,NaN,NaN
NaN,NaN,NaN,NaN
EOF
```

**Expected system behavior:** Upload succeeds with warning, training fails with error

#### Test Case 5: Invalid Data Types

**Objective:** Test handling of non-numeric data in numeric columns

**Create CSV:**
```bash
cat > invalid_data_types.csv << 'EOF'
age,height,weight,target
25,180,75,1
thirty,170,invalid,0
35,175,80,yes
forty-five,abc,90,1
50,185,def,0
EOF
```

**Expected system behavior:** Type conversion errors or data cleaning warnings

#### Test Case 6: Insufficient Data - Too Few Rows

**Objective:** Test behavior with insufficient data for train/val/test split

**Create CSV:**
```bash
cat > insufficient_rows.csv << 'EOF'
col1,col2,col3,target
1,2,3,4
5,6,7,8
EOF
```

**Expected system behavior:** Error during split (cannot split 2 rows into train/val/test)

#### Test Case 7: Special Characters and Injection Attempts

**Objective:** Test security/sanitization of special characters

**Create CSV:**
```bash
cat > special_chars.csv << 'EOF'
feature_1,feature_2,feature_3,target
1.5,2.3,3.7,0
4.2,5.8,6.1,1
7.9,<script>alert('xss')</script>,9.3,0
10.5,11.2,"'; DROP TABLE users; --",1
13.2,14.5,15.8,1
EOF
```

**Expected system behavior:** Characters escaped/sanitized, no code execution

#### Test Case 8: Exactly at 10MB Boundary

**Objective:** Verify 10MB limit is exactly 10,485,760 bytes

**Create CSV at exactly 10MB:**
```bash
python3 << 'EOF'
import pandas as pd
import numpy as np

target_bytes = 10 * 1024 * 1024  # Exactly 10MB
estimated_bytes_per_row = 50
rows = target_bytes // estimated_bytes_per_row

np.random.seed(42)
df = pd.DataFrame(
    np.random.randn(rows, 19),
    columns=[f'f{i}' for i in range(19)]
)
df['target'] = np.random.randint(0, 2, rows)

# Save and check size
df.to_csv('boundary_10mb.csv', index=False)

import os
size = os.path.getsize('boundary_10mb.csv')
print(f"File size: {size:,} bytes ({size/(1024*1024):.2f} MB)")

# Adjust if needed
if size > target_bytes:
    rows_to_remove = int((size - target_bytes) / estimated_bytes_per_row) + 1
    df = df[:-rows_to_remove]
    df.to_csv('boundary_10mb.csv', index=False)
    size = os.path.getsize('boundary_10mb.csv')
    print(f"Adjusted size: {size:,} bytes ({size/(1024*1024):.2f} MB)")
EOF
```

**Expected system behavior:** File should be accepted (≤10MB limit)

### Verify Test Data Creation

After running all scripts, verify:

```bash
ls -lh /workspaces/dream-ml-c/datasets/test_data/
```

Expected files:
- `large_file_11mb.csv` (>10MB)
- `invalid_no_headers.csv`
- `invalid_inconsistent_cols.csv`
- `invalid_all_nan.csv`
- `invalid_data_types.csv`
- `insufficient_rows.csv`
- `special_chars.csv`
- `boundary_10mb.csv` (~10MB)

---

## User Story #4 Tests: pipeline_config.json Tracking

### Test Suite 4.1: Classification - Complete Pipeline with All Steps

---

#### **Test 4.1.1: Classification with Logistic Regression - No Transformations**

**Priority:** CRITICAL
**Duration:** ~3-5 minutes
**Prerequisite:** Clean experiment directory

**Given** the user has started the application and all systems are initialized (Git, DVC, MLflow)

**When** the user performs a complete Classification experiment with the following configuration:

**Step 1 - Inicializar Sistemas:**
1. Navigate to http://localhost:3000
2. Open "Inicializar Sistemas" accordion
3. Click "Crear Directorio para Experimento" button
   - Wait for green checkmark indicator
   - Verify status message shows experiment directory path
4. Click "Inicializar Git y DVC" button
   - Wait for completion (should take ~10 seconds)
   - Verify green checkmark
5. Click "Configurar Almacenamiento Remoto DVC" button
   - Wait for completion
   - Verify green checkmark

**Step 2 - Navigate to Classification Tab:**
1. Click on "Experimento - Clasificación" tab (Tab 2)

**Step 3 - Upload and Clean CSV:**
1. Click "Seleccionar Archivo" button
2. Choose file: `/datasets/air+quality/arrhythmia_testing.csv`
3. Click "Previsualizar Columnas" button
4. Wait for columns to load (should show 11 columns)
5. Configure variables:
   - **Target Variable (Radio button):** Select `binaryClass`
   - **Input Features (Checkboxes):** System auto-selects all other columns (age, sex, height, weight, QRSduration, PRinterval, Q-Tinterval, Tinterval, Pinterval, QRS)
6. Configure cleaning options:
   - **Eliminar duplicados:** UNCHECKED
   - **Filtrar outliers:** UNCHECKED
   - **Relleno valores numéricos:** Select "Dejar valores faltantes"
7. Click "Subir y limpiar CSV" button
8. Wait for processing (progress bar should appear, ~15-20 seconds)
9. Verify success:
   - Green checkmark appears
   - Status shows "Archivo procesado exitosamente"
   - File paths displayed for EDA and training files

**Step 4 - Generate EDA:**
1. Scroll to "2. Generar Reporte EDA" card
2. Select dataset type: "EDA"
3. Click "Generar Reporte EDA" button
4. Wait for generation (~30-45 seconds)
5. Verify:
   - Progress bar completes
   - Success message appears
   - HTML report path shown

**Step 5 - Encode Data:**
1. Scroll to "3. Codificar Variables Categóricas" card
2. Click "Analizar CSV para Codificación" button
3. Wait for analysis
4. Configure encoding:
   - **Codificar target con One-Hot:** CHECKED (true)
   - **Codificar target con Label Encoding:** UNCHECKED (false)
5. Click "Codificar y Guardar" button
6. Wait for encoding (~10 seconds)
7. Verify success message

**Step 6 - Train Model:**
1. Scroll to "4. Entrenar Modelo" card
2. Configure training:
   - **Algoritmo:** Select "Logistic Regression"
   - **Nombre del modelo:** Enter "lg_test_1"
   - **Split ratios:**
     - Train: 0,70 (or adjust to 0.7)
     - Val: 0,15 (or adjust to 0.15)
     - Test: 0,15 (or adjust to 0.15)
   - **Método de optimización:** Select "Manual"
   - **Hyperparameters:**
     - Regularización (C): 1.0
     - Max Iteraciones: 100
     - Solver: "lbfgs"
3. Click "Entrenar Modelo" button
4. Wait for training (~20-40 seconds)
5. Verify:
   - Progress bar shows training progress
   - WebSocket updates show "training" status
   - Success message appears
   - Metrics plots displayed in UI (confusion matrix, ROC curve)

**Then** the user should see the following in the experiment's `pipeline_config.json` file:

**Location of file:**
```bash
# Find the experiment directory
docker exec -it dream-ml-backend ls /app/experimentos
# Should show: Exp_YYYYMMDD_HHMMSS_<hash>/

# View pipeline_config.json
docker exec -it dream-ml-backend cat /app/experimentos/Exp_*/pipeline_config.json
```

**Validation Steps:**

1. **Experiment metadata exists:**
   ```json
   {
     "experiment_id": "<UUID>",
     "experiment_name": "Exp_YYYYMMDD_HHMMSS_<hash>",
     "created_at": "<ISO timestamp>",
     "server_timezone": ["UTC", "UTC"],
     "steps": [...]
   }
   ```

   ✓ `experiment_id` is a valid UUID (32 hex characters)
   ✓ `experiment_name` matches directory name
   ✓ `created_at` is ISO 8601 format
   ✓ `steps` array exists and has 4 elements

2. **Step 1: data_cleaning step exists:**
   ```json
   {
     "step": "data_cleaning",
     "run_id": "<UUID>",
     "inputs": {},
     "outputs": {
       "raw_data": {
         "path": "raw/arrhythmia_testing.csv",
         "dvc_file": "raw/arrhythmia_testing.csv.dvc"
       },
       "processed_eda": {
         "path": "processed/processed_eda_arrhythmia_testing.csv",
         "dvc_file": "processed/processed_eda_arrhythmia_testing.csv.dvc"
       }
     },
     "parameters": {
       "eliminar_duplicados": false,
       "filtrar_outliers": false,
       "relleno_valores_numericos": "dejar",
       "valor_imputacion": null
     },
     "cleaning_report": {
       "initial_rows": 452,
       "final_rows": 452,
       "duplicates_removed": 0,
       ...
     },
     "energy_metrics": {
       "energy_consumed_total_kWh": <number>,
       "carbon_emission__kg": <number>
     }
   }
   ```

   ✓ `step` == "data_cleaning"
   ✓ `run_id` is valid UUID
   ✓ `parameters.eliminar_duplicados` == false
   ✓ `parameters.filtrar_outliers` == false
   ✓ `parameters.relleno_valores_numericos` == "dejar"
   ✓ `parameters.valor_imputacion` == null
   ✓ `cleaning_report.initial_rows` == 452
   ✓ `cleaning_report.final_rows` == 452 (no rows removed)
   ✓ `cleaning_report.duplicates_removed` == 0
   ✓ `outputs.raw_data.path` exists
   ✓ `outputs.processed_eda.path` exists
   ✓ `energy_metrics.energy_consumed_total_kWh` > 0
   ✓ `energy_metrics.carbon_emission__kg` > 0

   **Verify DVC files exist:**
   ```bash
   docker exec -it dream-ml-backend ls /app/experimentos/Exp_*/raw/arrhythmia_testing.csv.dvc
   docker exec -it dream-ml-backend ls /app/experimentos/Exp_*/processed/processed_eda_arrhythmia_testing.csv.dvc
   ```

3. **Step 2: generate_eda step exists:**
   ```json
   {
     "step": "generate_eda",
     "dataset_type": "eda",
     "input_csv": "processed/processed_eda_arrhythmia_testing.csv",
     "ydata_report_path": "eda_reports/ydata_report_eda.html",
     "energy_metrics": {...}
   }
   ```

   ✓ `step` == "generate_eda"
   ✓ `dataset_type` == "eda"
   ✓ `input_csv` matches processed file from step 1
   ✓ `ydata_report_path` ends with .html
   ✓ `energy_metrics` exists

   **Verify HTML report exists:**
   ```bash
   docker exec -it dream-ml-backend ls /app/experimentos/Exp_*/eda_reports/ydata_report_eda.html
   # Should show file path, not "No such file"
   ```

4. **Step 3: data_encoding step exists:**
   ```json
   {
     "step": "data_encoding",
     "raw_file_path": "processed/processed_eda_arrhythmia_testing.csv",
     "processed_train_path": "processed/processed_train_processed_eda_arrhythmia_testing.csv",
     "parameters": {
       "input_features": ["age", "sex", "height", "weight", "QRSduration", "PRinterval", "Q-Tinterval", "Tinterval", "Pinterval", "QRS"],
       "target_variables": ["binaryClass"],
       "encode_target_ohe": true,
       "encode_target_label": false
     },
     "energy_metrics": {...}
   }
   ```

   ✓ `step` == "data_encoding"
   ✓ `parameters.input_features` is array with 10 elements
   ✓ `parameters.target_variables` == ["binaryClass"]
   ✓ `parameters.encode_target_ohe` == true
   ✓ `parameters.encode_target_label` == false
   ✓ `processed_train_path` exists

5. **Step 4: train_logistic_regression step exists:**
   ```json
   {
     "step": "train_logistic_regression",
     "model_name": "lg_test_1",
     "input_features": ["age", "sex", ..., "binaryClass_N"],
     "target_variable": "binaryClass_P",
     "split_ratios": {
       "train": 0.7,
       "val": 0.15,
       "test": 0.15
     },
     "hyperparameters": {
       "C": 1.0,
       "max_iter": 100,
       "solver": "lbfgs",
       "random_state": 42
     },
     "hyperparameter_search_strategy": "none",
     "grid_search": {
       "use_grid_search": false,
       "best_params": null
     },
     "random_search": {
       "use_random_search": false,
       ...
     },
     "bayesian_search": {
       "use_bayesian_search": false,
       ...
     },
     "val_metrics": {
       "val_accuracy": <0.0-1.0>,
       "val_f1": <0.0-1.0>,
       "val_precision": <0.0-1.0>,
       "val_recall": <0.0-1.0>,
       "val_roc_auc": <0.0-1.0>
     },
     "test_metrics": {
       "test_accuracy": <0.0-1.0>,
       "test_f1": <0.0-1.0>,
       "test_precision": <0.0-1.0>,
       "test_recall": <0.0-1.0>,
       "test_roc_auc": <0.0-1.0>
     },
     "model_path": "/workspaces/dream-ml-c/experimentos/Exp_*/lg_test_1.pkl",
     "artifacts": {
       "val": {
         "confusion_matrix": "/workspaces/.../confusion_matrix_val.png",
         "roc_curve": "/workspaces/.../roc_curve_val.png"
       },
       "test": {
         "confusion_matrix": "/workspaces/.../confusion_matrix_test.png",
         "roc_curve": "/workspaces/.../roc_curve_test.png"
       }
     },
     "energy_metrics": {...}
   }
   ```

   ✓ `step` == "train_logistic_regression"
   ✓ `model_name` == "lg_test_1"
   ✓ `input_features` includes encoded columns (binaryClass_N)
   ✓ `target_variable` == "binaryClass_P" (encoded)
   ✓ `split_ratios.train` == 0.7
   ✓ `split_ratios.val` == 0.15
   ✓ `split_ratios.test` == 0.15
   ✓ `hyperparameters.C` == 1.0
   ✓ `hyperparameters.max_iter` == 100
   ✓ `hyperparameters.solver` == "lbfgs"
   ✓ `hyperparameter_search_strategy` == "none"
   ✓ `grid_search.use_grid_search` == false
   ✓ All val_metrics exist and are in range [0, 1]
   ✓ All test_metrics exist and are in range [0, 1]
   ✓ `model_path` is absolute path
   ✓ All 4 artifact paths exist

   **Verify model and artifact files exist:**
   ```bash
   docker exec -it dream-ml-backend ls /app/experimentos/Exp_*/lg_test_1.pkl
   docker exec -it dream-ml-backend ls /app/experimentos/Exp_*/confusion_matrix_val.png
   docker exec -it dream-ml-backend ls /app/experimentos/Exp_*/roc_curve_val.png
   docker exec -it dream-ml-backend ls /app/experimentos/Exp_*/confusion_matrix_test.png
   docker exec -it dream-ml-backend ls /app/experimentos/Exp_*/roc_curve_test.png
   ```

**And** the user should be able to view the metrics plots in the UI:
- Scroll to bottom of TrainCard
- Verify confusion matrix images are displayed
- Verify ROC curve images are displayed
- Metrics should be visible in UI panels

**Test PASSES if:**
- All 4 steps exist in pipeline_config.json
- All fields have expected values
- All files exist on filesystem
- Metrics are reasonable (accuracy > 0.5 for binary classification)
- No error messages in UI

---

#### **Test 4.1.2: Classification with All Data Transformations Enabled**

**Priority:** HIGH
**Duration:** ~3-5 minutes

**Given** the user has started the application and all systems are initialized

**When** the user performs a Classification experiment with ALL transformations enabled:

**Steps 1-2:** Same as Test 4.1.1 (Initialize systems, navigate to Classification tab)

**Step 3 - Upload CSV with ALL transformations:**
1. Upload `arrhythmia_testing.csv`
2. Preview columns
3. Select same target and features as Test 4.1.1
4. **Configure cleaning options (DIFFERENT):**
   - **Eliminar duplicados:** CHECKED ✓
   - **Filtrar outliers:** CHECKED ✓
   - **Relleno valores numéricos:** Select **"Imputar con la media"**
5. Click "Subir y limpiar CSV"
6. Wait for processing

**Steps 4-6:** Same as Test 4.1.1 (Generate EDA, Encode, Train)

**Then** in the `pipeline_config.json`:

**Validation Focus: data_cleaning step parameters:**

```json
{
  "step": "data_cleaning",
  "parameters": {
    "eliminar_duplicados": true,
    "filtrar_outliers": true,
    "relleno_valores_numericos": "media",
    "valor_imputacion": null
  },
  "cleaning_report": {
    "initial_rows": 452,
    "final_rows": <may be less than 452>,
    "duplicates_removed": <number >= 0>,
    "numeric_imputations": {
      "<column_name>": <count>
    },
    "outliers_removed": {
      "<column_name>": <count>
    },
    ...
  }
}
```

**Validation Checklist:**

✓ `parameters.eliminar_duplicados` == true
✓ `parameters.filtrar_outliers` == true
✓ `parameters.relleno_valores_numericos` == "media"
✓ `cleaning_report.duplicates_removed` >= 0
✓ `cleaning_report.final_rows` <= `initial_rows`
✓ If duplicates or outliers removed, `final_rows` < `initial_rows`
✓ `numeric_imputations` object exists (may be empty if no NaN values)
✓ `outliers_removed` object exists (may be empty if no outliers)
✓ All other steps (generate_eda, data_encoding, train_*) exist with correct structure

**Test PASSES if:**
- Transformation parameters correctly reflect UI selections
- Cleaning report shows transformation results
- Final row count <= initial row count (if transformations applied)
- All 4 steps exist in pipeline

---

#### **Test 4.1.3: Classification with Custom Value Imputation**

**Priority:** MEDIUM
**Duration:** ~3-5 minutes

**Given** the user has initialized systems

**When** the user performs Classification with custom imputation value:

**Step 3 - Upload CSV with custom value:**
1. Upload `arrhythmia_testing.csv`
2. Preview columns
3. Select target and features
4. **Configure cleaning options:**
   - **Eliminar duplicados:** UNCHECKED
   - **Filtrar outliers:** UNCHECKED
   - **Relleno valores numéricos:** Select **"Imputar con un valor"**
   - **Valor de imputación:** Enter **"999"**
5. Click "Subir y limpiar CSV"

**Steps 4-6:** Same as Test 4.1.1

**Then** in `pipeline_config.json`:

**Validation:**

```json
{
  "step": "data_cleaning",
  "parameters": {
    "relleno_valores_numericos": "valor",
    "valor_imputacion": 999
  }
}
```

✓ `relleno_valores_numericos` == "valor"
✓ `valor_imputacion` == 999 (numeric, not string)

**Test PASSES if:**
- Custom value correctly saved in pipeline_config.json
- All other steps exist

---

#### **Test 4.1.4: Classification with MLP Neural Network**

**Priority:** HIGH
**Duration:** ~4-6 minutes

**Given** the user has initialized systems

**When** the user trains with MLP algorithm:

**Steps 1-5:** Same as Test 4.1.1 (up to encoding)

**Step 6 - Train with MLP:**
1. Scroll to "4. Entrenar Modelo" card
2. **Configure training:**
   - **Algoritmo:** Select **"MLP (Multi-Layer Perceptron)"**
   - **Nombre del modelo:** "mlp_test_1"
   - **Tipo de problema:** Select "Binary Classification"
   - **Split ratios:** train=0.7, val=0.15, test=0.15
   - **Método de optimización:** "Manual"
   - **Hyperparameters:**
     - Hidden Layer Sizes: "10,5" (or use sliders/inputs for [10, 5])
     - Activation: "relu"
     - Solver: "adam"
     - Max Iterations: 200
3. Click "Entrenar Modelo"
4. Wait for training (~30-60 seconds, MLP may take longer)

**Then** in `pipeline_config.json`:

**Validation:**

```json
{
  "step": "train_mlp",
  "model_name": "mlp_test_1",
  "hyperparameters": {
    "hidden_layer_sizes": [10, 5],
    "activation": "relu",
    "solver": "adam",
    "max_iter": 200,
    ...
  },
  "val_metrics": {
    "val_accuracy": <0.0-1.0>,
    "val_f1": <0.0-1.0>,
    "val_precision": <0.0-1.0>,
    "val_recall": <0.0-1.0>,
    "val_roc_auc": <0.0-1.0>
  },
  "test_metrics": {...},
  "artifacts": {
    "val": {
      "confusion_matrix": "...",
      "roc_curve": "..."
    },
    "test": {...}
  }
}
```

✓ `step` == "train_mlp" (NOT train_logistic_regression)
✓ `model_name` == "mlp_test_1"
✓ `hyperparameters.hidden_layer_sizes` == [10, 5]
✓ `hyperparameters.activation` == "relu"
✓ `hyperparameters.solver` == "adam"
✓ `hyperparameters.max_iter` == 200
✓ Same metrics structure (accuracy, f1, precision, recall, roc_auc)
✓ Artifacts exist (confusion matrix, ROC curve)

**Test PASSES if:**
- Step name is "train_mlp"
- MLP-specific hyperparameters present
- Metrics and artifacts same structure as Logistic Regression

---

#### **Test 4.1.5: Classification with XGBoost**

**Priority:** HIGH
**Duration:** ~4-6 minutes

**Given** the user has initialized systems

**When** the user trains with XGBoost algorithm:

**Steps 1-5:** Same as Test 4.1.1

**Step 6 - Train with XGBoost:**
1. **Configure training:**
   - **Algoritmo:** Select **"XGBoost"**
   - **Nombre del modelo:** "xgb_test_1"
   - **Split ratios:** train=0.7, val=0.15, test=0.15
   - **Método de optimización:** "Manual"
   - **Hyperparameters:**
     - Learning Rate: 0.1
     - N Estimators: 100
     - Max Depth: 3
     - Subsample: 1.0 (default)
     - Colsample Bytree: 1.0 (default)
2. Click "Entrenar Modelo"
3. Wait for training

**Then** in `pipeline_config.json`:

**Validation:**

```json
{
  "step": "train_xgboost",
  "model_name": "xgb_test_1",
  "hyperparameters": {
    "learning_rate": 0.1,
    "n_estimators": 100,
    "max_depth": 3,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    ...
  },
  "val_metrics": {...},
  "test_metrics": {...},
  "artifacts": {...}
}
```

✓ `step` == "train_xgboost"
✓ `hyperparameters.learning_rate` == 0.1
✓ `hyperparameters.n_estimators` == 100
✓ `hyperparameters.max_depth` == 3
✓ Same metrics and artifacts structure

**Test PASSES if:**
- XGBoost-specific hyperparameters recorded
- Metrics within valid ranges

---

#### **Test 4.1.6: Classification with Random Search Hyperparameter Optimization**

**Priority:** HIGH
**Duration:** ~5-10 minutes (longer due to search)

**Given** the user has initialized systems

**When** the user trains with Random Search:

**Steps 1-5:** Same as Test 4.1.1

**Step 6 - Train with Random Search:**
1. **Configure training:**
   - **Algoritmo:** "Logistic Regression"
   - **Nombre del modelo:** "lg_random_1"
   - **Split ratios:** train=0.7, val=0.15, test=0.15
   - **Método de optimización:** Select **"Random Search"**
   - **N Random Iterations:** 100
   - **Parameter ranges (leave defaults or configure):**
     - C range: [0.001, 100.0]
     - Max iter range: [100, 1000]
     - Solver options: lbfgs, liblinear, saga
     - Penalty options: l2, none
2. Click "Entrenar Modelo"
3. Wait for Random Search (~2-5 minutes)

**Then** in `pipeline_config.json`:

**Validation:**

```json
{
  "step": "train_logistic_regression",
  "hyperparameter_search_strategy": "random",
  "grid_search": {
    "use_grid_search": false,
    "best_params": null
  },
  "random_search": {
    "use_random_search": true,
    "n_random_iterations": 100,
    "random_search_params": {
      "C_range": [0.001, 100],
      "max_iter_range": [100, 1000],
      "solver_options": ["lbfgs", "liblinear", "saga"],
      "penalty_options": ["l2", "none"]
    },
    "best_params": {
      "C": <numeric>,
      "max_iter": <integer>,
      "solver": "<string>",
      "penalty": "<string>",
      "random_state": 42,
      "val_accuracy": <0.0-1.0>
    }
  },
  "bayesian_search": {
    "use_bayesian_search": false
  },
  "hyperparameters": {
    "C": <matches best_params.C>,
    "max_iter": <matches best_params.max_iter>,
    "solver": <matches best_params.solver>,
    "penalty": <matches best_params.penalty>,
    ...
  }
}
```

✓ `hyperparameter_search_strategy` == "random"
✓ `random_search.use_random_search` == true
✓ `random_search.n_random_iterations` == 100
✓ `random_search.random_search_params` contains parameter ranges
✓ `random_search.best_params` exists and contains best parameters found
✓ `grid_search.use_grid_search` == false
✓ `bayesian_search.use_bayesian_search` == false
✓ Final `hyperparameters` match `random_search.best_params`

**Test PASSES if:**
- Random Search configuration recorded
- Best parameters found and applied
- Metrics reflect optimized hyperparameters

---

#### **Test 4.1.7: Classification with Bayesian Search Hyperparameter Optimization**

**Priority:** HIGH
**Duration:** ~5-10 minutes

**Given** the user has initialized systems

**When** the user trains with Bayesian Search:

**Steps 1-5:** Same as Test 4.1.1

**Step 6 - Train with Bayesian Search:**
1. **Configure training:**
   - **Algoritmo:** "Logistic Regression"
   - **Nombre del modelo:** "lg_bayesian_1"
   - **Método de optimización:** Select **"Bayesian Search"**
   - **N Bayesian Iterations:** 50
   - **Parameter space (configure):**
     - C: type=real, distribution=log-uniform, low=0.001, high=100.0
     - max_iter: type=integer, low=100, high=1000
     - solver: type=categorical, choices=[lbfgs, liblinear, saga]
     - penalty: type=categorical, choices=[l2, none]
2. Click "Entrenar Modelo"
3. Wait for Bayesian Search

**Then** in `pipeline_config.json`:

**Validation:**

```json
{
  "step": "train_logistic_regression",
  "hyperparameter_search_strategy": "bayesian",
  "bayesian_search": {
    "use_bayesian_search": true,
    "n_bayesian_iterations": 50,
    "bayesian_search_params": {
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
      },
      "penalty": {
        "type": "categorical",
        "choices": ["l2", "none"]
      }
    },
    "best_params": {
      "C": <numeric>,
      "max_iter": <integer>,
      "solver": "<string>",
      "penalty": "<string>",
      ...
    }
  },
  "grid_search": {
    "use_grid_search": false
  },
  "random_search": {
    "use_random_search": false
  },
  "hyperparameters": {
    "C": <matches best_params.C>,
    ...
  }
}
```

✓ `hyperparameter_search_strategy` == "bayesian"
✓ `bayesian_search.use_bayesian_search` == true
✓ `bayesian_search.n_bayesian_iterations` == 50
✓ `bayesian_search.bayesian_search_params` contains parameter space with types/distributions
✓ `bayesian_search.best_params` exists
✓ Final `hyperparameters` match best params from Bayesian optimization

**Test PASSES if:**
- Bayesian Search configuration fully recorded
- Parameter space definitions correct
- Best parameters applied

---

### Test Suite 4.2: Time Series - Complete Pipeline with All Steps

---

#### **Test 4.2.1: Time Series with ARIMA - No Date Standardization**

**Priority:** CRITICAL
**Duration:** ~4-6 minutes

**Given** the user has initialized systems (Git, DVC, MLflow)

**When** the user performs a complete Time Series experiment:

**Steps 1-2:** Initialize systems, then navigate to **"Experimento - Series de Tiempo"** tab (Tab 3)

**Step 3 - Upload and Clean CSV:**
1. Click "Seleccionar Archivo"
2. Choose file: `/datasets/air+quality/demo_single_id_1000.csv`
3. Click "Previsualizar Columnas"
4. Wait for columns to load (should show: series_id, timestamp, sin_day_of_week, cos_day_of_week, air_temperature, wind_speed)
5. Configure variables:
   - **Target Variable:** Select `wind_speed`
   - **Input Features:** System auto-selects all other columns
6. Configure cleaning options:
   - **Eliminar duplicados:** UNCHECKED
   - **Filtrar outliers:** UNCHECKED
   - **Estandarización de fechas:** Select **"No estandarizar fechas"** (none)
   - **Relleno valores numéricos:** "Dejar valores faltantes"
7. Click "Subir y limpiar CSV"
8. Wait for processing

**Step 4 - Generate EDA:**
1. Select dataset type: "EDA"
2. Click "Generar Reporte EDA"
3. Wait for completion

**Step 5 - Encode Data:**
1. Click "Analizar CSV para Codificación"
2. Configure encoding:
   - **Lag periods:** 1
   - **Lag NaN handling:** "leave_as_is"
   - **Date column:** "timestamp"
3. Click "Codificar y Guardar"

**Step 6 - Train Model:**
1. **Configure training:**
   - **Algoritmo:** Select **"ARIMA"**
   - **Nombre del modelo:** "arima_test_1"
   - **Date column:** "timestamp"
   - **Target variable:** "wind_speed" (auto-filled)
   - **Forecast horizon:** 5
   - **Split ratios:** train=0.7, val=0.15, test=0.15
   - **ARIMA order (p,d,q):**
     - p: 1
     - d: 1
     - q: 1
2. Click "Entrenar Modelo"
3. Wait for training (~30-60 seconds)

**Then** in `pipeline_config.json`:

**Validation:**

1. **Experiment metadata** (same structure as Classification)

2. **Step 1: data_cleaning (Time Series specific format):**
   ```json
   {
     "step": "data_cleaning",
     "run_id": "<UUID>",
     "parameters": {
       "optional_methods": "{\"cleaning_methods\":[{\"method\":\"fill_missing_numeric_values\",\"params\":{\"method\":\"dejar\",\"value\":null}}]}"
     },
     "cleaning_report": {
       "initial_rows": 366,
       "final_rows": 366,
       "is_time_series_regular": <boolean or null>,
       ...
     },
     "outputs": {
       "raw_data": {
         "path": "raw/demo_single_id_1000.csv",
         "dvc_file": "raw/demo_single_id_1000.csv.dvc"
       },
       "processed_eda": {
         "path": "processed/processed_eda_demo_single_id_1000.csv",
         "dvc_file": "processed/processed_eda_demo_single_id_1000.csv.dvc"
       }
     },
     "energy_metrics": {...}
   }
   ```

   ✓ `step` == "data_cleaning"
   ✓ `parameters.optional_methods` is JSON string
   ✓ Parse optional_methods: should contain cleaning_methods array
   ✓ cleaning_methods contains "fill_missing_numeric_values" with method="dejar"
   ✓ `cleaning_report.initial_rows` == 366
   ✓ `cleaning_report.is_time_series_regular` exists (may be null)
   ✓ `outputs.raw_data.path` and `.dvc_file` exist

3. **Step 2: generate_eda:**
   ```json
   {
     "step": "generate_eda",
     "dataset_type": "eda",
     "input_csv": "processed/processed_eda_demo_single_id_1000.csv",
     "ydata_report_path": "eda_reports/ydata_report_eda.html",
     "energy_metrics": {...}
   }
   ```

   ✓ Same structure as Classification
   ✓ HTML report exists

4. **Step 3: data_encoding (Time Series specific):**
   ```json
   {
     "step": "data_encoding",
     "raw_file_path": "processed/processed_eda_demo_single_id_1000.csv",
     "processed_train_path": "processed/processed_train_processed_eda_demo_single_id_1000.csv",
     "parameters": {
       "input_features": ["timestamp", "sin_day_of_week", "cos_day_of_week", "air_temperature"],
       "target_variables": ["wind_speed"],
       "encode_target_ohe": false,
       "encode_target_label": false,
       "lag_periods": 1,
       "lag_nan_handling": "leave_as_is",
       "date_column": "timestamp"
     },
     "energy_metrics": {...}
   }
   ```

   ✓ `parameters.lag_periods` == 1
   ✓ `parameters.lag_nan_handling` == "leave_as_is"
   ✓ `parameters.date_column` == "timestamp"
   ✓ `encode_target_ohe` == false (Time Series doesn't encode target)

5. **Step 4: train_arima (Time Series specific):**
   ```json
   {
     "step": "train_arima",
     "model_name": "arima_test_1",
     "date_col_name": "timestamp",
     "target_variable": "wind_speed",
     "forecast_horizon": 5,
     "split_ratios": {
       "train": 0.7,
       "val": 0.15,
       "test": 0.15
     },
     "hyperparameter_search_strategy": "none",
     "hyperparameters": {
       "order": [1, 1, 1]
     },
     "grid_search": {
       "use_grid_search": false,
       "best_params": null
     },
     "random_search": {
       "use_random_search": false,
       ...
     },
     "bayesian_search": {
       "use_bayesian_search": false,
       ...
     },
     "val_metrics": {
       "val_rmse": <numeric > 0>,
       "val_mae": <numeric > 0>,
       "val_mape": <numeric > 0>
     },
     "test_metrics": {
       "test_rmse": <numeric > 0>,
       "test_mae": <numeric > 0>,
       "test_mape": <numeric > 0>
     },
     "model_path": "/workspaces/.../arima_test_1.pkl",
     "artifacts": {
       "val": {
         "forecast_plot": "/workspaces/.../forecast_plot_val.png",
         "residuals_plot": "/workspaces/.../residuals_plot_val.png",
         "acf_pacf_plot": "/workspaces/.../acf_pacf_residuals_val.png"
       },
       "test": {
         "forecast_plot": "/workspaces/.../forecast_plot_test.png",
         "residuals_plot": null,
         "acf_pacf_plot": null
       }
     },
     "energy_metrics": {...}
   }
   ```

   ✓ `step` == "train_arima"
   ✓ `model_name` == "arima_test_1"
   ✓ `date_col_name` == "timestamp"
   ✓ `target_variable` == "wind_speed"
   ✓ `forecast_horizon` == 5
   ✓ `hyperparameters.order` == [1, 1, 1]
   ✓ **Time Series metrics (different from Classification):**
     - `val_metrics.val_rmse` > 0
     - `val_metrics.val_mae` > 0
     - `val_metrics.val_mape` > 0
     - `val_mae` should be < `val_rmse` (mathematical property)
   ✓ **Time Series artifacts (different from Classification):**
     - `forecast_plot` (not confusion_matrix)
     - `residuals_plot` (not roc_curve)
     - `acf_pacf_plot` (autocorrelation plots)
   ✓ All artifact files exist

**Verify artifacts:**
```bash
docker exec -it dream-ml-backend ls /app/experimentos/Exp_*/arima_test_1.pkl
docker exec -it dream-ml-backend ls /app/experimentos/Exp_*/forecast_plot_val.png
docker exec -it dream-ml-backend ls /app/experimentos/Exp_*/residuals_plot_val.png
docker exec -it dream-ml-backend ls /app/experimentos/Exp_*/acf_pacf_residuals_val.png
docker exec -it dream-ml-backend ls /app/experimentos/Exp_*/forecast_plot_test.png
```

**Test PASSES if:**
- All 4 steps exist with Time Series specific structures
- Metrics are RMSE/MAE/MAPE (not accuracy/f1/etc.)
- Artifacts are forecast plots (not confusion matrices)
- `val_mae` < `val_rmse` (sanity check)

---

#### **Test 4.2.2: Time Series with Date Standardization to UTC**

**Priority:** MEDIUM
**Duration:** ~4-6 minutes

**Given** the user has initialized systems

**When** the user performs Time Series with UTC date standardization:

**Step 3 - Upload CSV with date standardization:**
1. Upload `demo_single_id_1000.csv`
2. Preview columns
3. Select target: `wind_speed`, features: all others
4. **Configure cleaning options:**
   - **Estandarización de fechas:** Select **"Convertir a UTC"**
   - **Columna de fecha:** Select "timestamp" (auto-selected)
   - **Estrategia de imputación:** "Calcular intervalo promedio" (mean_timedelta)
   - Other options: same as Test 4.2.1
5. Click "Subir y limpiar CSV"

**Steps 4-6:** Same as Test 4.2.1

**Then** in `pipeline_config.json`:

**Validation Focus: data_cleaning step**

```json
{
  "step": "data_cleaning",
  "parameters": {
    "optional_methods": "{\"cleaning_methods\":[{\"method\":\"standardize_date_to_utc\",\"params\":{\"date_column\":\"timestamp\",\"imputation_strategy\":\"mean_timedelta\"}},{\"method\":\"fill_missing_numeric_values\",\"params\":{\"method\":\"dejar\",\"value\":null}}]}"
  }
}
```

✓ Parse `optional_methods` JSON string
✓ `cleaning_methods` array contains entry with:
  - `method` == "standardize_date_to_utc"
  - `params.date_column` == "timestamp"
  - `params.imputation_strategy` == "mean_timedelta"
✓ Array also contains "fill_missing_numeric_values" entry

**Test PASSES if:**
- Date standardization parameters recorded in optional_methods
- All other steps exist with correct structure

---

#### **Test 4.2.3: Time Series with LSTM Model**

**Priority:** HIGH
**Duration:** ~5-10 minutes (LSTM training is slower)

**Given** the user has initialized systems

**When** the user trains with LSTM algorithm:

**Steps 1-5:** Same as Test 4.2.1 (up to encoding)

**Step 6 - Train with LSTM:**
1. **Configure training:**
   - **Algoritmo:** Select **"LSTM"**
   - **Nombre del modelo:** "lstm_test_1"
   - **Date column:** "timestamp"
   - **Forecast horizon:** 5
   - **Split ratios:** train=0.7, val=0.15, test=0.15
   - **LSTM Hyperparameters:**
     - Sequence length: 10
     - Hidden units: 50
     - Epochs: 50
     - Learning rate: 0.001 (if configurable)
2. Click "Entrenar Modelo"
3. Wait for LSTM training (~2-5 minutes)

**Then** in `pipeline_config.json`:

**Validation:**

```json
{
  "step": "train_lstm",
  "model_name": "lstm_test_1",
  "hyperparameters": {
    "sequence_length": 10,
    "hidden_units": 50,
    "epochs": 50,
    "learning_rate": 0.001,
    ...
  },
  "val_metrics": {
    "val_rmse": <numeric>,
    "val_mae": <numeric>,
    "val_mape": <numeric>
  },
  "test_metrics": {...},
  "artifacts": {
    "val": {
      "forecast_plot": "...",
      "residuals_plot": "...",
      ...
    },
    "test": {...}
  }
}
```

✓ `step` == "train_lstm" (NOT train_arima)
✓ `hyperparameters.sequence_length` == 10
✓ `hyperparameters.hidden_units` == 50
✓ `hyperparameters.epochs` == 50
✓ Same Time Series metrics (RMSE, MAE, MAPE)
✓ Same Time Series artifacts (forecast plots)

**Test PASSES if:**
- LSTM-specific hyperparameters recorded
- Metrics and artifacts same structure as ARIMA

---

#### **Test 4.2.4: Time Series with Bayesian Hyperparameter Search**

**Priority:** MEDIUM
**Duration:** ~10-20 minutes (Bayesian + LSTM is very slow)

**Given** the user has initialized systems

**When** the user trains LSTM with Bayesian Search:

**Steps 1-5:** Same as Test 4.2.1

**Step 6 - Train with Bayesian Search:**
1. **Configure training:**
   - **Algoritmo:** "LSTM"
   - **Método de optimización:** "Bayesian Search"
   - **N Bayesian Iterations:** 20 (lower for testing speed)
   - **Parameter space:**
     - sequence_length: type=integer, low=5, high=20
     - hidden_units: type=integer, low=32, high=128
     - learning_rate: type=real, distribution=log-uniform, low=0.001, high=0.1
2. Click "Entrenar Modelo"
3. Wait for Bayesian optimization

**Then** in `pipeline_config.json`:

**Validation:**

```json
{
  "step": "train_lstm",
  "hyperparameter_search_strategy": "bayesian",
  "bayesian_search": {
    "use_bayesian_search": true,
    "n_bayesian_iterations": 20,
    "bayesian_search_params": {
      "sequence_length": {
        "type": "integer",
        "low": 5,
        "high": 20
      },
      "hidden_units": {
        "type": "integer",
        "low": 32,
        "high": 128
      },
      "learning_rate": {
        "type": "real",
        "distribution": "log-uniform",
        "low": 0.001,
        "high": 0.1
      }
    },
    "best_params": {
      "sequence_length": <integer 5-20>,
      "hidden_units": <integer 32-128>,
      "learning_rate": <float 0.001-0.1>,
      ...
    }
  },
  "hyperparameters": {
    "sequence_length": <matches best_params>,
    "hidden_units": <matches best_params>,
    "learning_rate": <matches best_params>,
    ...
  }
}
```

✓ `hyperparameter_search_strategy` == "bayesian"
✓ `bayesian_search.use_bayesian_search` == true
✓ `bayesian_search.n_bayesian_iterations` == 20
✓ Parameter space correctly defined
✓ Best parameters found and applied to final hyperparameters

**Test PASSES if:**
- Bayesian Search configuration recorded for Time Series
- Best parameters within defined ranges
- Final model uses optimized hyperparameters

---

## User Story #5 Tests: Reproducibility Verification

### Test Suite 5.1: Basic Reproducibility - Metrics Exact Match

---

#### **Test 5.1.1: Classification Experiment Reproducibility**

**Priority:** CRITICAL
**Duration:** ~6-10 minutes (run experiment twice)

**Given** the user has completed Test 4.1.1 (Classification with Logistic Regression) successfully
**And** has the experiment's `pipeline_config.json` file saved

**When** the user reproduces the experiment:

**Step 1 - Save original pipeline_config.json:**
```bash
# Copy the original pipeline_config.json to a safe location
docker exec -it dream-ml-backend bash -c "cp /app/experimentos/Exp_*/pipeline_config.json /app/original_pipeline_config.json"

# Or copy to host machine
docker cp dream-ml-backend:/app/experimentos/Exp_20251028_*/pipeline_config.json ./original_classification.json
```

**Step 2 - Navigate to "Ejecución de Pipeline" feature:**
1. In the application, click on the **"Ejecución de Pipeline"** tab (Tab 4 or main menu)
2. Verify the ExecutePipelineCard component is visible

**Step 3 - Configure execution:**
1. **Tipo de experimento:** Toggle to **"Clasificación Tabular"** (ensure switch is OFF/left position)
2. Verify the label shows "Clasificación Tabular"

**Step 4 - Upload pipeline_config.json:**
1. Click the file input under "Archivo pipeline_config.json"
2. Select the saved `original_classification.json` file
3. Click "Cargar Pipeline" button
4. Wait for parsing

**Step 5 - Verify pipeline loaded:**
1. Verify success message: "¡pipeline_config cargado con éxito!"
2. Verify the pipeline diagram appears showing all 4 steps:
   - data_cleaning
   - generate_eda
   - data_encoding
   - train_logistic_regression
3. Verify each step shows "PENDING" status initially

**Step 6 - Execute pipeline:**
1. Click "Ejecutar Pipeline" button
2. Observe progress:
   - Progress bar appears
   - WebSocket connections established
   - Steps change from "PENDING" → "OK" sequentially
   - Watch for any errors
3. Wait for completion (~2-4 minutes)
4. Verify final message: "Pipeline ejecutado correctamente"

**Step 7 - Locate new experiment directory:**
```bash
# List experiments, should now have TWO directories
docker exec -it dream-ml-backend ls -lt /app/experimentos

# Should show:
# Exp_20251028_HHMMSS_<new_hash>/  (NEW)
# Exp_20251028_HHMMSS_<old_hash>/  (ORIGINAL)
```

**Then** compare the two `pipeline_config.json` files:

**Step 8 - Extract both files for comparison:**
```bash
# Get original
docker cp dream-ml-backend:/app/experimentos/Exp_20251028_024536_*/pipeline_config.json ./original.json

# Get reproduced
docker cp dream-ml-backend:/app/experimentos/Exp_20251028_124125_*/pipeline_config.json ./reproduced.json
```

**Validation Checklist:**

**1. Different experiment metadata:**

```bash
# Compare experiment IDs (should be DIFFERENT)
jq '.experiment_id' original.json
jq '.experiment_id' reproduced.json
# UUIDs should NOT match

# Compare experiment names (should be DIFFERENT)
jq '.experiment_name' original.json
jq '.experiment_name' reproduced.json
# Timestamps and hashes should differ

# Compare created_at (should be DIFFERENT)
jq '.created_at' original.json
jq '.created_at' reproduced.json
# Timestamps should differ by ~minutes
```

✓ `experiment_id` is different
✓ `experiment_name` is different (new timestamp/hash)
✓ `created_at` is different (newer timestamp)

**2. Same pipeline structure:**

```bash
# Count steps (should both be 4)
jq '.steps | length' original.json
jq '.steps | length' reproduced.json

# Compare step names (should be identical)
jq '.steps[].step' original.json
jq '.steps[].step' reproduced.json
# Should both output:
# "data_cleaning"
# "generate_eda"
# "data_encoding"
# "train_logistic_regression"
```

✓ Both have 4 steps
✓ Step names in same order
✓ Step types match exactly

**3. Same transformations applied:**

```bash
# Compare data_cleaning parameters
jq '.steps[0].parameters' original.json
jq '.steps[0].parameters' reproduced.json
# Should be identical

# Compare data_encoding parameters
jq '.steps[2].parameters' original.json
jq '.steps[2].parameters' reproduced.json
# Should be identical (except run-specific IDs)

# Compare training hyperparameters
jq '.steps[3].hyperparameters' original.json
jq '.steps[3].hyperparameters' reproduced.json
# Should be identical
```

✓ data_cleaning parameters match
✓ data_encoding parameters match
✓ Training hyperparameters match
✓ Split ratios match

**4. Validation metrics EXACTLY match (CRITICAL TEST):**

```bash
# Extract val_metrics from both
echo "Original val_metrics:"
jq '.steps[3].val_metrics' original.json

echo "Reproduced val_metrics:"
jq '.steps[3].val_metrics' reproduced.json

# Compare each metric
jq '.steps[3].val_metrics.val_accuracy' original.json
jq '.steps[3].val_metrics.val_accuracy' reproduced.json
# Repeat for f1, precision, recall, roc_auc
```

**Python script for exact comparison:**
```python
import json

with open('original.json') as f:
    orig = json.load(f)
with open('reproduced.json') as f:
    repro = json.load(f)

orig_val = orig['steps'][3]['val_metrics']
repro_val = repro['steps'][3]['val_metrics']

print("Metric Comparison:")
for metric in ['val_accuracy', 'val_f1', 'val_precision', 'val_recall', 'val_roc_auc']:
    o = orig_val[metric]
    r = repro_val[metric]
    match = "✓" if o == r else "✗"
    print(f"{metric}: {o:.10f} vs {r:.10f} {match}")

    # Check if exactly equal (bitwise)
    if o != r:
        diff = abs(o - r)
        print(f"  Difference: {diff:.15e}")
```

✓ `val_accuracy` matches exactly (to at least 6 decimal places)
✓ `val_f1` matches exactly
✓ `val_precision` matches exactly
✓ `val_recall` matches exactly
✓ `val_roc_auc` matches exactly

**5. Test metrics EXACTLY match:**

```bash
# Same comparison for test_metrics
jq '.steps[3].test_metrics' original.json
jq '.steps[3].test_metrics' reproduced.json
```

✓ All test metrics match exactly

**6. Files exist in new experiment directory:**

```bash
# Check model file
docker exec -it dream-ml-backend ls /app/experimentos/Exp_<new>/lg_test_1.pkl

# Check artifacts
docker exec -it dream-ml-backend ls /app/experimentos/Exp_<new>/confusion_matrix_val.png
docker exec -it dream-ml-backend ls /app/experimentos/Exp_<new>/roc_curve_val.png
docker exec -it dream-ml-backend ls /app/experimentos/Exp_<new>/confusion_matrix_test.png
docker exec -it dream-ml-backend ls /app/experimentos/Exp_<new>/roc_curve_test.png

# Check EDA report
docker exec -it dream-ml-backend ls /app/experimentos/Exp_<new>/eda_reports/ydata_report_eda.html
```

✓ Model .pkl file exists
✓ All 4 artifact PNG files exist
✓ EDA HTML report exists

**Test PASSES if:**
- New experiment created with different ID/name
- All 4 steps reproduced with identical structure
- **ALL validation metrics match exactly (bit-for-bit)**
- **ALL test metrics match exactly**
- All files regenerated in new experiment directory
- No errors during pipeline execution

**Test FAILS if:**
- Any metric differs by more than machine epsilon (~1e-15)
- Any step is missing or has different parameters
- Files not created in new directory

---

#### **Test 5.1.2: Time Series Experiment Reproducibility**

**Priority:** CRITICAL
**Duration:** ~6-10 minutes

**Given** the user has completed Test 4.2.1 (Time Series with ARIMA)
**And** has saved the `pipeline_config.json` file

**When** the user reproduces the Time Series experiment:

**Steps 1-2:** Save original pipeline_config.json (same as 5.1.1)

**Step 3 - Navigate to "Ejecución de Pipeline"**

**Step 4 - Configure execution:**
1. **Tipo de experimento:** Toggle to **"Pronóstico de Series Temporales"** (switch ON/right position)
2. Verify label shows "Pronóstico de Series Temporales"

**Steps 5-7:** Upload pipeline_config, execute, wait for completion (same as 5.1.1)

**Then** compare the two pipeline_config.json files:

**Validation Checklist:**

**1. Different experiment metadata** (same as 5.1.1)

**2. Same pipeline structure:**
- Both have 4 steps
- Steps: data_cleaning, generate_eda, data_encoding, train_arima

**3. Same transformations applied:**
- Compare data_cleaning optional_methods
- Compare data_encoding parameters (lag_periods, date_column, etc.)
- Compare ARIMA hyperparameters (order)

**4. Time Series validation metrics EXACTLY match:**

```bash
# Extract val_metrics
jq '.steps[3].val_metrics' original_ts.json
jq '.steps[3].val_metrics' reproduced_ts.json
```

**Python comparison:**
```python
import json

with open('original_ts.json') as f:
    orig = json.load(f)
with open('reproduced_ts.json') as f:
    repro = json.load(f)

orig_val = orig['steps'][3]['val_metrics']
repro_val = repro['steps'][3]['val_metrics']

print("Time Series Metric Comparison:")
for metric in ['val_rmse', 'val_mae', 'val_mape']:
    o = orig_val[metric]
    r = repro_val[metric]
    match = "✓" if o == r else "✗"
    print(f"{metric}: {o:.10f} vs {r:.10f} {match}")
```

✓ `val_rmse` matches exactly
✓ `val_mae` matches exactly
✓ `val_mape` matches exactly

**5. Test metrics match:**

✓ `test_rmse` matches exactly
✓ `test_mae` matches exactly
✓ `test_mape` matches exactly

**6. Sanity check - Mathematical properties preserved:**

```python
# Verify RMSE >= MAE in both experiments
orig_rmse = orig['steps'][3]['val_metrics']['val_rmse']
orig_mae = orig['steps'][3]['val_metrics']['val_mae']
assert orig_rmse >= orig_mae, "Original: RMSE should be >= MAE"

repro_rmse = repro['steps'][3]['val_metrics']['val_rmse']
repro_mae = repro['steps'][3]['val_metrics']['val_mae']
assert repro_rmse >= repro_mae, "Reproduced: RMSE should be >= MAE"

print("✓ Mathematical properties preserved")
```

**7. Files exist:**
- ARIMA model .pkl
- Forecast plots (val and test)
- Residuals plot (val)
- ACF/PACF plot (val)

**Test PASSES if:**
- Time Series metrics (RMSE, MAE, MAPE) match exactly
- Mathematical properties preserved (RMSE ≥ MAE)
- All Time Series artifacts regenerated

---

### Test Suite 5.2: Advanced Reproducibility - File Checksum Verification (Bonus)

---

#### **Test 5.2.1: DVC File Hash Verification**

**Priority:** MEDIUM (Bonus test)
**Duration:** ~5 minutes (after reproducibility tests)

**Given** the user has completed Test 5.1.1 (reproduced Classification experiment)
**And** both original and reproduced experiment directories exist

**When** the tester examines DVC metadata files

**Then** the DVC hashes should match for identical data:

**Step 1 - Examine original DVC files:**

```bash
# Navigate to original experiment
ORIG_DIR=$(docker exec dream-ml-backend ls -t /app/experimentos | head -n 2 | tail -n 1)
echo "Original: $ORIG_DIR"

# View raw data DVC file
docker exec dream-ml-backend cat /app/experimentos/$ORIG_DIR/raw/arrhythmia_testing.csv.dvc

# Example output:
# outs:
# - md5: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
#   size: 50000
#   path: arrhythmia_testing.csv
```

**Step 2 - Examine reproduced DVC files:**

```bash
# Get reproduced experiment directory
REPRO_DIR=$(docker exec dream-ml-backend ls -t /app/experimentos | head -n 1)
echo "Reproduced: $REPRO_DIR"

# View raw data DVC file
docker exec dream-ml-backend cat /app/experimentos/$REPRO_DIR/raw/arrhythmia_testing.csv.dvc
```

**Step 3 - Compare MD5 hashes:**

```bash
# Extract and compare raw data hashes
ORIG_MD5=$(docker exec dream-ml-backend grep "md5:" /app/experimentos/$ORIG_DIR/raw/arrhythmia_testing.csv.dvc | awk '{print $2}')
REPRO_MD5=$(docker exec dream-ml-backend grep "md5:" /app/experimentos/$REPRO_DIR/raw/arrhythmia_testing.csv.dvc | awk '{print $2}')

echo "Original MD5:    $ORIG_MD5"
echo "Reproduced MD5:  $REPRO_MD5"

if [ "$ORIG_MD5" = "$REPRO_MD5" ]; then
    echo "✓ Raw data hashes MATCH - Same input data used"
else
    echo "✗ Raw data hashes DIFFER - Different input data!"
fi
```

**Step 4 - Compare processed data hashes:**

```bash
# Extract processed EDA data hashes
ORIG_PROC_MD5=$(docker exec dream-ml-backend grep "md5:" /app/experimentos/$ORIG_DIR/processed/processed_eda_*.csv.dvc | awk '{print $2}')
REPRO_PROC_MD5=$(docker exec dream-ml-backend grep "md5:" /app/experimentos/$REPRO_DIR/processed/processed_eda_*.csv.dvc | awk '{print $2}')

echo "Original processed MD5:    $ORIG_PROC_MD5"
echo "Reproduced processed MD5:  $REPRO_PROC_MD5"

if [ "$ORIG_PROC_MD5" = "$REPRO_PROC_MD5" ]; then
    echo "✓ Processed data hashes MATCH - Transformations produced identical output"
else
    echo "✗ Processed data hashes DIFFER - Transformations produced different output!"
fi
```

**Validation:**

✓ Raw data MD5 hashes match → Same input data
✓ Processed data MD5 hashes match → Deterministic transformations
✓ This proves data versioning works correctly via DVC

**Test PASSES if:**
- Raw data DVC hashes are identical
- Processed data DVC hashes are identical
- This confirms transformations are fully reproducible

**Test FAILS if:**
- Hashes differ (indicates non-deterministic behavior or data corruption)

---

#### **Test 5.2.2: Model File Checksum Verification**

**Priority:** LOW (Bonus test)
**Duration:** ~5 minutes

**Given** the user has completed Test 5.1.1

**When** the tester compares model file checksums

**Then** predictions should be identical even if binary checksums differ:

**Step 1 - Calculate model file checksums:**

```bash
# Get SHA256 checksums of model files
docker exec dream-ml-backend sha256sum /app/experimentos/$ORIG_DIR/lg_test_1.pkl
docker exec dream-ml-backend sha256sum /app/experimentos/$REPRO_DIR/lg_test_1.pkl
```

**Expected:** Checksums MAY differ due to:
- Pickle serialization includes timestamps
- Floating point precision differences
- Non-deterministic model internals

**Step 2 - Test prediction equivalence:**

Create a Python script to verify predictions are identical:

```bash
docker exec -it dream-ml-backend python3 << 'EOFPYTHON'
import pickle
import pandas as pd
import numpy as np

# Load both models
with open(f'/app/experimentos/{ORIG_DIR}/lg_test_1.pkl', 'rb') as f:
    orig_model = pickle.load(f)

with open(f'/app/experimentos/{REPRO_DIR}/lg_test_1.pkl', 'rb') as f:
    repro_model = pickle.load(f)

# Load test data (use processed train file)
test_data = pd.read_csv(f'/app/experimentos/{ORIG_DIR}/processed/processed_train_*.csv')

# Prepare test set (use same split as experiment)
from sklearn.model_selection import train_test_split
X = test_data.drop(['binaryClass_P'], axis=1)  # Adjust column name
y = test_data['binaryClass_P']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Generate predictions
orig_preds = orig_model.predict(X_test)
repro_preds = repro_model.predict(X_test)

orig_proba = orig_model.predict_proba(X_test)
repro_proba = repro_model.predict_proba(X_test)

# Compare predictions
preds_match = np.array_equal(orig_preds, repro_preds)
proba_match = np.allclose(orig_proba, repro_proba, atol=1e-10)

print(f"Predictions match: {preds_match}")
print(f"Probabilities match (within 1e-10): {proba_match}")

if preds_match and proba_match:
    print("✓ Model predictions are IDENTICAL - Full reproducibility achieved")
else:
    print("✗ Model predictions DIFFER")
    print(f"  Max difference in probabilities: {np.max(np.abs(orig_proba - repro_proba))}")

EOFPYTHON
```

**Validation:**

✓ Predictions (class labels) are identical
✓ Predicted probabilities are identical (or within 1e-10 tolerance)
✓ This proves functional equivalence even if binary files differ

**Test PASSES if:**
- Model predictions are identical on same test data
- This is the ultimate test of reproducibility

**Alternative verification (if script fails):**
- Compare metrics in pipeline_config.json (already done in 5.1.1)
- If metrics are identical, models are functionally equivalent

---

## Edge Case Tests

### Test Suite 6.1: File Size Validation

---

#### **Test 6.1.1: Upload File Exceeding 10MB Limit**

**Priority:** HIGH
**Duration:** ~2 minutes

**Given** the user has initialized systems
**And** has created the `large_file_11mb.csv` test file (>10MB)

**When** the user attempts to upload the oversized file:

**Step 1:** Navigate to Classification or Time Series tab

**Step 2:** Upload large file:
1. Click "Seleccionar Archivo"
2. Choose `/datasets/test_data/large_file_11mb.csv`
3. Observe file selection

**Step 3:** Attempt to preview:
1. Click "Previsualizar Columnas" button
2. Observe response

**Then** the system should reject the file:

**Expected Behaviors:**

✓ **Option A (Frontend validation):**
- File input shows error before upload
- "Previsualizar Columnas" button disabled
- Error message: "El archivo excede el límite de 10MB"
- Red error indicator displayed

✓ **Option B (Backend validation):**
- Upload proceeds but backend returns error
- Response: HTTP 400 or 413 (Payload Too Large)
- Error message displayed in UI: "El archivo excede el límite de 10MB"
- Upload status shows error state (red icon)

✓ **In both cases:**
- No columns loaded
- "Subir y limpiar CSV" button remains DISABLED (grayed out)
- Cannot proceed with experiment
- Status indicator shows error (red)

**Test PASSES if:**
- System rejects file >10MB
- Clear error message shown
- Cannot proceed with oversized file
- No partial data loaded

**Test FAILS if:**
- File accepted and columns loaded
- System crashes or hangs
- No error message shown

---

#### **Test 6.1.2: Upload File Exactly at 10MB Boundary**

**Priority:** MEDIUM
**Duration:** ~3 minutes

**Given** the user has created `boundary_10mb.csv` (~10MB, ≤10,485,760 bytes)

**When** the user uploads the boundary file:

1. Upload `boundary_10mb.csv`
2. Click "Previsualizar Columnas"
3. Observe response

**Then** the system should ACCEPT the file:

✓ File uploads successfully
✓ Columns loaded and displayed
✓ Can select target and features
✓ "Subir y limpiar CSV" button enabled
✓ Can proceed with experiment

**Verification:**
```bash
# Confirm file size is ≤10MB
ls -lh /workspaces/dream-ml-c/datasets/test_data/boundary_10mb.csv
stat -c%s /workspaces/dream-ml-c/datasets/test_data/boundary_10mb.csv
# Should be ≤ 10485760 bytes
```

**Test PASSES if:**
- File at exactly 10MB boundary is accepted
- No errors during upload or preview
- Can complete experiment workflow

---

### Test Suite 6.2: Invalid CSV Format Handling

---

#### **Test 6.2.1: Upload CSV Without Headers**

**Priority:** HIGH
**Duration:** ~2 minutes

**Given** the user has created `invalid_no_headers.csv` (no header row)

**When** the user uploads the file:

1. Upload `invalid_no_headers.csv`
2. Click "Previsualizar Columnas"
3. Observe response

**Then** the system should detect invalid format:

✓ Error message: "El archivo no contiene columnas válidas"
✓ No columns displayed in UI
✓ Upload status shows error (red indicator)
✓ Cannot select variables (no checkboxes/radio buttons shown)
✓ "Subir y limpiar CSV" button disabled

**Test PASSES if:**
- Invalid format detected
- Clear error message
- Cannot proceed

---

#### **Test 6.2.2: Upload CSV With Inconsistent Column Count**

**Priority:** MEDIUM
**Duration:** ~2 minutes

**Given** the user has created `invalid_inconsistent_cols.csv`

**When** the user uploads the file:

1. Upload file
2. Click "Previsualizar Columnas"

**Then** the system should:

✓ **Option A:** Reject with error "CSV malformado: número inconsistente de columnas"
✓ **Option B:** Accept but show warning, display only first row's column count
✓ **Option C:** pandas/backend handles gracefully with NaN padding

**Acceptable outcomes:**
- Error message shown
- OR warning shown but can proceed (pandas handles it)
- Should NOT crash or show no feedback

**Test PASSES if:**
- System doesn't crash
- Some feedback provided (error or warning)
- If allowed to proceed, no data corruption

---

#### **Test 6.2.3: Upload CSV With All Missing Values**

**Priority:** HIGH
**Duration:** ~3 minutes

**Given** the user has created `invalid_all_nan.csv` (all NaN)

**When** the user uploads and attempts full workflow:

**Step 1:** Upload file
- Expected: File uploads successfully

**Step 2:** Preview columns
- Expected: Columns show (col1, col2, col3, target)

**Step 3:** Select variables and cleaning
- Select target, features
- **Relleno valores numéricos:** "Dejar valores faltantes"
- Expected: Warning message displayed about missing values

**Step 4:** Click "Subir y limpiar CSV"
- Expected: Processing completes with warning

**Step 5:** Try to train model
- Expected: **Training FAILS**

**Then** the expected error handling:

✓ Upload and preview succeed (CSV structure valid)
✓ Warning shown during cleaning: "Advertencia: Dejar valores faltantes puede generar errores"
✓ **Training fails with error:**
  - "Error al entrenar: Cannot train model with all missing values"
  - OR "Error: Input contains NaN"
✓ Error displayed in TrainCard component
✓ Train button becomes disabled or shows error state
✓ No model file created

**Test PASSES if:**
- Upload/cleaning allowed but training fails
- Clear error message at training step
- System doesn't crash

---

### Test Suite 6.3: Experiment Type Switching

---

#### **Test 6.3.1: Attempt to Switch from Classification to Time Series Mid-Experiment**

**Priority:** CRITICAL
**Duration:** ~4 minutes

**Given** the user has started a Classification experiment
**And** has completed at least Step 1 (CSV uploaded and cleaned)
**And** `flow.cleaningDone = true` in AppContext

**When** the user tries to switch experiment types:

**Step 1:** Verify Classification experiment in progress:
- On Classification tab (Tab 2)
- "Subir y limpiar archivo CSV" card shows green checkmark
- Success message visible
- Context state: `flow.cleaningDone = true`

**Step 2:** Switch to Time Series tab:
1. Click "Experimento - Series de Tiempo" tab (Tab 3)
2. Observe tab switches (due to TabPanel behavior)

**Step 3:** Attempt to start Time Series experiment:
1. On Time Series tab, click "Seleccionar Archivo"
2. Choose a Time Series CSV (e.g., `demo_single_id_1000.csv`)
3. Click "Previsualizar Columnas"
4. Select target variable and features
5. Click "Subir y limpiar CSV"

**Then** the system should block the action:

**Expected Behavior:**

✓ **ERROR MODAL/ALERT appears with message:**
```
Ya existe un experimento de Clasificación en progreso.

Para iniciar un experimento de Series Temporales, debes:
```

✓ **Two options presented:**

**Opción 1:** "Volver al experimento de Clasificación actual"
- Button style: Secondary/Cancel

**Opción 2:** "Iniciar un nuevo experimento desde cero"
- Subtext: "(esto abandonará el experimento actual)"
- Button style: Primary/Destructive (red)

✓ **CSV upload does NOT proceed** (request blocked)

**When user selects "Volver al experimento de Clasificación actual":**

✓ Modal closes
✓ UI automatically switches to Classification tab (Tab 2)
✓ Classification experiment state preserved:
  - Green checkmark still visible
  - `flow.cleaningDone = true`
  - Can continue from where left off
✓ Time Series tab unchanged (no state modified)

**When user selects "Iniciar un nuevo experimento desde cero":**

✓ Modal closes
✓ System calls `resetFlow()` from AppContext
✓ All flow state reset:
  - `flow.cleaningDone = false`
  - `flow.edaDone = false`
  - `flow.encodeDone = false`
  - `flow.trainDone = false`
✓ Experiment directory cleared/reset (or new directory created)
✓ Classification tab resets (no green checkmarks)
✓ Time Series tab becomes available for fresh start
✓ User can now proceed with Time Series CSV upload

**Implementation Note:**
Check if this logic exists in the backend or frontend:
- Look for checks in `/api/ts/upload-and-clean-csv/` endpoint
- OR look for checks in TSUploadCsvCard.jsx uploadAndCleanCsv function
- May need to verify `experiment_type` tracking in backend

**Test PASSES if:**
- System detects conflict and blocks action
- Clear error modal with two options
- "Volver" preserves existing experiment
- "Iniciar nuevo" resets all state
- Cannot proceed without making choice

**Test FAILS if:**
- Can upload Time Series CSV while Classification active
- No error/warning shown
- Data gets mixed between experiment types
- App crashes

---

#### **Test 6.3.2: Attempt to Switch from Time Series to Classification Mid-Experiment**

**Priority:** CRITICAL
**Duration:** ~4 minutes

**Given** the user has started a Time Series experiment
**And** `flow.cleaningDone = true`

**When** the user switches to Classification tab and tries to upload Classification CSV

**Then** same error handling as Test 6.3.1 should occur:

✓ Error modal appears: "Ya existe un experimento de Series Temporales en progreso"
✓ Two options: "Volver" or "Iniciar nuevo"
✓ Same state preservation/reset logic
✓ Experiment type reversed but same behavior

**Test PASSES if:**
- Same error handling regardless of direction (CL→TS or TS→CL)

---

#### **Test 6.3.3: Switching Tabs Before Any Experiment Started**

**Priority:** MEDIUM
**Duration:** ~2 minutes

**Given** the user has initialized systems (Git, DVC, MLflow)
**But** has NOT started any experiment yet
**And** `flow.cleaningDone = false` for both types

**When** the user freely switches between tabs:

1. Click "Experimento - Clasificación" tab (Tab 2)
2. Observe state (no checkmarks, ready to start)
3. Click "Experimento - Series de Tiempo" tab (Tab 3)
4. Observe state (no checkmarks, ready to start)
5. Switch back to Classification tab
6. Repeat several times

**Then** the system should allow free navigation:

✓ No error messages displayed
✓ No warnings shown
✓ Both tabs show "ready to start" state
✓ Both upload cards available
✓ Both "Subir y limpiar CSV" buttons enabled (when conditions met)
✓ No experiment type lock until first CSV uploaded

**Test PASSES if:**
- Free navigation before experiment starts
- No errors or restrictions
- Both tabs equally accessible

---

### Test Suite 6.4: Missing Required Fields

---

#### **Test 6.4.1: Attempt to Upload CSV Without Selecting Target Variable**

**Priority:** HIGH
**Duration:** ~2 minutes

**Given** the user has uploaded and analyzed a CSV (columns loaded)

**When** the user configures variables incorrectly:

1. Upload valid CSV (e.g., `arrhythmia_testing.csv`)
2. Click "Previsualizar Columnas"
3. Wait for columns to load
4. **Select NO target variable** (no radio button selected)
5. **Check some input features** (select a few checkboxes)
6. Attempt to click "Subir y limpiar CSV"

**Then** the system should prevent submission:

✓ **Validation warning displayed:**
  - Red warning box appears
  - Message: "Debes seleccionar 1 variable de salida"

✓ **Button state:**
  - "Subir y limpiar CSV" button is DISABLED
  - Button appears grayed out
  - Cursor shows "not-allowed" on hover

✓ **Code validation:**
  - `isDisabled = true` due to `!targetVariable`
  - Button `disabled` attribute set

✓ **Cannot proceed:**
  - Clicking button has no effect
  - No API call made
  - No processing started

**Test PASSES if:**
- Button disabled when target not selected
- Clear validation message
- Cannot bypass validation

---

#### **Test 6.4.2: Attempt to Upload CSV Without Selecting Any Input Features**

**Priority:** HIGH
**Duration:** ~2 minutes

**Given** the user has uploaded and analyzed a CSV

**When** the user:

1. Upload and preview CSV
2. **Select a target variable** (radio button)
3. **Manually uncheck all input features** (if auto-selected, uncheck them)
4. Attempt to click "Subir y limpiar CSV"

**Then** the system should prevent submission:

✓ **Validation warning:**
  - Message: "Debes seleccionar al menos 1 variable de entrada"

✓ **Button disabled:**
  - `isDisabled = true` due to `!inputFeatures.length`

✓ **Cannot proceed**

**Test PASSES if:**
- Validation catches missing features
- Button disabled

---

#### **Test 6.4.3: Select Same Column as Both Input and Target**

**Priority:** MEDIUM
**Duration:** ~2 minutes

**Given** the user has uploaded and analyzed a CSV

**When** the user tries to select same column twice:

1. Upload and preview CSV
2. **Select "age" as target variable** (radio button)
   - System auto-selects all OTHER columns as features (expected behavior)
3. **Manually try to check "age" as input feature** (checkbox)

**Then** the system should prevent conflict:

✓ **Option A (UI prevention):**
  - "age" checkbox is DISABLED when "age" selected as target
  - Cannot check the box (grayed out)
  - Tooltip may show: "Cannot be both input and target"

✓ **Option B (Validation message):**
  - If somehow both selected, validation warning appears:
  - "Una columna no puede ser entrada y salida simultáneamente"
  - Button disabled

✓ **Code validation:**
  - Check in UploadCsvCard.jsx lines 180-183
  - `validationWarnings` includes overlap check

**Test PASSES if:**
- Cannot select same column for both roles
- Checkbox disabled OR validation message shown
- Button disabled when conflict exists

---

### Test Suite 6.5: Operation Timeout Handling

---

#### **Test 6.5.1: Training With Hyperparameter Search (Long Operation)**

**Priority:** MEDIUM
**Duration:** ~5-10 minutes

**Given** the user starts a training with Bayesian Search (known to take >60 seconds)

**When** the training is in progress:

1. Configure training with Bayesian Search (50+ iterations)
2. Click "Entrenar Modelo"
3. Observe UI during long-running operation

**Then** the system should handle long operations gracefully:

✓ **Progress indication:**
  - Progress bar displayed (ProgressBar component)
  - WebSocket connection established
  - Real-time status updates shown
  - Current status messages: "Ejecutando...", "Optimizando hiperparámetros...", etc.

✓ **No HTTP timeout:**
  - Backend handles async operations
  - Frontend doesn't timeout the request
  - Can run for >60 seconds without error

✓ **WebSocket updates:**
  - Periodic progress messages
  - Step status changes: PENDING → PROCESSING → OK
  - Training progress visible

✓ **User experience:**
  - Button disabled during training (shows "Procesando...")
  - Cannot trigger duplicate requests
  - Can still navigate UI (non-blocking)

**When training completes:**

✓ Final status message: "Entrenamiento completado"
✓ WebSocket sends "OK" status
✓ Progress bar reaches 100%
✓ Metrics appear in UI
✓ Pipeline_config.json updated
✓ Button re-enabled or shows "Completado"

**Test PASSES if:**
- Long operations complete successfully
- No timeouts even after 5-10 minutes
- Progress visible throughout
- Completion detected properly

---

#### **Test 6.5.2: Network Interruption During Pipeline Execution**

**Priority:** LOW
**Duration:** ~5 minutes

**Given** the user has started a pipeline execution

**When** network connectivity is lost mid-execution:

**Simulate network loss:**
```bash
# During pipeline execution, temporarily stop backend
docker-compose stop dream-ml-backend

# Wait 10-30 seconds

# Restart backend
docker-compose start dream-ml-backend
```

**Then** the system should handle disconnection:

✓ **During disconnection:**
  - WebSocket disconnection detected
  - UI shows error message: "Conexión perdida. Reconectando..."
  - OR: "WebSocket desconectado"
  - Last known step status preserved
  - Further interactions disabled/prevented

✓ **After reconnection:**
  - WebSocket attempts reconnection (see ExecutePipelineCard.jsx useEffect)
  - If reconnected, status updates resume
  - If backend completed work, final status appears
  - User sees current pipeline state

✓ **Worst case:**
  - If cannot reconnect, user must refresh page
  - Backend work may complete even if frontend disconnected
  - Can check pipeline_config.json manually to verify completion

**Test PASSES if:**
- System detects disconnection
- Shows error/warning message
- Attempts reconnection
- Doesn't crash or hang indefinitely

---

## Metrics Verification Guide

### Classification Metrics Validation

For each Classification test, verify metrics are within expected ranges and follow mathematical properties:

#### Metrics Reference Table

| Metric | Symbol | Range | Interpretation |
|--------|--------|-------|----------------|
| Accuracy | acc | [0.0, 1.0] | % of correct predictions |
| F1 Score | F1 | [0.0, 1.0] | Harmonic mean of precision & recall |
| Precision | P | [0.0, 1.0] | TP / (TP + FP) |
| Recall | R | [0.0, 1.0] | TP / (TP + FN) |
| ROC AUC | AUC | [0.0, 1.0] | Area under ROC curve |

#### Validation Checklist

**1. Range validation:**
```python
for metric in ['val_accuracy', 'val_f1', 'val_precision', 'val_recall', 'val_roc_auc']:
    value = metrics[metric]
    assert 0.0 <= value <= 1.0, f"{metric} out of range: {value}"
```

**2. Minimum quality thresholds:**
- Binary classification: accuracy > 0.5 (better than random)
- ROC AUC > 0.5 (better than random)
- If all metrics = 1.0, may indicate overfitting or very easy dataset

**3. Mathematical relationships:**
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
- F1 ≤ min(Precision, Recall)

**4. Test vs Validation:**
- Test metrics should be within ±0.1 of validation metrics
- If test_accuracy >> val_accuracy: possible data leakage
- If test_accuracy << val_accuracy: overfitting

**5. Energy metrics:**
- energy_consumed_total_kWh > 0
- carbon_emission__kg > 0
- Typically in range 1e-07 to 1e-02

### Time Series Metrics Validation

#### Metrics Reference Table

| Metric | Symbol | Range | Interpretation |
|--------|--------|-------|----------------|
| RMSE | RMSE | [0, ∞) | Root Mean Squared Error |
| MAE | MAE | [0, ∞) | Mean Absolute Error |
| MAPE | MAPE | [0, ∞) | Mean Absolute Percentage Error (%) |

#### Validation Checklist

**1. Range validation:**
```python
assert metrics['val_rmse'] > 0, "RMSE must be positive"
assert metrics['val_mae'] > 0, "MAE must be positive"
assert metrics['val_mape'] >= 0, "MAPE must be non-negative"
```

**2. Mathematical property - RMSE ≥ MAE:**
```python
rmse = metrics['val_rmse']
mae = metrics['val_mae']
assert rmse >= mae, f"RMSE ({rmse}) should be >= MAE ({mae})"
```

This must ALWAYS hold due to: √(mean(x²)) ≥ mean(|x|)

**3. MAPE interpretation:**
- MAPE < 10%: Excellent forecasting
- MAPE 10-20%: Good forecasting
- MAPE 20-50%: Reasonable forecasting
- MAPE > 50%: Poor forecasting (may indicate issues)

**4. Scale consistency:**
- Metrics should be reasonable relative to target variable scale
- Example: if wind_speed in range [0, 10], RMSE of 100 is impossible
- Check: RMSE < max(target) - min(target)

**5. Test vs Validation:**
- Time Series test metrics may differ more than Classification
- Acceptable if within 20-30% due to temporal distribution
- Large differences may indicate poor generalization

### Common Validation Script

**Python script for comprehensive validation:**

```python
import json
import sys

def validate_classification_metrics(metrics, metric_type='val'):
    """Validate Classification metrics structure and values"""
    required = [f'{metric_type}_accuracy', f'{metric_type}_f1',
                f'{metric_type}_precision', f'{metric_type}_recall',
                f'{metric_type}_roc_auc']

    for metric in required:
        if metric not in metrics:
            print(f"✗ Missing metric: {metric}")
            return False

        value = metrics[metric]
        if not (0.0 <= value <= 1.0):
            print(f"✗ {metric} out of range: {value}")
            return False

    # Check mathematical relationships
    precision = metrics[f'{metric_type}_precision']
    recall = metrics[f'{metric_type}_recall']
    f1 = metrics[f'{metric_type}_f1']

    expected_f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    if abs(f1 - expected_f1) > 0.01:
        print(f"⚠ F1 score inconsistent: {f1} vs expected {expected_f1}")

    print(f"✓ {metric_type.capitalize()} metrics valid")
    return True

def validate_timeseries_metrics(metrics, metric_type='val'):
    """Validate Time Series metrics structure and values"""
    required = [f'{metric_type}_rmse', f'{metric_type}_mae', f'{metric_type}_mape']

    for metric in required:
        if metric not in metrics:
            print(f"✗ Missing metric: {metric}")
            return False

        value = metrics[metric]
        if value <= 0:
            print(f"✗ {metric} must be positive: {value}")
            return False

    # Check RMSE >= MAE property
    rmse = metrics[f'{metric_type}_rmse']
    mae = metrics[f'{metric_type}_mae']

    if rmse < mae:
        print(f"✗ RMSE ({rmse}) should be >= MAE ({mae})")
        return False

    mape = metrics[f'{metric_type}_mape']
    if mape > 100:
        print(f"⚠ MAPE very high: {mape}% - check model quality")

    print(f"✓ {metric_type.capitalize()} metrics valid")
    return True

def validate_pipeline_config(filepath, experiment_type='classification'):
    """Validate entire pipeline_config.json file"""
    with open(filepath) as f:
        config = json.load(f)

    # Validate metadata
    assert 'experiment_id' in config
    assert 'experiment_name' in config
    assert 'created_at' in config
    assert 'steps' in config
    print("✓ Metadata valid")

    # Validate steps
    steps = config['steps']
    assert len(steps) >= 4, f"Expected 4+ steps, got {len(steps)}"

    # Find training step
    train_step = None
    for step in steps:
        if step['step'].startswith('train_'):
            train_step = step
            break

    assert train_step is not None, "No training step found"

    # Validate metrics based on experiment type
    if experiment_type == 'classification':
        assert validate_classification_metrics(train_step['val_metrics'], 'val')
        assert validate_classification_metrics(train_step['test_metrics'], 'test')
    elif experiment_type == 'timeseries':
        assert validate_timeseries_metrics(train_step['val_metrics'], 'val')
        assert validate_timeseries_metrics(train_step['test_metrics'], 'test')

    print("✓ All validations passed")
    return True

if __name__ == '__main__':
    filepath = sys.argv[1]
    exp_type = sys.argv[2] if len(sys.argv) > 2 else 'classification'
    validate_pipeline_config(filepath, exp_type)
```

**Usage:**
```bash
# Validate Classification pipeline_config.json
python3 validate_metrics.py /app/experimentos/Exp_*/pipeline_config.json classification

# Validate Time Series pipeline_config.json
python3 validate_metrics.py /app/experimentos/Exp_*/pipeline_config.json timeseries
```

---

## Test Execution Checklist

### Prerequisites Setup

- [ ] Docker and Docker Compose installed (version 20.10+)
- [ ] Python 3.9+ installed (for validation scripts)
- [ ] All services running: `docker-compose ps` shows all "Up"
- [ ] Frontend accessible: http://localhost:3000
- [ ] Backend accessible: http://localhost:8000
- [ ] MLflow accessible: http://localhost:5000
- [ ] Test datasets present in `/datasets/air+quality/`
- [ ] Invalid test datasets created in `/datasets/test_data/`
- [ ] Browser configured (Chrome 1920x1080, cache cleared)

### User Story #4: pipeline_config.json Tracking (11 tests)

**Classification Tests (7 tests):**
- [ ] 4.1.1: Complete pipeline with no transformations (CRITICAL)
- [ ] 4.1.2: All transformations enabled (HIGH)
- [ ] 4.1.3: Custom value imputation (MEDIUM)
- [ ] 4.1.4: MLP neural network (HIGH)
- [ ] 4.1.5: XGBoost algorithm (HIGH)
- [ ] 4.1.6: Random Search optimization (HIGH)
- [ ] 4.1.7: Bayesian Search optimization (HIGH)

**Time Series Tests (4 tests):**
- [ ] 4.2.1: ARIMA with no date standardization (CRITICAL)
- [ ] 4.2.2: UTC date standardization (MEDIUM)
- [ ] 4.2.3: LSTM model (HIGH)
- [ ] 4.2.4: Bayesian hyperparameter search (MEDIUM)

### User Story #5: Reproducibility (4 tests)

**Basic Reproducibility (2 tests):**
- [ ] 5.1.1: Classification metrics exact match (CRITICAL)
- [ ] 5.1.2: Time Series metrics exact match (CRITICAL)

**Advanced Reproducibility (2 tests - bonus):**
- [ ] 5.2.1: DVC file hash verification (MEDIUM)
- [ ] 5.2.2: Model file checksum verification (LOW)

### Edge Cases (14 tests)

**File Size (2 tests):**
- [ ] 6.1.1: >10MB file rejection (HIGH)
- [ ] 6.1.2: Exactly 10MB file acceptance (MEDIUM)

**Invalid CSV (3 tests):**
- [ ] 6.2.1: No headers (HIGH)
- [ ] 6.2.2: Inconsistent columns (MEDIUM)
- [ ] 6.2.3: All missing values (HIGH)

**Experiment Switching (3 tests):**
- [ ] 6.3.1: Classification → Time Series blocked (CRITICAL)
- [ ] 6.3.2: Time Series → Classification blocked (CRITICAL)
- [ ] 6.3.3: Switching before experiment started (MEDIUM)

**Missing Fields (3 tests):**
- [ ] 6.4.1: No target variable (HIGH)
- [ ] 6.4.2: No input features (HIGH)
- [ ] 6.4.3: Same column as input and target (MEDIUM)

**Timeouts (2 tests):**
- [ ] 6.5.1: Long training operation (MEDIUM)
- [ ] 6.5.2: Network interruption (LOW)

### Test Execution Summary

**Total:** 29 test cases
**Critical:** 7 tests
**High:** 11 tests
**Medium:** 9 tests
**Low:** 2 tests

**Estimated Total Execution Time:** 3-5 hours (if running all manually)

**Priority Order for Limited Time:**
1. All CRITICAL tests (7 tests, ~1.5 hours)
2. All HIGH tests (11 tests, ~2 hours)
3. MEDIUM tests as time permits
4. LOW tests are bonus

---

## CI/CD Integration Guide

### GitHub Actions Configuration

If you have time to set up CI/CD automation, here's how to integrate these tests:

#### Option A: Manual Execution Only (Week 1)

**Advantages:**
- Fast iteration during development
- Immediate feedback
- Easy debugging

**Execution:**
```bash
# Run tests locally
cd /workspaces/dream-ml-c
docker-compose up -d
# Execute tests manually following this document
```

#### Option B: CI/CD Automation (If Time Permits)

**Create `.github/workflows/e2e-tests.yml`:**

```yaml
name: E2E Tests

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
  workflow_dispatch:  # Manual trigger

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Start services
        run: |
          docker-compose up -d
          sleep 30  # Wait for services to be ready

      - name: Wait for services to be healthy
        run: |
          timeout 120 bash -c 'until curl -f http://localhost:3000; do sleep 2; done'
          timeout 120 bash -c 'until curl -f http://localhost:8000/api/; do sleep 2; done'

      - name: Run critical tests
        run: |
          # Run your test execution script
          bash e2e-tests/run_critical_tests.sh

      - name: Collect test results
        if: always()
        run: |
          docker-compose logs > docker-logs.txt

      - name: Upload artifacts
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: |
            docker-logs.txt
            e2e-tests/reports/

      - name: Cleanup
        if: always()
        run: docker-compose down -v
```

**Pros of CI/CD:**
- ✅ Automated testing on every commit
- ✅ Catches regressions early
- ✅ Thesis credibility (shows professional MLOps)
- ✅ Documentation (test logs as proof)

**Cons of CI/CD:**
- ❌ 2-3 hours setup time
- ❌ Debugging can be harder
- ❌ May have resource limits on GitHub Actions

### Recommendation

**Week 1 Strategy:**
1. **Days 1-5:** Focus on writing and running tests manually
2. **Day 6:** If time permits, set up basic CI (optional)
3. **Day 7:** Final testing and documentation

**For thesis defense:**
- Manual test execution is SUFFICIENT
- Show test results in defense presentation
- CI/CD is bonus points, not required

---

## Test Report Template

For each test execution, document results using this template:

```
===========================================
E2E TEST EXECUTION REPORT
===========================================

Test ID: 4.1.1
Test Name: Classification with Logistic Regression - No Transformations
Date: 2025-10-28
Tester: [Your Name]
Environment: Docker Compose, Chrome 120.0
Duration: 4 minutes 32 seconds

RESULT: ✓ PASS / ✗ FAIL

-------------------------------------------
STEPS EXECUTED:
-------------------------------------------
1. Initialized systems (Git, DVC, MLflow) - SUCCESS
2. Uploaded arrhythmia_testing.csv (452 rows) - SUCCESS
3. Selected target: binaryClass, features: 10 columns - SUCCESS
4. Cleaning: no transformations, "dejar" imputation - SUCCESS
5. Generated EDA report - SUCCESS
6. Encoded data with OHE - SUCCESS
7. Trained Logistic Regression (C=1.0, max_iter=100) - SUCCESS

-------------------------------------------
EXPECTED RESULTS:
-------------------------------------------
- pipeline_config.json contains 4 steps
- data_cleaning parameters: eliminar_duplicados=false, filtrar_outliers=false
- train_logistic_regression with val/test metrics
- All metrics in range [0, 1]
- Model file and artifacts exist

-------------------------------------------
ACTUAL RESULTS:
-------------------------------------------
- ✓ pipeline_config.json created successfully
- ✓ All 4 steps present (data_cleaning, generate_eda, data_encoding, train_logistic_regression)
- ✓ Parameters match expected values
- ✓ Metrics:
    val_accuracy: 1.0
    val_f1: 1.0
    val_precision: 1.0
    val_recall: 1.0
    val_roc_auc: 1.0
    test_accuracy: 1.0
    test_f1: 1.0
    (... all metrics ...)
- ✓ Model file exists: lg_test_1.pkl
- ✓ All 4 artifact PNGs exist

-------------------------------------------
SCREENSHOTS:
-------------------------------------------
[Attach screenshots if failure occurs]
- screenshot_1_upload_success.png
- screenshot_2_training_complete.png
- screenshot_3_metrics_displayed.png

-------------------------------------------
PIPELINE_CONFIG.JSON:
-------------------------------------------
Experiment: Exp_20251028_124536_abc123def
File location: /app/experimentos/Exp_20251028_124536_abc123def/pipeline_config.json
[Attach or paste relevant sections]

-------------------------------------------
NOTES / OBSERVATIONS:
-------------------------------------------
- Training completed in 28 seconds
- All metrics = 1.0 suggests very easy dataset or overfitting
- No errors encountered during execution
- Energy metrics recorded: 6.52e-06 kWh

-------------------------------------------
PASS/FAIL CRITERIA MET:
-------------------------------------------
✓ All 4 steps exist in pipeline_config.json
✓ Parameters match UI selections
✓ Metrics within valid ranges
✓ Files exist on filesystem
✓ No errors in UI

===========================================
```

---

## Conclusion

This E2E test plan provides comprehensive coverage of the DREAM-ML system's core MLOps capabilities:

**Coverage Summary:**
- ✅ **29 test cases** covering all critical workflows
- ✅ **User Story #4:** Complete pipeline_config.json tracking validation
- ✅ **User Story #5:** Reproducibility verification with exact metrics matching
- ✅ **Edge cases:** File size limits, invalid data, experiment switching
- ✅ **Both experiment types:** Classification and Time Series thoroughly tested
- ✅ **Multiple algorithms:** Logistic Regression, MLP, XGBoost, ARIMA, LSTM
- ✅ **Hyperparameter optimization:** Manual, Random Search, Bayesian Search
- ✅ **MLOps practices:** DVC versioning, MLflow tracking, reproducibility

**Thesis Value:**
- Validates core thesis claims (reproducibility, documentation, automation)
- Demonstrates professional software engineering practices
- Provides evidence for thesis defense
- Tests can be shown as part of quality assurance methodology

**Execution Recommendation:**
1. Start with CRITICAL tests (User Stories #4 & #5 basics)
2. Add HIGH priority tests (algorithm variations)
3. Include MEDIUM priority edge cases as time permits
4. Document all test executions for thesis appendix

**Good luck with your thesis defense! 🎓**

---

**Document Version:** 1.0
**Last Updated:** October 28, 2025
**Author:** AI Assistant (Claude) for Leonardo Espinoza Ortiz
**License:** GNU GPL v3 (matches DREAM-ML project license)
