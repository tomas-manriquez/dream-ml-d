# TSTrainCard Feature Selection Research
## Technical Documentation: Variable de Salida, Variables de Entrada, and Columna de Fecha

**Document Version:** 1.0
**Date:** 2025-12-17
**Component:** `DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx` (3641 lines)
**Target Audience:** Engineering team implementing robustness improvements

---

## 1. Executive Summary

### Overview
The TSTrainCard component implements a multi-stage form for time series model training configuration. The feature selection system consists of three interdependent form sections that manage column assignments from uploaded CSV files: target variable (output), input features (predictors), and date column (temporal index).

### Key Findings
1. **Algorithm-dependent behavior**: Target selection triggers automatic feature selection for ARIMA/XGBoost but not LSTM
2. **Separate state management**: LSTM maintains independent feature selection state (`lstmSelectedFeatures`)
3. **Reactive validation**: Five validation rules execute after every selection change
4. **Destructive operations**: Target deselection clears all features without confirmation
5. **No persistence layer**: Page reload results in complete state loss

### Critical Observations
- **State synchronization**: Three separate state variables (`targetVariable`, `inputFeatures`, `dateColumnName`) must remain synchronized through validation logic
- **User impact**: Data scientists may experience confusion from auto-selection behavior and lack of undo functionality
- **Edge case handling**: Minimal handling for CSV files with <3 columns, duplicate headers, or special characters in column names

### Risk Assessment for Data Scientists
| Risk Category | Impact Level | Description |
|--------------|--------------|-------------|
| Data loss | High | No persistence on page reload |
| Configuration errors | Medium | Auto-selection may overwhelm users with 100+ features |
| Workflow disruption | Medium | Toggle deselection clears features destructively |
| Algorithm confusion | Medium | LSTM behavior differs without clear indication |

---

## 2. Architecture Overview

### Component Hierarchy
```
TSTrainCard (main component)
├── State Management (React.useState)
│   ├── csvFile
│   ├── columns[]
│   ├── targetVariable
│   ├── inputFeatures[]
│   ├── dateColumnName
│   ├── lstmSelectedFeatures[] (LSTM-specific)
│   └── validationWarnings[]
├── Event Handlers
│   ├── handleFileChange()
│   ├── loadColumns()
│   ├── handleTargetChange()
│   ├── handleFeatureChange()
│   ├── handleDateColumnChange()
│   └── validateSelections()
├── UI Components (Material-UI)
│   ├── RadioGroup (Target selection)
│   ├── FormGroup + Checkboxes (Feature selection)
│   ├── RadioGroup (Date column selection)
│   └── ValidationSummary (Warning display)
└── Backend Integration
    ├── POST /api/analyze-csv/ (column loading)
    └── POST /api/ts/train-model/ (training submission)
```

### State Variables

| Variable | Type | Line | Purpose | Initial Value |
|----------|------|------|---------|---------------|
| `csvFile` | File | 73 | Uploaded CSV file object | `null` |
| `columns` | String[] | 74 | Column names from CSV headers | `[]` |
| `inputFeatures` | String[] | 75 | Selected feature columns | `[]` |
| `targetVariable` | String | 76 | Selected target column | `""` |
| `dateColumnName` | String | 77 | Selected date column | `""` |
| `lstmSelectedFeatures` | String[] | 216 | LSTM-specific feature selection | `[]` |
| `validationWarnings` | String[] | (implicit) | Current validation errors | `[]` |

### Data Flow Sequence
```
1. User uploads CSV file
   └─> handleFileChange() sets csvFile state

2. User clicks "Cargar Variables" button
   └─> loadColumns() POST to /api/analyze-csv/
       └─> Backend returns {columns: [...]}
           └─> setColumns() triggers form visibility

3. User selects target variable (radio button)
   └─> handleTargetChange(column)
       ├─> setTargetVariable(column)
       ├─> Auto-select remaining columns (if algorithm !== "lstm")
       ├─> setInputFeatures([...])
       └─> validateSelections()
           └─> setValidationWarnings([...])

4. User modifies feature selection (checkboxes)
   └─> handleFeatureChange(column)
       ├─> setInputFeatures(prev => toggle column)
       └─> validateSelections()

5. User selects date column (radio button)
   └─> handleDateColumnChange(column)
       ├─> setDateColumnName(column)
       ├─> Remove column from inputFeatures if present
       └─> validateSelections()

6. User clicks "Entrenar Modelo" button
   └─> handleTrain()
       ├─> Construct payload with features, target, dateColumn
       ├─> Override payload.input_features if algorithm === "lstm"
       └─> POST /api/ts/train-model/ with FormData
```

### Backend Integration Points

#### Column Loading Endpoint
- **URL**: `POST /api/analyze-csv/`
- **Handler**: `DREAM-ML-backend/GEML/api/views.py:442-472`
- **Logic**: `DREAM-ML-backend/GEML/api/utils.py:401-415` (`analyze_csv_logic()`)
- **Request**: `FormData` with `file` parameter
- **Response**: `{columns: string[]}`
- **Validation**:
  - File extension must be `.csv`
  - File size limit: 10MB
  - Pandas parsing errors caught

#### Training Submission Endpoint
- **URL**: `POST /api/ts/train-model/`
- **Handler**: `DREAM-ML-backend/GEML/apiTimeSeries/views.py:376-479`
- **Request**: `FormData` with:
  - `file`: CSV file
  - `data`: JSON string containing:
    ```json
    {
      "input_features": ["col1", "col2"],
      "target_variable": "target_col",
      "date_col_name": "date_col",
      "algorithm": "arima|xgboost|lstm",
      "training_mode": "univariate|multivariate" // LSTM only
    }
    ```

---

## 3. Target Variable Selection (Variable de Salida)

### Implementation Details

**Component Location**: `TSTrainCard.jsx:1017-1049`

**UI Component**: Material-UI `RadioGroup` (single selection)

**State Variable**:
```javascript
const [targetVariable, setTargetVariable] = useState(""); // Line 76
```

**Event Handler**: `handleTargetChange()` (Lines 337-356)
```javascript
const handleTargetChange = (column) => {
  if (targetVariable === column) {
    // If clicking the same target, deselect it
    setTargetVariable("");
    setInputFeatures([]);
    // Note: lstmSelectedFeatures cleared by useEffect
  } else {
    // Set new target
    setTargetVariable(column);

    // Only auto-select for ARIMA and XGBoost (NOT LSTM)
    // LSTM users must explicitly choose features for better control over univariate/multivariate modes
    if (algorithm !== "lstm") {
      const remainingColumns = columns.filter((col) => col !== column && col !== dateColumnName);
      const newFeatures = [...new Set([...inputFeatures, ...remainingColumns])];
      setInputFeatures(newFeatures);
    }
  }
  validateSelections();
};
```

### Behavior Analysis

#### Auto-Selection Logic
When a target is selected (and `algorithm !== "lstm"`):
1. Filter out the selected target column
2. Filter out the current date column (if set)
3. Add ALL remaining columns to `inputFeatures`
4. Use `Set` to ensure uniqueness

**Code Reference**: `TSTrainCard.jsx:350-352`

#### Toggle Deselection
- Clicking the **same** radio button again deselects the target
- **Side effect**: Clears all input features (`setInputFeatures([])`)
- **No confirmation dialog** presented to user
- **Destructive operation**: No undo mechanism

**Code Reference**: `TSTrainCard.jsx:338-342`

#### Algorithm-Specific Behavior

| Algorithm | Auto-selection | Reason |
|-----------|---------------|---------|
| ARIMA | ✅ Yes | Traditional time series forecasting assumes all features used |
| XGBoost | ✅ Yes | Tree-based models benefit from all features |
| LSTM | ❌ No | Requires explicit feature selection for univariate/multivariate control |

**Code Reference**: `TSTrainCard.jsx:349`

### UI Implementation
```javascript
<RadioGroup value={targetVariable} onChange={(e) => handleTargetChange(e.target.value)}>
  {columns.map((col) => (
    <FormControlLabel
      key={col}
      value={col}
      control={<Radio sx={variableSelectionStyles.radioButton} />}
      label={col}
      sx={variableSelectionStyles.formControlLabel}
    />
  ))}
</RadioGroup>
```
**Code Reference**: `TSTrainCard.jsx:1037-1047`

### Helper Text
Display text: `"Selecciona la columna que deseas predecir (variable objetivo). Al seleccionar, se auto-seleccionarán las variables de entrada restantes."`

**Code Reference**: `TSTrainCard.jsx:1032-1033`, defined in `variableSelectionStyles.js`

### Info Modal
Accessible via info icon button that opens `<InfoModal />` component with expanded explanation of target variables.

**Code Reference**: `TSTrainCard.jsx:1023-1029`

---

## 4. Input Features Selection (Variables de Entrada)

### Implementation Details

**Component Location**: `TSTrainCard.jsx:1054-1092`

**UI Component**: Material-UI `FormGroup` with `Checkbox` controls (multiple selection)

**State Variable**:
```javascript
const [inputFeatures, setInputFeatures] = useState([]); // Line 75
```

**Event Handler**: `handleFeatureChange()` (Lines 327-334)
```javascript
const handleFeatureChange = (column) => {
  setInputFeatures((prev) =>
    prev.includes(column)
      ? prev.filter((item) => item !== column)  // Remove if already selected
      : [...prev, column]  // Add if not selected
  );
  validateSelections();
};
```

### Behavior Analysis

#### Toggle Mechanism
- **Click to select**: If column not in `inputFeatures`, add it
- **Click to deselect**: If column in `inputFeatures`, remove it
- State update uses functional form of `setState` for safe concurrent updates

**Code Reference**: `TSTrainCard.jsx:328-332`

#### Disabled State Logic
Checkboxes are disabled when:
```javascript
disabled={col === targetVariable || col === dateColumnName}
```

| Condition | Disabled | Reason |
|-----------|----------|---------|
| `col === targetVariable` | ✅ Yes | Target cannot be an input feature |
| `col === dateColumnName` | ✅ Yes | Date column cannot be a feature |
| Otherwise | ❌ No | Available for selection |

**Code Reference**: `TSTrainCard.jsx:1083`

#### Validation Requirements
- **Minimum**: At least 1 feature must be selected if target is set
- Validated in `validateSelections()` at line 371-373

### UI Implementation
```javascript
<FormGroup>
  {columns.map((col) => (
    <FormControlLabel
      key={col}
      control={
        <Checkbox
          checked={inputFeatures.includes(col)}
          onChange={() => handleFeatureChange(col)}
          sx={variableSelectionStyles.checkbox}
          disabled={col === targetVariable || col === dateColumnName}
        />
      }
      label={col}
      sx={variableSelectionStyles.formControlLabel}
    />
  ))}
</FormGroup>
```
**Code Reference**: `TSTrainCard.jsx:1074-1091`

### Helper Text
Display text: `"Selecciona las columnas que usarás como variables de entrada (features) para el modelo. Puedes seleccionar múltiples variables."`

**Code Reference**: `TSTrainCard.jsx:1069-1070`, defined in `variableSelectionStyles.js`

### Scrolling Container
Features are rendered in a scrollable box with:
- **Max height**: 150px
- **Overflow**: auto
- **Border**: 2px solid teal (#00796b)

**Code Reference**: `variableSelectionStyles.js` (`variableBox` style)

### LSTM Exception
When `algorithm === "lstm"`, the component uses a **separate state** (`lstmSelectedFeatures`) instead of `inputFeatures` for the final payload. See Section 6 for details.

---

## 5. Date Column Selection (Columna de Fecha)

### Implementation Details

**Component Location**: `TSTrainCard.jsx:1094-1117`

**UI Component**: Material-UI `RadioGroup` (single selection)

**State Variable**:
```javascript
const [dateColumnName, setDateColumnName] = useState(""); // Line 77
```

**Event Handler**: `handleDateColumnChange()` (Lines 359-364)
```javascript
const handleDateColumnChange = (column) => {
  setDateColumnName(column);
  // Remove from features if it was selected
  setInputFeatures((prev) => prev.filter((item) => item !== column));
  validateSelections();
};
```

### Behavior Analysis

#### Auto-Cleanup Logic
When a date column is selected:
1. Set `dateColumnName` to the selected column
2. **Automatically remove** the column from `inputFeatures` if it was previously selected
3. Trigger validation

**Code Reference**: `TSTrainCard.jsx:360-362`

**Rationale**: Date columns represent temporal indices and should not be used as predictive features in time series models.

#### Requirement Level
- **Mandatory**: Date column is required for all time series forecasting
- Enforced in train button disabled logic

### UI Implementation
```javascript
<RadioGroup value={dateColumnName} onChange={(e) => handleDateColumnChange(e.target.value)}>
  {columns.map((col) => (
    <FormControlLabel
      key={col}
      value={col}
      control={<Radio sx={variableSelectionStyles.radioButton} />}
      label={col}
      sx={variableSelectionStyles.formControlLabel}
    />
  ))}
</RadioGroup>
```
**Code Reference**: `TSTrainCard.jsx:1105-1115`

### Helper Text
Display text: `"Selecciona la columna que contiene las fechas para el análisis de series temporales"`

**Code Reference**: `TSTrainCard.jsx:1100-1102`

### No Info Modal
Unlike target and features sections, date column selection does not have an info modal icon.

---

## 6. LSTM-Specific Feature Selection

### Overview
LSTM algorithm maintains **completely separate** feature selection state from ARIMA/XGBoost. This architectural decision provides explicit control over univariate vs multivariate forecasting modes.

### Separate State Management

**State Variable**:
```javascript
// LSTM-specific feature selection (separate from global inputFeatures) - Phase 4
const [lstmSelectedFeatures, setLstmSelectedFeatures] = useState([]); // Line 216
```

**Key Difference**:
- `inputFeatures` (line 75): Used for ARIMA/XGBoost
- `lstmSelectedFeatures` (line 216): Used for LSTM

### Auto-Selection Behavior Difference

**ARIMA/XGBoost**: Auto-select all remaining columns when target is selected
```javascript
if (algorithm !== "lstm") {
  const remainingColumns = columns.filter((col) => col !== column && col !== dateColumnName);
  const newFeatures = [...new Set([...inputFeatures, ...remainingColumns])];
  setInputFeatures(newFeatures);
}
```
**Code Reference**: `TSTrainCard.jsx:349-352`

**LSTM**: No auto-selection
- User must **manually** select features
- Enables explicit choice between univariate and multivariate modes

### Univariate vs Multivariate Mode Logic

During training submission:
```javascript
if (algorithm === "lstm") {
  // Override input_features with LSTM-specific selection
  payload.input_features = lstmSelectedFeatures;
  payload.training_mode = lstmSelectedFeatures.length === 0 ? "univariate" : "multivariate";
}
```
**Code Reference**: `TSTrainCard.jsx:761-762`

| `lstmSelectedFeatures.length` | Training Mode | Behavior |
|-------------------------------|---------------|----------|
| 0 | `"univariate"` | Model uses only target variable's history |
| ≥ 1 | `"multivariate"` | Model uses target + additional features |

### State Cleanup: useEffect Hook

**Implementation**:
```javascript
// Clear LSTM features when algorithm, target, or columns change (Phase 4)
useEffect(() => {
  setLstmSelectedFeatures([]);
}, [algorithm, targetVariable, columns]);
```
**Code Reference**: `TSTrainCard.jsx:636-638`

**Trigger Conditions**:
1. **Algorithm changes**: Switching from/to LSTM clears LSTM features
2. **Target variable changes**: New target requires re-evaluation of features
3. **Columns change**: New CSV uploaded, invalidates previous selections

**Timing**: Executes **after** render cycle when dependencies change

### Target Deselection Side Effect
When target is deselected via toggle:
```javascript
if (targetVariable === column) {
  setTargetVariable("");
  setInputFeatures([]);
  // Note: lstmSelectedFeatures cleared by useEffect
}
```
**Code Reference**: `TSTrainCard.jsx:338-342`

**Note at line 342**: Comment indicates reliance on `useEffect` for cleanup, but this creates a dependency on React's effect scheduling.

### Payload Construction Difference

**ARIMA/XGBoost Payload**:
```javascript
const payload = {
  input_features: inputFeatures,  // From line 75 state
  target_variable: targetVariable,
  // ...
};
```

**LSTM Payload Override**:
```javascript
if (algorithm === "lstm") {
  payload.input_features = lstmSelectedFeatures;  // Override
  payload.training_mode = lstmSelectedFeatures.length === 0 ? "univariate" : "multivariate";
}
```
**Code Reference**: `TSTrainCard.jsx:755-763`

### UI Rendering
LSTM feature selection likely rendered in a separate UI section (not shown in lines 1054-1092). This separate UI would bind to `lstmSelectedFeatures` state instead of `inputFeatures`.

---

## 7. Validation System

### Validation Function

**Function**: `validateSelections()` (Lines 367-396)

**Invocation Points**:
- After target selection: `TSTrainCard.jsx:355`
- After feature toggle: `TSTrainCard.jsx:333`
- After date column selection: `TSTrainCard.jsx:363`

**Return Value**: `boolean` (true if no warnings, false if warnings exist)

### Validation Rules

#### Rule 1: Minimum Features
```javascript
if (inputFeatures.length === 0 && targetVariable) {
  warnings.push("Debes seleccionar al menos 1 variable de entrada");
}
```
**Code Reference**: `TSTrainCard.jsx:371-373`

**Condition**: Target is set but no features selected
**Message**: "Debes seleccionar al menos 1 variable de entrada"

#### Rule 2: Target Required with Features
```javascript
if (!targetVariable && inputFeatures.length > 0) {
  warnings.push("Debes seleccionar 1 variable de salida");
}
```
**Code Reference**: `TSTrainCard.jsx:376-378`

**Condition**: Features selected but no target set
**Message**: "Debes seleccionar 1 variable de salida"

#### Rule 3: No Overlap Between Target and Features
```javascript
if (targetVariable && inputFeatures.includes(targetVariable)) {
  warnings.push("Una columna no puede ser entrada y salida simultáneamente");
}
```
**Code Reference**: `TSTrainCard.jsx:381-383`

**Condition**: Target column also selected as feature
**Message**: "Una columna no puede ser entrada y salida simultáneamente"

**Note**: This condition should be **impossible** due to checkbox disabled logic at line 1083, but validation provides defense-in-depth.

#### Rule 4: Date Column Cannot Be Feature
```javascript
if (dateColumnName && inputFeatures.includes(dateColumnName)) {
  warnings.push("La columna de fecha no puede ser una variable de entrada");
}
```
**Code Reference**: `TSTrainCard.jsx:386-388`

**Condition**: Date column also selected as feature
**Message**: "La columna de fecha no puede ser una variable de entrada"

**Note**: This condition should be **impossible** due to:
1. Checkbox disabled logic (line 1083)
2. Auto-cleanup in `handleDateColumnChange` (line 362)

Validation provides additional safety layer.

#### Rule 5: Date Column Cannot Be Target
```javascript
if (dateColumnName && targetVariable === dateColumnName) {
  warnings.push("La columna de fecha no puede ser la variable de salida");
}
```
**Code Reference**: `TSTrainCard.jsx:390-392`

**Condition**: Date column selected as target
**Message**: "La columna de fecha no puede ser la variable de salida"

**Note**: This condition **is possible** since target selection has no disabled logic preventing date column selection.

### Validation State Update
```javascript
setValidationWarnings(warnings);
return warnings.length === 0;
```
**Code Reference**: `TSTrainCard.jsx:394-395`

Updates `validationWarnings` state array, triggering re-render of `<ValidationSummary />` component.

### ValidationSummary Component

**Component**: `DREAM-ML-frontend/frontend/src/components/ValidationSummary.jsx`

**Rendering Location**: `TSTrainCard.jsx:1052`
```javascript
<ValidationSummary warnings={validationWarnings} />
```

**Visibility**: Only renders if `warnings.length > 0` (early return at line 26-28)

**Styling**:
- Background: `#fff3e0` (orange-tinted cream)
- Border: `2px solid #ff6f00` (dark orange)
- Icon: `<WarningIcon />` in orange (`#ff6f00`)
- Title: "ADVERTENCIAS DE VALIDACIÓN"
- Each warning prefixed with `▸` symbol

**Code Reference**: `ValidationSummary.jsx:31-89`

### Train Button Disabled Logic

The train button is disabled when any of the following conditions are true:
```javascript
const isDisabled =
  trainInProgress ||
  !experimentDir ||
  !runId ||
  !flow.encodeDone ||
  flow.trainDone ||
  (algorithm !== "lstm" && !inputFeatures.length) ||  // Features required for ARIMA/XGBoost
  !targetVariable ||  // Target required
  !dateColumnName ||  // Date column required
  !modelName.trim() ||
  targetVariable === dateColumnName ||  // Cannot be same
  inputFeatures.includes(dateColumnName) ||  // Date cannot be feature
  !isRandomSearchParamsValid() ||
  !isLSTMParamsValid() ||
  validationWarnings.length > 0 ||  // Block if ANY warnings
  !splitRatiosValid;
```
**Code Reference**: Location not shown in excerpts, but referenced in exploration findings

**Key Feature Selection Conditions**:
1. `!targetVariable` - Target must be selected
2. `!dateColumnName` - Date column must be selected
3. `algorithm !== "lstm" && !inputFeatures.length` - Features required for ARIMA/XGBoost
4. `targetVariable === dateColumnName` - Target and date cannot be same column
5. `inputFeatures.includes(dateColumnName)` - Date column cannot be feature
6. `validationWarnings.length > 0` - No validation warnings allowed

---

## 8. Data Flow Analysis

### Phase 1: CSV Upload

**User Action**: Selects CSV file via file input

**Handler**: `handleFileChange()` (location not shown in excerpts)

**State Update**:
```javascript
setCsvFile(fileObject);
```

### Phase 2: Column Loading

**User Action**: Clicks "Cargar Variables" button

**Button Render**: `TSTrainCard.jsx:1004-1012`
```javascript
<Button
  variant="contained"
  onClick={loadColumns}
  disabled={!csvFile || trainInProgress}
  sx={variableSelectionStyles.button}
>
  {trainInProgress ? <CircularProgress size={24} sx={{ color: "#fff" }} /> : "Cargar Variables"}
</Button>
```

**Handler**: `loadColumns()` (Lines 283-324)
```javascript
const loadColumns = async () => {
  if (!csvFile) {
    setTrainStatus("⚠️ Selecciona un archivo CSV primero.");
    return;
  }

  setTrainInProgress(true);
  setTrainStatus("⏳ Cargando columnas del archivo...");

  try {
    const formData = new FormData();
    formData.append("file", csvFile);

    const response = await axios.post("/analyze-csv/", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    if (response.data && response.data.columns) {
      setColumns(response.data.columns);
      setTrainStatus("✅ Columnas cargadas. Ahora selecciona target, features y columna de fecha.");
    } else {
      setTrainStatus("❌ No se pudieron obtener las columnas del archivo.");
    }
  } catch (error) {
    console.error("Error al cargar columnas:", error);
    setTrainStatus("❌ Error al cargar las columnas del archivo.");
  } finally {
    setTrainInProgress(false);
  }
};
```
**Code Reference**: `TSTrainCard.jsx:283-324`

**Backend Endpoint**: `POST /api/analyze-csv/`

**Backend Handler**: `DREAM-ML-backend/GEML/api/views.py:442-472`
```python
@csrf_exempt
def analyze_csv(request):
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            csv_file = request.FILES['file']

            # Validate file extension
            if not csv_file.name.endswith('.csv'):
                return JsonResponse({"error": "El archivo debe ser un CSV."}, status=400)

            # File size limit: 10MB
            max_size = 10 * 1024 * 1024
            if csv_file.size > max_size:
                return JsonResponse({"error": "El archivo excede el tamaño máximo permitido de 10MB."}, status=400)

            result = analyze_csv_logic(csv_file)
            return JsonResponse(result, status=200)
        except pd.errors.EmptyDataError:
            return JsonResponse({"error": "El archivo CSV está vacío."}, status=400)
        except pd.errors.ParserError:
            return JsonResponse({"error": "Error al parsear el archivo CSV."}, status=400)
        except Exception as e:
            return JsonResponse({"error": f"Error al analizar el archivo: {str(e)}"}, status=500)

    return JsonResponse({"error": "Método no permitido"}, status=405)
```
**Code Reference**: `api/views.py:442-472`

**Backend Logic**: `DREAM-ML-backend/GEML/api/utils.py` (analyze_csv_logic function)
- Reads CSV headers only (optimized with `nrows=0`)
- Returns `{columns: [...]}`

**Success Response**:
```json
{
  "columns": ["date", "temperature", "humidity", "target_variable"]
}
```

**Error Responses**:
- File not CSV: `{"error": "El archivo debe ser un CSV."}`
- File too large: `{"error": "El archivo excede el tamaño máximo permitido de 10MB."}`
- Empty CSV: `{"error": "El archivo CSV está vacío."}`
- Parse error: `{"error": "Error al parsear el archivo CSV. Asegúrate de que está bien formateado."}`

**Frontend State Update**:
```javascript
setColumns(response.data.columns);
```

**UI Effect**: Form sections at lines 1015-1119 become visible due to condition:
```javascript
{columns.length > 0 && (
  // Render target, features, and date selection forms
)}
```
**Code Reference**: `TSTrainCard.jsx:1015`

### Phase 3: Feature Selection

User interacts with three form sections:
1. Target variable selection (triggers `handleTargetChange`)
2. Input features selection (triggers `handleFeatureChange`)
3. Date column selection (triggers `handleDateColumnChange`)

Each handler calls `validateSelections()` after state updates.

### Phase 4: Training Submission

**User Action**: Clicks "Entrenar Modelo" button

**Handler**: `handleTrain()` (Lines 641+, excerpt at 750-799)

**Payload Construction**:
```javascript
const payload = {
  model_name: modelName,
  input_features: inputFeatures,  // Array of column names
  target_variable: targetVariable,  // Single column name
  date_col_name: dateColumnName,  // Single column name
  experiment_dir: experimentDir,
  split_ratios: splitRatios,
  run_id: runId,
  algorithm: algorithm,
  problem_type: problemType, // "ts_forecasting"
  forecast_horizon: forecastHorizon,
};
```

**LSTM Override**:
```javascript
if (algorithm === "lstm") {
  payload.sequence_length = sequenceLength;
  payload.early_stopping_patience = earlyStoppingPatience;
  payload.optimization_metric = "mse";

  // Override input_features with LSTM-specific selection
  payload.input_features = lstmSelectedFeatures;
  payload.training_mode = lstmSelectedFeatures.length === 0 ? "univariate" : "multivariate";
}
```
**Code Reference**: `TSTrainCard.jsx:755-763`

**FormData Construction**:
```javascript
const formData = new FormData();
formData.append("file", csvFile);
formData.append("data", JSON.stringify(payload));
```
**Code Reference**: `TSTrainCard.jsx:792-794`

**API Call**:
```javascript
const response = await axios.post("/ts/train-model/", formData, {
  headers: {
    "Content-Type": "multipart/form-data",
  },
});
```

**Backend Endpoint**: `POST /api/ts/train-model/`

**Backend Handler**: `DREAM-ML-backend/GEML/apiTimeSeries/views.py:376-479`

**Success Response** (expected structure):
```json
{
  "status": "success",
  "run_id": "...",
  "metrics": {},
  "model_path": "...",
  "mlflow_ui": "..."
}
```

**Error Handling** (Lines 822-833, from exploration):
```javascript
catch (error) {
  const errorMessage = error.response?.data?.message || "Error desconocido";
  const errorDetails = error.response?.data?.error_details;

  let fullErrorMessage = `❌ Error durante el entrenamiento: ${errorMessage}`;
  if (errorDetails) {
    fullErrorMessage += `\n📝 Detalles: ${errorDetails}`;
  }
  setTrainStatus(fullErrorMessage);
}
```

---

## 9. Edge Cases & Failure Modes

### Category 1: Issues That Block Work

#### EC-1: CSV with Insufficient Columns
**Scenario**: CSV file contains only 1-2 columns (e.g., date + target)

**Reproduction**:
1. Upload CSV with columns: `["date", "target"]`
2. Select `"target"` as target variable
3. Select `"date"` as date column
4. Feature selection has 0 available columns (both disabled)

**Current Handling**:
- Validation rule 1 triggers: "Debes seleccionar al menos 1 variable de entrada"
- Train button disabled due to `validationWarnings.length > 0`

**Impact**: User cannot proceed with univariate time series model even though it's a valid configuration.

**Technical Detail**: Validation at `TSTrainCard.jsx:371-373` does not account for intentional univariate models.

---

#### EC-2: Empty CSV File
**Scenario**: Uploaded CSV file has no data rows or is completely empty

**Reproduction**:
1. Upload empty CSV file
2. Click "Cargar Variables"

**Current Handling**:
- Backend catches `pd.errors.EmptyDataError`
- Returns error: `{"error": "El archivo CSV está vacío."}`
- Frontend displays: "❌ Error al cargar las columnas del archivo."

**Code Reference**: `api/views.py:462-464`

**Impact**: Blocks progress appropriately with clear error message.

---

#### EC-3: CSV Parsing Errors
**Scenario**: Malformed CSV file (incorrect delimiters, unmatched quotes, etc.)

**Reproduction**:
1. Upload CSV with malformed structure
2. Click "Cargar Variables"

**Current Handling**:
- Backend catches `pd.errors.ParserError`
- Returns: `{"error": "Error al parsear el archivo CSV. Asegúrate de que está bien formateado."}`
- Frontend displays: "❌ Error al cargar las columnas del archivo."

**Code Reference**: `api/views.py:465-467`

**Impact**: Blocks with generic error; does not specify what's wrong with formatting.

---

#### EC-4: File Size Exceeds 10MB Limit
**Scenario**: Large CSV file exceeds backend size limit

**Reproduction**:
1. Upload CSV file >10MB
2. Click "Cargar Variables"

**Current Handling**:
- Backend rejects with: `{"error": "El archivo excede el tamaño máximo permitido de 10MB."}`
- Frontend displays: "❌ Error al cargar las columnas del archivo."

**Code Reference**: `api/views.py:455-458`

**Impact**: Blocks with clear explanation of limit.

---

#### EC-5: Backend Validation Mismatch
**Scenario**: Frontend validation passes but backend rejects configuration

**Reproduction**: Unknown - requires knowledge of backend validation rules not visible in frontend code

**Current Handling**: Error caught and displayed via generic error message

**Impact**: User completes form thinking configuration is valid, then receives unexpected error on submission.

---

### Category 2: Issues That Cause Confusion

#### EC-6: Auto-Selection Overwhelm
**Scenario**: CSV with 100+ columns; selecting target auto-selects 99+ features

**Reproduction**:
1. Upload CSV with 100 columns
2. Select any column as target
3. Observe all 99 remaining columns selected as features (if algorithm !== "lstm")

**Current Handling**:
- Auto-selection logic at `TSTrainCard.jsx:350-352` has no column count limit
- Scrollable container shows first ~10 features; rest require scrolling

**Impact**:
- Data scientists may not realize 99 features are selected
- Overwhelming to manually deselect unwanted features
- Contradicts principle of explicit feature engineering

---

#### EC-7: Toggle Deselection Without Confirmation
**Scenario**: Accidentally clicking target radio button again clears all features

**Reproduction**:
1. Select target variable (auto-selects features)
2. Manually adjust feature selection (e.g., deselect 20 features)
3. Accidentally click same target radio button
4. All features cleared; manual work lost

**Current Handling**:
- Toggle logic at `TSTrainCard.jsx:338-342`
- No confirmation dialog
- No undo mechanism

**Impact**: Destructive operation without warning; frustrating for users who spent time curating features.

---

#### EC-8: LSTM Mode Indication Missing
**Scenario**: User switches to LSTM algorithm; no indication that feature selection behavior differs

**Reproduction**:
1. Select ARIMA, select target (features auto-selected)
2. Switch algorithm dropdown to LSTM
3. No visible indication that feature selection now uses separate state

**Current Handling**:
- `useEffect` at `TSTrainCard.jsx:636-638` clears `lstmSelectedFeatures`
- No UI indication of mode change
- Helper text does not mention algorithm-specific behavior

**Impact**: User may assume selected features apply to LSTM, but `lstmSelectedFeatures` is empty, resulting in univariate mode unexpectedly.

---

#### EC-9: Date Column as Target Not Prevented
**Scenario**: User can select date column as target variable

**Reproduction**:
1. Load columns
2. Select date column as target
3. Select same column as date column

**Current Handling**:
- Target selection has no disabled logic preventing date column
- Validation rule 5 triggers: "La columna de fecha no puede ser la variable de salida"
- Train button disabled

**Impact**: User can make invalid selection; validation catches it but doesn't prevent it proactively. Poor UX compared to feature checkboxes which disable target/date columns.

---

#### EC-10: Duplicate Column Names
**Scenario**: CSV has duplicate column headers

**Reproduction**:
1. Upload CSV with headers: `["date", "temp", "temp", "target"]`
2. Pandas reads as: `["date", "temp", "temp.1", "target"]` OR `["date", "temp", "temp", "target"]` depending on version

**Current Handling**: No explicit handling; behavior depends on Pandas version

**Impact**:
- React `key` warnings if duplicate keys in map
- Potential state corruption if two columns have same name

---

#### EC-11: Special Characters in Column Names
**Scenario**: CSV columns contain spaces, quotes, or unicode characters

**Reproduction**:
1. Upload CSV with columns: `["Date Time", "Temp °C", "Target [kg]"]`
2. Select columns normally

**Current Handling**: No sanitization or validation of column names

**Impact**:
- May cause backend processing errors if special characters not handled
- JSON serialization issues possible with certain characters

---

### Category 3: Minor Annoyances

#### EC-12: No State Persistence on Page Reload
**Scenario**: User completes feature selection, accidentally refreshes page

**Reproduction**:
1. Complete all feature selections
2. Refresh browser page (F5 or Cmd+R)

**Current Handling**: All React state reset to initial values

**Impact**: User must re-upload CSV and re-select everything. No localStorage or sessionStorage persistence.

---

#### EC-13: No Undo Mechanism
**Scenario**: User wants to revert to previous feature selection

**Reproduction**:
1. Select target and features
2. Make changes to feature selection
3. Want to restore previous state

**Current Handling**: No undo/redo functionality

**Impact**: User must manually remember and re-select previous configuration.

---

#### EC-14: Large Column Set Scrolling
**Scenario**: CSV with 50+ columns requires extensive scrolling in selection boxes

**Reproduction**:
1. Upload CSV with 50+ columns
2. Try to view all available features

**Current Handling**:
- Scrollable container with `maxHeight: "150px"`
- No search/filter functionality

**Impact**: Difficult to navigate large feature sets; no way to quickly find specific columns.

---

#### EC-15: Validation Timing (Post-Selection)
**Scenario**: Validation warnings appear after invalid selection made

**Reproduction**:
1. Select features before selecting target
2. Validation warning appears: "Debes seleccionar 1 variable de salida"

**Current Handling**: Reactive validation via `validateSelections()` called after state changes

**Impact**: User makes invalid selection, then sees warning. Preventive UI (disabled controls) would be better UX in some cases.

---

#### EC-16: Race Condition in Rapid Selection
**Scenario**: User rapidly clicks multiple selections before React state updates complete

**Reproduction**:
1. Rapidly click target radio button multiple times
2. Rapidly toggle multiple feature checkboxes

**Current Handling**:
- State updates queued by React
- Functional setState form used in `handleFeatureChange` provides some protection

**Impact**: Possible state inconsistencies if updates overlap, though React's batching mitigates this.

**Technical Detail**: `handleTargetChange` uses direct state access (`inputFeatures`) rather than functional form when auto-selecting, creating potential for stale state reads.

**Code Reference**: `TSTrainCard.jsx:350-352` uses `inputFeatures` directly instead of `prev => ...`

---

#### EC-17: LSTM State Cleanup Timing
**Scenario**: `useEffect` clears `lstmSelectedFeatures` after dependency changes, but timing may not be immediate

**Reproduction**:
1. Select LSTM features
2. Change algorithm to ARIMA
3. Immediately change back to LSTM before effect fires

**Current Handling**:
- `useEffect` at line 636-638 schedules cleanup
- React effect timing not guaranteed to be synchronous

**Impact**: Possible edge case where old `lstmSelectedFeatures` briefly persists before cleanup.

---

## 10. Code References Table

### State Variables

| Variable | Type | Line | Purpose | Initial Value |
|----------|------|------|---------|---------------|
| `csvFile` | File \| null | 73 | Uploaded CSV file object | `null` |
| `columns` | string[] | 74 | Column names from CSV headers | `[]` |
| `inputFeatures` | string[] | 75 | Selected feature columns (ARIMA/XGBoost) | `[]` |
| `targetVariable` | string | 76 | Selected target column | `""` |
| `dateColumnName` | string | 77 | Selected date column | `""` |
| `lstmSelectedFeatures` | string[] | 216 | LSTM-specific feature selection | `[]` |
| `validationWarnings` | string[] | (implicit) | Current validation errors | `[]` |

### Event Handlers

| Function | Lines | Purpose | Triggers Validation |
|----------|-------|---------|---------------------|
| `handleFileChange()` | (not shown) | Store uploaded CSV file | No |
| `loadColumns()` | 283-324 | POST to /analyze-csv/, populate columns state | No |
| `handleTargetChange(column)` | 337-356 | Set target, auto-select features (if not LSTM), toggle deselect | Yes (line 355) |
| `handleFeatureChange(column)` | 327-334 | Toggle feature in/out of selection | Yes (line 333) |
| `handleDateColumnChange(column)` | 359-364 | Set date column, remove from features | Yes (line 363) |
| `validateSelections()` | 367-396 | Execute 5 validation rules, update warnings state | N/A (is validation) |
| `handleTrain()` | 641+ (excerpt 750-799) | Construct payload, POST to /ts/train-model/ | No |

### Validation Rules

| Rule | Lines | Condition | Error Message |
|------|-------|-----------|---------------|
| Rule 1 | 371-373 | `inputFeatures.length === 0 && targetVariable` | "Debes seleccionar al menos 1 variable de entrada" |
| Rule 2 | 376-378 | `!targetVariable && inputFeatures.length > 0` | "Debes seleccionar 1 variable de salida" |
| Rule 3 | 381-383 | `targetVariable && inputFeatures.includes(targetVariable)` | "Una columna no puede ser entrada y salida simultáneamente" |
| Rule 4 | 386-388 | `dateColumnName && inputFeatures.includes(dateColumnName)` | "La columna de fecha no puede ser una variable de entrada" |
| Rule 5 | 390-392 | `dateColumnName && targetVariable === dateColumnName` | "La columna de fecha no puede ser la variable de salida" |

### UI Components

| Component | Lines | Purpose | Binds To State |
|-----------|-------|---------|----------------|
| File input | (not shown) | Upload CSV file | `csvFile` |
| "Cargar Variables" button | 1004-1012 | Trigger column loading | Disabled when `!csvFile \|\| trainInProgress` |
| Target RadioGroup | 1037-1047 | Select target variable | `targetVariable` |
| Features FormGroup | 1074-1091 | Multi-select features | `inputFeatures` |
| Date RadioGroup | 1105-1115 | Select date column | `dateColumnName` |
| ValidationSummary | 1052 | Display validation warnings | `validationWarnings` |
| "Entrenar Modelo" button | (not shown) | Submit training request | Disabled by complex logic |

### Backend Endpoints

| Endpoint | Handler File:Line | Purpose | Request | Response |
|----------|-------------------|---------|---------|----------|
| `POST /api/analyze-csv/` | `api/views.py:442-472` | Extract column names from CSV | FormData with `file` | `{columns: string[]}` or error |
| `POST /api/ts/train-model/` | `apiTimeSeries/views.py:376-479` | Train time series model | FormData with `file` and JSON `data` | Success with metrics or error |

### React Effects

| Effect | Lines | Dependencies | Purpose |
|--------|-------|--------------|---------|
| Grid combinations update | 624-633 | `[algorithm, optimizationMethod, lstmGridOptions]` | Recalculate LSTM grid search combinations |
| LSTM features cleanup | 636-638 | `[algorithm, targetVariable, columns]` | Clear `lstmSelectedFeatures` when context changes |

---

## 11. Open Questions

### Q1: Backend Validation Rules
**Question**: What additional validation does the backend perform on `input_features`, `target_variable`, and `date_col_name` that is not enforced in the frontend?

**Context**: Frontend validation allows certain selections (e.g., date column as target before date selection made) that backend may reject.

**Impact**: Understanding backend validation would allow frontend to provide proactive error prevention.

---

### Q2: LSTM Univariate Minimum Features
**Question**: For LSTM univariate mode (`lstmSelectedFeatures.length === 0`), is there a backend check that ensures target variable is valid for univariate forecasting?

**Context**: Code at `TSTrainCard.jsx:762` sets `training_mode` based on feature count, but doesn't validate target suitability.

---

### Q3: Column Name Sanitization
**Question**: Does the backend sanitize or validate column names for special characters, reserved keywords, or length limits?

**Context**: Frontend accepts any column names returned by Pandas, including those with spaces, unicode, or special characters.

**Impact**: If backend has restrictions, frontend should validate to provide immediate feedback.

---

### Q4: Duplicate Column Handling
**Question**: How does the system handle CSV files with duplicate column names?

**Context**: Pandas' behavior varies by version:
- Older versions: Allow duplicates
- Newer versions: Rename to `column`, `column.1`, `column.2`

**Impact**: React key warnings and potential state corruption if duplicates exist.

---

### Q5: Maximum Column Count
**Question**: Is there a practical limit to the number of columns supported?

**Context**: Frontend will render hundreds of radio buttons/checkboxes if CSV has hundreds of columns, causing performance and UX issues.

**Impact**: Large column sets create poor UX; may need pagination or search functionality.

---

### Q6: Feature Selection for ARIMA vs XGBoost
**Question**: Are there algorithm-specific restrictions on feature selection for ARIMA vs XGBoost that frontend should enforce?

**Context**: Both algorithms use `inputFeatures` state and share auto-selection logic, but may have different requirements.

---

### Q7: Date Column Format Validation
**Question**: Does the backend validate that the selected date column contains parseable date/time values?

**Context**: Frontend only allows selection by column name; doesn't inspect values.

**Impact**: User may select non-date column as date column, causing backend parsing errors.

---

### Q8: LSTM State Cleanup Race Conditions
**Question**: Are there scenarios where the `useEffect` cleanup at line 636-638 executes too late, allowing stale `lstmSelectedFeatures` to be submitted?

**Context**: React effects are scheduled asynchronously after render.

**Impact**: Potential for submitting wrong features in edge cases.

---

### Q9: Feature Config for XGBoost and LSTM
**Question**: What is the structure and purpose of `feature_config` added to payload at line 789?

**Context**: Code shows `payload.feature_config = featureConfig` but state variable `featureConfig` not shown in excerpts.

**Impact**: Unknown dependency on feature selection system.

---

### Q10: Split Ratios Validation
**Question**: How does `splitRatiosValid` (used in train button disabled logic) relate to feature selection?

**Context**: Split ratios validation is separate concern but blocks training.

**Impact**: User may complete feature selection but still be blocked by split ratio issues.

---

## 12. Identified Issues for Robustness

### Issues That Block Work

#### I-1: Univariate Model Configuration Impossible
**Location**: Validation rule 1 (`TSTrainCard.jsx:371-373`)

**Issue**: Validation requires at least 1 feature even when target-only (univariate) models are valid time series configurations. CSV with only date and target columns cannot proceed.

**User Impact**: Data scientists with genuinely univariate datasets (single time series) cannot use the system.

---

#### I-2: Generic Backend Error Messages
**Location**: `loadColumns()` catch block (`TSTrainCard.jsx:319-321`)

**Issue**: All backend errors displayed as generic "❌ Error al cargar las columnas del archivo" without specific error details from response.

**User Impact**: Users receive unhelpful error message when backend provides specific error in `response.data.error`.

---

#### I-3: Backend Validation Mismatch Risk
**Location**: Frontend validation vs unknown backend validation

**Issue**: Frontend validation rules may not match backend validation, allowing users to complete form with configuration that backend will reject.

**User Impact**: Wasted effort completing form only to receive error on submission.

---

### Issues That Cause Confusion

#### I-4: Auto-Selection Without User Control
**Location**: `handleTargetChange()` (`TSTrainCard.jsx:349-352`)

**Issue**: Selecting target automatically selects ALL remaining features for ARIMA/XGBoost without user confirmation or control. No maximum column threshold.

**User Impact**:
- Unexpected behavior for users accustomed to manual feature selection
- Overwhelming with large column sets (50+ features)
- Contradicts explicit feature engineering workflow

---

#### I-5: Destructive Toggle Deselection
**Location**: `handleTargetChange()` toggle logic (`TSTrainCard.jsx:338-342`)

**Issue**: Clicking same target radio button deselects target and clears ALL features without confirmation dialog. No undo mechanism.

**User Impact**: Accidental click destroys manual feature curation work.

---

#### I-6: LSTM Mode Change Not Visible
**Location**: LSTM-specific feature selection system (separate state at line 216)

**Issue**: No UI indication that LSTM uses different feature selection state. Helper text doesn't mention algorithm-specific behavior.

**User Impact**: Users switch to LSTM and don't realize their feature selections are no longer applied, resulting in unexpected univariate mode.

---

#### I-7: Date Column Can Be Selected as Target
**Location**: Target RadioGroup (`TSTrainCard.jsx:1037-1047`)

**Issue**: Unlike feature checkboxes which disable target and date columns, target radio buttons have no disabled logic preventing date column selection.

**User Impact**: User can make invalid selection; validation catches it reactively but doesn't prevent it proactively. Inconsistent with feature selection UX.

---

#### I-8: No Column Name Validation
**Location**: Column loading (`TSTrainCard.jsx:306`)

**Issue**: Frontend accepts any column names returned by Pandas without validation for special characters, length, reserved keywords, or duplicates.

**User Impact**:
- React key warnings with duplicate columns
- Potential backend errors with special characters
- Poor error messages when backend rejects certain column names

---

#### I-9: Validation Timing (Reactive Not Preventive)
**Location**: All selection handlers trigger `validateSelections()` after state change

**Issue**: Validation occurs after invalid selection made rather than preventing it. Train button disabling is only enforcement.

**User Impact**: Users make invalid selections and see warnings, rather than UI preventing invalid actions.

---

### Minor Annoyances

#### I-10: No State Persistence
**Location**: All React state variables

**Issue**: No localStorage or sessionStorage persistence. Page reload or navigation away loses all selections.

**User Impact**: Users must re-upload CSV and re-select everything after accidental refresh.

---

#### I-11: No Undo/Redo Functionality
**Location**: All selection handlers

**Issue**: No mechanism to revert to previous selections.

**User Impact**: Users must manually remember and recreate previous configurations if they make mistakes.

---

#### I-12: Large Column Sets Not Paginated
**Location**: All selection RadioGroups and FormGroups (`TSTrainCard.jsx:1037-1115`)

**Issue**: Fixed height scrolling container (150px) with no search, filter, or pagination for large column sets.

**User Impact**: Difficult to navigate and select from 50+ columns. No way to quickly find specific columns.

---

#### I-13: Race Condition Potential in Auto-Selection
**Location**: `handleTargetChange()` auto-selection (`TSTrainCard.jsx:350`)

**Issue**: Uses current `inputFeatures` state value rather than functional setState form, creating risk of stale state in rapid interactions.

**Technical Detail**:
```javascript
const newFeatures = [...new Set([...inputFeatures, ...remainingColumns])];
```
Should be:
```javascript
setInputFeatures(prev => [...new Set([...prev, ...remainingColumns])]);
```

**User Impact**: Edge case where rapid clicking might produce incorrect feature selections.

---

#### I-14: LSTM useEffect Cleanup Timing
**Location**: `useEffect` hook (`TSTrainCard.jsx:636-638`)

**Issue**: Effect cleanup is scheduled asynchronously; rapid algorithm switching might briefly submit stale `lstmSelectedFeatures`.

**User Impact**: Low probability edge case but possible configuration corruption.

---

#### I-15: No Loading State During Validation
**Location**: `validateSelections()` function

**Issue**: Validation is synchronous and fast, but no loading indicator if validation logic becomes more complex.

**User Impact**: Currently not an issue, but potential future problem if validation includes async checks.

---

#### I-16: Missing Error Details in Display
**Location**: Error handling throughout component

**Issue**: Backend error responses may contain detailed error messages (`error_details`) but frontend only shows generic messages in several places.

**User Impact**: Users lack information needed to diagnose and fix problems.

---

#### I-17: No Accessibility Attributes
**Location**: All form controls

**Issue**: RadioGroups and Checkboxes lack ARIA labels, descriptions, and error announcements for screen readers.

**User Impact**: Inaccessible to users with visual disabilities.

---

### Technical Debt Observations

#### TD-1: Magic Algorithm String
**Location**: Multiple locations checking `algorithm !== "lstm"`

**Issue**: String literal `"lstm"` used throughout instead of constants or enum.

**Impact**: Refactoring algorithm names requires changes in multiple locations; typo risk.

---

#### TD-2: Validation Logic Separate from UI
**Location**: Validation rules in separate function, disabled logic in JSX

**Issue**: Validation rules (lines 367-396) duplicates some logic already enforced by disabled checkboxes (line 1083).

**Impact**: Two sources of truth for what selections are valid; maintenance burden.

---

#### TD-3: Comment Relies on useEffect Timing
**Location**: Line 342 comment: `"// Note: lstmSelectedFeatures cleared by useEffect"`

**Issue**: Code relies on implicit behavior (effect scheduling) rather than explicit cleanup in handler.

**Impact**: Fragile dependency on React's effect timing; unclear execution order.

---

#### TD-4: Inconsistent Error Message Styling
**Location**: Various setTrainStatus calls

**Issue**: Status messages use emoji prefixes inconsistently (⚠️, ❌, ✅, ⏳) with no centralized message formatting.

**Impact**: Inconsistent UX; difficult to style or internationalize.

---

#### TD-5: No TypeScript or PropTypes
**Location**: Entire component

**Issue**: JavaScript without type checking; state types and handler signatures not enforced.

**Impact**: Runtime errors from type mismatches; poor IDE autocomplete.

---

