# LSTM Training Implementation Plan

**Research Reference:** [2025-11-06_lstm-training-implementation.md](../research/2025-11-06_lstm-training-implementation.md)
**Status:** 🟢 Phase 1 Complete | 🟢 Phase 2A Complete | 🟢 Phase 2B Complete | 🟢 Phase 3A-B Complete | ❌ Phase 3C Out of Scope | 🟡 Phase 4-5 Pending
**Total Estimated Time:** 10-14 hours (Reduced: Phase 3C out of scope)
**Last Updated:** 2025-11-13
**Prerequisites:** Python 3.8+, TensorFlow 2.x, MLflow, DVC, React, Material-UI, psutil

---

## Implementation Status

### Phase 1: Core LSTM Implementation ✅ COMPLETED (2025-11-07)
- ✅ All 6 LSTM functions implemented (~820 lines)
- ✅ Test suite created with 11 tests - ALL PASSED
- ✅ Manual verification completed successfully
- ✅ MLflow integration verified
- ✅ Checkpoint cleanup verified
- ✅ Import from services.py working
- **Files Modified:**
  - [train.py](../../DREAM-ML-backend/GEML/apiTimeSeries/train.py) - Lines 1531-2351 added
  - [test_lstm_phase1.py](../../DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_lstm_phase1.py) - Created

### Phase 2A: Grid Search Hyperparameter Optimization ✅ COMPLETED (2025-11-10)
- ✅ Backend Grid Search implementation complete (train.py lines 2160-2327)
- ✅ Memory profiling with psutil integrated
- ✅ Progress tracking every 10 iterations
- ✅ Configurable grid warning threshold (default: 50 combinations)
- ✅ Conservative default grid parameters (8 combinations)
- ✅ Best model selection based on val_loss
- ✅ Memory cleanup between iterations (verified < 500MB increase)
- ✅ MLflow logging: best_iteration, best_val_loss, grid_iterations_total, memory metrics
- ✅ Manual verification completed (16 combinations, 479.83 MB memory increase)
- ✅ Logging enhancement applied (INFO logs now visible in console)
- **Files Modified:**
  - [train.py](../../DREAM-ML-backend/GEML/apiTimeSeries/train.py) - Lines 2160-2327 added
  - [settings.py](../../DREAM-ML-backend/GEML/GEML/settings.py) - LOGGING configuration added
  - [views.py](../../DREAM-ML-backend/GEML/apiTimeSeries/views.py) - Logger level restriction removed
  - [services.py](../../DREAM-ML-backend/GEML/apiTimeSeries/services.py) - Logger level restriction removed
- **Verification Results:**
  - Grid iterations: 16/16 completed
  - Memory increase: 479.83 MB (✅ < 500 MB threshold)
  - Best iteration: 11
  - Best val_loss: 0.5336
  - All MLflow metrics and parameters logged correctly

### Phase 2B: Random Search Hyperparameter Optimization ✅ COMPLETED (2025-11-11)
- ✅ Backend Random Search implementation complete (train.py lines 2414-2578)
- ✅ `generate_random_lstm_params()` function implemented with conservative defaults
- ✅ Memory profiling with conditional psutil tracking (follows Phase 2A pattern)
- ✅ Progress tracking every 10 iterations
- ✅ Warning threshold for n_random_iterations > 200
- ✅ Best model selection based on val_loss
- ✅ Memory cleanup between iterations (verified < 500MB increase for 20 iterations)
- ✅ MLflow logging: best_iteration, best_val_loss, random_iterations_total, memory metrics
- ✅ Log-uniform distribution for learning_rate (better sampling across magnitudes)
- ✅ Manual verification completed (10 iterations, all metrics logged correctly)
- ✅ Test suite created with 11 tests - ALL PASSED
- **Files Modified:**
  - [train.py](../../DREAM-ML-backend/GEML/apiTimeSeries/train.py) - Lines 1942-2025 (generate_random_lstm_params), Lines 2414-2578 (random search block)
  - [test_lstm_phase2b.py](../../DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_lstm_phase2b.py) - Created
- **Verification Results:**
  - Random iterations: 10/10 completed
  - Memory increase: < 500 MB (verified with 20 iterations)
  - Best model selection working correctly
  - All MLflow metrics and parameters logged correctly
  - Warning threshold functioning (> 200 iterations)
  - Progress logs every 10 iterations
- **Known Issues (To Fix in Phase 3):**
  - ⚠️ UI random search parameter format mismatch (min-max format vs. options list)
  - ⚠️ `pipeline_config.json` missing complete metrics (only val_rmse, test_rmse logged)
  - ⚠️ `pipeline_config.json` missing hyperparameter search strategy and configuration (CRITICAL)

### Phase 3: Frontend Enhancements + Resume 🟡 READY (Requires Phase 2B fixes)
### Phase 4: External Features Support 🔴 BLOCKED (Requires Phases 2A, 2B, 3)
### Phase 5: Multi-Step Seq2Seq 🔴 DEFERRED

---

## Table of Contents

- [Implementation Conventions](#implementation-conventions)
- [Executive Summary](#executive-summary)
- [Phase Dependencies](#phase-dependencies)
- [Phase 1: Core LSTM Implementation](#phase-1-core-lstm-implementation)
- [Phase 2: Hyperparameter Search](#phase-2-hyperparameter-search)
- [Phase 3: Frontend Enhancements + Resume](#phase-3-frontend-enhancements--resume)
- [Phase 4: External Features Support](#phase-4-external-features-support)
- [Phase 5: Multi-Step Seq2Seq](#phase-5-multi-step-seq2seq)
- [Code Patterns Reference](#code-patterns-reference)
- [Risk Mitigation](#risk-mitigation)

---

## Implementation Conventions

### Code Comment Patterns

Throughout this plan, code snippets use these markers to guide implementation:

- **`# ACTUAL CODE:`** - Implement this code as written. This is production-ready code that should be copied/adapted directly.
- **`# PSEUDOCODE:`** - Implementation hint or high-level logic. Adapt this to the specific context and implement the described functionality.
- **`# CRITICAL:`** - Must implement exactly as specified (e.g., memory management, cleanup operations). Deviation may cause bugs.

**Example:**
```python
# ACTUAL CODE: Extract parameters with defaults
lstm_params = data.get("lstm_params", {...})

# PSEUDOCODE: Validate minimum samples in each set
if len(X_train) < 10 or len(X_val) < 5:
    logger.warning("Conjuntos muy pequeños detectados")

# CRITICAL: Memory cleanup after each iteration
del model
tf.keras.backend.clear_session()
gc.collect()
```

### Existing Helper Functions

These functions already exist in the codebase and should be used (not reimplemented):

| Function | Location | Purpose |
|----------|----------|---------|
| `convert_numpy_to_python()` | [train.py:289-323](../../DREAM-ML-backend/GEML/apiTimeSeries/train.py#L289-L323) | Converts numpy types to Python natives for JSON serialization |
| `load_and_validate_ts_data()` | [train.py:142-203](../../DREAM-ML-backend/GEML/apiTimeSeries/train.py#L142-L203) | Loads CSV and validates datetime index, columns |
| `log_energy_metrics()` | [train.py:281-287](../../DREAM-ML-backend/GEML/apiTimeSeries/train.py#L281-L287) | Logs CodeCarbon energy consumption to MLflow |
| `save_pipeline_config()` | [train.py:325-353](../../DREAM-ML-backend/GEML/apiTimeSeries/train.py#L325-L353) | Updates pipeline_config.json with step metadata |
| `ts_train_val_test_split()` | [train.py:112-137](../../DREAM-ML-backend/GEML/apiTimeSeries/train.py#L112-L137) | Temporal split for time series (reference pattern) |

### Integration Points

**WebSocket Notifications:**
- WebSocket integration is handled by `services.py` (TrainModelService class)
- Training functions don't directly send WebSocket messages
- Progress updates are logged via `logger.info()` and picked up by service layer

**MLflow Client:**
- MLflow is already initialized and available in global scope
- Use `mlflow.start_run()`, `mlflow.log_params()`, etc. directly
- No need to configure tracking URI (already set)

**CodeCarbon Tracker:**
- Import: `from codecarbon import EmissionsTracker`
- Pattern: Initialize → start() → [training] → stop() → log_energy_metrics()

### Function Evolution Across Phases

**Important:** Long functions (e.g., `train_lstm_model`) will be built incrementally:

- **Phase 1:** Implement manual parameters path only (lines 819-878)
- **Phase 2:** Replace placeholder with grid/random search (lines 1417-1647)
- **Phase 3:** No changes to train_lstm_model (frontend/services only)
- **Phase 4:** Modify sequence creation for feature selection

Don't implement the entire 400-line function at once. Build the scaffold in Phase 1, then extend in Phase 2.

---

## Executive Summary

### Objective

Implement complete LSTM model training infrastructure for time series forecasting, following established ARIMA/XGBoost patterns. This includes backend training functions, frontend UI enhancements, hyperparameter optimization, and training resumption capability.

### Current State

- ❌ `train_lstm_model` function imported but not implemented (ImportError at runtime)
- ✅ Frontend UI state variables already exist in TSTrainCard.jsx
- ✅ Service layer routing ready at services.py:1036
- ✅ ARIMA and XGBoost provide complete reference implementations

### Success Criteria

- ✅ LSTM training completes without ImportError
- ✅ Manual, Grid, and Random search strategies work
- ✅ MLflow logs model with signature, metrics, and artifacts
- ✅ DVC versions model files
- ✅ Resume training after crash/cancel
- ✅ Univariate and multivariate sequences supported
- ✅ Memory usage < 500MB increase during random search

### Phase Breakdown

| Phase | Description | Time Estimate | Dependencies |
|-------|-------------|---------------|--------------|
| 1 | Core LSTM (manual params, single-step) | 4-6 hours | None |
| 2A | Grid Search Hyperparameter Optimization | 3-4.5 hours | Phase 1 |
| 2B | Random Search Hyperparameter Optimization | 2-3 hours | Phase 2A |
| 3A | Critical Fixes (pipeline_config) | 45-60 min | Phase 2A, 2B |
| 3B | UI Enhancements (warnings, tooltips) | 1.5-2 hours | Phase 3A |
| 3C | Resume/Cancel (OUT OF SCOPE) | N/A | Deferred (reproducibility conflict) |
| 4 | External Features Support | 2-3 hours | Phases 2A, 2B, 3A, 3B |
| 5 | Multi-Step Seq2Seq (outline) | N/A (deferred) | Phases 1-4 |

**Total Estimated Time:** 10-14 hours (Reduced: Phase 3C out of scope)

---

## Phase Dependencies

```
Phase 1: Core LSTM (Backend)
    ↓
    ├─→ Phase 2A: Grid Search (Backend) ───────────┐
    │        ↓                                      │
    │   Phase 2B: Random Search (Backend) ─────────┤
    │                                               │
    └─→ Phase 3: Frontend + Resume ────────────────┤
                                                    ↓
                                       Phase 4: External Features
                                                    ↓
                                       Phase 5: Seq2Seq (Outline)
```

**Parallelization Opportunities:**
- Phase 2A and Phase 3 can be developed in parallel if separate developers (backend vs frontend)
- Phase 2B depends on Phase 2A completion (shares memory management patterns)
- Phase 3 frontend work can begin once Phase 1 is complete
- Phase 5 is conceptual outline only, no implementation in this plan

---

## Phase 1: Core LSTM Implementation

### Overview

**Objective:** Implement single-step LSTM forecasting with user-provided hyperparameters (manual mode).

**Scope:**
- Sequence creation (3D tensor conversion)
- Train/val/test splitting for LSTM
- Model building (Keras Sequential)
- Callbacks configuration (EarlyStopping, ModelCheckpoint)
- Evaluation with metrics and plots
- MLflow model registration with signature
- Aggressive checkpoint cleanup

**Time Estimate:** 4-6 hours

### Prerequisites

- ARIMA and XGBoost training functions working
- TensorFlow 2.x installed
- MLflow tracking server running
- Research document reviewed

### Prerequisites Verification

Before starting Phase 1 implementation, run these verification commands:

```bash
# 1. Verify TensorFlow installation and version
python -c "import tensorflow as tf; print(f'✓ TensorFlow {tf.__version__}')"
# Expected: TensorFlow 2.x.x (e.g., 2.13.0, 2.15.0)

# 2. Verify Keras availability (part of TensorFlow 2.x)
python -c "from tensorflow import keras; print('✓ Keras available')"
# Expected: ✓ Keras available

# 3. Verify existing training functions are importable
python -c "from apiTimeSeries.train import train_arima_model, train_xgboost_model; print('✓ Existing training functions OK')"
# Expected: ✓ Existing training functions OK

# 4. Verify helper functions exist
python -c "from apiTimeSeries.train import load_and_validate_ts_data, convert_numpy_to_python, log_energy_metrics, save_pipeline_config; print('✓ Helper functions OK')"
# Expected: ✓ Helper functions OK

# 5. Check MLflow server is accessible
curl -s http://localhost:5000 > /dev/null && echo "✓ MLflow server running" || echo "✗ MLflow server not accessible"
# Expected: ✓ MLflow server running
# If not running: Start with `mlflow ui --port 5000`

# 6. Verify CodeCarbon is installed
python -c "from codecarbon import EmissionsTracker; print('✓ CodeCarbon available')"
# Expected: ✓ CodeCarbon available

# 7. Check current working directory
pwd
# Expected: Should be in DREAM-ML-backend/GEML or repository root
```

**If any verification fails:**
- TensorFlow missing: `pip install tensorflow>=2.0`
- Imports fail: Check you're in correct directory (`cd DREAM-ML-backend/GEML`)
- MLflow not running: `mlflow ui --port 5000` in separate terminal
- CodeCarbon missing: `pip install codecarbon`

### Files Modified

1. **`DREAM-ML-backend/GEML/apiTimeSeries/train.py`**
   - Add ~500 lines after line 1529
   - Import statements already present (lines 47-51)

2. **`DREAM-ML-backend/GEML/apiTimeSeries/services.py`**
   - Verify import at line 48 (should already exist)
   - No changes needed if import present

### Implementation Details

#### A. Sequence Creation Function

**Location:** `train.py`, after line 1529

**Function Signature:**
```python
def create_sequences_for_lstm(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
```

**Implementation:**
```python
def create_sequences_for_lstm(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte DataFrame de series temporales en secuencias 3D para LSTM.

    Args:
        df: Datos con índice datetime ordenado temporalmente
        feature_cols: Lista de columnas de características de entrada
        target_col: Nombre de la columna objetivo
        sequence_length: Longitud de la ventana temporal (número de timesteps)
        forecast_horizon: Pasos adelante a predecir (default: 1 para single-step)

    Returns:
        X: Secuencias de entrada, shape (n_sequences, sequence_length, n_features)
        y: Valores objetivo, shape (n_sequences,)

    Example:
        >>> df = pd.DataFrame({'date': ..., 'temp': ..., 'sales': ...})
        >>> X, y = create_sequences_for_lstm(df, ['temp'], 'sales', 10)
        >>> X.shape
        (190, 10, 1)
        >>> y.shape
        (190,)
    """
    # PSEUDOCODE: Validate minimum data requirements to ensure enough sequences
    min_sequences = 50
    max_sequence_length = len(df) - forecast_horizon - min_sequences

    if sequence_length > max_sequence_length:
        # AUTO-FALLBACK: Adjust sequence_length with warning
        logger.warning(
            f"sequence_length {sequence_length} excede el máximo válido {max_sequence_length}. "
            f"Usando {max_sequence_length} en su lugar para garantizar al menos {min_sequences} secuencias."
        )
        sequence_length = max_sequence_length

        # PSEUDOCODE: Send WebSocket notification to frontend (via services.py)
        # This will be handled by services.py when calling this function

    if sequence_length < 1:
        raise ValueError(
            f"Dataset insuficiente para crear secuencias. "
            f"Se requieren al menos {min_sequences + forecast_horizon + 1} muestras. "
            f"Dataset actual: {len(df)} muestras."
        )

    # ACTUAL CODE: Extract feature and target arrays
    features = df[feature_cols].values
    target = df[target_col].values

    # PSEUDOCODE: Create sequences using sliding window approach
    # For each position i, extract window [i:i+sequence_length] as input
    # and target[i+sequence_length+forecast_horizon-1] as output
    X_sequences = []
    y_sequences = []

    # ACTUAL CODE: Sliding window iteration
    for i in range(len(df) - sequence_length - forecast_horizon + 1):
        # Input sequence: sequence_length timesteps
        X_seq = features[i:i + sequence_length]

        # Target value: forecast_horizon steps ahead
        y_seq = target[i + sequence_length + forecast_horizon - 1]

        X_sequences.append(X_seq)
        y_sequences.append(y_seq)

    # PSEUDOCODE: Convert to numpy arrays with proper shapes
    X = np.array(X_sequences)  # Shape: (n_sequences, sequence_length, n_features)
    y = np.array(y_sequences)  # Shape: (n_sequences,)

    logger.info(
        f"Secuencias creadas exitosamente - X: {X.shape}, y: {y.shape} "
        f"(sequence_length={sequence_length}, forecast_horizon={forecast_horizon})"
    )

    return X, y
```

**Key Implementation Notes:**
- Auto-fallback for sequence_length validation (logs warning)
- Maintains temporal order (no shuffling)
- Supports single-step forecasting (forecast_horizon=1)
- Returns 3D array for X, 1D array for y

---

#### B. LSTM Train/Val/Test Split Function

**Location:** `train.py`, after `create_sequences_for_lstm`

**Function Signature:**
```python
def lstm_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    split_ratios: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

**Implementation:**
```python
def lstm_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    split_ratios: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    División temporal para secuencias LSTM (mantiene forma 3D).

    Respeta el orden temporal - NO realiza shuffling aleatorio.

    Args:
        X: Secuencias de entrada, shape (n_sequences, sequence_length, n_features)
        y: Valores objetivo, shape (n_sequences,)
        split_ratios: Diccionario con proporciones {"train": 0.7, "val": 0.15, "test": 0.15}

    Returns:
        Tupla de 6 elementos:
        - X_train: Secuencias de entrenamiento (3D)
        - y_train: Objetivos de entrenamiento (1D)
        - X_val: Secuencias de validación (3D)
        - y_val: Objetivos de validación (1D)
        - X_test: Secuencias de prueba (3D)
        - y_test: Objetivos de prueba (1D)

    Raises:
        ValueError: Si los ratios no suman aproximadamente 1.0
    """
    # PSEUDOCODE: Validate split ratios sum to 1.0
    total_ratio = sum(split_ratios.values())
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(
            f"La suma de split_ratios debe ser 1.0, actual: {total_ratio}. "
            f"Proporciones recibidas: {split_ratios}"
        )

    # ACTUAL CODE: Calculate split indices based on ratios
    n = len(X)
    train_size = int(n * split_ratios["train"])
    val_size = int(n * split_ratios["val"])

    # PSEUDOCODE: Split maintaining temporal order (critical for time series)
    # Train: earliest data
    # Val: middle data
    # Test: most recent data
    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    logger.info(
        f"División temporal completada - "
        f"Train: {len(X_train)} ({split_ratios['train']*100:.1f}%), "
        f"Val: {len(X_val)} ({split_ratios['val']*100:.1f}%), "
        f"Test: {len(X_test)} ({split_ratios['test']*100:.1f}%)"
    )

    # PSEUDOCODE: Validate minimum samples in each set
    if len(X_train) < 10 or len(X_val) < 5 or len(X_test) < 5:
        logger.warning(
            f"Conjuntos muy pequeños detectados. "
            f"Se recomienda tener al menos 50 secuencias totales."
        )

    return X_train, y_train, X_val, y_val, X_test, y_test
```

**Key Implementation Notes:**
- Preserves 3D shape throughout split
- No random shuffling (maintains temporal order)
- Validates minimum samples per set
- Follows same pattern as `ts_train_val_test_split` (line 112)

---

#### C. Model Building Function

**Location:** `train.py`, after `lstm_train_val_test_split`

**Function Signature:**
```python
def build_lstm_model(params: Dict, input_shape: Tuple[int, int]) -> keras.Model:
```

**Implementation:**
```python
def build_lstm_model(params: Dict, input_shape: Tuple[int, int]) -> keras.Model:
    """
    Construye modelo Keras LSTM desde hiperparámetros.

    Soporta arquitecturas de una o múltiples capas LSTM.

    Args:
        params: Diccionario con hiperparámetros:
            - lstm_units: List[int] (e.g., [64] para 1 capa, [64, 32] para 2 capas)
            - dropout_rate: float (0.0 - 0.5)
            - recurrent_dropout_rate: float (0.0 - 0.5)
            - learning_rate: float (típicamente 0.0001 - 0.01)
        input_shape: Tupla (sequence_length, n_features)

    Returns:
        Modelo Sequential de Keras compilado con Adam optimizer y MSE loss

    Example:
        >>> params = {"lstm_units": [64, 32], "dropout_rate": 0.2,
        ...           "recurrent_dropout_rate": 0.2, "learning_rate": 0.001}
        >>> model = build_lstm_model(params, input_shape=(10, 2))
        >>> model.summary()
    """
    # PSEUDOCODE: Initialize Sequential model
    model = Sequential(name="LSTM_TimeSeriesModel")

    # ACTUAL CODE: Extract parameters with defaults
    lstm_units = params.get("lstm_units", [64])
    dropout_rate = params.get("dropout_rate", 0.2)
    recurrent_dropout_rate = params.get("recurrent_dropout_rate", 0.2)
    learning_rate = params.get("learning_rate", 0.001)

    logger.info(
        f"Construyendo modelo LSTM - Arquitectura: {lstm_units}, "
        f"Dropout: {dropout_rate}, Recurrent Dropout: {recurrent_dropout_rate}, "
        f"Learning Rate: {learning_rate}"
    )

    # PSEUDOCODE: Add LSTM layers (handle single vs multi-layer architectures)
    if len(lstm_units) == 1:
        # Single LSTM layer - no return_sequences needed
        model.add(LSTM(
            units=lstm_units[0],
            input_shape=input_shape,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
            name="LSTM_Layer"
        ))
    else:
        # Multiple LSTM layers - return_sequences=True for all except last
        for i, units in enumerate(lstm_units[:-1]):
            model.add(LSTM(
                units=units,
                return_sequences=True,  # Pass sequences to next LSTM layer
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout_rate,
                input_shape=input_shape if i == 0 else None,  # Only first layer needs input_shape
                name=f"LSTM_Layer_{i+1}"
            ))

        # Last LSTM layer - no return_sequences (output goes to Dense)
        model.add(LSTM(
            units=lstm_units[-1],
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
            name=f"LSTM_Layer_{len(lstm_units)}"
        ))

    # ACTUAL CODE: Output layer for single-step forecasting
    model.add(Dense(1, name="Output_Layer"))

    # PSEUDOCODE: Compile model with Adam optimizer and MSE loss
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae", "mse"]  # Track both MAE and MSE during training
    )

    total_params = model.count_params()
    logger.info(
        f"Modelo LSTM compilado exitosamente - "
        f"Total de parámetros: {total_params:,}"
    )

    return model
```

**Key Implementation Notes:**
- Supports single-layer and multi-layer architectures
- Uses `return_sequences=True` for intermediate LSTM layers
- Adam optimizer with configurable learning rate
- MSE loss appropriate for regression

---

#### D. Callbacks Configuration Function

**Location:** `train.py`, after `build_lstm_model`

**Function Signature:**
```python
def create_lstm_callbacks(
    experiment_dir: str,
    early_stopping_patience: int,
    checkpoint_filename: str = "best_lstm_checkpoint.h5"
) -> Tuple[List[keras.callbacks.Callback], str]:
```

**Implementation:**
```python
def create_lstm_callbacks(
    experiment_dir: str,
    early_stopping_patience: int,
    checkpoint_filename: str = "best_lstm_checkpoint.h5"
) -> Tuple[List[keras.callbacks.Callback], str]:
    """
    Crea callbacks de Keras para entrenamiento LSTM.

    Args:
        experiment_dir: Directorio del experimento para guardar checkpoints temporales
        early_stopping_patience: Número de épocas a esperar sin mejora antes de detener
        checkpoint_filename: Nombre del archivo de checkpoint (default: "best_lstm_checkpoint.h5")

    Returns:
        Tupla de:
        - Lista de callbacks de Keras configurados
        - Ruta completa al archivo de checkpoint

    Callbacks incluidos:
        - EarlyStopping: Detiene entrenamiento si val_loss no mejora
        - ModelCheckpoint: Guarda mejor modelo basado en val_loss
        - ReduceLROnPlateau: Reduce learning rate si val_loss se estanca
    """
    # PSEUDOCODE: Create temporary checkpoint directory
    # Checkpoints will be deleted after training (aggressive cleanup)
    checkpoint_dir = os.path.join(experiment_dir, "temp_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    logger.info(f"Directorio de checkpoints temporales: {checkpoint_dir}")

    # ACTUAL CODE: EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,  # Load best weights when stopping
        verbose=1,
        mode="min"
    )

    # ACTUAL CODE: ModelCheckpoint callback (saves best model only)
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,  # Only save when val_loss improves
        verbose=0,  # Quiet mode (logged via logger instead)
        mode="min"
    )

    # ACTUAL CODE: ReduceLROnPlateau callback
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,  # Multiply learning rate by 0.5
        patience=early_stopping_patience // 2,  # Half of early stopping patience
        min_lr=1e-7,  # Minimum learning rate threshold
        verbose=1,
        mode="min"
    )

    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    logger.info(
        f"Callbacks configurados - "
        f"EarlyStopping patience: {early_stopping_patience}, "
        f"ReduceLR patience: {early_stopping_patience // 2}"
    )

    return callbacks, checkpoint_path
```

**Key Implementation Notes:**
- Checkpoint saved to `temp_checkpoints/` subdirectory
- All checkpoints deleted after training (aggressive cleanup strategy)
- ReduceLROnPlateau helps escape local minima
- EarlyStopping with `restore_best_weights=True` ensures best model used

---

#### E. Evaluation Function

**Location:** `train.py`, after `create_lstm_callbacks`

**Function Signature:**
```python
def evaluate_lstm_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    prefix: str,
    experiment_dir: str
) -> Tuple[Dict[str, float], List[str]]:
```

**Implementation:**
```python
def evaluate_lstm_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    prefix: str,
    experiment_dir: str
) -> Tuple[Dict[str, float], List[str]]:
    """
    Evalúa modelo LSTM y genera gráficos de diagnóstico.

    Args:
        model: Modelo Keras entrenado
        X_test: Secuencias de prueba, shape (n_samples, sequence_length, n_features)
        y_test: Objetivos de prueba, shape (n_samples,)
        prefix: Prefijo para métricas ("val" o "test")
        experiment_dir: Directorio para guardar gráficos

    Returns:
        Tupla de:
        - metrics: Diccionario {f"{prefix}_rmse": float, f"{prefix}_mae": float, ...}
        - artifacts: Lista de rutas a archivos de gráficos generados
    """
    # ACTUAL CODE: Generate predictions
    logger.info(f"Generando predicciones para conjunto {prefix}...")
    y_pred = model.predict(X_test, verbose=0).flatten()

    # PSEUDOCODE: Calculate regression metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # ACTUAL CODE: Calculate MAPE (avoid division by zero)
    mask = y_test != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    else:
        mape = None
        logger.warning(f"No se puede calcular MAPE para {prefix}: todos los valores objetivo son cero")

    metrics = {
        f"{prefix}_rmse": float(rmse),
        f"{prefix}_mae": float(mae),
        f"{prefix}_mape": float(mape) if mape is not None else None
    }

    logger.info(
        f"Métricas {prefix} - RMSE: {rmse:.4f}, MAE: {mae:.4f}" +
        (f", MAPE: {mape:.2f}%" if mape is not None else "")
    )

    # PSEUDOCODE: Generate diagnostic plots
    artifacts = []

    # Plot 1: Predictions vs Actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Real", alpha=0.7, linewidth=2)
    plt.plot(y_pred, label="Predicción LSTM", alpha=0.7, linewidth=2)
    plt.title(f"LSTM - Predicciones vs Valores Reales ({prefix.upper()})")
    plt.xlabel("Índice de Muestra")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True, alpha=0.3)
    forecast_path = os.path.join(experiment_dir, f"lstm_{prefix}_forecast.png")
    plt.savefig(forecast_path, dpi=150, bbox_inches="tight")
    plt.close()
    artifacts.append(forecast_path)
    logger.info(f"Gráfico de pronóstico guardado: {forecast_path}")

    # Plot 2: Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 6))
    plt.plot(residuals, alpha=0.7, linewidth=1)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
    plt.title(f"LSTM - Residuos ({prefix.upper()})")
    plt.xlabel("Índice de Muestra")
    plt.ylabel("Residuo (Real - Predicción)")
    plt.grid(True, alpha=0.3)
    residuals_path = os.path.join(experiment_dir, f"lstm_{prefix}_residuals.png")
    plt.savefig(residuals_path, dpi=150, bbox_inches="tight")
    plt.close()
    artifacts.append(residuals_path)
    logger.info(f"Gráfico de residuos guardado: {residuals_path}")

    # Plot 3: Residuals distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, alpha=0.5)
    plt.title(f"LSTM - Distribución de Residuos ({prefix.upper()})")
    plt.xlabel("Residuo")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3, axis='y')
    residuals_dist_path = os.path.join(experiment_dir, f"lstm_{prefix}_residuals_distribution.png")
    plt.savefig(residuals_dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    artifacts.append(residuals_dist_path)
    logger.info(f"Gráfico de distribución de residuos guardado: {residuals_dist_path}")

    return metrics, artifacts
```

**Key Implementation Notes:**
- Generates 3 diagnostic plots (forecast, residuals, distribution)
- Handles division by zero for MAPE
- Follows same pattern as `evaluate_xgboost_model` (line 483)
- All plots saved to experiment_dir

---

#### F. Main Training Function

**Location:** `train.py`, after `evaluate_lstm_model`

**Function Signature:**
```python
def train_lstm_model(dataset_path: str, data: Dict, experiment_dir: str) -> Dict:
```

**Implementation (Manual Params Only for Phase 1):**
```python
def train_lstm_model(dataset_path: str, data: Dict, experiment_dir: str) -> Dict:
    """
    Entrena y registra un modelo LSTM para pronóstico de series temporales.

    Sigue el mismo contrato que train_arima_model y train_xgboost_model.
    Soporta búsqueda manual de hiperparámetros en Fase 1.

    Args:
        dataset_path: Ruta al archivo CSV con los datos
        data: Diccionario con configuración del entrenamiento, incluyendo:
            - date_col_name: str
            - target_variable: str
            - input_features: List[str]
            - model_name: str
            - forecast_horizon: int (default: 1)
            - split_ratios: Dict (default: {"train": 0.7, "val": 0.15, "test": 0.15})
            - hyperparameter_search_strategy: "none" para manual (Fase 1)
            - sequence_length: int (default: 10)
            - early_stopping_patience: int (default: 20)
            - lstm_params: Dict con hiperparámetros manuales
        experiment_dir: Directorio para guardar artefactos

    Returns:
        Diccionario con resultados del entrenamiento:
        {
            "status": "success" | "error",
            "val_metrics": {"rmse": float, "mae": float, "mape": float},
            "test_metrics": {"rmse": float, "mae": float, "mape": float},
            "model_path": str,
            "run_id": str,
            "features_used": List[str]
        }

    Raises:
        ValueError: Errores de validación de parámetros
        RuntimeError: Errores de ejecución durante entrenamiento
    """
    try:
        # ======================
        # 1. EXTRACCIÓN DE PARÁMETROS
        # ======================

        # ACTUAL CODE: Extract required parameters
        date_col_name = data.get("date_col_name")
        target_variable = data.get("target_variable")
        input_features = data.get("input_features", [target_variable])  # Default: univariate
        model_name = data.get("model_name")

        # ACTUAL CODE: Extract optional parameters with defaults
        forecast_horizon = data.get("forecast_horizon", 1)
        split_ratios = data.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
        sequence_length = data.get("sequence_length", 10)
        early_stopping_patience = data.get("early_stopping_patience", 20)
        hyperparameter_search_strategy = data.get("hyperparameter_search_strategy", "none")

        # PSEUDOCODE: Validate required parameters
        if not all([date_col_name, target_variable, model_name]):
            raise ValueError(
                "Parámetros requeridos faltantes. Se requieren: "
                "date_col_name, target_variable, model_name"
            )

        # ACTUAL CODE: CPU warning
        logger.warning(
            "⚠️ Entrenamiento LSTM usa CPU solamente (sin soporte GPU en esta versión). "
            "Tiempo de entrenamiento esperado: 30-60 minutos para 100 épocas. "
            "Considere reducir 'epochs' si el tiempo es excesivo."
        )

        # ======================
        # 2. INICIALIZACIÓN DE MLFLOW
        # ======================

        # PSEUDOCODE: Verify no active MLflow run
        if mlflow.active_run():
            mlflow.end_run()
            logger.warning("Run activa de MLflow detectada y finalizada")

        # ACTUAL CODE: Start MLflow run
        run_id = str(uuid.uuid4())[:8]
        mlflow.start_run(run_name=f"lstm_manual_{run_id}")
        mlflow_run_id = mlflow.active_run().info.run_id

        logger.info(f"Iniciando entrenamiento LSTM en run: {mlflow_run_id}")

        # ACTUAL CODE: Log parameters to MLflow
        mlflow.log_params({
            "model_type": "LSTM",
            "date_col_name": date_col_name,
            "target_variable": target_variable,
            "input_features": input_features,
            "forecast_horizon": forecast_horizon,
            "sequence_length": sequence_length,
            "early_stopping_patience": early_stopping_patience,
            "split_ratios": split_ratios,
            "hyperparameter_search_strategy": hyperparameter_search_strategy,
            "cpu_only": True
        })

        # ======================
        # 3. CARGA Y VALIDACIÓN DE DATOS
        # ======================

        # ACTUAL CODE: Load and validate data
        logger.info("Cargando y validando dataset...")
        df = load_and_validate_ts_data(dataset_path, date_col_name, target_variable)

        # PSEUDOCODE: Validate input features exist
        for feature in input_features:
            if feature not in df.columns:
                raise ValueError(f"Característica de entrada no encontrada: {feature}")

        logger.info(f"Dataset cargado: {len(df)} muestras, características: {input_features}")

        # ======================
        # 4. CREACIÓN DE SECUENCIAS
        # ======================

        # ACTUAL CODE: Create LSTM sequences
        logger.info(f"Creando secuencias LSTM (sequence_length={sequence_length})...")
        X, y = create_sequences_for_lstm(
            df=df,
            feature_cols=input_features,
            target_col=target_variable,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )

        # PSEUDOCODE: Log sequence creation results
        mlflow.log_params({
            "n_sequences": len(X),
            "sequence_shape": str(X.shape),
            "n_features": X.shape[2]
        })

        # ======================
        # 5. DIVISIÓN TRAIN/VAL/TEST
        # ======================

        # ACTUAL CODE: Split data
        logger.info("Dividiendo dataset en train/val/test...")
        X_train, y_train, X_val, y_val, X_test, y_test = lstm_train_val_test_split(
            X=X,
            y=y,
            split_ratios=split_ratios
        )

        # ======================
        # 6. INICIALIZACIÓN DE TRACKER DE ENERGÍA
        # ======================

        # ACTUAL CODE: Start energy tracking
        tracker = EmissionsTracker(
            project_name=f"LSTM_{model_name}",
            output_dir=experiment_dir,
            log_level="warning"
        )
        tracker.start()

        # ======================
        # 7. ENTRENAMIENTO CON PARÁMETROS MANUALES
        # ======================

        if hyperparameter_search_strategy == "none":
            # ACTUAL CODE: Extract manual parameters
            lstm_params = data.get("lstm_params", {
                "lstm_units": [64],
                "dropout_rate": 0.2,
                "recurrent_dropout_rate": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100
            })

            logger.info(f"Entrenando con parámetros manuales: {lstm_params}")

            # PSEUDOCODE: Build model
            input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, n_features)
            model = build_lstm_model(lstm_params, input_shape)

            # PSEUDOCODE: Create callbacks
            callbacks, checkpoint_path = create_lstm_callbacks(
                experiment_dir=experiment_dir,
                early_stopping_patience=early_stopping_patience
            )

            # ACTUAL CODE: Train model
            logger.info("Iniciando entrenamiento del modelo...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=lstm_params.get("epochs", 100),
                batch_size=lstm_params.get("batch_size", 32),
                callbacks=callbacks,
                verbose=1  # Show progress bar
            )

            # PSEUDOCODE: Extract best validation metrics from history
            best_val_loss = min(history.history["val_loss"])
            best_epoch = history.history["val_loss"].index(best_val_loss) + 1

            logger.info(
                f"Entrenamiento completado - "
                f"Mejor val_loss: {best_val_loss:.4f} en época {best_epoch}"
            )

            # ACTUAL CODE: Log training metrics
            mlflow.log_params({
                "best_epoch": best_epoch,
                "total_epochs_trained": len(history.history["loss"])
            })
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("final_train_loss", history.history["loss"][-1])

            best_model = model
            best_params = lstm_params

        else:
            # PLACEHOLDER: Grid/Random search will be added in Phase 2
            raise ValueError(
                f"hyperparameter_search_strategy '{hyperparameter_search_strategy}' "
                f"no soportado en Fase 1. Use 'none' para parámetros manuales."
            )

        # ======================
        # 8. DETENER TRACKER DE ENERGÍA
        # ======================

        # ACTUAL CODE: Stop energy tracking
        tracker.stop()
        energy_kwh, emissions_kg = log_energy_metrics(tracker)

        logger.info(
            f"Consumo de energía: {energy_kwh:.4f} kWh, "
            f"Emisiones de carbono: {emissions_kg:.6f} kg CO2"
        )

        # ======================
        # 9. EVALUACIÓN EN CONJUNTO DE VALIDACIÓN
        # ======================

        # ACTUAL CODE: Evaluate on validation set
        logger.info("Evaluando modelo en conjunto de validación...")
        val_metrics, val_artifacts = evaluate_lstm_model(
            model=best_model,
            X_test=X_val,
            y_test=y_val,
            prefix="val",
            experiment_dir=experiment_dir
        )

        # ======================
        # 10. EVALUACIÓN EN CONJUNTO DE PRUEBA
        # ======================

        # ACTUAL CODE: Evaluate on test set
        logger.info("Evaluando modelo en conjunto de prueba...")
        test_metrics, test_artifacts = evaluate_lstm_model(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
            prefix="test",
            experiment_dir=experiment_dir
        )

        # ======================
        # 11. REGISTRO DE MÉTRICAS EN MLFLOW
        # ======================

        # ACTUAL CODE: Log all metrics
        for metric_name, metric_value in {**val_metrics, **test_metrics}.items():
            if metric_value is not None:
                mlflow.log_metric(metric_name, metric_value)

        # ACTUAL CODE: Log best parameters
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # ======================
        # 12. REGISTRO DE ARTEFACTOS EN MLFLOW
        # ======================

        # ACTUAL CODE: Log plot artifacts
        for artifact_path in val_artifacts + test_artifacts:
            if os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path, "plots")

        # PSEUDOCODE: Generate and log training history plot
        if 'history' in locals():
            plt.figure(figsize=(12, 6))
            plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
            plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
            plt.title("LSTM - Curva de Aprendizaje")
            plt.xlabel("Época")
            plt.ylabel("Loss (MSE)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            history_path = os.path.join(experiment_dir, "lstm_training_history.png")
            plt.savefig(history_path, dpi=150, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(history_path, "plots")

        # ======================
        # 13. GUARDADO Y REGISTRO DEL MODELO
        # ======================

        # ACTUAL CODE: Save model to experiment directory
        model_save_path = os.path.join(experiment_dir, "lstm_model")
        best_model.save(model_save_path, save_format="tf")  # SavedModel format
        logger.info(f"Modelo guardado en: {model_save_path}")

        # PSEUDOCODE: Infer signature for MLflow
        # Use a small sample for signature inference
        sample_input = X_train[:5]
        sample_output = best_model.predict(sample_input, verbose=0)
        signature = infer_signature(sample_input, sample_output)

        # ACTUAL CODE: Register model in MLflow
        mlflow.keras.log_model(
            model=best_model,
            artifact_path="lstm_model",
            signature=signature,
            registered_model_name=model_name,
            metadata={
                "dataset": os.path.basename(dataset_path),
                "target": target_variable,
                "features": input_features,
                "date_column": date_col_name,
                "forecast_horizon": forecast_horizon,
                "sequence_length": sequence_length,
                "architecture": str(best_params.get("lstm_units")),
                "cpu_only": True
            }
        )

        logger.info(f"Modelo registrado en MLflow: {model_name}")

        # ======================
        # 14. LIMPIEZA DE CHECKPOINTS TEMPORALES
        # ======================

        # ACTUAL CODE: Aggressive cleanup - delete all temporary checkpoints
        checkpoint_dir = os.path.join(experiment_dir, "temp_checkpoints")
        if os.path.exists(checkpoint_dir):
            import shutil
            shutil.rmtree(checkpoint_dir)
            logger.info(f"Checkpoints temporales eliminados: {checkpoint_dir}")

        # ======================
        # 15. ACTUALIZACIÓN DE PIPELINE CONFIG
        # ======================

        # ACTUAL CODE: Save pipeline configuration
        pipeline_step_config = {
            "step_name": "train_model",
            "algorithm": "lstm",
            "params": convert_numpy_to_python(best_params),
            "metrics": {
                "val_rmse": val_metrics.get("val_rmse"),
                "test_rmse": test_metrics.get("test_rmse")
            },
            "lstm_metadata": {
                "sequence_length": sequence_length,
                "model_architecture": str(best_params.get("lstm_units")),
                "training_time_seconds": None,  # Can be added if tracked
                "early_stopped": best_epoch < best_params.get("epochs", 100),
                "stopped_at_epoch": best_epoch,
                "total_params": best_model.count_params(),
                "cpu_only": True,
                "energy_kwh": energy_kwh,
                "carbon_emissions_kg": emissions_kg
            }
        }

        save_pipeline_config(experiment_dir, pipeline_step_config)

        # ======================
        # 16. FINALIZACIÓN DE RUN DE MLFLOW
        # ======================

        # ACTUAL CODE: End MLflow run
        mlflow.end_run()
        logger.info("Run de MLflow finalizada exitosamente")

        # ======================
        # 17. RETORNO DE RESULTADOS
        # ======================

        return {
            "status": "success",
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_path": os.path.relpath(model_save_path, experiment_dir),
            "run_id": mlflow_run_id,
            "features_used": input_features,
            "sequence_length": sequence_length,
            "best_params": best_params
        }

    except Exception as e:
        # ACTUAL CODE: Error handling
        logger.error(f"Error en entrenamiento LSTM: {e}", exc_info=True)

        # PSEUDOCODE: End MLflow run if active
        if mlflow.active_run():
            mlflow.end_run()

        # PSEUDOCODE: Re-raise as RuntimeError with context
        raise RuntimeError(f"Error en entrenamiento LSTM: {e}") from e
```

**Key Implementation Notes:**
- Phase 1 only supports `hyperparameter_search_strategy="none"`
- Follows exact same structure as `train_xgboost_model`
- Aggressive checkpoint cleanup (deletes temp directory)
- Comprehensive MLflow logging
- Energy tracking with CodeCarbon
- SavedModel format for Keras models

---

### Automated Verification

**Test File:** Create `DREAM-ML-backend/GEML/apiTimeSeries/tests/test_lstm_phase1.py`

```python
import pytest
import numpy as np
import pandas as pd
from apiTimeSeries.train import (
    create_sequences_for_lstm,
    lstm_train_val_test_split,
    build_lstm_model,
    train_lstm_model
)

def test_create_sequences_basic():
    """Test basic sequence creation"""
    # Generate synthetic data
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'value': np.sin(np.linspace(0, 10, 100))
    })
    df.set_index('date', inplace=True)

    X, y = create_sequences_for_lstm(df, ['value'], 'value', sequence_length=10)

    assert X.shape == (90, 10, 1)  # 100 - 10 = 90 sequences
    assert y.shape == (90,)
    assert X.dtype == np.float64


def test_lstm_train_val_test_split():
    """Test temporal split maintains 3D shape"""
    X = np.random.rand(100, 10, 2)
    y = np.random.rand(100)

    X_train, y_train, X_val, y_val, X_test, y_test = lstm_train_val_test_split(
        X, y, {"train": 0.7, "val": 0.15, "test": 0.15}
    )

    assert X_train.shape == (70, 10, 2)
    assert X_val.shape == (15, 10, 2)
    assert X_test.shape == (15, 10, 2)


def test_build_lstm_model_single_layer():
    """Test single-layer LSTM model"""
    params = {
        "lstm_units": [64],
        "dropout_rate": 0.2,
        "learning_rate": 0.001
    }

    model = build_lstm_model(params, input_shape=(10, 2))

    assert model is not None
    assert len(model.layers) == 2  # LSTM + Dense


def test_build_lstm_model_multi_layer():
    """Test multi-layer LSTM model"""
    params = {
        "lstm_units": [64, 32],
        "dropout_rate": 0.2,
        "learning_rate": 0.001
    }

    model = build_lstm_model(params, input_shape=(10, 2))

    assert len(model.layers) == 3  # LSTM + LSTM + Dense
```

**Run Tests:**
```bash
cd DREAM-ML-backend/GEML
python -m pytest apiTimeSeries/tests/test_lstm_phase1.py -v
```

---

### Manual Verification

1. **Start Backend:**
   ```bash
   cd DREAM-ML-backend/GEML
   python manage.py runserver
   ```

2. **Open Frontend:**
   - Navigate to TSTrainCard component
   - Select "LSTM (Deep Learning)" from algorithm dropdown

3. **Configure Training:**
   - Upload CSV dataset (e.g., time series with date + target)
   - Select target variable
   - Keep default manual parameters
   - Set sequence_length: 10
   - Set epochs: 20 (for quick test)

4. **Train Model:**
   - Click "Train Model" button
   - Monitor console logs for progress
   - Verify training completes without errors

5. **Verify MLflow:**
   - Open MLflow UI (typically http://localhost:5000)
   - Find LSTM run
   - Check:
     - ✅ Parameters logged (model_type, sequence_length, etc.)
     - ✅ Metrics logged (val_rmse, test_rmse, etc.)
     - ✅ Artifacts present (plots folder)
     - ✅ Model registered with signature

6. **Verify Experiment Directory:**
   ```bash
   ls <experiment_dir>/
   # Should contain:
   # - lstm_model/ (SavedModel directory)
   # - lstm_val_forecast.png
   # - lstm_test_forecast.png
   # - lstm_*_residuals.png
   # - pipeline_config.json
   ```

---

### Success Criteria

- ✅ No ImportError when importing `train_lstm_model` from services.py
- ✅ Training completes without errors
- ✅ MLflow logs model with correct signature
- ✅ All evaluation plots generated
- ✅ SavedModel directory created
- ✅ Pipeline config updated with LSTM metadata
- ✅ Energy metrics tracked and logged
- ✅ Temporary checkpoints cleaned up

---

### Rollback Strategy

**If Phase 1 fails:**

**Option A: Git Rollback**
```bash
cd DREAM-ML-backend/GEML
git checkout HEAD~1 -- apiTimeSeries/train.py
git checkout HEAD~1 -- apiTimeSeries/services.py  # if modified
```

**Option B: Manual Removal**
1. Open `train.py`
2. Remove functions added after line 1529:
   - `create_sequences_for_lstm`
   - `lstm_train_val_test_split`
   - `build_lstm_model`
   - `create_lstm_callbacks`
   - `evaluate_lstm_model`
   - `train_lstm_model`
3. Verify `services.py` import at line 48 (comment out if needed)

**Option C: Restore from Backup**
```bash
cp train.py.backup train.py
```

---

### Dependencies

**Python Packages (already installed):**
- tensorflow>=2.0
- keras (included in TensorFlow 2.x)
- mlflow
- pandas
- numpy
- matplotlib
- codecarbon
- scikit-learn

**Verify Installation:**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

---

## Phase 2A: Grid Search Hyperparameter Optimization

**Status:** 🔄 In Progress
**Time Estimate:** 3-4.5 hours
**Prerequisites:** Phase 1 completed and verified

### ⚠️ Pattern Consistency Checklist (Phase 1 → Phase 2A)

Before implementing Phase 2A, ensure the following patterns from Phase 1 are maintained:

#### Memory Management Patterns
- [ ] **Critical:** Use `del model` before `tf.keras.backend.clear_session()` in search loops
- [ ] **Critical:** Call `gc.collect()` after clear_session in each iteration
- [ ] Verify memory increase < 500MB during grid search iterations (use psutil)
- [ ] Delete previous best model when new best is found: `if best_model is not None: del best_model`

#### Best Model Selection Patterns
- [ ] Track `best_val_loss` as float('inf') initially
- [ ] Use `min(history.history["val_loss"])` for iteration comparison
- [ ] Store `best_iteration` (1-indexed, not 0-indexed)
- [ ] Only save best model, not last model

#### MLflow Logging Patterns
- [ ] Log `best_iteration` as metric
- [ ] Log `best_val_loss` as metric
- [ ] Log `grid_iterations_total` as metric
- [ ] Log best params with `f"best_{k}"` prefix
- [ ] Convert numpy types with `convert_numpy_to_python()` before logging
- [ ] Log memory metrics if profiling enabled: `memory_usage_mb`, `memory_increase_mb`

#### Search Loop Patterns
- [ ] Use `verbose=0` (quiet mode) for model.fit() in search loops
- [ ] Generate unique checkpoint filenames: `f"grid_checkpoint_{i}.h5"`
- [ ] Wrap iteration in try-except, continue on error (don't fail entire search)
- [ ] Log progress every 10 iterations: `logger.info(f"Grid Search Progress: {i+1}/{total} iterations completed")`

#### Grid Search Specific
- [ ] Use `from sklearn.model_selection import ParameterGrid`
- [ ] Calculate `n_combinations = len(grid)`
- [ ] Warn if `n_combinations > grid_warning_threshold` (configurable, default 50)
- [ ] Validate at least one model trained successfully before returning
- [ ] Provide conservative defaults if grid_search_params not provided (8 combinations)

#### Memory Profiling (Conditional)
- [ ] Import psutil only when `enable_memory_profiling=True`
- [ ] Track initial_memory_mb before training loop
- [ ] Track final_memory_mb and memory_increase_mb after loop
- [ ] Log warning if memory_increase_mb > 500MB
- [ ] Log metrics to MLflow when profiling enabled

#### Validation Patterns (from Phase 1)
- [ ] Maintain sequence_length auto-fallback with warning
- [ ] Validate hyperparameter_search_strategy in ["none", "grid"]
- [ ] Extract grid_warning_threshold with default value

#### Code Organization
- [ ] Keep Phase 1 "none" strategy code unchanged
- [ ] Add grid search as elif block after "none"
- [ ] Maintain same structure: build → callbacks → fit → evaluate → cleanup

---

### Overview

**Objective:** Implement Grid Search hyperparameter optimization for LSTM training with configurable parameters, memory profiling, and frontend UI controls.

**Scope:**
- Grid search with configurable combination count warning threshold
- Conservative default parameter grid (8 combinations)
- Conditional memory profiling with psutil
- Progress tracking every 10 iterations
- Best model selection based on val_loss
- Frontend UI controls for grid parameters
- Memory cleanup between iterations (prevent leaks)

**Time Estimate:** 3-4.5 hours

<!--
ARCHIVED: Original Phase 2 Combined Scope (Grid + Random Search)
This section preserved for reference. Implementation split into Phase 2A (Grid) and Phase 2B (Random).

**Original Objective:** Add Grid Search and Random Search hyperparameter optimization strategies.
**Original Scope:**
- Grid search with combination count warning
- Random search with parameter generation
- Memory cleanup between iterations (prevent leaks)
- Progress tracking via WebSocket
- Best model selection based on val_loss
**Original Time Estimate:** 3-4 hours
-->

### Prerequisites

- Phase 1 completed and verified
- Manual params training working correctly

### Memory Profiling Setup

Phase 2 introduces iterative training (grid/random search), which requires monitoring memory usage to prevent leaks. The success criterion is **memory increase < 500MB** during 100 random search iterations.

**Installation:**
```bash
pip install psutil
```

**Usage in Code (for testing):**
```python
import psutil
import os

# Initialize process monitor
process = psutil.Process(os.getpid())

# Measure memory before training loop
initial_memory_mb = process.memory_info().rss / 1024 / 1024  # Convert to MB
print(f"Initial memory: {initial_memory_mb:.1f} MB")

# ... run grid/random search training loop ...

# Measure memory after training loop
final_memory_mb = process.memory_info().rss / 1024 / 1024
memory_increase_mb = final_memory_mb - initial_memory_mb

print(f"Final memory: {final_memory_mb:.1f} MB")
print(f"Memory increase: {memory_increase_mb:.1f} MB")

# Verify memory leak threshold
assert memory_increase_mb < 500, f"Memory leak detected: {memory_increase_mb:.1f}MB increase exceeds 500MB threshold"
```

**Integration with Existing Test (test_lstm_phase2.py, line 1687):**

The test file already includes a memory monitoring test. Ensure the following:

1. **Before implementing Phase 2:** Baseline memory usage by running Phase 1 training
2. **During implementation:** Add memory profiling to grid/random search loops
3. **After implementation:** Run `test_random_search_memory_usage()` to verify <500MB

**Expected Memory Profile:**
- Phase 1 (manual params): ~200-300MB increase (single model training)
- Phase 2 Grid Search (10 iterations): ~300-400MB increase (with cleanup)
- Phase 2 Random Search (100 iterations): ~400-500MB increase (with cleanup)

**If memory exceeds threshold:**
1. Verify `tf.keras.backend.clear_session()` is called after each iteration
2. Check that `del model` is executed before clear_session
3. Add explicit `gc.collect()` after cleanup
4. Reduce batch size if still problematic

### Files Modified

1. **`DREAM-ML-backend/GEML/apiTimeSeries/train.py`**
   - Add `generate_random_lstm_params` function
   - Modify `train_lstm_model` to support grid/random strategies

### Implementation Details

#### A. Random Parameter Generator Function

**Location:** `train.py`, before `train_lstm_model`

```python
def generate_random_lstm_params(random_search_params: Dict) -> Dict:
    """
    Genera hiperparámetros LSTM aleatorios desde rangos especificados.

    Asegura que todos los valores retornados sean tipos Python nativos
    (no numpy) para serialización JSON.

    Args:
        random_search_params: Diccionario con rangos y opciones:
            - lstm_units_options: List[List[int]] (e.g., [[32], [64], [128], [64, 32]])
            - dropout_rate_range: [float, float] (e.g., [0.0, 0.3])
            - recurrent_dropout_rate_range: [float, float]
            - learning_rate_range: [float, float] (usa distribución log-uniforme)
            - batch_size_options: List[int] (e.g., [16, 32, 64])
            - epochs_range: [int, int] (e.g., [20, 100])

    Returns:
        Diccionario con parámetros aleatorios (tipos Python nativos)

    Example:
        >>> search_params = {
        ...     "lstm_units_options": [[32], [64], [128]],
        ...     "dropout_rate_range": [0.0, 0.3],
        ...     "learning_rate_range": [0.001, 0.01],
        ...     "batch_size_options": [16, 32],
        ...     "epochs_range": [20, 50]
        ... }
        >>> params = generate_random_lstm_params(search_params)
        >>> params
        {'lstm_units': [64], 'dropout_rate': 0.234, ...}
    """
    import random

    # ACTUAL CODE: Random sampling from options and ranges
    params = {
        # Categorical: random choice from list
        "lstm_units": random.choice(random_search_params["lstm_units_options"]),

        # Uniform distribution for dropout rates
        "dropout_rate": float(np.random.uniform(
            random_search_params["dropout_rate_range"][0],
            random_search_params["dropout_rate_range"][1]
        )),

        "recurrent_dropout_rate": float(np.random.uniform(
            random_search_params["recurrent_dropout_rate_range"][0],
            random_search_params["recurrent_dropout_rate_range"][1]
        )),

        # Log-uniform distribution for learning rate (better sampling across orders of magnitude)
        "learning_rate": float(np.exp(np.random.uniform(
            np.log(random_search_params["learning_rate_range"][0]),
            np.log(random_search_params["learning_rate_range"][1])
        ))),

        # Categorical: random choice for batch size
        "batch_size": int(random.choice(random_search_params["batch_size_options"])),

        # Uniform integer distribution for epochs
        "epochs": int(np.random.randint(
            random_search_params["epochs_range"][0],
            random_search_params["epochs_range"][1]
        ))
    }

    # PSEUDOCODE: Convert any remaining numpy types to Python natives
    params = convert_numpy_to_python(params)

    return params
```

---

#### B. Modify train_lstm_model Function

**Location:** `train.py`, in `train_lstm_model`, replace Phase 1 placeholder with:**

```python
        # ======================
        # 7. ESTRATEGIA DE BÚSQUEDA DE HIPERPARÁMETROS
        # ======================

        if hyperparameter_search_strategy == "none":
            # MANUAL PARAMS (from Phase 1 - keep existing code)
            lstm_params = data.get("lstm_params", {
                "lstm_units": [64],
                "dropout_rate": 0.2,
                "recurrent_dropout_rate": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100
            })

            logger.info(f"Entrenando con parámetros manuales: {lstm_params}")

            # Build, train, evaluate (existing Phase 1 code)
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = build_lstm_model(lstm_params, input_shape)
            callbacks, checkpoint_path = create_lstm_callbacks(
                experiment_dir=experiment_dir,
                early_stopping_patience=early_stopping_patience
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=lstm_params.get("epochs", 100),
                batch_size=lstm_params.get("batch_size", 32),
                callbacks=callbacks,
                verbose=1
            )

            best_val_loss = min(history.history["val_loss"])
            best_epoch = history.history["val_loss"].index(best_val_loss) + 1

            mlflow.log_params({"best_epoch": best_epoch})
            mlflow.log_metric("best_val_loss", best_val_loss)

            best_model = model
            best_params = lstm_params

        elif hyperparameter_search_strategy == "grid":
            # GRID SEARCH IMPLEMENTATION
            logger.info("Iniciando Grid Search de hiperparámetros...")

            # PSEUDOCODE: Extract grid search parameters
            grid_search_params = data.get("grid_search_params", {})
            if not grid_search_params:
                raise ValueError("grid_search_params requerido para estrategia 'grid'")

            # ACTUAL CODE: Generate parameter grid
            from sklearn.model_selection import ParameterGrid

            # Convert lstm_units to proper format for ParameterGrid
            param_grid = {
                "lstm_units": grid_search_params.get("lstm_units_options", [[64]]),
                "dropout_rate": grid_search_params.get("dropout_rate_options", [0.2]),
                "recurrent_dropout_rate": grid_search_params.get("recurrent_dropout_rate_options", [0.2]),
                "learning_rate": grid_search_params.get("learning_rate_options", [0.001]),
                "batch_size": grid_search_params.get("batch_size_options", [32]),
                "epochs": grid_search_params.get("epochs_options", [100])
            }

            grid = list(ParameterGrid(param_grid))
            n_combinations = len(grid)

            logger.info(f"Grid Search: {n_combinations} combinaciones a evaluar")

            # PSEUDOCODE: Warn if combinations exceed threshold
            if n_combinations > 100:
                logger.warning(
                    f"⚠️ Grid Search generará {n_combinations} combinaciones. "
                    f"Esto puede tomar varias horas. Considere usar Random Search en su lugar."
                )

            # PSEUDOCODE: Initialize best model tracking
            best_val_loss = float('inf')
            best_model = None
            best_params = None
            best_iteration = None

            input_shape = (X_train.shape[1], X_train.shape[2])

            # ACTUAL CODE: Iterate through grid
            for i, params in enumerate(grid):
                logger.info(f"Grid Search - Iteración {i+1}/{n_combinations}: {params}")

                try:
                    # Build model
                    model = build_lstm_model(params, input_shape)

                    # Create callbacks
                    callbacks, checkpoint_path = create_lstm_callbacks(
                        experiment_dir=experiment_dir,
                        early_stopping_patience=early_stopping_patience,
                        checkpoint_filename=f"grid_checkpoint_{i}.h5"
                    )

                    # Train model
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=params["epochs"],
                        batch_size=params["batch_size"],
                        callbacks=callbacks,
                        verbose=0  # Quiet mode for grid search
                    )

                    # Extract best val_loss from this iteration
                    iteration_val_loss = min(history.history["val_loss"])

                    logger.info(
                        f"Grid Search - Iteración {i+1} completada. "
                        f"val_loss: {iteration_val_loss:.4f}"
                    )

                    # PSEUDOCODE: Update best model if improved
                    if iteration_val_loss < best_val_loss:
                        best_val_loss = iteration_val_loss

                        # Delete previous best model to save memory
                        if best_model is not None:
                            del best_model

                        best_model = model
                        best_params = params
                        best_iteration = i + 1

                        logger.info(
                            f"✓ Nuevo mejor modelo encontrado en iteración {best_iteration}: "
                            f"val_loss={best_val_loss:.4f}"
                        )
                    else:
                        # Not the best, delete to free memory
                        del model

                    # CRITICAL: Memory cleanup after each iteration
                    tf.keras.backend.clear_session()
                    gc.collect()

                except Exception as e:
                    logger.error(f"Error en iteración {i+1}: {e}")
                    # Continue to next iteration
                    continue

            # PSEUDOCODE: Verify at least one model trained successfully
            if best_model is None:
                raise RuntimeError("No se pudo entrenar ningún modelo en Grid Search")

            logger.info(
                f"Grid Search completado - Mejor modelo: iteración {best_iteration}, "
                f"val_loss={best_val_loss:.4f}, params={best_params}"
            )

            # Log best results to MLflow
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("grid_iterations_total", n_combinations)
            mlflow.log_metric("best_iteration", best_iteration)

        elif hyperparameter_search_strategy == "random":
            # RANDOM SEARCH IMPLEMENTATION
            logger.info("Iniciando Random Search de hiperparámetros...")

            # PSEUDOCODE: Extract random search parameters
            n_random_iterations = data.get("n_random_iterations", 100)
            random_search_params = data.get("random_search_params", {})

            if not random_search_params:
                raise ValueError("random_search_params requerido para estrategia 'random'")

            logger.info(f"Random Search: {n_random_iterations} iteraciones")

            # PSEUDOCODE: Warn if iterations very high
            if n_random_iterations > 200:
                logger.warning(
                    f"n_random_iterations es muy alto ({n_random_iterations}). "
                    f"Considere usar un valor menor (<200) para mejorar el rendimiento."
                )

            # PSEUDOCODE: Initialize best model tracking
            best_val_loss = float('inf')
            best_model = None
            best_params = None
            best_iteration = None

            input_shape = (X_train.shape[1], X_train.shape[2])

            # ACTUAL CODE: Random search loop
            for i in range(n_random_iterations):
                # Generate random parameters
                params = generate_random_lstm_params(random_search_params)

                logger.info(
                    f"Random Search - Iteración {i+1}/{n_random_iterations}: {params}"
                )

                try:
                    # Build model
                    model = build_lstm_model(params, input_shape)

                    # Create callbacks
                    callbacks, checkpoint_path = create_lstm_callbacks(
                        experiment_dir=experiment_dir,
                        early_stopping_patience=early_stopping_patience,
                        checkpoint_filename=f"random_checkpoint_{i}.h5"
                    )

                    # Train model
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=params["epochs"],
                        batch_size=params["batch_size"],
                        callbacks=callbacks,
                        verbose=0  # Quiet mode
                    )

                    # Extract best val_loss
                    iteration_val_loss = min(history.history["val_loss"])

                    logger.info(
                        f"Random Search - Iteración {i+1} completada. "
                        f"val_loss: {iteration_val_loss:.4f}"
                    )

                    # PSEUDOCODE: Update best model
                    if iteration_val_loss < best_val_loss:
                        best_val_loss = iteration_val_loss

                        if best_model is not None:
                            del best_model

                        best_model = model
                        best_params = params
                        best_iteration = i + 1

                        logger.info(
                            f"✓ Nuevo mejor modelo: iteración {best_iteration}, "
                            f"val_loss={best_val_loss:.4f}"
                        )
                    else:
                        del model

                    # CRITICAL: Memory cleanup (prevent memory leak)
                    tf.keras.backend.clear_session()
                    gc.collect()

                except Exception as e:
                    logger.error(f"Error en iteración {i+1}: {e}")
                    continue

            # PSEUDOCODE: Verify success
            if best_model is None:
                raise RuntimeError("No se pudo entrenar ningún modelo en Random Search")

            logger.info(
                f"Random Search completado - Mejor modelo: iteración {best_iteration}, "
                f"val_loss={best_val_loss:.4f}"
            )

            # Log results
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("random_iterations_total", n_random_iterations)
            mlflow.log_metric("best_iteration", best_iteration)

        else:
            raise ValueError(
                f"hyperparameter_search_strategy '{hyperparameter_search_strategy}' "
                f"no soportado. Opciones válidas: 'none', 'grid', 'random'"
            )
```

---

### Automated Verification

**Add to test file:** `test_lstm_phase2.py`

```python
import pytest
from apiTimeSeries.train import generate_random_lstm_params

def test_generate_random_lstm_params():
    """Test random parameter generation"""
    search_params = {
        "lstm_units_options": [[32], [64], [128]],
        "dropout_rate_range": [0.0, 0.3],
        "recurrent_dropout_rate_range": [0.0, 0.3],
        "learning_rate_range": [0.001, 0.01],
        "batch_size_options": [16, 32],
        "epochs_range": [20, 50]
    }

    params = generate_random_lstm_params(search_params)

    # Verify all keys present
    assert "lstm_units" in params
    assert "dropout_rate" in params
    assert "learning_rate" in params

    # Verify types (Python natives, not numpy)
    assert isinstance(params["dropout_rate"], float)
    assert isinstance(params["batch_size"], int)

    # Verify ranges
    assert 0.0 <= params["dropout_rate"] <= 0.3
    assert params["batch_size"] in [16, 32]


def test_random_search_memory_usage():
    """Test memory doesn't leak during random search"""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Run small random search
    # ... (training code)

    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory

    # Should be < 500MB increase
    assert memory_increase < 500, f"Memory leak detected: {memory_increase:.1f}MB increase"
```

---

### Manual Verification

1. **Grid Search Test:**
   - Frontend: Select "Grid Search"
   - Configure small grid (2×2×2 = 8 combinations)
   - Train model
   - Verify: Warning if >100 combinations
   - Check MLflow: `best_iteration` logged

2. **Random Search Test:**
   - Frontend: Select "Random Search"
   - Set iterations: 10 (for quick test)
   - Train model
   - Monitor memory usage: `watch -n 1 'ps aux | grep python'`
   - Verify memory stable (no continuous growth)

3. **Best Model Selection:**
   - Verify best model saved (not last model)
   - Check MLflow metrics match best iteration

---

### Success Criteria

- ✅ Grid search completes without errors
- ✅ Random search completes without errors
- ✅ Warning shown when grid combinations > 100
- ✅ Memory increase < 500MB during random search
- ✅ Best model selected correctly (lowest val_loss)
- ✅ MLflow logs best parameters and iteration
- ✅ Memory cleanup prevents leaks

---

### Rollback Strategy

**If Phase 2 fails:**
- Revert `train_lstm_model` changes to Phase 1 version
- Remove `generate_random_lstm_params` function
- Keep Phase 1 manual params working

```bash
git diff train.py  # Review changes
git checkout <phase1_commit> -- apiTimeSeries/train.py
```

---

## Phase 2B: Random Search Hyperparameter Optimization

**Status:** 🟡 READY (Phase 2A Complete)
**Time Estimate:** 2-3 hours
**Prerequisites:** Phase 2A completed and verified

### ⚠️ Pattern Consistency Checklist (Phase 2A → Phase 2B)

Before implementing Phase 2B, ensure the following patterns from Phase 2A are maintained:

#### Memory Management Patterns (Carry over from 2A)
- [ ] **Critical:** Use `del model` before `tf.keras.backend.clear_session()` in search loops
- [ ] **Critical:** Call `gc.collect()` after clear_session in each iteration
- [ ] Verify memory increase < 500MB during 100 random search iterations (use psutil)
- [ ] Delete previous best model when new best is found: `if best_model is not None: del best_model`

#### Best Model Selection Patterns (Carry over from 2A)
- [ ] Track `best_val_loss` as float('inf') initially
- [ ] Use `min(history.history["val_loss"])` for iteration comparison
- [ ] Store `best_iteration` (1-indexed, not 0-indexed)
- [ ] Only save best model, not last model
- [ ] Set `best_epoch = None` for random search (not tracked per iteration)

#### MLflow Logging Patterns (Carry over from 2A)
- [ ] Log `best_iteration` as metric
- [ ] Log `best_val_loss` as metric
- [ ] Log `random_iterations_total` as metric (instead of grid_iterations_total)
- [ ] Log best params with `f"best_{k}"` prefix
- [ ] Convert numpy types with `convert_numpy_to_python()` before logging
- [ ] Log memory metrics if profiling enabled: `memory_usage_mb`, `memory_increase_mb`

#### Search Loop Patterns (Carry over from 2A)
- [ ] Use `verbose=0` (quiet mode) for model.fit() in search loops
- [ ] Generate unique checkpoint filenames: `f"random_checkpoint_{i}.h5"`
- [ ] Wrap iteration in try-except, continue on error (don't fail entire search)
- [ ] Log progress every 10 iterations: `logger.info(f"Random Search Progress: {i+1}/{total} iterations completed")`

#### Random Search Specific (NEW for Phase 2B)
- [ ] Implement `generate_random_lstm_params()` function first
- [ ] Use log-uniform distribution for learning rate: `np.exp(np.random.uniform(np.log(min), np.log(max)))`
- [ ] Use uniform distribution for dropout rates: `np.random.uniform(low, high)`
- [ ] Use categorical choice for lstm_units: `random.choice(options)`
- [ ] Use categorical choice for batch_size: `random.choice(options)`
- [ ] Use uniform integer range for epochs: `np.random.randint(low, high+1)`
- [ ] Return Python native types (not numpy) from parameter generator
- [ ] Convert all generated params with `convert_numpy_to_python()` before returning

#### Memory Profiling (Reuse from 2A)
- [ ] Import psutil only when `enable_memory_profiling=True`
- [ ] Track initial_memory_mb before training loop
- [ ] Track final_memory_mb and memory_increase_mb after loop
- [ ] Log warning if memory_increase_mb > 500MB
- [ ] Log metrics to MLflow when profiling enabled

#### Validation Patterns (Carry over from 2A)
- [ ] Maintain sequence_length auto-fallback with warning
- [ ] Validate hyperparameter_search_strategy in ["none", "grid", "random"]
- [ ] Extract n_random_iterations with default value (e.g., 100)
- [ ] Warn if n_random_iterations > 200

#### Code Organization (Build on 2A)
- [ ] Keep Phase 1 "none" strategy code unchanged
- [ ] Keep Phase 2A "grid" strategy code unchanged
- [ ] Add random search as elif block after "grid"
- [ ] Maintain same structure: build → callbacks → fit → evaluate → cleanup
- [ ] Reuse memory profiling logic from Phase 2A

#### Integration with Frontend (Reuse existing)
- [ ] Random search UI already exists (TSTrainCard.jsx lines 250-257, 2270+)
- [ ] Payload construction already includes random_search_params (lines 740-750)
- [ ] No frontend changes needed for Phase 2B

---

### Overview

**Objective:** Implement Random Search hyperparameter optimization for LSTM training, reusing memory profiling and patterns from Phase 2A.

**Scope:**
- Random parameter generation function (`generate_random_lstm_params`)
- Random search loop with configurable iterations (default: 100)
- Memory profiling (reuse from Phase 2A)
- Progress tracking every 10 iterations
- Best model selection based on val_loss
- Memory cleanup between iterations (prevent leaks)
- Frontend UI already exists (no changes needed)

**Time Estimate:** 2-3 hours

### Prerequisites

- Phase 2A completed and verified
- Grid search memory patterns working correctly

### Files Modified

1. **Backend**: `DREAM-ML-backend/GEML/apiTimeSeries/train.py` (~200 lines added)
   - Add `generate_random_lstm_params()` function before `train_lstm_model()`
   - Add random search elif block in `train_lstm_model()` after grid search

2. **Tests**: `DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_lstm_phase2b.py` (new file, ~400 lines)
   - Test random parameter generation
   - Test random search with 10 iterations
   - Test memory profiling (conditional)
   - Test memory cleanup (< 500MB for 100 iterations)
   - Test parameter types (Python natives, not numpy)

### Implementation Notes

**Frontend Integration:**
- Random search UI already exists and working (implemented before Phase 2A)
- No frontend changes needed
- Focus purely on backend implementation

**Pattern Reuse from Phase 2A:**
- Memory profiling logic: Copy pattern exactly
- Progress logging: Same format, replace "Grid" with "Random"
- MLflow logging: Same pattern, replace `grid_iterations_total` with `random_iterations_total`
- Memory cleanup: Identical pattern

**Key Differences from Grid Search:**
- No ParameterGrid needed (generate params on-the-fly)
- Need `generate_random_lstm_params()` helper function
- Use log-uniform distribution for learning_rate
- Use uniform distribution for dropout rates
- Use random.choice for categorical params

---

## Phase 3A: Critical Fixes for Reproducibility

**Status:** ✅ COMPLETED (2025-11-13)
**Time Estimate:** 45-60 minutes (Actual: ~50 minutes)
**Prerequisites:** Phase 2A and 2B completed and verified
**Priority:** CRITICAL (Required for reproducibility objective)

### Objective

Fix `pipeline_config.json` to include complete metrics and hyperparameter search metadata, ensuring any user can fully reproduce experiments from the configuration file alone.

### Scope

1. Add all 6 metrics to pipeline_config.json (val_mae, val_mape, test_mae, test_mape in addition to existing RMSE metrics)
2. Add comprehensive hyperparameter_search metadata section
3. Add schema versioning (v1.1) for future compatibility
4. Implement custom validation function for schema compliance
5. Write automated tests for pipeline_config completeness
6. Manual verification with both synthetic and realistic datasets

### Files Modified

**Backend:**
- `DREAM-ML-backend/GEML/apiTimeSeries/train.py` (lines ~2714-2735)

**Tests:**
- `DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_lstm_phase3a.py` (new file)

### Implementation Details

#### A. Fix Complete Metrics in pipeline_config.json

**Location:** `train.py`, lines 2714-2735 (in `train_lstm_model` function)

**Current Code (Incomplete):**
```python
pipeline_step_config = {
    "step_name": "train_model",
    "algorithm": "lstm",
    "params": convert_numpy_to_python(best_params),
    "metrics": {
        "val_rmse": val_metrics.get("val_rmse"),
        "test_rmse": test_metrics.get("test_rmse")
    },
    "lstm_metadata": {
        "sequence_length": sequence_length,
        # ... other metadata
    }
}
```

**Fixed Code (Complete):**
```python
pipeline_step_config = {
    "schema_version": "1.1",  # NEW: Schema versioning
    "step_name": "train_model",
    "algorithm": "lstm",
    "params": convert_numpy_to_python(best_params),
    "metrics": {
        # FIXED: All 6 metrics included
        "val_rmse": val_metrics.get("val_rmse"),
        "val_mae": val_metrics.get("val_mae"),
        "val_mape": val_metrics.get("val_mape"),
        "test_rmse": test_metrics.get("test_rmse"),
        "test_mae": test_metrics.get("test_mae"),
        "test_mape": test_metrics.get("test_mape")
    },
    "hyperparameter_search": {  # NEW: Search metadata
        "strategy": hyperparameter_search_strategy,
        "iterations_total": None,  # Set below based on strategy
        "best_iteration": None,  # Set below based on strategy
        "best_val_loss": None,  # Set below based on strategy
        "grid_search_params": None,
        "random_search_params": None,
        "n_random_iterations": None,
        "memory_profiling": None
    },
    "lstm_metadata": {
        "sequence_length": sequence_length,
        "model_architecture": str(best_params.get("lstm_units")),
        "training_time_seconds": None,
        "early_stopped": None,  # Set based on strategy
        "stopped_at_epoch": None,  # Set based on strategy
        "total_params": best_model.count_params(),
        "cpu_only": True,
        "energy_kwh": energy_kwh,
        "carbon_emissions_kg": emissions_kg
    }
}

# Populate hyperparameter_search section based on strategy
if hyperparameter_search_strategy == "none":
    pipeline_step_config["hyperparameter_search"]["iterations_total"] = 1
    pipeline_step_config["hyperparameter_search"]["best_iteration"] = 1
    pipeline_step_config["hyperparameter_search"]["best_val_loss"] = float(best_val_loss)

elif hyperparameter_search_strategy == "grid":
    pipeline_step_config["hyperparameter_search"]["iterations_total"] = n_combinations
    pipeline_step_config["hyperparameter_search"]["best_iteration"] = best_iteration
    pipeline_step_config["hyperparameter_search"]["best_val_loss"] = float(best_val_loss)
    pipeline_step_config["hyperparameter_search"]["grid_search_params"] = convert_numpy_to_python(grid_search_params)

    # Add memory profiling if available
    if 'memory_increase_mb' in locals():
        pipeline_step_config["hyperparameter_search"]["memory_profiling"] = {
            "enabled": True,
            "initial_memory_mb": initial_memory_mb,
            "final_memory_mb": final_memory_mb,
            "memory_increase_mb": memory_increase_mb
        }

elif hyperparameter_search_strategy == "random":
    pipeline_step_config["hyperparameter_search"]["iterations_total"] = n_random_iterations
    pipeline_step_config["hyperparameter_search"]["best_iteration"] = best_iteration
    pipeline_step_config["hyperparameter_search"]["best_val_loss"] = float(best_val_loss)
    pipeline_step_config["hyperparameter_search"]["random_search_params"] = convert_numpy_to_python(random_search_params)
    pipeline_step_config["hyperparameter_search"]["n_random_iterations"] = n_random_iterations

    # Add memory profiling if available
    if 'memory_increase_mb' in locals():
        pipeline_step_config["hyperparameter_search"]["memory_profiling"] = {
            "enabled": True,
            "initial_memory_mb": initial_memory_mb,
            "final_memory_mb": final_memory_mb,
            "memory_increase_mb": memory_increase_mb
        }
```

#### B. Schema Validation Function

**Location:** `train.py`, add after helper functions (around line 350)

```python
def validate_pipeline_config_schema(config: Dict, strict: bool = False) -> bool:
    """
    Validates pipeline_config.json schema for completeness and type correctness.

    Supports version-aware validation:
    - v1.0: Relaxed validation (legacy format)
    - v1.1: Strict validation (requires all fields)

    Args:
        config: Pipeline configuration dictionary
        strict: If True, raise exceptions on validation errors

    Returns:
        True if validation passes

    Raises:
        ValueError: If strict=True and validation fails
    """
    version = config.get("schema_version", "1.0")
    errors = []

    # Required fields for v1.1
    if version == "1.1":
        required_top_level = ["schema_version", "step_name", "algorithm", "params", "metrics", "hyperparameter_search", "lstm_metadata"]

        for field in required_top_level:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Validate metrics completeness
        if "metrics" in config:
            required_metrics = ["val_rmse", "val_mae", "val_mape", "test_rmse", "test_mae", "test_mape"]
            for metric in required_metrics:
                if metric not in config["metrics"]:
                    errors.append(f"Missing required metric: {metric}")
                elif config["metrics"][metric] is not None:
                    if not isinstance(config["metrics"][metric], (int, float)):
                        errors.append(f"Invalid type for {metric}: expected float, got {type(config['metrics'][metric])}")

        # Validate hyperparameter_search structure
        if "hyperparameter_search" in config:
            hs = config["hyperparameter_search"]
            required_hs_fields = ["strategy", "iterations_total", "best_iteration", "best_val_loss"]

            for field in required_hs_fields:
                if field not in hs:
                    errors.append(f"Missing hyperparameter_search field: {field}")

            # Strategy-specific validation
            if "strategy" in hs:
                strategy = hs["strategy"]
                if strategy not in ["none", "grid", "random"]:
                    errors.append(f"Invalid strategy: {strategy}")

                if strategy == "grid" and "grid_search_params" not in hs:
                    errors.append("Missing grid_search_params for grid strategy")

                if strategy == "random" and "random_search_params" not in hs:
                    errors.append("Missing random_search_params for random strategy")

        # Validate lstm_metadata
        if "lstm_metadata" in config:
            required_metadata = ["sequence_length", "model_architecture", "total_params", "cpu_only"]
            for field in required_metadata:
                if field not in config["lstm_metadata"]:
                    errors.append(f"Missing lstm_metadata field: {field}")

    elif version == "1.0":
        # Relaxed validation for legacy format
        logger.info("Validating legacy pipeline_config schema (v1.0)")
        if "step_name" not in config:
            errors.append("Missing required field: step_name")
        if "algorithm" not in config:
            errors.append("Missing required field: algorithm")

    else:
        errors.append(f"Unknown schema version: {version}")

    # Handle errors
    if errors:
        error_msg = f"Pipeline config schema validation failed ({len(errors)} errors):\n" + "\n".join(f"  - {e}" for e in errors)
        if strict:
            raise ValueError(error_msg)
        else:
            logger.warning(error_msg)
            return False

    logger.info(f"Pipeline config schema validation passed (version {version})")
    return True
```

#### C. Call Validation After Saving Config

**Location:** `train.py`, after `save_pipeline_config()` call (around line 2740)

```python
# Save and validate pipeline configuration
save_pipeline_config(experiment_dir, pipeline_step_config)

# Validate schema (non-strict for logging purposes)
try:
    validate_pipeline_config_schema(pipeline_step_config, strict=False)
except Exception as e:
    logger.warning(f"Pipeline config validation encountered issues: {e}")
```

---

### Automated Verification

**Test File:** `DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_lstm_phase3a.py`

```python
import pytest
import json
from apiTimeSeries.train import validate_pipeline_config_schema


def test_pipeline_config_schema_version():
    """Test schema version is present and correct"""
    config = {
        "schema_version": "1.1",
        "step_name": "train_model",
        "algorithm": "lstm",
        "params": {},
        "metrics": {
            "val_rmse": 0.5, "val_mae": 0.3, "val_mape": 5.0,
            "test_rmse": 0.6, "test_mae": 0.4, "test_mape": 6.0
        },
        "hyperparameter_search": {
            "strategy": "none",
            "iterations_total": 1,
            "best_iteration": 1,
            "best_val_loss": 0.5,
            "grid_search_params": None,
            "random_search_params": None,
            "n_random_iterations": None,
            "memory_profiling": None
        },
        "lstm_metadata": {
            "sequence_length": 10,
            "model_architecture": "[64]",
            "total_params": 5000,
            "cpu_only": True
        }
    }

    assert validate_pipeline_config_schema(config, strict=True) == True
    assert config["schema_version"] == "1.1"


def test_pipeline_config_all_metrics_present():
    """Test all 6 metrics are included in config"""
    # Similar structure, verify all metrics present
    pass


def test_pipeline_config_hyperparameter_search_grid():
    """Test hyperparameter_search section for grid strategy"""
    # Verify grid search metadata structure
    pass


# Additional tests omitted for brevity - see full implementation plan
```

**Run Tests:**
```bash
cd DREAM-ML-backend/GEML
python -m pytest tests/apiTimeSeries_tests/test_lstm_phase3a.py -v
```

---

### Manual Verification

Detailed step-by-step scenarios with synthetic and realistic data - see full implementation document for complete verification procedures.

---

### Success Criteria

- ✅ Schema version "1.1" present in all new pipeline_config.json files
- ✅ All 6 metrics (val_rmse, val_mae, val_mape, test_rmse, test_mae, test_mape) logged
- ✅ Hyperparameter_search section complete with strategy, iterations, best_iteration, best_val_loss
- ✅ Grid search configs include grid_search_params
- ✅ Random search configs include random_search_params and n_random_iterations
- ✅ Validation function correctly identifies complete vs incomplete configs
- ✅ All automated tests pass
- ✅ Manual verification confirms reproducibility from pipeline_config alone

---

### Rollback Strategy

**If Phase 3A fails:**

```bash
cd DREAM-ML-backend/GEML
git diff apiTimeSeries/train.py  # Review changes
git checkout HEAD -- apiTimeSeries/train.py
```

---

### Phase 3A Completion Summary

**Completion Date:** 2025-11-13
**Implementation Time:** ~50 minutes
**Test Results:** 24/24 automated tests passed ✅
**Manual Verification:** All scenarios (A, B, C, D) completed successfully ✅

**Key Achievements:**
- ✅ Schema version 1.1 implemented and validated
- ✅ All 6 metrics (val/test RMSE/MAE/MAPE) now logged to pipeline_config.json
- ✅ Comprehensive hyperparameter_search metadata section added
- ✅ Memory profiling field always included (None when disabled)
- ✅ MAPE null values handled correctly (division by zero case)
- ✅ Validation function with strict/non-strict modes
- ✅ Backward compatibility with v1.0 configs maintained
- ✅ Edge cases fully tested (malformed configs, type errors, high iterations)

**Files Modified:**
- `apiTimeSeries/train.py`: Added `validate_pipeline_config_schema()` function and enhanced pipeline_config creation
- `tests/apiTimeSeries_tests/test_lstm_phase3a.py`: Created comprehensive test suite (24 tests)

**No Deviations:** Implementation followed plan exactly as specified.

---

## Phase 3B: UI Enhancements

**Status:** ✅ COMPLETED (2025-11-13)
**Time Estimate:** 1.5-2 hours (Actual: ~1.5 hours)
**Prerequisites:** Phase 3A completed and verified ✅
**Priority:** HIGH (User experience improvements)

### ⚠️ Pattern Consistency Checklist (Phase 3A → Phase 3B)

Before implementing Phase 3B, ensure the following patterns from Phase 3A are maintained:

#### Schema Version Patterns (Carry over from 3A)
- [ ] **Critical:** Continue using schema version "1.1" (do NOT create v1.2)
- [ ] UI changes must preserve all existing pipeline_config fields
- [ ] New UI parameters must be added to pipeline_config if they affect reproducibility
- [ ] Sequence length parameter must persist to pipeline_config.json
- [ ] No breaking changes to existing v1.1 schema structure

#### Validation Patterns (Carry over from 3A)
- [ ] If adding new required fields to pipeline_config, update `validate_pipeline_config_schema()`
- [ ] Use non-strict validation mode in production (warnings only)
- [ ] Use strict validation mode in tests
- [ ] Log validation results with clear messages

#### UI Data Flow Patterns
- [ ] UI controls → request payload → train_lstm_model() → pipeline_config.json
- [ ] Ensure sequence_length from UI overrides default value
- [ ] Warning banners should NOT modify training logic
- [ ] Warning banners are UI-only (informational, not blocking)

#### Testing Patterns (Maintain from 3A)
- [ ] Create test file: `test_lstm_phase3b.py`
- [ ] Test UI payload construction with new fields
- [ ] Test that pipeline_config includes new configurable parameters
- [ ] Test warning banner logic (thresholds, dismissal state)
- [ ] Follow existing test fixture patterns (configure_test_logging, synthetic_lstm_dataset)

#### Documentation Patterns
- [ ] Add inline comments for schema-specific behavior
- [ ] Document non-editable parameters with clear explanations
- [ ] Helper text should explain "why" not just "what"
- [ ] Tooltips should be educational and concise

#### Backward Compatibility
- [ ] Existing training workflows must continue working
- [ ] UI defaults must match backend defaults
- [ ] Old experiments without new fields should still load correctly

### Objective

Add UI controls for sequence length configuration, warning banners for grid search and CPU limitations, and comprehensive helper text/tooltips for Random Search parameters.

### Scope

1. Add sequence_length configurable TextField
2. Add grid search warning banner (>50 combinations, dismissible, reactive)
3. Add CPU-only warning banner (persistent, informational)
4. Add learning rate distribution selector (display-only, educational)
5. Add comprehensive English helper text for all Random Search parameters
6. Document limitation for non-editable parameters

### Files Modified

**Frontend:**
- `DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx`

**Tests:**
- `DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_lstm_phase3b.py`

### Implementation Details

Complete implementation code provided in full plan document.

---

### Success Criteria

- ✅ Sequence length TextField functional and persists to pipeline_config
- ✅ Grid warning banner appears when combinations > 50
- ✅ Grid warning is dismissible and resets on parameter change
- ✅ Real-time grid combination count displays
- ✅ CPU-only warning banner displays for LSTM (persistent)
- ✅ Learning rate distribution selector is display-only with clear explanation
- ✅ All Random Search parameters have comprehensive helper text (Spanish concise style maintained)
- ✅ Non-editable parameters documented with alternatives

---

### Phase 3B Completion Summary

**Completion Date:** 2025-11-13
**Implementation Time:** ~1.5 hours
**Test Results:** 9/9 automated tests passed ✅
**Manual Verification:** All scenarios completed successfully ✅

**Key Achievements:**
- ✅ CPU-only warning banner implemented (persistent info Alert)
- ✅ Grid search warning banner with real-time combination calculator (client-side)
- ✅ Dismissible warning that resets on parameter change
- ✅ Memory profiling Tooltip with comprehensive explanation
- ✅ Learning rate distribution display-only field in Random Search
- ✅ Client-side grid combination calculation with graceful fallback
- ✅ useEffect hook for reactive warning updates
- ✅ All Material-UI components properly imported and styled
- ✅ Backward compatibility maintained (no breaking changes)

**Files Modified:**
- `DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx`: Added warnings, tooltips, distribution field
- `tests/apiTimeSeries_tests/test_lstm_phase3b.py`: Created comprehensive test suite (9 tests)

**Pattern Consistency:**
- ✅ Schema version 1.1 unchanged (no new pipeline_config fields)
- ✅ UI-only changes (informational, not blocking)
- ✅ Helper text maintained in concise Spanish style
- ✅ Validation patterns followed (payload construction tests)
- ✅ Material-UI component patterns consistent with existing code

**No Deviations:** Implementation followed plan exactly as specified.

---

## Phase 3C: Resume/Cancel Training Features

**Status:** ❌ OUT OF SCOPE (Conflicts with reproducibility objective)
**Original Time Estimate:** 4-5 hours
**Prerequisites:** Phase 3A and 3B completed ✅
**Priority:** DEFERRED (Fundamental requirements conflict)
**Decision Date:** 2025-11-13

---

### Reproducibility Conflict Analysis

#### Project Objective (Primary Requirement)
"Allow users to acquire automated documentation and `pipeline_config.json` files that allow **any other user** running this project to **fully reproduce their experiment** (in terms of actual model training, measured by model configuration and validation metrics)."

#### Why Resume/Cancel Breaks Reproducibility

**Problem:** Resume functionality introduces non-deterministic user actions that cannot be reproduced from `pipeline_config.json` alone.

**Example Scenario:**
```json
// pipeline_config.json with resume artifacts
{
  "hyperparameter_search": {
    "strategy": "grid",
    "iterations_total": 10,
    "best_iteration": 8,
    "cancelled_at_iteration": 5,        // ← User action artifact
    "resumed_from_session_id": "abc123", // ← Operational detail
    "training_interrupted": true         // ← Non-deterministic flag
  }
}
```

**Reproduction Challenge:** Another user would need to:
1. Start grid search with 10 combinations
2. **Manually cancel at iteration 5** ← Non-algorithmic user action
3. Resume training
4. Hope RNG state produces same results

**Issues:**
- ❌ Cancellation point is user-decided, not algorithmic
- ❌ `pipeline_config.json` mixes training configuration with operational artifacts
- ❌ Workflow is: "configuration + user interaction" instead of pure "configuration"
- ❌ Non-reproducible across different users/environments

#### Clean Reproducibility Requirement

**Correct Approach:**
```json
// Clean pipeline_config.json (no resume artifacts)
{
  "hyperparameter_search": {
    "strategy": "grid",
    "iterations_total": 10,
    "best_iteration": 8,
    "grid_search_params": {
      "lstm_units_options": [[32], [64], [128]],
      "dropout_rate_options": [0.1, 0.2, 0.3],
      "learning_rate_options": [0.001, 0.01]
    }
  }
}
```

**Reproduction:** Another user runs exact same configuration → deterministic result
- ✅ No manual intervention required
- ✅ Pure algorithmic workflow
- ✅ `pipeline_config.json` is complete specification
- ✅ Reproducible across users/environments

---

### Decision: Mark Phase 3C as Out of Scope

**Rationale:**
1. **Maintains Scientific Reproducibility** - Primary project objective preserved
2. **Clean Configuration Files** - `pipeline_config.json` remains pure algorithmic spec
3. **Simplified Implementation** - Saves 6 hours of complex threading/state management
4. **Forces Good Practices** - Encourages reasonable experiment design (10-20 iterations for testing)

**Trade-offs Accepted:**
- ⚠️ No recovery from server crashes (must restart from scratch)
- ⚠️ Long grid searches (100+ iterations) become risky
- ⚠️ Users lose progress if training interrupted

**Mitigation Strategies:**
1. **Grid Search Warning** (Phase 3B ✅): Warns if combinations > 50
2. **Random Search Warning** (Phase 2B ✅): Warns if iterations > 200
3. **Conservative Defaults**: Recommend 10-20 iterations for testing, 50-100 for production
4. **Progress Logging** (Phase 2A/2B ✅): Log every 10 iterations for monitoring
5. **Future GPU Support**: Reduce training time from hours to minutes (future consideration)

---

### Reference Implementation (Not Implemented)

**Note:** This section preserved for future reference if reproducibility requirements change. Current decision: **DO NOT IMPLEMENT** due to conflict with project objectives.

#### High-Level Architecture (If Requirements Change)

**Components:**
- **LSTMResumeState Class**: In-memory dataclass storing training state
  - Fields: `best_model`, `best_params`, `best_val_loss`, `completed_iterations`, RNG state, MLflow run ID
  - Storage: Module-level dictionary `_lstm_resume_states = {}` in `services.py`

- **Cancellation System**: Threading-based flag system
  - `threading.Event` flags checked at grid/random search iteration boundaries
  - Graceful cancellation (completes current iteration, saves state)
  - MLflow tagging: `training_status="cancelled"`, `cancelled_at_iteration=N`

- **Backend Endpoints** (3 new views in `views.py`):
  - `POST /ts/cancel-training/` - Sets cancellation flag
  - `POST /ts/resume-training/` - Retrieves state, continues training
  - `GET /ts/training-status/{session_id}/` - Polls progress (frontend)

- **Frontend Components** (TSTrainCard.jsx):
  - Cancel button (visible during training)
  - Resume button (visible after cancellation)
  - Polling mechanism (2-second interval for status updates)
  - Session ID state management

**Key Files to Modify:**
- `services.py`: +150 lines (LSTMResumeState, state storage, flag management)
- `views.py`: +60 lines (3 new endpoints)
- `urls.py`: +2 routes
- `train.py`: +50 lines (cancellation checks in loops)
- `TSTrainCard.jsx`: +100 lines (UI components, polling)
- `test_lstm_phase3c.py`: +400 lines (comprehensive test suite)

**Estimated Implementation Time:** 6 hours
- Backend state management: 2 hours
- Endpoints + threading: 2 hours
- Frontend UI + polling: 1.5 hours
- Testing: 0.5 hours

**Main Technical Challenges:**
1. Thread-safe state management (multiple concurrent trainings)
2. RNG state preservation (numpy + TensorFlow random states)
3. MLflow run continuity across cancel/resume
4. Model serialization (64-256MB in-memory or disk storage)
5. Session timeout/cleanup (prevent memory leaks)

**Implementation Decisions (If Reconsidered):**
- Architecture: Threading (not Celery/background tasks)
- Communication: HTTP polling (not WebSocket)
- Storage: In-memory only (not disk persistence)
- Cancel behavior: Immediate at iteration boundary
- MLflow: Continue same run on resume
- Grid resume: Store full grid, skip completed iterations
- Session timeout: 24 hours
- Model storage: Disk-based (`{experiment_dir}/resume_checkpoint.keras`)

---

### Alternative: Cancel-Only (No Resume)

**If operational convenience needed without sacrificing reproducibility:**

Implement **graceful cancellation only** (no resume):
- Allow users to stop training early (saves compute)
- Mark cancelled runs as `training_status="cancelled"` in MLflow
- **Do not save to `pipeline_config.json`** (incomplete experiment)
- Cancelled runs are invalid for reproduction

**Benefits:**
- ✅ Maintains reproducibility (cancelled = incomplete/invalid)
- ✅ Operational convenience (stop bad configurations early)
- ✅ Simpler implementation (~2 hours vs 6 hours)
- ✅ No state management complexity

**Implementation Time:** 2 hours
- Cancellation flag system: 1 hour
- Frontend cancel button: 0.5 hours
- MLflow tagging: 0.5 hours

**Recommendation:** Consider this alternative if users frequently need to stop trainings early during experimentation phase.

---

### Success Criteria (If Implemented in Future)

- ✅ LSTMResumeState class functional
- ✅ Cancellation graceful at iteration boundaries
- ✅ Resume continues from exact iteration
- ✅ RNG state preserved (deterministic resume)
- ✅ MLflow tagging complete
- ✅ Frontend cancel/resume buttons working
- ✅ Automated tests pass (8-10 tests minimum)
- ⚠️ **Reproducibility documented as "Requires Manual Steps"** in pipeline_config

---

### Rollback Strategy (Not Applicable)

Since Phase 3C is not being implemented, no rollback needed.

If implemented in future and needs rollback:
```bash
git checkout HEAD -- apiTimeSeries/services.py
git checkout HEAD -- apiTimeSeries/views.py
git checkout HEAD -- apiTimeSeries/urls.py
git checkout HEAD -- apiTimeSeries/train.py
git checkout HEAD -- frontend/src/components/TSTrainCard.jsx
```

---


## Phase 3 Summary

**Total Implementation Time:** 2-2.5 hours (Reduced from original 6-8 hours)
- Phase 3A (Critical Fixes): ✅ 50 min (COMPLETED 2025-11-13)
- Phase 3B (UI Enhancements): ✅ 1.5 hours (COMPLETED 2025-11-13)
- Phase 3C (Resume/Cancel): ❌ OUT OF SCOPE (Reproducibility conflict)

**Git Commit Strategy:**
1. ✅ Phase 3A: `git commit -m "feat: Phase 3A - Fix pipeline_config.json completeness and add schema versioning"`
2. ✅ Phase 3B: `git commit -m "feat: Phase 3B - Add LSTM UI enhancements (warnings, tooltips, distribution field)"`
3. ❌ Phase 3C: Not implemented (out of scope)

**Key Achievements (Phase 3A + 3B):**
- ✅ Pipeline config is complete and reproducible (schema v1.1)
- ✅ User experience significantly improved with warning banners
- ✅ Memory profiling tooltip added for debugging
- ✅ Grid search combination calculator with real-time feedback
- ✅ CPU-only warning banner for user awareness
- ✅ Learning rate distribution display field (educational)
- ✅ Comprehensive test coverage (24 + 9 = 33 tests total)
- ✅ **Reproducibility maintained** (no resume artifacts in pipeline_config)
- ✅ Clear rollback strategies
- ✅ Backward compatibility maintained

**Phase 3C Decision:**
- ❌ Resume/Cancel functionality deferred due to reproducibility conflict
- ✅ Mitigation: Grid/Random search warnings prevent overly long trainings
- ✅ Best practice: Test with 10-20 iterations, production with 50-100 max
- 📋 Future consideration: Cancel-only (no resume) as lightweight alternative

---

## ⚠️ Pattern Consistency Checklist (Phase 3B → Phase 4)

**Note:** Phase 3C (Resume/Cancel) was marked out of scope. This checklist transitions from Phase 3B directly to Phase 4.

Before implementing Phase 4 (External Features Support), ensure the following patterns from Phase 3B are maintained:

#### Schema Version Patterns (Carry over from 3A/3B)
- [ ] **Critical:** Continue using schema version "1.1" (do NOT create v1.2 unless required)
- [ ] New UI parameters must be added to pipeline_config if they affect reproducibility
- [ ] Feature selection parameters must persist to pipeline_config.json
- [ ] No breaking changes to existing v1.1 schema structure

#### Validation Patterns (Carry over from 3A)
- [ ] If adding new required fields to pipeline_config, update `validate_pipeline_config_schema()`
- [ ] Use non-strict validation mode in production (warnings only)
- [ ] Use strict validation mode in tests
- [ ] Log validation results with clear messages

#### UI Data Flow Patterns (Carry over from 3B)
- [ ] UI controls → request payload → train_lstm_model() → pipeline_config.json
- [ ] Ensure feature selection from UI overrides defaults
- [ ] Warning banners should NOT modify training logic
- [ ] Warning banners are UI-only (informational, not blocking)

#### Feature Selection Patterns (NEW for Phase 4)
- [ ] Support univariate mode (empty feature list → use target only)
- [ ] Support multivariate mode (selected features → use in sequences)
- [ ] Validate features exist in dataset before training
- [ ] Log mode clearly: "Modo univariante" vs "Modo multivariante con N características"
- [ ] Shape validation: univariate (n, seq_len, 1) vs multivariate (n, seq_len, N)

#### Testing Patterns (Maintain from 3A/3B)
- [ ] Create test file: `test_lstm_phase4.py`
- [ ] Test univariate sequence creation
- [ ] Test multivariate sequence creation
- [ ] Test feature validation (missing features, invalid features)
- [ ] Test that pipeline_config includes feature selection parameters
- [ ] Follow existing test fixture patterns (configure_test_logging, synthetic_lstm_dataset)

#### Backend Integration Patterns (Maintain from all phases)
- [ ] Modify `create_sequences_for_lstm()` to handle empty feature_cols
- [ ] Auto-fallback: empty features → [target_col]
- [ ] Validation errors should be user-friendly
- [ ] Log feature count and mode for debugging

#### Backward Compatibility (Critical)
- [ ] Existing training workflows must continue working
- [ ] Default behavior: univariate (target only) if no features specified
- [ ] UI defaults must match backend defaults
- [ ] Schema version 1.1 unchanged (feature selection is optional)

#### Reproducibility (Maintain from Phase 3 decisions)
- [ ] Feature selection must be fully specified in pipeline_config
- [ ] Another user can reproduce experiment from config alone
- [ ] No user interaction artifacts (only algorithmic configuration)
- [ ] Clear documentation of which features were used

---

## Phase 4: External Features Support

**Status:** 🟡 READY TO IMPLEMENT (Comprehensive planning completed 2025-11-13)
**Time Estimate:** 3.5-4 hours (Revised after architecture analysis)
**Prerequisites:** Phase 1, 2A, 2B, 3A, 3B completed ✅
**Priority:** HIGH (Core functionality for univariate/multivariate LSTM)

---

### 🎯 Implementation Decisions (From Planning Session 2025-11-13)

This phase was comprehensively planned with detailed architecture analysis. All design decisions have been made and documented below.

#### Key Decisions Summary:
1. **State Management:** Create separate `lstmSelectedFeatures` state (clean separation, no cross-algorithm pollution)
2. **Default Behavior:** Empty selection = univariate mode (users must explicitly add features for multivariate)
3. **UI Placement:** After sequence length, before hyperparameters (logical data flow)
4. **Warning Severity:** `severity="info"` for univariate mode (it's a valid choice, not an error)
5. **Target in Features:** **Allow** selecting target variable as input feature (enables target history in sequences - very common for LSTM)
6. **Testing Strategy:** Both synthetic (automated) and realistic CSV (manual verification)
7. **Schema Version:** Keep "1.1" - backward compatible (feature selection is optional)
8. **Pipeline Config:** Add explicit `training_mode` and `n_input_features` fields (better reproducibility)
9. **Error Handling:** Both frontend and backend validation (defense in depth)
10. **Feature Population:** On CSV upload (consistent with XGBoost pattern)

#### Architecture Refactoring Approach: **Option C+ (Enhanced C)**

After comprehensive analysis of TSTrainCard.jsx (2000+ lines), we identified the root cause of confusion:
- **Problem:** Auto-selection of features when target is selected (lines 360-374) is too aggressive for LSTM
- **Current behavior:** When user selects a target, ALL remaining columns are automatically added to `inputFeatures`
- **Impact on LSTM:** Forces multivariate mode, users can't easily do univariate without manually unchecking everything

**Solution (Option C+):**
- Disable auto-selection ONLY for LSTM algorithm
- Keep auto-selection for ARIMA and XGBoost (backward compatible)
- Add UI indicators for univariate/multivariate mode
- Add `training_mode` field to payload for backend clarity
- Minimal refactoring, maximum clarity

---

### Overview

**Objective:** Enable univariate and multivariate LSTM training with explicit feature selection, following the refactored architecture pattern.

**Scope:**
- LSTM-specific feature state (separate from global `inputFeatures`)
- Disable auto-selection for LSTM (user control)
- Univariate mode support (empty features → target only)
- Multivariate mode with flexible feature selection (including target history)
- Mode indicator alerts (info-level for univariate, success-level for multivariate)
- Backend sequence creation with selected features
- Training mode metadata in pipeline_config.json

**Time Estimate:** 3.5-4 hours
- Backend: 30-45 minutes (~20 lines)
- Frontend: 1.5-2 hours (~90 lines)
- Testing: 1-1.5 hours (6 automated tests)
- Manual verification: 30 minutes (4 scenarios)

---

### Prerequisites

**Required Phases:**
- ✅ Phase 1: Core LSTM Implementation (sequence creation works)
- ✅ Phase 2A: Grid Search (memory patterns established)
- ✅ Phase 2B: Random Search (completed)
- ✅ Phase 3A: Schema v1.1 + Reproducibility (pipeline_config complete)
- ✅ Phase 3B: UI Enhancements (warning banner patterns established)

**Phase 3C (Resume/Cancel):** ❌ OUT OF SCOPE (not required for Phase 4)

---

### Files Modified

1. **Backend:** `DREAM-ML-backend/GEML/apiTimeSeries/train.py`
   - Modify `create_sequences_for_lstm` function (~10 lines added)
   - Enhance `train_lstm_model` logging (~7 lines added)
   - Update pipeline_config metadata (~3 lines modified)

2. **Frontend:** `DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx`
   - Add `lstmSelectedFeatures` state (1 line)
   - Modify `handleTargetChange` to disable auto-selection for LSTM (~5 lines)
   - Add `handleLstmFeatureToggle` handler (~7 lines)
   - Add LSTM feature selector UI with mode indicators (~70 lines)
   - Update payload construction (~7 lines)
   - Update validation logic (~3 lines)

3. **Tests:** `DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_lstm_phase4.py` (NEW)
   - 6 comprehensive tests (~180 lines)

**Total Code Changes:** ~290 lines across 3 files

---

### 📋 Detailed Implementation Plan

---

## Part 1: Backend Implementation (~30-45 min)

### File: `train.py`

#### A. Modify `create_sequences_for_lstm` (Lines 1649+)

**Current State:** Function exists but doesn't handle empty `feature_cols`

**Add univariate fallback logic at function start (after docstring, around line 1665):**

```python
# Handle univariate mode (empty feature_cols)
if not feature_cols or len(feature_cols) == 0:
    logger.info("Modo univariante detectado - usando solo variable objetivo")
    feature_cols = [target_col]
    univariate_mode = True
else:
    univariate_mode = False
    logger.info(f"Modo multivariante - usando {len(feature_cols)} características: {feature_cols}")
```

**Update final logging (around line 1700, replace existing log):**

```python
logger.info(
    f"Secuencias creadas exitosamente - X: {X.shape}, y: {y.shape} "
    f"({'univariante' if univariate_mode else 'multivariante con ' + str(len(feature_cols)) + ' features'})"
)
```

**Estimated addition:** 8 lines

---

#### B. Enhance `train_lstm_model` logging (Lines 2275+)

**Add mode detection after dataset loading (after line 2275):**

```python
# Log training mode explicitly
if len(input_features) == 1 and input_features[0] == target_variable:
    logger.info("Entrenando LSTM en modo univariante (solo variable objetivo)")
elif len(input_features) == 0:
    logger.warning("input_features vacío - forzando univariante con target")
    input_features = [target_variable]
else:
    logger.info(f"Entrenando LSTM en modo multivariante con {len(input_features)} características: {input_features}")
```

**Estimated addition:** 7 lines

---

#### C. Update `pipeline_config.json` (Lines 2847+)

**Modify lstm_metadata section to add training mode fields:**

```python
"lstm_metadata": {
    "sequence_length": sequence_length,
    "model_architecture": str(best_params.get("lstm_units")),
    "training_mode": "univariate" if len(input_features) == 1 else "multivariate",  # NEW
    "n_input_features": len(input_features),  # NEW
    "training_time_seconds": None,
    "early_stopped": None,
    "stopped_at_epoch": None,
    "total_params": best_model.count_params(),
    "cpu_only": True,
    "energy_kwh": energy_kwh,
    "carbon_emissions_kg": emissions_kg
}
```

**Estimated modification:** 2 lines added, 0 lines removed

**Total Backend Changes:** ~17 lines

---

## Part 2: Frontend Implementation (~1.5-2 hours)

### File: `TSTrainCard.jsx`

#### A. Add LSTM-Specific Feature State (After line 235)

```javascript
// LSTM-specific feature selection (separate from global inputFeatures)
const [lstmSelectedFeatures, setLstmSelectedFeatures] = useState([]);
```

**Location:** After line 235 (after `externalFeatures` state)
**Estimated addition:** 1 line

---

#### B. Modify Auto-Selection Logic (Lines 360-374)

**Replace `handleTargetChange` function:**

```javascript
const handleTargetChange = (column) => {
  if (targetVariable === column) {
    // Deselecting target
    setTargetVariable("");
    setInputFeatures([]);
    setLstmSelectedFeatures([]);  // Clear LSTM features too
  } else {
    // Selecting new target
    setTargetVariable(column);

    // Only auto-select for ARIMA and XGBoost (NOT LSTM)
    if (algorithm !== "lstm") {
      const remainingColumns = columns.filter((col) => col !== column && col !== dateColumnName);
      const newFeatures = [...new Set([...inputFeatures, ...remainingColumns])];
      setInputFeatures(newFeatures);
    }
    // For LSTM: Don't auto-select, let user explicitly choose features
  }
  validateSelections();
};
```

**Rationale:** Disabling auto-selection for LSTM gives users explicit control over univariate vs multivariate mode
**Estimated modification:** 5 lines added to existing function

---

#### C. Add LSTM Feature Toggle Handler (After line 468)

```javascript
// Handler for LSTM feature selection (allows selecting target variable)
const handleLstmFeatureToggle = (column) => {
  setLstmSelectedFeatures((prev) =>
    prev.includes(column)
      ? prev.filter((item) => item !== column)
      : [...prev, column]
  );
};
```

**Location:** After line 468 (after `handleExternalFeatureChange`)
**Estimated addition:** 7 lines

---

#### D. Add LSTM Feature Selector UI (After line 2084, before hyperparameters)

**Insert between "Early Stopping Patience" and "Optimization Method" sections:**

```javascript
{/* LSTM Feature Selector - Phase 4 */}
{algorithm === "lstm" && columns.length > 0 && (
  <Box sx={{ mt: 3, mb: 3, p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#f0f7ff" }}>
    <Typography variant="subtitle1" sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>
      Características de Entrada (Input Features)
    </Typography>

    <Typography variant="body2" sx={{ color: "#666", mb: 2, fontStyle: "italic" }}>
      Selecciona las características para crear secuencias LSTM.
      Deja vacío para modo univariante (solo target), o selecciona variables para modo multivariante.
    </Typography>

    {/* Feature checkboxes */}
    <FormGroup>
      {columns
        .filter((col) => col !== dateColumnName)  // Show all except date (including target - Option A)
        .map((column) => (
          <FormControlLabel
            key={column}
            control={
              <Checkbox
                checked={lstmSelectedFeatures.includes(column)}
                onChange={() => handleLstmFeatureToggle(column)}
                disabled={trainInProgress}
              />
            }
            label={
              <span>
                {column}
                {column === targetVariable && (
                  <span style={{ color: "#ff6b6b", fontWeight: "bold", marginLeft: "8px" }}>
                    (Target - Historia)
                  </span>
                )}
              </span>
            }
            sx={{ mb: 0.5 }}
          />
        ))}
    </FormGroup>

    {/* Mode indicator alert */}
    {lstmSelectedFeatures.length === 0 ? (
      <Alert severity="info" sx={{ mt: 2 }}>
        <strong>Modo Univariante:</strong> Solo se usará la variable objetivo ({targetVariable || "target"})
        para crear secuencias. Forma de entrada: (n, {sequenceLength}, 1)
      </Alert>
    ) : (
      <Alert severity="success" sx={{ mt: 2 }}>
        <strong>Modo Multivariante:</strong> Usando {lstmSelectedFeatures.length} características
        {lstmSelectedFeatures.includes(targetVariable) ? " (incluye historia del target)" : ""}.
        Forma de entrada: (n, {sequenceLength}, {lstmSelectedFeatures.length})
      </Alert>
    )}
  </Box>
)}
```

**Location:** After line 2084, before "Método de optimización" section
**Estimated addition:** 70 lines

**Key Features:**
- Shows ALL columns except date (including target - per Decision #5)
- Target variable labeled with "(Target - Historia)" indicator
- Info alert for univariate (empty selection)
- Success alert for multivariate (with feature count)
- Shows input shape dynamically

---

#### E. Update Payload Construction (Lines 793-797)

**Replace LSTM payload section:**

```javascript
// LSTM-specific parameters
if (algorithm === "lstm") {
  payload.sequence_length = sequenceLength;
  payload.early_stopping_patience = earlyStoppingPatience;
  payload.optimization_metric = "mse";

  // Override input_features with LSTM-specific selection
  payload.input_features = lstmSelectedFeatures;  // Use LSTM state instead of global
  payload.training_mode = lstmSelectedFeatures.length === 0 ? "univariate" : "multivariate";
}
```

**Rationale:** Sends LSTM-specific features instead of global `inputFeatures`, adds explicit training mode
**Estimated modification:** 3 lines modified, 2 lines added

---

#### F. Update Validation (Lines 1068+)

**Modify `isDisabled` condition to allow empty features for LSTM:**

```javascript
const isDisabled =
  trainInProgress ||
  !experimentDir ||
  !runId ||
  !flow.encodeDone ||
  flow.trainDone ||
  (algorithm !== "lstm" && !inputFeatures.length) ||  // LSTM can have empty features (univariate)
  !targetVariable ||
  !dateColumnName ||
  !modelName.trim() ||
  targetVariable === dateColumnName ||
  inputFeatures.includes(dateColumnName) ||
  !isXGBoostParamsValid() ||
  !isRandomSearchParamsValid() ||
  !isBayesianSearchParamsValid() ||
  !isLSTMParamsValid() ||
  validationWarnings.length > 0 ||
  !splitRatiosValid;
```

**Rationale:** Allows empty `inputFeatures` for LSTM (univariate mode is valid)
**Estimated modification:** 1 line modified

**Total Frontend Changes:** ~94 lines

---

## Part 3: Testing (~1-1.5 hours)

### File: `test_lstm_phase4.py` (NEW)

**Location:** `DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_lstm_phase4.py`

**Create comprehensive test suite:**

```python
"""
Phase 4 Tests: External Features Support for LSTM
Tests univariate/multivariate sequence creation and feature validation.
"""
import pytest
import numpy as np
import pandas as pd
from apiTimeSeries.train import create_sequences_for_lstm, train_lstm_model


def test_univariate_sequences_empty_features():
    """Test univariate mode with empty feature_cols (auto-fallback to target)"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'sales': np.sin(np.linspace(0, 10, 100))
    })
    df.set_index('date', inplace=True)

    # Empty features → should auto-use target
    X, y = create_sequences_for_lstm(df, [], 'sales', sequence_length=10)

    assert X.shape == (90, 10, 1), f"Expected (90, 10, 1), got {X.shape}"
    assert y.shape == (90,)
    print("✓ Test passed: Univariate mode (empty features)")


def test_univariate_sequences_target_only():
    """Test univariate mode with explicit target as feature (target history)"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'sales': np.sin(np.linspace(0, 10, 100)),
        'temp': np.random.rand(100)
    })
    df.set_index('date', inplace=True)

    # Explicit target-only
    X, y = create_sequences_for_lstm(df, ['sales'], 'sales', sequence_length=10)

    assert X.shape == (90, 10, 1), "Should use only sales column"
    assert y.shape == (90,)
    print("✓ Test passed: Univariate mode (explicit target)")


def test_multivariate_sequences_with_target():
    """Test multivariate with target + external features (Decision #5: Option A)"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'sales': np.sin(np.linspace(0, 10, 100)),
        'temp': np.random.rand(100),
        'humidity': np.random.rand(100)
    })
    df.set_index('date', inplace=True)

    # Target + 2 external features (using target history)
    X, y = create_sequences_for_lstm(
        df, ['sales', 'temp', 'humidity'], 'sales', sequence_length=10
    )

    assert X.shape == (90, 10, 3), f"Expected 3 features, got {X.shape[2]}"
    assert y.shape == (90,)
    print("✓ Test passed: Multivariate with target history")


def test_multivariate_sequences_without_target():
    """Test multivariate with external features only (no target history)"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'sales': np.sin(np.linspace(0, 10, 100)),
        'temp': np.random.rand(100),
        'humidity': np.random.rand(100)
    })
    df.set_index('date', inplace=True)

    # External features only (predict sales from temp + humidity, no sales history)
    X, y = create_sequences_for_lstm(
        df, ['temp', 'humidity'], 'sales', sequence_length=10
    )

    assert X.shape == (90, 10, 2), "Should use 2 external features"
    assert y.shape == (90,)
    # Verify sales column is NOT in the input sequences
    print("✓ Test passed: Multivariate without target history")


def test_feature_validation_missing():
    """Test error when feature doesn't exist in dataset"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'sales': np.sin(np.linspace(0, 10, 100))
    })
    df.set_index('date', inplace=True)

    with pytest.raises(ValueError, match="no encontrada en dataset"):
        create_sequences_for_lstm(df, ['nonexistent_feature'], 'sales', sequence_length=10)

    print("✓ Test passed: Missing feature validation")


def test_shape_validation():
    """Test that sequence shapes are correct for different modes"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
        'target': np.sin(np.linspace(0, 10, 100))
    })
    df.set_index('date', inplace=True)

    # Test different feature combinations
    test_cases = [
        ([], 1, "empty → univariate"),
        (['target'], 1, "target only"),
        (['feature1'], 1, "1 external feature"),
        (['target', 'feature1'], 2, "target + 1 feature"),
        (['feature1', 'feature2', 'feature3'], 3, "3 external features"),
        (['target', 'feature1', 'feature2', 'feature3'], 4, "target + 3 features"),
    ]

    for features, expected_n_features, description in test_cases:
        X, y = create_sequences_for_lstm(df, features, 'target', sequence_length=10)
        assert X.shape[2] == expected_n_features, f"Failed for {description}: expected {expected_n_features}, got {X.shape[2]}"
        print(f"✓ Shape test passed: {description} → {X.shape}")


# Run with: pytest test_lstm_phase4.py -v -s
```

**Estimated:** 6 tests, ~180 lines

**Test Coverage:**
- ✅ Univariate mode (empty features)
- ✅ Univariate mode (explicit target)
- ✅ Multivariate with target history (Decision #5)
- ✅ Multivariate without target history
- ✅ Feature validation (missing features)
- ✅ Shape validation (comprehensive scenarios)

---

## Part 4: Manual Verification (~30 min)

### Scenario A: Univariate LSTM (Empty Features - Default)
**Objective:** Verify that users can train univariate LSTM by not selecting any features

1. Upload CSV with `[Date, Sales, Temperature]`
2. Click "Cargar Variables"
3. Select algorithm: **LSTM (Deep Learning)**
4. Select target: **Sales**
5. **Verify:** No auto-selection happens (LSTM features remain empty)
6. **Do NOT select any features** in LSTM feature selector
7. **Verify:** Alert shows "Modo Univariante: Solo se usará la variable objetivo (Sales)"
8. **Verify:** Alert shows "Forma de entrada: (n, 10, 1)"
9. Set sequence_length: 10
10. Click "Entrenar Modelo"
11. **Check backend logs:** Should see "Modo univariante detectado - usando solo variable objetivo"
12. **Check training logs:** Should see "Creando secuencias univariantes con Sales"
13. **Verify:** Training completes successfully
14. **Check MLflow:** Verify sequences shape is (n, 10, 1)
15. **Check pipeline_config.json:** Verify `training_mode: "univariate"`, `n_input_features: 1`

---

### Scenario B: Univariate LSTM (Target History Explicit)
**Objective:** Verify that users can explicitly select target as an input feature

1. Same CSV
2. Select algorithm: LSTM
3. Select target: Sales
4. **Check only "Sales (Target - Historia)"** in feature selector
5. **Verify:** Alert shows "Modo Multivariante: Usando 1 características (incluye historia del target)"
6. **Verify:** Alert shows "Forma de entrada: (n, 10, 1)"
7. Train model
8. **Check logs:** "Modo multivariante con 1 características: ['Sales']"
9. **Verify:** Sequences shape (n, 10, 1) - same as Scenario A but logged differently
10. **Check pipeline_config.json:** Verify `training_mode: "multivariate"`, `n_input_features: 1`

**Expected Difference from Scenario A:**
- Backend logs show "multivariante" instead of "univariante"
- pipeline_config has `training_mode: "multivariate"` (because user explicitly selected features)
- Functionally identical (both use only Sales column)

---

### Scenario C: Multivariate LSTM (Target + External Features)
**Objective:** Verify that target can be included with external features

1. Same CSV
2. Select target: Sales
3. **Check both "Sales (Target - Historia)" AND "Temperature"**
4. **Verify:** Alert shows "Modo Multivariante: Usando 2 características (incluye historia del target)"
5. **Verify:** Alert shows "Forma de entrada: (n, 10, 2)"
6. Train model
7. **Check logs:** "Modo multivariante con 2 características: ['Sales', 'Temperature']"
8. **Verify:** Sequences shape (n, 10, 2)
9. **Check pipeline_config.json:** Verify `training_mode: "multivariate"`, `n_input_features: 2`

---

### Scenario D: Multivariate LSTM (External Only, No Target History)
**Objective:** Verify prediction using only external features (no target history)

1. Same CSV
2. Select target: Sales
3. **Check only "Temperature"** (NOT Sales)
4. **Verify:** Alert shows "Modo Multivariante: Usando 1 características"
5. **Verify:** Alert does NOT show "(incluye historia del target)"
6. **Verify:** Alert shows "Forma de entrada: (n, 10, 1)"
7. Train model
8. **Check logs:** "Modo multivariante con 1 características: ['Temperature']"
9. **Verify:** LSTM uses Temperature sequences to predict Sales (no Sales history)
10. **Verify:** Training completes (this tests that target exclusion works)

---

### Verification Checklist

**Backend:**
- [ ] Univariate fallback works (empty → target)
- [ ] Multivariate accepts selected features (including target)
- [ ] Feature validation (missing features error)
- [ ] Training mode logged correctly
- [ ] pipeline_config includes `training_mode` and `n_input_features`

**Frontend:**
- [ ] LSTM feature selector displays after sequence length
- [ ] Checkboxes include target variable (labeled with "Target - Historia")
- [ ] Mode indicator shows univariate (info alert) when empty
- [ ] Mode indicator shows multivariate (success alert) with feature count
- [ ] Auto-selection disabled for LSTM (manual verification: switch from ARIMA → LSTM)
- [ ] Empty features allowed (no validation error on Train button)

**Payload:**
- [ ] `input_features` uses `lstmSelectedFeatures` (not global `inputFeatures`)
- [ ] `training_mode` field added to payload

**Integration:**
- [ ] ARIMA auto-selection still works (backward compatibility)
- [ ] XGBoost auto-selection still works (backward compatibility)
- [ ] No breaking changes to existing algorithms

---

## Part 5: Documentation Updates

### Update This Implementation Plan

**Add completion summary after this section:**

```markdown
### Phase 4 Completion Summary

**Completion Date:** [Date]
**Implementation Time:** [Actual time] (Estimated: 3.5-4 hours)
**Test Results:** [X/6] automated tests passed ✅
**Manual Verification:** [All 4 scenarios completed] ✅

**Key Achievements:**
- ✅ LSTM-specific feature state (`lstmSelectedFeatures`) - clean separation
- ✅ Disabled auto-selection for LSTM (user control)
- ✅ Univariate mode support (empty features → target auto-fallback)
- ✅ Multivariate mode with flexible feature selection
- ✅ Allow target variable as input feature (Option A - enables target history)
- ✅ Training mode logging and pipeline_config metadata
- ✅ Frontend validation allows empty features for LSTM
- ✅ Clear UI indicators for univariate/multivariate mode
- ✅ Backward compatible with ARIMA/XGBoost auto-selection

**Files Modified:**
- `train.py`: Enhanced `create_sequences_for_lstm` and `train_lstm_model` (~17 lines)
- `TSTrainCard.jsx`: Added LSTM feature selector with mode indicators (~94 lines)
- `test_lstm_phase4.py`: Created comprehensive test suite (6 tests, ~180 lines)

**Architecture Decisions:**
- Followed Option C+ (Enhanced C) refactoring approach from architecture analysis
- Separate state for LSTM features (no cross-algorithm pollution)
- Auto-selection disabled for LSTM (explicit user control)
- Explicit training mode in payload and pipeline_config
- Schema version 1.1 unchanged (backward compatible)

**No Deviations:** Implementation followed plan exactly as specified.
```

---

## Success Criteria Checklist

### Backend Implementation:
- [ ] `create_sequences_for_lstm` handles empty `feature_cols` (univariate fallback)
- [ ] Multivariate accepts selected features (including target)
- [ ] Feature validation raises clear error for missing features
- [ ] Training mode logged correctly in console
- [ ] pipeline_config includes `training_mode` and `n_input_features` fields

### Frontend Implementation:
- [ ] LSTM feature selector displays after sequence length
- [ ] Checkboxes include ALL columns except date (including target - Decision #5)
- [ ] Target variable labeled with "(Target - Historia)"
- [ ] Mode indicator shows "Modo Univariante" (info) when empty
- [ ] Mode indicator shows "Modo Multivariante" (success) with feature count
- [ ] Auto-selection disabled for LSTM (verify by switching algorithms)
- [ ] Empty features allowed (Train button enabled)

### Payload & Integration:
- [ ] `input_features` uses `lstmSelectedFeatures` (not global state)
- [ ] `training_mode` field added to payload
- [ ] ARIMA auto-selection still works (backward compatibility)
- [ ] XGBoost auto-selection still works (backward compatibility)

### Testing:
- [ ] All 6 automated tests pass
- [ ] Scenario A (univariate empty) verified manually
- [ ] Scenario B (univariate explicit) verified manually
- [ ] Scenario C (multivariate with target) verified manually
- [ ] Scenario D (multivariate without target) verified manually

### Documentation:
- [ ] Implementation plan updated with completion summary
- [ ] All design decisions documented
- [ ] Architecture analysis preserved in plan

---

## Rollback Strategy

**If Phase 4 fails or needs to be reverted:**

### Backend Rollback:
```bash
cd DREAM-ML-backend/GEML
git diff apiTimeSeries/train.py
# Review changes, then:
git checkout HEAD -- apiTimeSeries/train.py
```

**Rollback impact:** Backend reverts to Phase 3B state
- `create_sequences_for_lstm` will not handle empty features (will fail or create wrong shape)
- Training mode not logged
- pipeline_config missing `training_mode` field

### Frontend Rollback:
```bash
cd DREAM-ML-frontend/frontend
git diff src/components/TSTrainCard.jsx
# Review changes, then:
git checkout HEAD -- src/components/TSTrainCard.jsx
```

**Rollback impact:** Frontend reverts to Phase 3B state
- No LSTM feature selector
- Auto-selection re-enabled for LSTM
- No mode indicators

### Test File Cleanup:
```bash
rm DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_lstm_phase4.py
```

### Verification After Rollback:
1. **ARIMA training:** Should work unchanged (auto-selection intact)
2. **XGBoost training:** Should work unchanged (auto-selection intact)
3. **LSTM training:** Will use global `inputFeatures` (auto-selected features from previous algorithm selection)

**Safe fallback:** Phase 1-3B functionality preserved, LSTM returns to previous behavior

---

## Notes for Implementation Session

### Context from Planning Session (2025-11-13)

**Key Insights from Architecture Analysis:**
1. **Root cause identified:** Auto-selection in `handleTargetChange` (lines 360-374) forces multivariate mode
2. **Current behavior:** When target selected, all remaining columns auto-added to `inputFeatures`
3. **Impact:** LSTM inherits this behavior, making univariate mode difficult
4. **Solution:** Disable auto-selection ONLY for LSTM (keep ARIMA/XGBoost unchanged)

**Design Decision Rationale:**
- **Option A (Target in features):** Enables LSTM target history - very common pattern in time series LSTM
- **Option C+ (Disable auto-select):** Minimal refactoring, maximum clarity, backward compatible
- **Separate state:** `lstmSelectedFeatures` prevents cross-algorithm pollution

**Important Implementation Notes:**
1. Frontend changes are isolated to LSTM sections (no impact on ARIMA/XGBoost)
2. Backend already supports empty `input_features` via Phase 1 default behavior
3. Schema version stays 1.1 (feature selection is optional metadata)
4. Testing covers all 4 use cases: univariate (empty/explicit), multivariate (with/without target)

### Ready to Implement!

**Implementation Order:**
1. Start with backend (quickest, ~30 min)
2. Test backend with manual Python script
3. Implement frontend (~1.5 hours)
4. Run automated tests
5. Manual verification (4 scenarios)
6. Update documentation

**Files to Open:**
- Backend: `/Users/tomasmanriquez/git/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/train.py`
- Frontend: `/Users/tomasmanriquez/git/dream-ml-c/DREAM-ML-frontend/frontend/src/components/TSTrainCard.jsx`
- Tests: `/Users/tomasmanriquez/git/dream-ml-c/DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_lstm_phase4.py` (new)

**All decisions made, comprehensive planning complete. Ready to execute in fresh session!**

---

## Phase 5: Multi-Step Seq2Seq

### Overview

**Objective:** Provide detailed architecture outline for multi-step sequence-to-sequence forecasting.

**Scope:** Conceptual outline only - implementation deferred to future plan.

**Reference Document:** See [2025-11-06_lstm-seq2seq-architecture-outline.md](./2025-11-06_lstm-seq2seq-architecture-outline.md)

**Key Topics Covered:**
- Encoder-decoder architecture
- Multi-step output generation
- Attention mechanism (optional)
- New hyperparameters (encoder_units, decoder_units)
- UI changes for forecast_horizon configuration
- Backward compatibility with single-step models

**Time Estimate:** N/A (deferred)

---

## Code Patterns Reference

### Docstring Format (Spanish, Google-style)

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Breve descripción en una línea.

    Descripción más detallada si es necesaria.
    Puede incluir múltiples párrafos.

    Args:
        param1: Descripción del parámetro 1
        param2: Descripción del parámetro 2

    Returns:
        Descripción del valor de retorno

    Raises:
        ErrorType: Cuándo se lanza este error

    Example:
        >>> result = function_name(value1, value2)
        >>> result
        expected_output
    """
```

### Logging Patterns

```python
# Info: Major operations, progress milestones
logger.info(f"Iniciando entrenamiento LSTM en run: {run_id}")

# Warning: Recoverable issues, missing optional data
logger.warning(f"sequence_length {value} excede el máximo. Usando {max_value}")

# Error: Failures requiring attention
logger.error(f"Error en entrenamiento: {e}", exc_info=True)
```

### Exception Handling

```python
# Validation errors
if invalid_condition:
    raise ValueError(
        f"Parámetro inválido: {param_name}. "
        f"Opciones válidas: {valid_options}. "
        f"Recibido: {actual_value}"
    )

# Execution errors
try:
    # risky operation
except SpecificException as e:
    logger.error(f"Contexto del error: {e}", exc_info=True)
    raise RuntimeError(f"Mensaje amigable para usuario: {e}") from e
```

### MLflow Logging

```python
# Batch parameters
mlflow.log_params({
    "model_type": "LSTM",
    "param1": value1,
    "param2": value2
})

# Metrics
mlflow.log_metric("val_rmse", rmse_value)
mlflow.log_metric("best_val_loss", best_loss)

# Artifacts with subfolders
mlflow.log_artifact(plot_path, "plots")

# Model with signature and metadata
mlflow.keras.log_model(
    model=model,
    artifact_path="lstm_model",
    signature=signature,
    registered_model_name=model_name,
    metadata={
        "dataset": dataset_name,
        "target": target_variable
    }
)
```

### Memory Management (LSTM-Specific)

```python
# After each iteration in grid/random search
del model
tf.keras.backend.clear_session()
gc.collect()
```

---

## Risk Mitigation

### Known Issues & Solutions

**1. Memory Leak in Iterative Search**
- **Issue:** Keras models not cleared between iterations
- **Solution:** `del model; tf.keras.backend.clear_session(); gc.collect()` after each iteration
- **Verification:** Memory increase < 500MB during 100 iterations

**2. Sequence Length Too Large**
- **Issue:** User requests sequence_length exceeding dataset size
- **Solution:** Auto-fallback with warning (WebSocket + response)
- **Verification:** Training continues with adjusted value

**3. Batch Size OOM**
- **Issue:** Batch size too large for available CPU memory
- **Solution:** Error message suggests fallback values (batch_size // 2)
- **Verification:** Clear error message with suggested fix

**4. Long Training Times**
- **Issue:** CPU-only training can take 30-60+ minutes
- **Solution:** Warnings in frontend, backend logs, and help text
- **Verification:** User informed before starting

**5. Resume State Lost on Server Restart**
- **Issue:** Session-only storage doesn't survive restarts
- **Solution:** Document limitation, future phase can add persistent storage
- **Verification:** Clear error if resume state not found

---

## Summary

This implementation plan provides a complete roadmap for adding LSTM training to the DREAM-ML platform, following established patterns from ARIMA and XGBoost implementations. Each phase builds incrementally, with clear verification steps and rollback strategies.

**Key Features:**
- ✅ Single-step LSTM forecasting (Phase 1)
- ✅ Grid and Random hyperparameter search (Phase 2)
- ✅ UI enhancements and resume capability (Phase 3)
- ✅ Univariate and multivariate support (Phase 4)
- 📋 Multi-step seq2seq architecture outline (Phase 5)

**Total Estimated Time:** 12-17 hours

**Next Steps:**
1. Review this plan with team
2. Begin Phase 1 implementation
3. Test thoroughly at each phase
4. Proceed to next phase after verification

For questions or clarifications, refer to the research document: [2025-11-06_lstm-training-implementation.md](../research/2025-11-06_lstm-training-implementation.md)
