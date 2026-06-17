# Manually found errors after implementing phase 4 of LSTM training implementation

Implementation plan document: `thoughts/shared/plans/2025-11-06_lstm-training-implementation.md` (phase 4)

All errors were found by manual user verification through the projects frontend.

## Manual verification steps
(Part 1 was skipped because it completed succesfully)

You now need to perform manual end-to-end testing to complete Phase 4. Here's what to do:

Part 2: LSTM End-to-End Testing (~75 min)
Dataset 1: Synthetic CSV (test_lstm_phase4.csv)
Scenario A: Univariate (Empty Features) - 3 Strategies (~25 min)
1. Upload test_lstm_phase4.csv
2. Click "Cargar Variables"
3. Select algorithm: LSTM
4. Select target: Sales
5. ✓ Verify: lstmSelectedFeatures empty (no auto-selection)
6. ✓ Verify: Alert shows "Modo Univariante: Solo se usará la variable objetivo (Sales)"
7. ✓ Verify: Alert shows "Forma de entrada: (n, 10, 1)"
8. Set sequence_length: 10, epochs: 10
9. Test Manual params: Train → ✓ Verify success
10. Test Grid search (lstm_units: [32,64], dropout: [0.1,0.2]): Train → ✓ Verify success
11. Test Random search (5 iterations): Train → ✓ Verify success
12. ✓ Check pipeline_config.json: All 3 should show training_mode: "univariate", n_input_features: 1
Scenario B: Multivariate (Target + External) - 3 Strategies (~25 min)
1. Same CSV
2. Select target: Sales
3. Check features: Sales (Target - Historia) ✓ + Temperature ✓
4. ✓ Verify: Alert shows "Modo Multivariante: Usando 2 características (incluye historia del target)"
5. ✓ Verify: Alert shows "Forma de entrada: (n, 10, 2)"
6. Test Manual/Grid/Random (same configs as Scenario A)
7. ✓ Check pipeline_config.json: All 3 should show training_mode: "multivariate", n_input_features: 2
Dataset 2: Real CSV (AirlinePassengers.csv)
Scenario C: Univariate (Explicit Target) (~10 min)
1. Upload AirlinePassengers.csv
2. Select target: Passengers
3. Check ONLY: Passengers (Target - Historia) ✓
4. ✓ Verify: Alert shows "Modo Multivariante: Usando 1 características (incluye historia del target)"
5. Set sequence_length: 10, epochs: 10
6. Train with Manual params
7. ✓ Check logs: Should show "Modo multivariante con 1 características: ['Passengers']"
8. ✓ Check pipeline_config.json: training_mode: "multivariate", n_input_features: 1
Scenario D: Multivariate (External Only) (~15 min)
1. Upload test_lstm_phase4.csv again
2. Select target: Sales
3. Check ONLY: Temperature ✓ (NOT Sales)
4. ✓ Verify: Alert shows "Modo Multivariante: Usando 1 características"
5. ✓ Verify: Alert does NOT mention "(incluye historia del target)"
6. Train with Manual params
7. ✓ Check logs: Should show "Modo multivariante con 1 características: ['Temperature']"
8. ✓ Verify: LSTM predicts Sales from Temperature only (no Sales history)
9. ✓ Verify: Training completes successfully

Manual Verification Checklist
Please test the scenarios above and report back with: Backend:
*  Univariate fallback works (empty → target)
*  Multivariate accepts selected features (including target)
*  Training mode logged correctly in console
*  pipeline_config includes training_mode and n_input_features
Frontend:
*  LSTM feature selector displays correctly
*  Mode indicators show univariate/multivariate correctly
*  Auto-selection disabled for LSTM (verify by switching from ARIMA)
*  Empty features allowed (Train button enabled)
Integration:
*  ARIMA auto-selection still works ✓
*  XGBoost auto-selection still works ✓
*  All 4 LSTM scenarios complete successfully
Once you've completed the manual testing, let me know the results and I'll update the implementation plan documentation!

## Error 1
While performing part 2, Scenario A, step 9.
Given the following formData:
```
data: {"model_name":"lstm_phase4_1","input_features":["Temperature","Temperature_lag_1"],"target_variable":"Sales","date_col_name":"Date","experiment_dir":"/workspaces/dream-ml-c/experimentos/Exp_20251114_153443_eb8072d6","split_ratios":{"train":0.7,"val":0.15,"test":0.15},"run_id":"bf5cd07b62e74d33b76c064da3e27555","algorithm":"lstm","params":{"lstm_units":[64],"dropout_rate":0.2,"recurrent_dropout_rate":0.2,"learning_rate":0.001,"batch_size":32,"epochs":10},"use_grid_search":false,"use_random_search":false,"problem_type":"ts_forecasting","forecast_horizon":3,"sequence_length":10,"early_stopping_patience":20,"optimization_metric":"mse","training_mode":"multivariate","hyperparameter_search_strategy":"none","feature_config":{"lag_periods":[1,2,3,4,5],"rolling_windows":[3,7,14],"external_features":[]}}
```
, I got the following error in the backend:
```python
[WARNING] apiTimeSeries.train - ⚠️ Entrenamiento LSTM usa CPU solamente (sin soporte GPU en esta versión). Tiempo de entrenamiento esperado: 30-60 minutos para 100 épocas. Considere reducir 'epochs' si el tiempo es excesivo.
2025/11/14 15:37:57 INFO mlflow.system_metrics.system_metrics_monitor: Stopping system metrics monitoring...
2025/11/14 15:37:57 INFO mlflow.system_metrics.system_metrics_monitor: Successfully terminated system metrics monitoring!
[WARNING] apiTimeSeries.train - Run activa de MLflow detectada y finalizada
[INFO] apiTimeSeries.train - Iniciando entrenamiento LSTM en run: 626db2a2513f4aca9e0ba0b1f080b8a0
[INFO] apiTimeSeries.train - Cargando y validando dataset...
[INFO] apiTimeSeries.train - Dataset cargado: 100 muestras, características: ['Temperature', 'Temperature_lag_1']
[INFO] apiTimeSeries.train - Entrenando LSTM en modo multivariate con 2 características: ['Temperature', 'Temperature_lag_1']
[INFO] apiTimeSeries.train - Creando secuencias LSTM (sequence_length=10)...
[INFO] apiTimeSeries.train - Modo multivariante - usando 2 características: ['Temperature', 'Temperature_lag_1']
[INFO] apiTimeSeries.train - Secuencias creadas exitosamente - X: (88, 10, 2), y: (88,) (multivariante con 2 features)
[INFO] apiTimeSeries.train - Dividiendo dataset en train/val/test...
[INFO] apiTimeSeries.train - División temporal completada - Train: 61 (70.0%), Val: 13 (15.0%), Test: 14 (15.0%)
[codecarbon WARNING @ 15:37:57] No CPU tracking mode found. Falling back on CPU constant mode. 
 Linux OS detected: Please ensure RAPL files exist at \sys\class\powercap\intel-rapl to measure CPU

[codecarbon WARNING @ 15:37:57] We saw that you have a - but we don't know it. Please contact us.
[INFO] apiTimeSeries.train - Entrenando con parámetros manuales: {'lstm_units': [64], 'dropout_rate': 0.2, 'recurrent_dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 32, 'epochs': 100}
[INFO] apiTimeSeries.train - Construyendo modelo LSTM - Arquitectura: [64], Dropout: 0.2, Recurrent Dropout: 0.2, Learning Rate: 0.001
[INFO] apiTimeSeries.train - Modelo LSTM compilado exitosamente - Total de parámetros: 17,217
[INFO] apiTimeSeries.train - Directorio de checkpoints temporales: /workspaces/dream-ml-c/experimentos/Exp_20251114_153443_eb8072d6/temp_checkpoints
[INFO] apiTimeSeries.train - Callbacks configurados - EarlyStopping patience: 20, ReduceLR patience: 10
[INFO] apiTimeSeries.train - Iniciando entrenamiento del modelo...
Epoch 1/100
2025-11-14 15:38:01.408973: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_15}}
1/2 ━━━━━━━━━━━━━━━━━━━━ 0s 762ms/step - loss: nan - mae: nan - mse: nan[WARNING] absl - You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
2/2 ━━━━━━━━━━━━━━━━━━━━ 1s 155ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 0.0010
Epoch 2/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 0.0010
Epoch 3/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 0.0010
Epoch 4/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 0.0010
Epoch 5/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 0.0010
Epoch 6/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 0.0010
Epoch 7/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 0.0010
Epoch 8/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 0.0010
Epoch 9/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 0.0010
Epoch 10/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 0.0010
Epoch 11/100
1/2 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: nan - mae: nan - mse: nan
Epoch 11: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 0.0010
Epoch 12/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 5.0000e-04
Epoch 13/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 5.0000e-04
Epoch 14/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 5.0000e-04
Epoch 15/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 5.0000e-04
Epoch 16/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 5.0000e-04
Epoch 17/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 5.0000e-04
Epoch 18/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 5.0000e-04
Epoch 19/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 5.0000e-04
Epoch 20/100
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 5.0000e-04
Epoch 21/100
1/2 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: nan - mae: nan - mse: nan
Epoch 21: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - loss: nan - mae: nan - mse: nan - val_loss: nan - val_mae: nan - val_mse: nan - learning_rate: 5.0000e-04
Epoch 21: early stopping
Restoring model weights from the end of the best epoch: 1.
[INFO] apiTimeSeries.train - Entrenamiento completado - Mejor val_loss: nan en época 1
[INFO] apiTimeSeries.train - Consumo de energía: 0.0000 kWh, Emisiones de carbono: 0.000006 kg CO2
[INFO] apiTimeSeries.train - Evaluando modelo en conjunto de validación...
[INFO] apiTimeSeries.train - Generando predicciones para conjunto val...
[ERROR] apiTimeSeries.train - Error en entrenamiento LSTM: Input contains NaN.
Traceback (most recent call last):
  File "/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/train.py", line 2801, in train_lstm_model
    val_metrics, val_artifacts = evaluate_lstm_model(
                                 ^^^^^^^^^^^^^^^^^^^^
  File "/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/train.py", line 2030, in evaluate_lstm_model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/utils/_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/metrics/_regression.py", line 565, in mean_squared_error
    _check_reg_targets_with_floating_dtype(
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/metrics/_regression.py", line 198, in _check_reg_targets_with_floating_dtype
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
                                          ^^^^^^^^^^^^^^^^^^^
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/metrics/_regression.py", line 106, in _check_reg_targets
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/utils/validation.py", line 1107, in check_array
    _assert_all_finite(
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/utils/validation.py", line 120, in _assert_all_finite
    _assert_all_finite_element_wise(
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/utils/validation.py", line 169, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input contains NaN.
[ERROR] apiTimeSeries.services - Error durante el entrenamiento: Error en entrenamiento LSTM: Input contains NaN.
Traceback (most recent call last):
  File "/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/train.py", line 2801, in train_lstm_model
    val_metrics, val_artifacts = evaluate_lstm_model(
                                 ^^^^^^^^^^^^^^^^^^^^
  File "/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/train.py", line 2030, in evaluate_lstm_model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/utils/_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/metrics/_regression.py", line 565, in mean_squared_error
    _check_reg_targets_with_floating_dtype(
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/metrics/_regression.py", line 198, in _check_reg_targets_with_floating_dtype
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
                                          ^^^^^^^^^^^^^^^^^^^
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/metrics/_regression.py", line 106, in _check_reg_targets
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/utils/validation.py", line 1107, in check_array
    _assert_all_finite(
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/utils/validation.py", line 120, in _assert_all_finite
    _assert_all_finite_element_wise(
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/utils/validation.py", line 169, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input contains NaN.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/services.py", line 1035, in train_model_logic
    result = train_lstm_model(
             ^^^^^^^^^^^^^^^^^
  File "/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/train.py", line 3041, in train_lstm_model
    raise RuntimeError(f"Error en entrenamiento LSTM: {e}") from e
RuntimeError: Error en entrenamiento LSTM: Input contains NaN.
[ERROR] apiTimeSeries.views - Error de ejecución: Error en el proceso de entrenamiento: Error en entrenamiento LSTM: Input contains NaN.
Traceback (most recent call last):
  File "/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/train.py", line 2801, in train_lstm_model
    val_metrics, val_artifacts = evaluate_lstm_model(
                                 ^^^^^^^^^^^^^^^^^^^^
  File "/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/train.py", line 2030, in evaluate_lstm_model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/utils/_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/metrics/_regression.py", line 565, in mean_squared_error
    _check_reg_targets_with_floating_dtype(
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/metrics/_regression.py", line 198, in _check_reg_targets_with_floating_dtype
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
                                          ^^^^^^^^^^^^^^^^^^^
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/metrics/_regression.py", line 106, in _check_reg_targets
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/utils/validation.py", line 1107, in check_array
    _assert_all_finite(
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/utils/validation.py", line 120, in _assert_all_finite
    _assert_all_finite_element_wise(
  File "/home/vscode/.local/lib/python3.12/site-packages/sklearn/utils/validation.py", line 169, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input contains NaN.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/services.py", line 1035, in train_model_logic
    result = train_lstm_model(
             ^^^^^^^^^^^^^^^^^
  File "/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/train.py", line 3041, in train_lstm_model
    raise RuntimeError(f"Error en entrenamiento LSTM: {e}") from e
RuntimeError: Error en entrenamiento LSTM: Input contains NaN.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/views.py", line 413, in train_model
    result = trainModelService.train_model_logic(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspaces/dream-ml-c/DREAM-ML-backend/GEML/apiTimeSeries/services.py", line 1090, in train_model_logic
    raise RuntimeError(f"Error en el proceso de entrenamiento: {str(e)}")
RuntimeError: Error en el proceso de entrenamiento: Error en entrenamiento LSTM: Input contains NaN.
Internal Server Error: /api/ts/train-model/
[ERROR] django.request - Internal Server Error: /api/ts/train-model/
[14/Nov/2025 15:38:03] "POST /api/ts/train-model/ HTTP/1.1" 500 171
```


## Error 2 + 3
While performing Part 2, Scenario A, step 10:

With the following dataset: `experimentos/Exp_20251114_154447_4a86a4a1/processed/processed_train_processed_eda_test_lstm_phase4.csv`,
And the following variables selection:
* Características de Entrada (Input Features): Temperature
* Características Externas (opcionales): None
* Variable de Salida (Target): Sales
* Variables de Entrada (Features): Temperature (o None)
* Columna de Fecha: Date
, the user form would not allow me to click the 'ENTRENAR MODELO' button. There was a warning ('ADVERTENCIAS DE VALIDACIÓN') that said: "Debes seleccionar al menos 1 variable de entrada" when 'Temperature' was selected as a Feature variable. When this wasnt selected, this warning disappeared but I still couldnt finish training.

Error 3: the same situation happen when performing Scenario A step 11 (Random Search instead of Grid Search)

For both 2 and 3: I tested again with "Características de Entrada (Input Features): None" but still got the same error

# Errors for Scenario C + D

The training could not be finished, because with only 2 columns I could select all "Variable de Salida (Target)" and "Columna de Fecha" and "Variables de Entrada (Features)". Similar situation to Errors 2 + 3.

---

## Related Analysis Documents

For comprehensive root cause analysis of these errors, see:

- **Error 1 (NaN values)**: [Backend Analysis - Feature Engineering](2025-11-14_backend-analysis.md) (lines 166-179)
  - Root cause: Lag-of-lag feature creation in `data_encoding_utils.py:139`
  - Explains NaN cascade mechanism from double-shift operations

- **Errors 2-3 (validation blocking)**: [Frontend Analysis - Bug #2](2025-11-14_frontend-analysis.md) (lines 410-447)
  - Root cause: Frontend validation logic blocking empty LSTM features
  - User experience walkthrough and state management issues

- **All bugs - Proposed fixes**: [Recommendations](2025-11-14_recommendations.md)
  - Option 1A: Filter lagged features from LSTM UI
  - Option 1C: Backend auto-detect lagged features (defense-in-depth)
  - Bug #2 fix: Update validation logic to allow empty LSTM features

- **Validation rules**: [Validation Logic](2025-11-14_validation-logic.md)
  - Complete validation rule analysis for all algorithms
  - Frontend-backend comparison tables
  - Post-encoding validation gaps

- **Feature selection research**: [Variable Selection Research](2025-11-14_variable-selection-research.md)
  - Industry best practices (AWS, Azure, Google Cloud)
  - LSTM univariate vs multivariate modes
  - Critical findings on current implementation issues