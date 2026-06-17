/*
 * Copyright (C) 2025 Leonardo Espinoza Ortiz <leonardo.espinoza.o@usach.cl>
 *
 * This file is part of DREAM ML.
 *
 * DREAM ML is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DREAM ML is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DREAM ML. If not, see <https://www.gnu.org/licenses/>.
 */


import React, { useState, useContext, useEffect } from "react";
import InfoModal from './InfoModal';
import ValidationSummary from './ValidationSummary';
import ProgressBar from './ProgressBar';
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  CircularProgress,
  Collapse,
  FormControl,
  FormControlLabel,
  FormGroup,
  FormHelperText,
  FormLabel,
  Grid,
  IconButton,
  InputLabel,
  MenuItem,
  Radio,
  RadioGroup,
  Select,
  Slider,
  Switch,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Autorenew as AutorenewIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
} from "@mui/icons-material";
import axios from "../utils/axiosConfig";
import { AppContext } from "../AppContext";
import { variableSelectionStyles, infoContent, helperTextStrings } from "../styles/variableSelectionStyles";

const TSTrainCard = () => {
  const { 
    experimentDir, 
    flow, 
    markStepDone, 
    runId, 
    trainInProgress, 
    setTrainInProgress,
    trainStatus,
    setTrainStatus,
  } = useContext(AppContext);

  const [csvFile, setCsvFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [inputFeatures, setInputFeatures] = useState([]);
  const [targetVariable, setTargetVariable] = useState("");
  const [dateColumnName, setDateColumnName] = useState("");

  const [algorithm, setAlgorithm] = useState("arima");
  const problemType = "ts_forecasting"; // Para ARIMA
  const [modelName, setModelName] = useState("");
  const [splitRatios, setSplitRatios] = useState({
    train: 0.7,
    val: 0.15,
    test: 0.15,
  });

  // Split ratio validation state
  const [splitRatiosValid, setSplitRatiosValid] = useState(true);
  const [validationMessage, setValidationMessage] = useState("");
  const [showValidation, setShowValidation] = useState(false);
  const [validationTimer, setValidationTimer] = useState(null);

  // Input display values (raw strings for free typing)
  const [inputDisplayValues, setInputDisplayValues] = useState({
    train: "0,70",
    val: "0,15",
    test: "0,15"
  });

  // ARIMA specific parameters
  const [forecastHorizon, setForecastHorizon] = useState(12);
  const [enableSeasonalParams, setEnableSeasonalParams] = useState(false);
  const [arimaParams, setArimaParams] = useState({
    p: "1",
    d: "1",
    q: "1",
    seasonal_P: "1",
    seasonal_D: "1",
    seasonal_Q: "1",
    seasonal_s: "12",
    trend: "n",
    enforce_stationarity: "True",
    enforce_invertibility: "True",
  });

  // Optimization method: "manual", "grid", "random"
  const [optimizationMethod, setOptimizationMethod] = useState("manual");

  // Legacy support - derive from optimizationMethod
  const useGridSearch = optimizationMethod === "grid";
  const useRandomSearch = optimizationMethod === "random";

  // Info modal states
  const [showTargetInfo, setShowTargetInfo] = useState(false);
  const [showFeatureInfo, setShowFeatureInfo] = useState(false);

  // Validation warnings
  const [validationWarnings, setValidationWarnings] = useState([]);

  // Random Search parameters
  const [nRandomIterations, setNRandomIterations] = useState(100);

  // Bayesian Search UI state
  const [showBayesianAdvanced, setShowBayesianAdvanced] = useState(false);

  // ARIMA Random Search parameter ranges
  const [arimaRandomRanges, setArimaRandomRanges] = useState({
    p_range: [0, 4],
    d_range: [0, 3],
    q_range: [0, 4],
    seasonal_P_range: [0, 3],
    seasonal_D_range: [0, 3],
    seasonal_Q_range: [0, 3],
  });
  // ARIMA Random Search parameters
  const [arimaGridSearchParams, setArimaGridSearchParams] = useState({
    p: [],
    d: [],
    q: [],
    P: [],
    D: [],
    Q: [],
    s: [],
    trend: ["n"],
    enforce_stationarity: [true],
    enforce_invertibility: [true],
  })
  const [optimizationMetric, setOptimizationMetric] = useState("val_rmse")
  const [useSeasonalParamsHT, setUseSeasonalParamsHT] = useState(false)

  //XGBoost Manual parameters
  const [xgBoostParams, setXgboostParams] = useState({
    "n_estimators": 0,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 0.3,
    "gamma": 2.0,
    "min_child_weight": 1
  });
  //XGBoost Manual parameters
  const [xgBoostGridParams, setXgboostGridParams] = useState({
    "n_estimators": [],
    "max_depth": [],
    "learning_rate": [],
    "subsample": [],
    "colsample_bytree": [],
    "gamma": [],
    "min_child_weight": []
  })

  // XGBoost Random Search parameter ranges
  const [xgboostRandomRanges, setXgboostRandomRanges] = useState({
    n_estimators_range: [50, 1000],
    max_depth_range: [3, 10],
    learning_rate_range: [0.01, 0.3],
    subsample_range: [0.5, 1.0],
    colsample_bytree_range: [0.5, 1.0],
    gamma_range: [0.0, 5.0],
    min_child_weight_range: [1, 10],
    reg_alpha_range: [0.0, 1.0],
    reg_lambda_range: [0.0, 1.0],
  });


  // Bayesian Search advanced configuration
  const [bayesianConfig, setBayesianConfig] = useState({
    n_trials: 50,                       // Number of optimization trials
    n_initial_points: 10,
    acq_func: "ei",                     // Lowercase for backend compatibility
    max_memory_mb: null,
    timeout_seconds: null,
    convergence_tolerance: null,        // Phase 7: Minimum improvement threshold (backend default: 0.001)
    convergence_patience: null          // Phase 7: Consecutive trials without improvement (backend default: 5)
  });

  // Bayesian Search parameter ranges toggle
  const [showBayesianParamRanges, setShowBayesianParamRanges] = useState(false);

  // ARIMA Bayesian Search parameter ranges
  const [arimaBayesianRanges, setArimaBayesianRanges] = useState({
    p: { min: 0, max: 3 },
    d: { min: 0, max: 1 },
    q: { min: 0, max: 3 },
    P: { min: 0, max: 2 },
    D: { min: 0, max: 1 },
    Q: { min: 0, max: 2 },
    s: { min: 2, max: 24 },
    trend: { choices: ["n", "c", "t", "ct"] }
  });

  // XGBoost Bayesian Search parameter ranges
  const [xgboostBayesianRanges, setXgboostBayesianRanges] = useState({
    n_estimators: { min: 50, max: 500 },
    max_depth: { min: 3, max: 10 },
    learning_rate: { min: 0.001, max: 0.1, log: true },
    subsample: { min: 0.5, max: 1.0 },
    colsample_bytree: { min: 0.5, max: 1.0 },
    gamma: { min: 0.0, max: 1.0 },
    min_child_weight: { min: 1, max: 10 }
  });

  // LSTM specific parameters
  const [sequenceLength, setSequenceLength] = useState(10);
  const [earlyStoppingPatience, setEarlyStoppingPatience] = useState(20);

  // LSTM-specific feature selection (separate from global inputFeatures) - Phase 4
  const [lstmSelectedFeatures, setLstmSelectedFeatures] = useState([]);

  // LSTM manual training parameters
  const [lstmManualParams, setLstmManualParams] = useState({
    lstm_units: "[64]",  // String for display, will parse to array
    dropout_rate: "0.2",
    recurrent_dropout_rate: "0.2",
    learning_rate: "0.001",
    batch_size: "32",
    epochs: "100"
  });

  // LSTM Random Search parameter ranges
  const [lstmRandomRanges, setLstmRandomRanges] = useState({
    lstm_units_options: ["[32]", "[64]", "[128]", "[64,32]", "[128,64]"],
    dropout_rate_range: [0.0, 0.5],
    recurrent_dropout_rate_range: [0.0, 0.5],
    learning_rate_range: [0.0001, 0.01],
    batch_size_options: [16, 32, 64],
    epochs_range: [50, 300]
  });

  // LSTM Bayesian Search parameter ranges (Phase 9: Configurable Parameter Ranges)
  const [lstmBayesianRanges, setLstmBayesianRanges] = useState({
    lstm_units: { choices: [32, 64, 128] },
    dropout_rate: { min: 0.1, max: 0.4 },
    recurrent_dropout_rate: { min: 0.1, max: 0.4 },  // UI only, not sent to backend
    learning_rate: { min: 0.0001, max: 0.01, log: true },
    batch_size: { choices: [16, 32, 64] },
    epochs: { min: 30, max: 100 },
    time_steps: { min: 5, max: 30 }
  });

  // LSTM Grid Search parameter options (Phase 2A)
  const [lstmGridOptions, setLstmGridOptions] = useState({
    lstm_units_options: "[64], [128]",
    dropout_rate_options: "0.2, 0.3",
    recurrent_dropout_rate_options: "0.2",
    learning_rate_options: "0.001, 0.01",
    batch_size_options: "32",
    epochs_options: "100"
  });

  const [enableMemoryProfiling, setEnableMemoryProfiling] = useState(false);
  const [gridWarningThreshold, setGridWarningThreshold] = useState(50);
  const [gridWarningDismissed, setGridWarningDismissed] = useState(false);
  const [gridCombinationsCount, setGridCombinationsCount] = useState(0);

  // PatchTSMixer channel selection (checkboxes for all variables)
  const [patchTSMixerChannels, setPatchTSMixerChannels] = useState([]);

  // PatchTSMixer essential hyperparameters (10 params)
  const [patchTSMixerParams, setPatchTSMixerParams] = useState({
    context_length: 512,
    patch_length: 8,
    patch_stride: 8,  // Should equal patch_length for non-overlapping
    d_model: 32,
    num_layers: 8,
    dropout: 0.2,
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 100,
    early_stopping_patience: 10,
  });

  // PatchTSMixer advanced hyperparameters (9 params, collapsed by default)
  const [patchTSMixerAdvanced, setPatchTSMixerAdvanced] = useState({
    expansion_factor: 2,
    head_dropout: 0.2,
    mode: "common_channel",        // Options: common_channel, mix_channel
    gated_attn: true,              // Boolean
    self_attn: false,              // Boolean
    scaling: "std",                // Options: std, mean, none
    norm_mlp: "LayerNorm",         // Options: LayerNorm, BatchNorm, none
    loss: "mse",                   // Options: mse, mae
    distribution_output: "student_t", // Options: student_t, normal, negative_binomial, none
  });

  // PatchTSMixer UI state
  const [showPatchTSMixerAdvanced, setShowPatchTSMixerAdvanced] = useState(false);
  const [patchTSMixerPreset, setPatchTSMixerPreset] = useState("");

  // PatchTSMixer preset loading function
  const loadPatchTSMixerPreset = (presetName) => {
    const presets = {
      small: {
        context_length: 512,
        patch_length: 16,
        patch_stride: 16,
        d_model: 16,
        num_layers: 6,
        dropout: 0.2,
        learning_rate: 0.001,
        batch_size: 32,
        epochs: 100,
        early_stopping_patience: 10,
      },
      medium: {
        context_length: 512,
        patch_length: 8,
        patch_stride: 8,
        d_model: 32,
        num_layers: 8,
        dropout: 0.2,
        learning_rate: 0.001,
        batch_size: 32,
        epochs: 100,
        early_stopping_patience: 10,
      },
      large: {
        context_length: 512,
        patch_length: 8,
        patch_stride: 8,
        d_model: 64,
        num_layers: 12,
        dropout: 0.3,
        learning_rate: 0.0005,
        batch_size: 16,
        epochs: 150,
        early_stopping_patience: 15,
      },
    };

    if (presets[presetName]) {
      setPatchTSMixerParams(presets[presetName]);
      setPatchTSMixerPreset(presetName);
      console.log(`[PatchTSMixer] Loaded preset "${presetName}":`, presets[presetName]);
    }
  };

  // DEBUG: Expose preset function to window for console testing (Step 5)
  // TODO: Remove after Phase 7a verification
  if (typeof window !== 'undefined') {
    window.loadPatchTSMixerPreset = loadPatchTSMixerPreset;
  }

  // Eliminamos el estado local de status para usar el global trainStatus

  // Manejo del archivo CSV
  const handleFileChange = (event) => {
    setCsvFile(event.target.files[0]);
    setColumns([]);
    setInputFeatures([]);
    setTargetVariable("");
    setDateColumnName("");
    setTrainStatus("📂 Archivo CSV seleccionado.");
  };

  // Función para cargar las columnas del CSV usando el endpoint /analyze-csv/
  const loadColumns = async () => {
    if (!csvFile) {
      setTrainStatus("⚠️ Por favor, selecciona un archivo CSV primero.");
      return;
    }
    setTrainInProgress(true);
    setTrainStatus("📊 Cargando columnas del archivo...");
    const formData = new FormData();
    formData.append("file", csvFile);
    try {
      const response = await axios.post("/analyze-csv/", formData);
      if (response.data.columns && response.data.columns.length > 0) {
        setColumns(response.data.columns);
        setTrainStatus("✅ Columnas cargadas. Ahora selecciona las variables de entrada y la variable target.");
      } else {
        setTrainStatus("❌ El archivo no contiene columnas válidas.");
      }
    } catch (error) {
      console.error("Error al cargar columnas:", error);
      setTrainStatus("❌ Error al cargar las columnas del archivo.");
    } finally {
      setTrainInProgress(false);
    }
  };

  // Alternar selección de variables de entrada (checkboxes)
  const handleFeatureChange = (column) => {
    setInputFeatures((prev) =>
      prev.includes(column)
        ? prev.filter((item) => item !== column)
        : [...prev, column]
    );
    validateSelections();
  };

  // Seleccionar variable de salida (radio buttons) con auto-selección
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

  // Seleccionar columna de fecha (radio buttons)
  const handleDateColumnChange = (column) => {
    setDateColumnName(column);
    // Remove from features if it was selected
    setInputFeatures((prev) => prev.filter((item) => item !== column));
    validateSelections();
  };

  // Validation function
  const validateSelections = () => {
    const warnings = [];

    // Allow empty features for ARIMA and LSTM (univariate models)
    // XGBoost requires at least 1 feature
    if (inputFeatures.length === 0 && targetVariable && algorithm === "xgboost") {
      warnings.push("XGBoost requiere al menos 1 variable de entrada. Para modelos univariados, usa ARIMA o LSTM.");
    }

    // Check if target is selected
    if (!targetVariable && inputFeatures.length > 0) {
      warnings.push("Debes seleccionar 1 variable de salida");
    }

    // Check for overlap
    if (targetVariable && inputFeatures.includes(targetVariable)) {
      warnings.push("Una columna no puede ser entrada y salida simultáneamente");
    }

    // Check date column doesn't overlap
    if (dateColumnName && inputFeatures.includes(dateColumnName)) {
      warnings.push("La columna de fecha no puede ser una variable de entrada");
    }

    if (dateColumnName && targetVariable === dateColumnName) {
      warnings.push("La columna de fecha no puede ser la variable de salida");
    }

    // PatchTSMixer-specific validation
    if (algorithm === "patchtsmixer") {
      // Validate at least 1 channel selected
      if (patchTSMixerChannels.length === 0) {
        warnings.push("PatchTSMixer requires at least 1 variable selected");
      }

      // Validate context_length and patch_length divisibility
      const contextLen = parseInt(patchTSMixerParams.context_length);
      const patchLen = parseInt(patchTSMixerParams.patch_length);
      if (contextLen % patchLen !== 0) {
        warnings.push("Warning: Context length should be divisible by patch length for optimal performance");
      }

      // Validate all numeric params are positive
      const numericParams = ['context_length', 'patch_length', 'patch_stride', 'd_model',
                             'num_layers', 'batch_size', 'epochs', 'early_stopping_patience'];
      numericParams.forEach(param => {
        if (parseInt(patchTSMixerParams[param]) <= 0) {
          warnings.push(`${param} must be a positive integer`);
        }
      });

      const floatParams = ['dropout', 'learning_rate'];
      floatParams.forEach(param => {
        const val = parseFloat(patchTSMixerParams[param]);
        if (val < 0) {
          warnings.push(`${param} must be non-negative`);
        }
      });

      // Validate dropout range
      if (parseFloat(patchTSMixerParams.dropout) > 1) {
        warnings.push("Dropout must be between 0 and 1");
      }

      // Validate advanced params if shown
      if (showPatchTSMixerAdvanced) {
        if (parseInt(patchTSMixerAdvanced.expansion_factor) <= 0) {
          warnings.push("Expansion factor must be positive");
        }
        const headDropout = parseFloat(patchTSMixerAdvanced.head_dropout);
        if (headDropout < 0 || headDropout > 1) {
          warnings.push("Head dropout must be between 0 and 1");
        }
      }

      // DEBUG: Log PatchTSMixer validation warnings (Step 6)
      // TODO: Remove after Phase 7a verification
      if (warnings.length > 0) {
        console.log("[PatchTSMixer] Validation warnings:", warnings);
      } else {
        console.log("[PatchTSMixer] Validation passed - no warnings");
      }
    }

    setValidationWarnings(warnings);
    return warnings.length === 0;
  };

  // Utility functions for comma decimal formatting
  const formatToComma = (value) => {
    return value.toFixed(2).replace('.', ',');
  };

  const parseFromComma = (value) => {
    if (typeof value === 'string') {
      // Replace comma with period for parsing
      const normalized = value.replace(',', '.');
      const parsed = parseFloat(normalized);
      return isNaN(parsed) ? 0 : parsed;
    }
    return value;
  };

  // Validation function for split ratios with debounce
  const validateSplitRatios = (ratios) => {
    // Clear existing timer
    if (validationTimer) {
      clearTimeout(validationTimer);
    }

    // Set new timer for debounced validation
    const timer = setTimeout(() => {
      const sum = Math.round((ratios.train + ratios.val + ratios.test) * 100) / 100;
      const allZeros = ratios.train === 0 && ratios.val === 0 && ratios.test === 0;

      if (allZeros) {
        setSplitRatiosValid(false);
        setValidationMessage("Los valores no pueden ser todos ceros");
        setShowValidation(true);
      } else if (sum !== 1.0) {
        setSplitRatiosValid(false);
        setValidationMessage(`La suma debe ser 1,00 (actual: ${formatToComma(sum)})`);
        setShowValidation(true);
      } else {
        setSplitRatiosValid(true);
        setValidationMessage("✓ Suma correcta: 1,00");
        setShowValidation(true);
      }
    }, 500);

    setValidationTimer(timer);
  };

  // Handler for LSTM feature selection (allows selecting target variable) - Phase 4
  const handleLstmFeatureToggle = (column) => {
    setLstmSelectedFeatures((prev) =>
      prev.includes(column)
        ? prev.filter((item) => item !== column)
        : [...prev, column]
    );
  };

  // Helper function to determine current state
  const getCurrentState = () => {
    if (!experimentDir || !runId) {
      return {
        type: 'warning',
        message: 'Requiere experimento creado y run_id',
        icon: <InfoIcon fontSize="small" />
      };
    }
    if (!flow.encodeDone) {
      return {
        type: 'warning',
        message: 'Esperando completar codificación de datos',
        icon: <InfoIcon fontSize="small" />
      };
    }
    if (flow.trainDone) {
      return {
        type: 'success',
        message: 'Modelo entrenado exitosamente',
        icon: <CheckCircleIcon fontSize="small" />
      };
    }
    if (trainInProgress) {
      return {
        type: 'processing',
        message: 'Entrenando modelo...',
        icon: <AutorenewIcon fontSize="small" sx={{ animation: 'spin 1s linear infinite', '@keyframes spin': { '0%': { transform: 'rotate(0deg)' }, '100%': { transform: 'rotate(360deg)' } } }} />
      };
    }
    if (trainStatus && (trainStatus.includes("❌") || trainStatus.includes("Error"))) {
      return {
        type: 'error',
        message: trainStatus.replace("❌", "").trim(),
        icon: <ErrorIcon fontSize="small" />
      };
    }
    return {
      type: 'info',
      message: 'Listo para entrenar modelo',
      icon: <InfoIcon fontSize="small" />
    };
  };

  // Helper function to get state color
  const getStateColor = (type) => {
    switch (type) {
      case 'info': return '#0288d1';      // Blue
      case 'success': return '#2e7d32';    // Green
      case 'warning': return '#ed6c02';    // Orange
      case 'error': return '#d32f2f';      // Red
      case 'processing': return '#ed6c02'; // Orange
      default: return '#0288d1';
    }
  };

  // Manejar cambios en las proporciones de división del dataset
  // Handler for input focus (auto-select all text)
  const handleInputFocus = (name) => (event) => {
    event.target.select();
  };

  // Handler for Enter key press (blur to trigger validation)
  const handleInputKeyDown = (name) => (event) => {
    if (event.key === 'Enter') {
      event.target.blur(); // Blur triggers handleInputBlur automatically
    }
  };

  // Handler for input changes (free typing with character filtering)
  const handleInputChange = (name) => (event) => {
    const inputValue = event.target.value;

    // Filter: only allow digits, comma, and period
    const filtered = inputValue.replace(/[^0-9.,]/g, '');

    // Update display value only (not splitRatios)
    setInputDisplayValues(prev => ({
      ...prev,
      [name]: filtered
    }));

    // Do NOT update splitRatios
    // Do NOT trigger validation
  };

  // Handler for input blur (validate, parse, clamp, format)
  const handleInputBlur = (name) => (event) => {
    const inputValue = inputDisplayValues[name];

    // Check for multiple decimal separators
    const commaCount = (inputValue.match(/,/g) || []).length;
    const periodCount = (inputValue.match(/\./g) || []).length;
    const hasMultipleSeparators = (commaCount + periodCount) > 1;

    // If invalid format, revert to previous value
    if (hasMultipleSeparators || inputValue.trim() === '') {
      setInputDisplayValues(prev => ({
        ...prev,
        [name]: formatToComma(splitRatios[name])
      }));
      return;
    }

    // Parse value (convert comma to period)
    const normalized = inputValue.replace(',', '.');
    const parsed = parseFloat(normalized);

    // If parsing failed, revert to previous value
    if (isNaN(parsed)) {
      setInputDisplayValues(prev => ({
        ...prev,
        [name]: formatToComma(splitRatios[name])
      }));
      return;
    }

    // Clamp to [0, 1] and round to 2 decimals
    const clampedValue = Math.max(0, Math.min(1, parsed));
    const roundedValue = Math.round(clampedValue * 100) / 100;

    // Update numeric state
    const newRatios = { ...splitRatios, [name]: roundedValue };
    setSplitRatios(newRatios);

    // Update display state with formatted value
    setInputDisplayValues(prev => ({
      ...prev,
      [name]: formatToComma(roundedValue)
    }));

    // Trigger validation
    validateSplitRatios(newRatios);
  };

  // Handler for slider changes (update both numeric and display states)
  const handleSliderChange = (name) => (event, newValue) => {
    // Update numeric state
    const newRatios = { ...splitRatios, [name]: newValue };
    setSplitRatios(newRatios);

    // Update display state with formatted value immediately
    setInputDisplayValues(prev => ({
      ...prev,
      [name]: formatToComma(newValue)
    }));

    // Trigger validation
    validateSplitRatios(newRatios);
  };

  // Calculate grid search combinations count (client-side with graceful fallback)
  const calculateGridCombinations = () => {
    try {
      if (!lstmGridOptions) return 0;

      const counts = [
        lstmGridOptions.lstm_units_options.split(',').filter(x => x.trim()).length,
        lstmGridOptions.dropout_rate_options.split(',').filter(x => x.trim()).length,
        lstmGridOptions.recurrent_dropout_rate_options.split(',').filter(x => x.trim()).length,
        lstmGridOptions.learning_rate_options.split(',').filter(x => x.trim()).length,
        lstmGridOptions.batch_size_options.split(',').filter(x => x.trim()).length,
        lstmGridOptions.epochs_options.split(',').filter(x => x.trim()).length
      ];

      return counts.reduce((a, b) => a * b, 1);
    } catch (error) {
      return null; // Graceful fallback - show "?" in UI
    }
  };

  // Update grid combinations count when grid options change
  useEffect(() => {
    if (algorithm === "lstm" && optimizationMethod === "grid") {
      const newCount = calculateGridCombinations();
      if (newCount !== gridCombinationsCount) {
        setGridCombinationsCount(newCount);
        // Reset warning dismissal when count changes
        setGridWarningDismissed(false);
      }
    }
  }, [algorithm, optimizationMethod, lstmGridOptions]);

  // Clear LSTM features when algorithm, target, or columns change (Phase 4)
  useEffect(() => {
    setLstmSelectedFeatures([]);
  }, [algorithm, targetVariable, columns]);

  // Auto-select manual optimization for PatchTSMixer (Phase 7b-II)
  useEffect(() => {
    if (algorithm === "patchtsmixer" && optimizationMethod !== "manual") {
      setOptimizationMethod("manual");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [algorithm]); // Intentionally only depend on algorithm to avoid infinite loop

  // Validation function for Bayesian Search configuration
  const validateBayesianConfig = () => {
    if (optimizationMethod !== "bayesian") return true;

    const {
      n_trials,
      n_initial_points,
      timeout_seconds,
      convergence_tolerance,
      convergence_patience
    } = bayesianConfig;

    // Validate n_trials
    if (!n_trials || n_trials < 1) {
      alert("n_trials debe ser al menos 1");
      return false;
    }

    // Validate n_initial_points < n_trials
    if (n_initial_points >= n_trials) {
      alert(`n_initial_points (${n_initial_points}) debe ser menor que n_trials (${n_trials})`);
      return false;
    }

    // Validate timeout if provided
    if (timeout_seconds !== null && timeout_seconds < 0) {
      alert("timeout_seconds debe ser positivo o nulo");
      return false;
    }

    // Validate convergence_tolerance if provided (Phase 7)
    if (convergence_tolerance !== null) {
      if (convergence_tolerance <= 0) {
        alert("convergence_tolerance debe ser mayor que 0 (ej. 0.001)");
        return false;
      }
      if (convergence_tolerance > 1.0) {
        alert("convergence_tolerance parece muy alto (> 1.0). Esto podría detener la optimización inmediatamente.");
        return false;
      }
    }

    // Validate convergence_patience if provided (Phase 7)
    if (convergence_patience !== null) {
      if (convergence_patience < 1) {
        alert("convergence_patience debe ser al menos 1");
        return false;
      }
      if (convergence_patience >= n_trials) {
        alert(`convergence_patience (${convergence_patience}) debe ser menor que n_trials (${n_trials})`);
        return false;
      }
    }

    return true;
  };

  // Función para enviar la solicitud de entrenamiento
  const handleTrain = async () => {
    if (!csvFile) {
      setTrainStatus("⚠️ Por favor, selecciona un archivo CSV.");
      return;
    }
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
    if (targetVariable === dateColumnName) {
      setTrainStatus("⚠️ La variable target y la columna de fecha deben ser diferentes.");
      return;
    }
    if (inputFeatures.includes(dateColumnName)) {
      setTrainStatus("⚠️ La columna de fecha no puede ser una variable de entrada.");
      return;
    }
    if (!experimentDir) {
      setTrainStatus("⚠️ El directorio del experimento no está configurado.");
      return;
    }
    if (!modelName.trim()) {
      setTrainStatus("⚠️ Por favor, proporciona un nombre para el modelo.");
      return;
    }
    if (!runId) {
      setTrainStatus("❌ Error: No se encontró un run_id. Completa primero el paso anterior.");
      return;
    }
    if (Object.values(splitRatios).reduce((a, b) => a + b, 0) !== 1) {
      setTrainStatus("⚠️ La suma de las proporciones de división debe ser igual a 1.");
      return;
    }

    // Validate bayesian config
    if (!validateBayesianConfig()) {
      return;
    }

    // Validate param_ranges if Bayesian Search
    const paramRangeErrors = validateParamRanges();
    if (paramRangeErrors.length > 0) {
      setTrainStatus("error");
      alert("Errores en rangos de parámetros:\n\n" + paramRangeErrors.join("\n"));
      return;
    }

    setTrainInProgress(true);
    setTrainStatus("🚀 Entrenando modelo, por favor espera...");

    // Preparar parámetros específicos del algoritmo
    let finalParams = {};
    let featureConfig = {};

    if (algorithm === "arima") {
      finalParams = { ...arimaParams };
      if (!enableSeasonalParams) {
        // Remove seasonal parameters if not enabled
        delete finalParams.seasonal_P;
        delete finalParams.seasonal_D;
        delete finalParams.seasonal_Q;
        delete finalParams.seasonal_s;
      }
    } else if (algorithm === "xgboost"){
      if (optimizationMethod === "manual") {
        finalParams = {
          n_estimators: parseInt(xgBoostParams.n_estimators),
          max_depth: parseInt(xgBoostParams.max_depth),
          learning_rate: parseFloat(xgBoostParams.learning_rate),
          subsample: parseFloat(xgBoostParams.subsample),
          gamma: parseFloat(xgBoostParams.gamma),
          min_child_weight: parseInt(xgBoostParams.min_child_weight)
        };
      }
    } else if (algorithm === "lstm") {
      // Parse lstm_units string to array
      const parseLstmUnits = (unitsStr) => {
        try {
          return JSON.parse(unitsStr);
        } catch {
          return [64]; // default
        }
      };

      if (optimizationMethod === "manual") {
        finalParams = {
          lstm_units: parseLstmUnits(lstmManualParams.lstm_units),
          dropout_rate: parseFloat(lstmManualParams.dropout_rate),
          recurrent_dropout_rate: parseFloat(lstmManualParams.recurrent_dropout_rate),
          learning_rate: parseFloat(lstmManualParams.learning_rate),
          batch_size: parseInt(lstmManualParams.batch_size),
          epochs: parseInt(lstmManualParams.epochs)
        };
      }
    }

    // Preparar el payload con los datos de entrenamiento
    const payload = {
      model_name: modelName,
      input_features: inputFeatures,
      target_variable: targetVariable,
      date_col_name: dateColumnName,
      experiment_dir: experimentDir,
      split_ratios: splitRatios,
      run_id: runId,
      algorithm: algorithm,
      manual_params: finalParams,
      params: finalParams,
      grid_search: arimaGridSearchParams,
      n_random_iterations: useRandomSearch ? nRandomIterations : undefined,
      random_search_params: useRandomSearch ?
        (algorithm === "arima" ? arimaRandomRanges :
         algorithm === "xgboost" ? xgboostRandomRanges :
         algorithm === "lstm" ? {
           lstm_units_options: lstmRandomRanges.lstm_units_options.map(s => JSON.parse(s)),
           dropout_rate_range: lstmRandomRanges.dropout_rate_range,
           recurrent_dropout_rate_range: lstmRandomRanges.recurrent_dropout_rate_range,
           learning_rate_range: lstmRandomRanges.learning_rate_range,
           batch_size_options: lstmRandomRanges.batch_size_options,
           epochs_range: lstmRandomRanges.epochs_range
         } : undefined) :
        undefined,
      optimization_metric: optimizationMetric,
      problem_type: problemType, // "ts_forecasting"
      forecast_horizon: forecastHorizon,
    };

    // LSTM-specific parameters
    if (algorithm === "lstm") {
      payload.sequence_length = sequenceLength;
      payload.early_stopping_patience = earlyStoppingPatience;
      payload.optimization_metric = "mse"; // Default for LSTM

      // Override input_features with LSTM-specific selection
      payload.input_features = lstmSelectedFeatures;
      payload.training_mode = lstmSelectedFeatures.length === 0 ? "univariate" : "multivariate";
    }

    // PatchTSMixer-specific parameters
    if (algorithm === "patchtsmixer") {
      payload.patchtsmixer_channels = patchTSMixerChannels;

      payload.manual_params = {
        // Essential params (10)
        context_length: parseInt(patchTSMixerParams.context_length),
        patch_length: parseInt(patchTSMixerParams.patch_length),
        patch_stride: parseInt(patchTSMixerParams.patch_stride),
        d_model: parseInt(patchTSMixerParams.d_model),
        num_layers: parseInt(patchTSMixerParams.num_layers),
        dropout: parseFloat(patchTSMixerParams.dropout),
        learning_rate: parseFloat(patchTSMixerParams.learning_rate),
        batch_size: parseInt(patchTSMixerParams.batch_size),
        epochs: parseInt(patchTSMixerParams.epochs),
        early_stopping_patience: parseInt(patchTSMixerParams.early_stopping_patience),
      };

      // Include advanced params if shown (9)
      if (showPatchTSMixerAdvanced) {
        payload.manual_params.expansion_factor = parseInt(patchTSMixerAdvanced.expansion_factor);
        payload.manual_params.head_dropout = parseFloat(patchTSMixerAdvanced.head_dropout);
        payload.manual_params.mode = patchTSMixerAdvanced.mode;
        payload.manual_params.gated_attn = patchTSMixerAdvanced.gated_attn;
        payload.manual_params.self_attn = patchTSMixerAdvanced.self_attn;
        payload.manual_params.scaling = patchTSMixerAdvanced.scaling;
        payload.manual_params.norm_mlp = patchTSMixerAdvanced.norm_mlp;
        payload.manual_params.loss = patchTSMixerAdvanced.loss;
        payload.manual_params.distribution_output = patchTSMixerAdvanced.distribution_output;
      }

      // DEBUG: Log PatchTSMixer payload (Step 7)
      // TODO: Remove after Phase 7a verification
      console.log("[PatchTSMixer] Payload constructed:", {
        patchtsmixer_channels: payload.patchtsmixer_channels,
        manual_params: payload.manual_params,
        showPatchTSMixerAdvanced: showPatchTSMixerAdvanced
      });
    }

   if (optimizationMethod === "grid") {
      payload.hyperparameter_search_strategy = "grid";

      // Add LSTM Grid Search parameters (Phase 2A)
      if (algorithm === "lstm") {
        payload.grid_search_params = {
          lstm_units_options: lstmGridOptions.lstm_units_options.split(',').map(s => JSON.parse(s.trim())),
          dropout_rate_options: lstmGridOptions.dropout_rate_options.split(',').map(s => parseFloat(s.trim())),
          recurrent_dropout_rate_options: lstmGridOptions.recurrent_dropout_rate_options.split(',').map(s => parseFloat(s.trim())),
          learning_rate_options: lstmGridOptions.learning_rate_options.split(',').map(s => parseFloat(s.trim())),
          batch_size_options: lstmGridOptions.batch_size_options.split(',').map(s => parseInt(s.trim())),
          epochs_options: lstmGridOptions.epochs_options.split(',').map(s => parseInt(s.trim()))
        };
        payload.enable_memory_profiling = enableMemoryProfiling;
        payload.grid_warning_threshold = gridWarningThreshold;
      }
    } else if (optimizationMethod === "random") {
      payload.hyperparameter_search_strategy = "random";
      payload.n_random_iterations = nRandomIterations;
    } else if (optimizationMethod === "bayesian") {
      // Send bayesian configuration
      payload.hyperparameter_search_strategy = "bayesian";

      // Build param_ranges based on algorithm and user preference
      let param_ranges = {};

      if (showBayesianParamRanges) {
        if (algorithm === "arima") {
          param_ranges = buildArimaParamRanges(arimaBayesianRanges, enableSeasonalParams);
        } else if (algorithm === "xgboost") {
          param_ranges = buildXGBoostParamRanges(xgboostBayesianRanges);
        } else if (algorithm === "lstm") {
          param_ranges = buildLSTMParamRanges(lstmBayesianRanges);
        }
      }


      // Include param_ranges only if not empty
      payload.bayesian_config = {
        ...bayesianConfig,
        ...(Object.keys(param_ranges).length > 0 && { param_ranges })
      };
    } else {
      payload.hyperparameter_search_strategy = "manual";
    }

    // Agregar feature_config para XGBoost y LSTM
    if (algorithm === "xgboost" || algorithm === "lstm") {
      payload.feature_config = featureConfig;
    }

    const formData = new FormData();
    formData.append("file", csvFile);
    formData.append("data", JSON.stringify(payload));
    // Iterate over the entries and log them
    for (const pair of formData.entries()) {
      console.log(`${pair[0]}: ${pair[1]}`);
    }

    try {
      const response = await axios.post("/ts/train-model/", formData);
      // Extract data from response
      const { status, model_path, metrics, mlflow_ui, run_id } = response.data;
      if (status === "success"){
        let successMessage = "✅ Modelo entrenado correctamente.";
    
        // Add model path if exists
        if (model_path) {
          successMessage += `\n📁 Modelo guardado en: ${model_path}`;
        }
        
        // Add metrics if exists
        if (metrics && Object.keys(metrics).length > 0) {
          successMessage += "\n📊 Métricas de validación:";
          Object.entries(metrics).forEach(([key, value]) => {
            successMessage += `\n  • ${key}: ${typeof value === 'number' ? value.toFixed(4) : value}`;
          });
        }
        setTrainStatus(successMessage);
        markStepDone("trainDone");
    }
    } catch (error) {
      console.error("Error al entrenar el modelo:", error);
        // Use the structured error response from backend
      const errorMessage = error.response?.data?.message || "Error desconocido";
      const errorDetails = error.response?.data?.error_details;
      
      let fullErrorMessage = `❌ Error durante el entrenamiento: ${errorMessage}`;
      
      if (errorDetails) {
        fullErrorMessage += `\n📝 Detalles: ${errorDetails}`;
      }
       setTrainStatus(fullErrorMessage);
    } finally {
      setTrainInProgress(false);
    }
  };


  // Función auxiliar para validar rangos de Random Search
  const isRandomSearchParamsValid = () => {
    if (!useRandomSearch) return true;

    // Validar número de iteraciones
    if (nRandomIterations < 1 || nRandomIterations > 1000) return false;

    if (algorithm === "arima") {
      // Validar rangos ARIMA
      const ranges = arimaRandomRanges;
      return (
        ranges.p_range[0] <= ranges.p_range[1] &&
        ranges.d_range[0] <= ranges.d_range[1] &&
        ranges.q_range[0] <= ranges.q_range[1] &&
        ranges.seasonal_P_range[0] <= ranges.seasonal_P_range[1] &&
        ranges.seasonal_D_range[0] <= ranges.seasonal_D_range[1] &&
        ranges.seasonal_Q_range[0] <= ranges.seasonal_Q_range[1] &&
        ranges.p_range[0] >= 0 && ranges.p_range[1] >= 0 &&
        ranges.d_range[0] >= 0 && ranges.d_range[1] >= 0 &&
        ranges.q_range[0] >= 0 && ranges.q_range[1] >= 0 &&
        ranges.seasonal_P_range[0] >= 0 && ranges.seasonal_P_range[1] >= 0 &&
        ranges.seasonal_D_range[0] >= 0 && ranges.seasonal_D_range[1] >= 0 &&
        ranges.seasonal_Q_range[0] >= 0 && ranges.seasonal_Q_range[1] >= 0
      );
    } else if (algorithm === "xgboost") {
      // Validar rangos XGBoost
      const ranges = xgboostRandomRanges;
      return (
        ranges.n_estimators_range[0] <= ranges.n_estimators_range[1] &&
        ranges.max_depth_range[0] <= ranges.max_depth_range[1] &&
        ranges.learning_rate_range[0] <= ranges.learning_rate_range[1] &&
        ranges.subsample_range[0] <= ranges.subsample_range[1] &&
        ranges.colsample_bytree_range[0] <= ranges.colsample_bytree_range[1] &&
        ranges.gamma_range[0] <= ranges.gamma_range[1] &&
        ranges.min_child_weight_range[0] <= ranges.min_child_weight_range[1] &&
        ranges.reg_alpha_range[0] <= ranges.reg_alpha_range[1] &&
        ranges.reg_lambda_range[0] <= ranges.reg_lambda_range[1] &&
        ranges.n_estimators_range[0] > 0 && ranges.n_estimators_range[1] > 0 &&
        ranges.max_depth_range[0] > 0 && ranges.max_depth_range[1] > 0 &&
        ranges.learning_rate_range[0] > 0 && ranges.learning_rate_range[1] > 0 &&
        ranges.subsample_range[0] > 0 && ranges.subsample_range[1] <= 1 &&
        ranges.colsample_bytree_range[0] > 0 && ranges.colsample_bytree_range[1] <= 1 &&
        ranges.gamma_range[0] >= 0 && ranges.gamma_range[1] >= 0 &&
        ranges.min_child_weight_range[0] > 0 && ranges.min_child_weight_range[1] > 0 &&
        ranges.reg_alpha_range[0] >= 0 && ranges.reg_alpha_range[1] >= 0 &&
        ranges.reg_lambda_range[0] >= 0 && ranges.reg_lambda_range[1] >= 0
      );
    }
    return true;
  };


  // Función auxiliar para validar parámetros de LSTM
  const isLSTMParamsValid = () => {
    if (algorithm !== "lstm") return true;

    // Validar sequence_length
    if (sequenceLength < 1) return false;

    // Validar early_stopping_patience
    if (earlyStoppingPatience < 1) return false;

    // Validar manual params si es modo manual
    if (optimizationMethod === "manual") {
      const dropout = parseFloat(lstmManualParams.dropout_rate);
      const recurrentDropout = parseFloat(lstmManualParams.recurrent_dropout_rate);
      const lr = parseFloat(lstmManualParams.learning_rate);
      const batch = parseInt(lstmManualParams.batch_size);
      const epochs = parseInt(lstmManualParams.epochs);

      return (
        !isNaN(dropout) && dropout >= 0 && dropout <= 0.8 &&
        !isNaN(recurrentDropout) && recurrentDropout >= 0 && recurrentDropout <= 0.8 &&
        !isNaN(lr) && lr > 0 &&
        !isNaN(batch) && batch > 0 &&
        !isNaN(epochs) && epochs >= 10 && epochs <= 500
      );
    }

    // Validar random search params
    if (optimizationMethod === "random") {
      const ranges = lstmRandomRanges;
      return (
        ranges.dropout_rate_range[0] <= ranges.dropout_rate_range[1] &&
        ranges.recurrent_dropout_rate_range[0] <= ranges.recurrent_dropout_rate_range[1] &&
        ranges.learning_rate_range[0] <= ranges.learning_rate_range[1] &&
        ranges.epochs_range[0] <= ranges.epochs_range[1] &&
        ranges.dropout_rate_range[0] >= 0 && ranges.dropout_rate_range[1] <= 0.8 &&
        ranges.recurrent_dropout_rate_range[0] >= 0 && ranges.recurrent_dropout_rate_range[1] <= 0.8 &&
        ranges.learning_rate_range[0] > 0 && ranges.learning_rate_range[1] > 0 &&
        ranges.epochs_range[0] > 0 && ranges.epochs_range[1] > 0
      );
    }

    return true;
  };

  // Helper functions to transform Bayesian param_ranges to backend format
  const buildArimaParamRanges = (ranges, includeSeasonalParams) => {
    const param_ranges = {};

    // Basic ARIMA params
    if (ranges.p) param_ranges.p = { min: ranges.p.min, max: ranges.p.max };
    if (ranges.d) param_ranges.d = { min: ranges.d.min, max: ranges.d.max };
    if (ranges.q) param_ranges.q = { min: ranges.q.min, max: ranges.q.max };

    // Seasonal params (conditional)
    if (includeSeasonalParams) {
      if (ranges.P) param_ranges.P = { min: ranges.P.min, max: ranges.P.max };
      if (ranges.D) param_ranges.D = { min: ranges.D.min, max: ranges.D.max };
      if (ranges.Q) param_ranges.Q = { min: ranges.Q.min, max: ranges.Q.max };
      if (ranges.s) param_ranges.s = { min: ranges.s.min, max: ranges.s.max };
    }

    // Categorical params
    if (ranges.trend && ranges.trend.choices.length > 0) {
      param_ranges.trend = { choices: ranges.trend.choices };
    }

    return param_ranges;
  };

  const buildXGBoostParamRanges = (ranges) => {
    const param_ranges = {};

    // Integer params
    if (ranges.n_estimators) {
      param_ranges.n_estimators = { min: ranges.n_estimators.min, max: ranges.n_estimators.max };
    }
    if (ranges.max_depth) {
      param_ranges.max_depth = { min: ranges.max_depth.min, max: ranges.max_depth.max };
    }
    if (ranges.min_child_weight) {
      param_ranges.min_child_weight = { min: ranges.min_child_weight.min, max: ranges.min_child_weight.max };
    }

    // Float params with optional log scale
    if (ranges.learning_rate) {
      param_ranges.learning_rate = {
        min: ranges.learning_rate.min,
        max: ranges.learning_rate.max,
        log: ranges.learning_rate.log || false
      };
    }
    if (ranges.subsample) {
      param_ranges.subsample = { min: ranges.subsample.min, max: ranges.subsample.max };
    }
    if (ranges.colsample_bytree) {
      param_ranges.colsample_bytree = { min: ranges.colsample_bytree.min, max: ranges.colsample_bytree.max };
    }
    if (ranges.gamma) {
      param_ranges.gamma = { min: ranges.gamma.min, max: ranges.gamma.max };
    }

    return param_ranges;
  };

  const buildLSTMParamRanges = (ranges) => {
    const param_ranges = {};

    // Categorical params - directly use choices array
    if (ranges.lstm_units && ranges.lstm_units.choices && ranges.lstm_units.choices.length > 0) {
      param_ranges.lstm_units = { choices: ranges.lstm_units.choices };
    }
    if (ranges.batch_size && ranges.batch_size.choices && ranges.batch_size.choices.length > 0) {
      param_ranges.batch_size = { choices: ranges.batch_size.choices };
    }

    // Float params - use min/max directly
    if (ranges.dropout_rate) {
      param_ranges.dropout_rate = {
        min: ranges.dropout_rate.min,
        max: ranges.dropout_rate.max
      };
    }

    // CRITICAL: Do NOT include recurrent_dropout_rate in param_ranges
    // Backend doesn't accept it (see train.py:4458 known_params)
    // Backend hardcodes it to match dropout_rate (see train.py:4599)
    // We keep it in UI state for user visibility but don't send it

    if (ranges.learning_rate) {
      param_ranges.learning_rate = {
        min: ranges.learning_rate.min,
        max: ranges.learning_rate.max,
        log: ranges.learning_rate.log || false
      };
    }

    // Integer params - use min/max directly
    if (ranges.epochs) {
      param_ranges.epochs = {
        min: ranges.epochs.min,
        max: ranges.epochs.max
      };
    }
    if (ranges.time_steps) {
      param_ranges.time_steps = {
        min: ranges.time_steps.min,
        max: ranges.time_steps.max
      };
    }

    return param_ranges;
  };

  // Validate param_ranges before submission
  const validateParamRanges = () => {
    const errors = [];

    if (optimizationMethod !== "bayesian" || !showBayesianParamRanges) return errors;

    if (algorithm === "arima") {
      // Validate min < max for numeric params
      const numericParams = ['p', 'd', 'q', 'P', 'D', 'Q', 's'];
      numericParams.forEach(param => {
        if (arimaBayesianRanges[param]) {
          const { min, max } = arimaBayesianRanges[param];
          if (min !== undefined && max !== undefined && min >= max) {
            errors.push(`ARIMA ${param}: min (${min}) debe ser menor que max (${max})`);
          }
        }
      });

      // Validate categorical params
      if (arimaBayesianRanges.trend && arimaBayesianRanges.trend.choices.length === 0) {
        errors.push("ARIMA trend: debe seleccionar al menos una opción");
      }
    }

    if (algorithm === "xgboost") {
      const params = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'gamma', 'min_child_weight'];
      params.forEach(param => {
        if (xgboostBayesianRanges[param]) {
          const { min, max } = xgboostBayesianRanges[param];
          if (min >= max) {
            errors.push(`XGBoost ${param}: min (${min}) debe ser menor que max (${max})`);
          }
        }
      });

      // Validate log scale only for positive ranges
      if (xgboostBayesianRanges.learning_rate && xgboostBayesianRanges.learning_rate.log) {
        const { min, max } = xgboostBayesianRanges.learning_rate;
        if (min <= 0 || max <= 0) {
          errors.push("XGBoost learning_rate: escala logarítmica requiere valores positivos");
        }
      }
    }

    if (algorithm === "lstm") {
      // Validate numeric params (min < max)
      const numericParams = ['dropout_rate', 'recurrent_dropout_rate', 'learning_rate', 'epochs', 'time_steps'];
      numericParams.forEach(param => {
        if (lstmBayesianRanges[param]) {
          const { min, max } = lstmBayesianRanges[param];
          if (min !== undefined && max !== undefined && min >= max) {
            errors.push(`LSTM ${param}: min (${min}) debe ser menor que max (${max})`);
          }
        }
      });

      // Validate categorical params
      if (lstmBayesianRanges.lstm_units && lstmBayesianRanges.lstm_units.choices.length === 0) {
        errors.push("LSTM lstm_units: debe tener al menos una opción");
      }
      if (lstmBayesianRanges.batch_size && lstmBayesianRanges.batch_size.choices.length === 0) {
        errors.push("LSTM batch_size: debe tener al menos una opción");
      }

      // Validate log scale for learning_rate
      if (lstmBayesianRanges.learning_rate && lstmBayesianRanges.learning_rate.log) {
        const { min, max } = lstmBayesianRanges.learning_rate;
        if (min <= 0 || max <= 0) {
          errors.push("LSTM learning_rate: escala logarítmica requiere valores positivos");
        }
      }
    }

    return errors;
  };

  // Se deshabilita la acción si:
  // - El proceso de entrenamiento está en curso.
  // - No hay experimento o run_id.
  // - No se han completado los pasos previos obligatorios (en este caso, la codificación debe estar completada).
  // - O ya se entrenó el modelo.
  // - Faltan campos requeridos para el algoritmo seleccionado.
  // - Los parámetros de Random Search son inválidos.
  // - Los parámetros de Bayesian Search son inválidos.
  // - Los parámetros de LSTM son inválidos.
  // - Hay advertencias de validación (nuevo)
  const isDisabled =
    trainInProgress ||
    !experimentDir ||
    !runId ||
    !flow.encodeDone ||
    flow.trainDone ||
    (algorithm === "xgboost" && !inputFeatures.length) ||  // XGBoost requires features
    !targetVariable ||
    !dateColumnName ||
    !modelName.trim() ||
    targetVariable === dateColumnName ||
    inputFeatures.includes(dateColumnName) ||
    !isRandomSearchParamsValid() ||
    !isLSTMParamsValid() ||
    validationWarnings.length > 0 ||
    !splitRatiosValid;

  return (
    <Card
      sx={{
        backgroundColor: "#e0f7fa",
        borderRadius: "12px",
        padding: "30px",
        textAlign: "center",
        boxShadow: "0 4px 12px rgba(0, 121, 107, 0.3)",
        margin: "20px",
        border: "1px solid #00796b",
      }}
    >
      <CardContent>
        <Typography
          variant="h5"
          sx={{
            mb: 2,
            color: "#004d40",
            textAlign: "center",
            fontWeight: "bold",
          }}
        >
          4. Entrenar Modelo
        </Typography>
        <Typography variant="body1" sx={{ mb: 2, color: "#004d40", textAlign: "center" }}>
          Selecciona un archivo CSV, ajusta los hiperparámetros y entrena un modelo. Los resultados y métricas se
          guardarán en MLflow.
        </Typography>

        {/* Botón de selección de archivo */}
        <Button variant="outlined" component="label" sx={{ mb: 2 }}>
          Seleccionar Archivo
          <input type="file" accept=".csv" hidden onChange={handleFileChange} />
        </Button>

        {/* Botón para cargar columnas */}
        <Button
          variant="contained"
          onClick={loadColumns}
          disabled={!csvFile || trainInProgress}
          sx={{
            mb: 2,
            backgroundColor: "#00796b",
            "&:hover": { backgroundColor: "#004d40" },
            width: "100%",
          }}
        >
          {trainInProgress ? <CircularProgress size={24} sx={{ color: "#fff" }} /> : "Cargar Variables"}
        </Button>

        {/* Feature Selection Forms - shown when columns are loaded */}
        {columns.length > 0 && (
          <>
            {/* VARIABLE DE SALIDA PRIMERO - con info modal y helper text */}
            <Box sx={variableSelectionStyles.sectionContainer}>
              <Box sx={variableSelectionStyles.sectionTitleContainer}>
                <Typography variant="subtitle1" sx={variableSelectionStyles.sectionTitle}>
                  Variable de Salida (Target)
                </Typography>
                <IconButton
                  sx={variableSelectionStyles.infoButton}
                  onClick={() => setShowTargetInfo(true)}
                  size="small"
                >
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Box>

              <Typography sx={variableSelectionStyles.helperText}>
                {helperTextStrings.variablesDeSalida}
              </Typography>

              <Box sx={variableSelectionStyles.variableBox}>
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
              </Box>
            </Box>

            {/* VALIDATION SUMMARY */}
            <ValidationSummary warnings={validationWarnings} />

            {/* VARIABLES DE ENTRADA - con info modal y helper text */}
            <Box sx={variableSelectionStyles.sectionContainer}>
              <Box sx={variableSelectionStyles.sectionTitleContainer}>
                <Typography variant="subtitle1" sx={variableSelectionStyles.sectionTitle}>
                  Variables de Entrada (Features)
                </Typography>
                <IconButton
                  sx={variableSelectionStyles.infoButton}
                  onClick={() => setShowFeatureInfo(true)}
                  size="small"
                >
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Box>

              <Typography sx={variableSelectionStyles.helperText}>
                {algorithm === "xgboost"
                  ? "Selecciona las columnas que usarás como variables de entrada (features) para el modelo. XGBoost requiere al menos 1 variable."
                  : "Selecciona las columnas que usarás como variables de entrada (features) para el modelo. Para modelos univariados (solo histórico del target), deja vacío."}
              </Typography>

              <Box sx={variableSelectionStyles.variableBox}>
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
              </Box>
            </Box>

            {/* COLUMNA DE FECHA */}
            <Box sx={variableSelectionStyles.sectionContainer}>
              <Typography variant="subtitle1" sx={variableSelectionStyles.sectionTitle}>
                Columna de Fecha
              </Typography>

              <Typography sx={variableSelectionStyles.helperText}>
                Selecciona la columna que contiene las fechas para el análisis de series temporales
              </Typography>

              <Box sx={variableSelectionStyles.variableBox}>
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
              </Box>
            </Box>
          </>
        )}

        <Box sx={{ textAlign: "left", mt: 2 }}>
          <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>Algoritmo:</Typography>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Selecciona un algoritmo</InputLabel>
            <Select
              value={algorithm}
              onChange={(e) => {
                setAlgorithm(e.target.value);
              }}
              label="Selecciona un algoritmo"
            >
              <MenuItem value="arima">ARIMA (Time Series)</MenuItem>
              <MenuItem value="xgboost">XGBoost (Time Series)</MenuItem>
              <MenuItem value="lstm">LSTM (Deep Learning)</MenuItem>
              <MenuItem value="patchtsmixer">PatchTSMixer (Transformer)</MenuItem>
            </Select>
          </FormControl>

          <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>
            Nombre del Modelo:
          </Typography>
          <TextField
            fullWidth
            placeholder="Introduce un nombre para el modelo"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            sx={{ mb: 2 }}
          />

          <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>
            Horizonte de Pronóstico:
          </Typography>
          <TextField
            fullWidth
            type="number"
            placeholder="Número de períodos a pronosticar"
            value={forecastHorizon}
            onChange={(e) => setForecastHorizon(parseInt(e.target.value) || 12)}
            slotProps={{
              input: {
                min: 1,
                max: 1000
              }
            }}
            sx={{ mb: 2 }}
          />

          {/* Optimization method selection */}
          <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>
            Método de optimización:
          </Typography>
          <FormControl component="fieldset" sx={{ mb: 2 }}>
            {["manual", "grid", "random", "bayesian"].map((method) => {
              // Disable non-manual methods for PatchTSMixer
              const isDisabledForPatchTSMixer = algorithm === "patchtsmixer" && method !== "manual";

              return (
                <Tooltip
                  key={method}
                  title={isDisabledForPatchTSMixer ? "PatchTSMixer solo soporta configuración manual de hiperparámetros" : ""}
                  placement="right"
                >
                  <span>
                    <FormControlLabel
                      control={
                        <input
                          type="radio"
                          name="optimizationMethod"
                          checked={optimizationMethod === method}
                          onChange={() => setOptimizationMethod(method)}
                          disabled={isDisabledForPatchTSMixer}
                          style={{ marginRight: "8px", transform: "scale(1.2)" }}
                        />
                      }
                      label={
                        method === "manual" ? "Parámetros manuales" :
                        method === "grid" ? "Grid Search (búsqueda automática)" :
                        method === "random" ? "Random Search (búsqueda aleatoria)" :
                        "Bayesian Search (optimización bayesiana)"
                      }
                      sx={{
                        display: "block",
                        padding: "5px 0",
                        opacity: isDisabledForPatchTSMixer ? 0.5 : 1
                      }}
                    />
                  </span>
                </Tooltip>
              );
            })}
          </FormControl>

          {/* Random Search configuration */}
          {useRandomSearch && (
            <Box sx={{ mb: 2, p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
              <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>
                Configuración Random Search:
              </Typography>
              <TextField
                type="number"
                label="Número de iteraciones"
                value={nRandomIterations}
                onChange={(e) => setNRandomIterations(Math.max(1, Math.min(1000, parseInt(e.target.value) || 100)))}
                slotProps={{
                  input: {
                    min: 1,
                    max: 1000
                  }
                }}
                helperText="Entre 1 y 1000 iteraciones (recomendado: 50-200)"
                sx={{ width: "100%", mb: 2 }}
              />
            </Box>
          )}

          {/* Bayesian Search configuration */}
          {optimizationMethod === "bayesian" && (
            <Box sx={{ mb: 2, p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
              <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>
                Configuración Bayesian Search:
              </Typography>
              <TextField
                type="number"
                label="Número de pruebas (n_trials)"
                value={bayesianConfig.n_trials}
                onChange={(e) => setBayesianConfig({
                  ...bayesianConfig,
                  n_trials: parseInt(e.target.value) || 50
                })}
                slotProps={{
                  input: {
                    min: 10,
                    max: 500,
                    step: 10
                  }
                }}
                helperText="Número total de pruebas de optimización Bayesiana (por defecto: 50)"
                sx={{ width: "100%", mb: 2 }}
              />

              {/* Advanced Settings - Collapsible */}
              <Box sx={{ mt: 2 }}>
                <Button
                  onClick={() => setShowBayesianAdvanced(!showBayesianAdvanced)}
                  variant="outlined"
                  size="small"
                  sx={{ mb: showBayesianAdvanced ? 2 : 0, textTransform: "none" }}
                >
                  {showBayesianAdvanced ? "▼" : "▶"} Configuración Avanzada (Opcional)
                </Button>

                {showBayesianAdvanced && (
                  <Box sx={{ p: 2, border: "1px solid #ccc", borderRadius: "8px", backgroundColor: "#fff" }}>
                    <Typography variant="body2" sx={{ color: "#666", mb: 2, fontStyle: "italic" }}>
                      Configuración avanzada de optimización Bayesiana. Los valores predeterminados funcionan bien en la mayoría de casos.
                    </Typography>

                    {/* n_initial_points */}
                    <TextField
                      type="number"
                      label="Puntos iniciales (n_initial_points)"
                      value={bayesianConfig.n_initial_points}
                      onChange={(e) => setBayesianConfig({
                        ...bayesianConfig,
                        n_initial_points: Math.max(1, parseInt(e.target.value) || 10)
                      })}
                      slotProps={{ input: { min: 1 } }}
                      helperText="Número de evaluaciones aleatorias antes de iniciar optimización"
                      size="small"
                      sx={{ width: "100%", mb: 2 }}
                    />

                    {/* acq_func */}
                    <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                      <InputLabel>Función de adquisición</InputLabel>
                      <Select
                        value={bayesianConfig.acq_func}
                        onChange={(e) => setBayesianConfig({
                          ...bayesianConfig,
                          acq_func: e.target.value
                        })}
                        label="Función de adquisición"
                      >
                        <MenuItem value="ei">EI (Expected Improvement)</MenuItem>
                        <MenuItem value="pi">PI (Probability of Improvement)</MenuItem>
                        <MenuItem value="ucb">UCB (Upper Confidence Bound)</MenuItem>
                        <MenuItem value="lcb">LCB (Lower Confidence Bound)</MenuItem>
                      </Select>
                      <Typography variant="caption" sx={{ color: "#666", mt: 0.5, ml: 1.5 }}>
                        Estrategia para elegir el próximo punto a evaluar
                      </Typography>
                    </FormControl>

                    {/* max_memory_mb */}
                    <TextField
                      type="number"
                      label="Límite de memoria (MB)"
                      value={bayesianConfig.max_memory_mb || ""}
                      onChange={(e) => setBayesianConfig({
                        ...bayesianConfig,
                        max_memory_mb: e.target.value ? parseInt(e.target.value) : null
                      })}
                      slotProps={{ input: { min: 100 } }}
                      helperText="Límite de memoria en MB (vacío = sin límite)"
                      size="small"
                      sx={{ width: "100%", mb: 2 }}
                    />

                    {/* timeout_seconds */}
                    <TextField
                      type="number"
                      label="Timeout (segundos)"
                      value={bayesianConfig.timeout_seconds || ""}
                      onChange={(e) => setBayesianConfig({
                        ...bayesianConfig,
                        timeout_seconds: e.target.value ? parseInt(e.target.value) : null
                      })}
                      slotProps={{ input: { min: 60 } }}
                      helperText="Tiempo máximo de ejecución en segundos (vacío = sin límite)"
                      size="small"
                      sx={{ width: "100%", mb: 2 }}
                    />

                    {/* Convergence Early-Stopping Section - Phase 7 */}
                    <Typography variant="body2" sx={{ color: "#666", mt: 2, mb: 1, fontWeight: "bold" }}>
                      Detección de Convergencia (Parada Temprana)
                    </Typography>

                    {/* convergence_tolerance */}
                    <TextField
                      type="number"
                      label="Tolerancia de convergencia"
                      value={bayesianConfig.convergence_tolerance ?? ""}
                      onChange={(e) => setBayesianConfig({
                        ...bayesianConfig,
                        convergence_tolerance: e.target.value ? parseFloat(e.target.value) : null
                      })}
                      slotProps={{
                        htmlInput: {
                          min: 0.0001,
                          max: 1.0,
                          step: 0.0001,
                          placeholder: "0.001 (default)"
                        }
                      }}
                      helperText="Mejora mínima para continuar optimización (vacío = 0.001). Valores típicos: 0.0001 - 0.01"
                      size="small"
                      sx={{ width: "100%", mb: 2 }}
                    />

                    {/* convergence_patience */}
                    <TextField
                      type="number"
                      label="Paciencia de convergencia"
                      value={bayesianConfig.convergence_patience || ""}
                      onChange={(e) => setBayesianConfig({
                        ...bayesianConfig,
                        convergence_patience: e.target.value ? parseInt(e.target.value) : null
                      })}
                      slotProps={{
                        input: {
                          min: 1,
                          step: 1
                        },
                        htmlInput: {
                          placeholder: "5 (default)"
                        }
                      }}
                      helperText="Número de trials sin mejora antes de detener (vacío = 5). Valores típicos: 3 - 10"
                      size="small"
                      sx={{ width: "100%", mb: 2 }}
                    />
                  </Box>
                )}
              </Box>

              {/* ARIMA Bayesian Parameter Ranges - Collapsible */}
              {algorithm === "arima" && (
                <Box sx={{ mt: 2 }}>
                  <Button
                    onClick={() => setShowBayesianParamRanges(!showBayesianParamRanges)}
                    variant="outlined"
                    size="small"
                    sx={{ mb: showBayesianParamRanges ? 2 : 0, textTransform: "none" }}
                  >
                    {showBayesianParamRanges ? "▼" : "▶"} Rangos de Parámetros (Opcional)
                  </Button>

                  {showBayesianParamRanges && (
                    <Box sx={{ p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#fff" }}>
                      <Typography variant="body2" sx={{ color: "#666", mb: 2, fontStyle: "italic" }}>
                        Personaliza los rangos de búsqueda para cada parámetro. Si no se especifica, se usan valores predeterminados.
                      </Typography>

                      {/* Basic ARIMA parameters: p, d, q */}
                      <Typography variant="body2" sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>
                        Parámetros básicos:
                      </Typography>
                      <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                        <TextField
                          label="p (min)"
                          type="number"
                          value={arimaBayesianRanges.p.min}
                          onChange={(e) => setArimaBayesianRanges(prev => ({
                            ...prev,
                            p: { ...prev.p, min: parseInt(e.target.value) || 0 }
                          }))}
                          slotProps={{ input: { min: 0, max: 5 } }}
                          size="small"
                          helperText="Orden autorregresivo"
                        />
                        <TextField
                          label="p (max)"
                          type="number"
                          value={arimaBayesianRanges.p.max}
                          onChange={(e) => setArimaBayesianRanges(prev => ({
                            ...prev,
                            p: { ...prev.p, max: parseInt(e.target.value) || 3 }
                          }))}
                          slotProps={{ input: { min: 0, max: 5 } }}
                          size="small"
                        />
                        <TextField
                          label="d (min)"
                          type="number"
                          value={arimaBayesianRanges.d.min}
                          onChange={(e) => setArimaBayesianRanges(prev => ({
                            ...prev,
                            d: { ...prev.d, min: parseInt(e.target.value) || 0 }
                          }))}
                          slotProps={{ input: { min: 0, max: 2 } }}
                          size="small"
                          helperText="Orden de diferenciación"
                        />
                        <TextField
                          label="d (max)"
                          type="number"
                          value={arimaBayesianRanges.d.max}
                          onChange={(e) => setArimaBayesianRanges(prev => ({
                            ...prev,
                            d: { ...prev.d, max: parseInt(e.target.value) || 1 }
                          }))}
                          slotProps={{ input: { min: 0, max: 2 } }}
                          size="small"
                        />
                        <TextField
                          label="q (min)"
                          type="number"
                          value={arimaBayesianRanges.q.min}
                          onChange={(e) => setArimaBayesianRanges(prev => ({
                            ...prev,
                            q: { ...prev.q, min: parseInt(e.target.value) || 0 }
                          }))}
                          slotProps={{ input: { min: 0, max: 5 } }}
                          size="small"
                          helperText="Orden media móvil"
                        />
                        <TextField
                          label="q (max)"
                          type="number"
                          value={arimaBayesianRanges.q.max}
                          onChange={(e) => setArimaBayesianRanges(prev => ({
                            ...prev,
                            q: { ...prev.q, max: parseInt(e.target.value) || 3 }
                          }))}
                          slotProps={{ input: { min: 0, max: 5 } }}
                          size="small"
                        />
                      </Box>

                      {/* Trend - Categorical */}
                      <Typography variant="body2" sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>
                        Trend (categórico):
                      </Typography>
                      <FormGroup row sx={{ mb: 2 }}>
                        {["n", "c", "t", "ct"].map(option => (
                          <FormControlLabel
                            key={option}
                            control={
                              <Checkbox
                                checked={arimaBayesianRanges.trend.choices.includes(option)}
                                onChange={(e) => {
                                  const newChoices = e.target.checked
                                    ? [...arimaBayesianRanges.trend.choices, option]
                                    : arimaBayesianRanges.trend.choices.filter(c => c !== option);
                                  setArimaBayesianRanges(prev => ({
                                    ...prev,
                                    trend: { choices: newChoices }
                                  }));
                                }}
                              />
                            }
                            label={option === "n" ? "none" : option}
                          />
                        ))}
                      </FormGroup>

                      {/* Seasonal parameters - conditional */}
                      {enableSeasonalParams && (
                        <>
                          <Typography variant="body2" sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>
                            Parámetros estacionales:
                          </Typography>
                          <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                            <TextField
                              label="P (min)"
                              type="number"
                              value={arimaBayesianRanges.P.min}
                              onChange={(e) => setArimaBayesianRanges(prev => ({
                                ...prev,
                                P: { ...prev.P, min: parseInt(e.target.value) || 0 }
                              }))}
                              slotProps={{ input: { min: 0, max: 3 } }}
                              size="small"
                            />
                            <TextField
                              label="P (max)"
                              type="number"
                              value={arimaBayesianRanges.P.max}
                              onChange={(e) => setArimaBayesianRanges(prev => ({
                                ...prev,
                                P: { ...prev.P, max: parseInt(e.target.value) || 2 }
                              }))}
                              slotProps={{ input: { min: 0, max: 3 } }}
                              size="small"
                            />
                            <TextField
                              label="D (min)"
                              type="number"
                              value={arimaBayesianRanges.D.min}
                              onChange={(e) => setArimaBayesianRanges(prev => ({
                                ...prev,
                                D: { ...prev.D, min: parseInt(e.target.value) || 0 }
                              }))}
                              slotProps={{ input: { min: 0, max: 2 } }}
                              size="small"
                            />
                            <TextField
                              label="D (max)"
                              type="number"
                              value={arimaBayesianRanges.D.max}
                              onChange={(e) => setArimaBayesianRanges(prev => ({
                                ...prev,
                                D: { ...prev.D, max: parseInt(e.target.value) || 1 }
                              }))}
                              slotProps={{ input: { min: 0, max: 2 } }}
                              size="small"
                            />
                            <TextField
                              label="Q (min)"
                              type="number"
                              value={arimaBayesianRanges.Q.min}
                              onChange={(e) => setArimaBayesianRanges(prev => ({
                                ...prev,
                                Q: { ...prev.Q, min: parseInt(e.target.value) || 0 }
                              }))}
                              slotProps={{ input: { min: 0, max: 3 } }}
                              size="small"
                            />
                            <TextField
                              label="Q (max)"
                              type="number"
                              value={arimaBayesianRanges.Q.max}
                              onChange={(e) => setArimaBayesianRanges(prev => ({
                                ...prev,
                                Q: { ...prev.Q, max: parseInt(e.target.value) || 2 }
                              }))}
                              slotProps={{ input: { min: 0, max: 3 } }}
                              size="small"
                            />
                            <TextField
                              label="s (min)"
                              type="number"
                              value={arimaBayesianRanges.s.min}
                              onChange={(e) => setArimaBayesianRanges(prev => ({
                                ...prev,
                                s: { ...prev.s, min: parseInt(e.target.value) || 2 }
                              }))}
                              slotProps={{ input: { min: 2, max: 24 } }}
                              size="small"
                              helperText="Período estacional"
                            />
                            <TextField
                              label="s (max)"
                              type="number"
                              value={arimaBayesianRanges.s.max}
                              onChange={(e) => setArimaBayesianRanges(prev => ({
                                ...prev,
                                s: { ...prev.s, max: parseInt(e.target.value) || 24 }
                              }))}
                              slotProps={{ input: { min: 2, max: 24 } }}
                              size="small"
                            />
                          </Box>
                        </>
                      )}
                    </Box>
                  )}
                </Box>
              )}

              {/* ARIMA-specific: Seasonal parameters checkbox */}
              {algorithm === "arima" && (
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={enableSeasonalParams}
                      onChange={(e) => setEnableSeasonalParams(e.target.checked)}
                      color="primary"
                    />
                  }
                  label="Habilitar parámetros estacionales (SARIMA)"
                  sx={{ mt: 2 }}
                />
              )}
            </Box>
          )}

          {/* Para ARIMA */}
          {algorithm === "arima" && (
            <>
              {optimizationMethod === "manual" ? (
                <>
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mt: 2 }}>
                    Hiperparámetros - ARIMA
                  </Typography>
                  <Box sx={{ display: "flex", gap: "10px", mb: 2 }}>
                    <TextField
                      type="number"
                      label="p (AR order)"
                      value={arimaParams.p}
                      onChange={(e) => setArimaParams({ ...arimaParams, p: e.target.value })}
                      slotProps={{
                        input: {
                          min: 0,
                          max: 5
                        }
                      }}
                      sx={{ width: "33%" }}
                    />
                    <TextField
                      type="number"
                      label="d (Diferenciación)"
                      value={arimaParams.d}
                      onChange={(e) => setArimaParams({ ...arimaParams, d: e.target.value })}
                      slotProps={{
                        input: {
                          min: 0,
                          max: 2
                        }
                      }}
                      sx={{ width: "33%" }}
                    />
                    <TextField
                      type="number"
                      label="q (MA order)"
                      value={arimaParams.q}
                      onChange={(e) => setArimaParams({ ...arimaParams, q: e.target.value })}
                      slotProps={{
                        input: {
                          min: 0,
                          max: 5
                        }
                      }}
                      sx={{ width: "33%" }}
                    />
                  </Box>

                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={enableSeasonalParams}
                        onChange={(e) => setEnableSeasonalParams(e.target.checked)}
                        color="primary"
                      />
                    }
                    label="Habilitar parámetros estacionales (SARIMA)"
                  />

                  {enableSeasonalParams && (
                    <>
                      <Typography sx={{ fontWeight: "bold", color: "#004d40", mt: 2 }}>
                        Parámetros Estacionales
                      </Typography>
                      <Box sx={{ display: "flex", gap: "10px", mb: 2 }}>
                        <TextField
                          type="number"
                          label="P (Seasonal AR)"
                          value={arimaParams.seasonal_P}
                          onChange={(e) => setArimaParams({ ...arimaParams, seasonal_P: e.target.value })}
                          slotProps={{
                            input: {
                              min: 0,
                              max: 2
                            }
                          }}
                          sx={{ width: "25%" }}
                        />
                        <TextField
                          type="number"
                          label="D (Seasonal Diff)"
                          value={arimaParams.seasonal_D}
                          onChange={(e) => setArimaParams({ ...arimaParams, seasonal_D: e.target.value })}
                          slotProps={{
                            input: {
                              min: 0,
                              max: 2
                            }
                          }}
                          sx={{ width: "25%" }}
                        />
                        <TextField
                          type="number"
                          label="Q (Seasonal MA)"
                          value={arimaParams.seasonal_Q}
                          onChange={(e) => setArimaParams({ ...arimaParams, seasonal_Q: e.target.value })}
                          slotProps={{
                            input: {
                              min: 0,
                              max: 2
                            }
                          }}
                          sx={{ width: "25%" }}
                        />
                        <TextField
                          type="number"
                          label="s (Periodo)"
                          value={arimaParams.seasonal_s}
                          onChange={(e) => setArimaParams({ ...arimaParams, seasonal_s: e.target.value })}
                          slotProps={{
                            input: {
                              min: 1
                            }
                          }}
                          sx={{ width: "25%" }}
                        />
                      </Box>
                    </>
                  )}

                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mt: 2 }}>
                    Parámetros Adicionales
                  </Typography>

                  <FormControl sx={{ width: "100%", mb: 2 }}>
                    <InputLabel>Trend</InputLabel>
                    <Select
                      value={arimaParams.trend}
                      onChange={(e) => setArimaParams({ ...arimaParams, trend: e.target.value })}
                      label="Trend"
                    >
                      <MenuItem value="n">n (No trend)</MenuItem>
                      <MenuItem value="c">c (Constant)</MenuItem>
                      <MenuItem value="t">t (Linear trend)</MenuItem>
                      <MenuItem value="ct">ct (Constant + linear trend)</MenuItem>
                    </Select>
                  </FormControl>

                  <Box sx={{ display: "flex", flexDirection: "column", gap: 1, mb: 2 }}>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={arimaParams.enforce_stationarity === "True"}
                          onChange={(e) => setArimaParams({
                            ...arimaParams,
                            enforce_stationarity: e.target.checked ? "True" : "False"
                          })}
                          color="primary"
                        />
                      }
                      label="Enforce Stationarity"
                    />
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={arimaParams.enforce_invertibility === "True"}
                          onChange={(e) => setArimaParams({
                            ...arimaParams,
                            enforce_invertibility: e.target.checked ? "True" : "False"
                          })}
                          color="primary"
                        />
                      }
                      label="Enforce Invertibility"
                    />
                  </Box>
                </>
              ) : optimizationMethod === "grid" ? (
                <Box sx={{ mt: 2 }}>
                  {/* ARIMA Hyperparameters Accordion */}
                  <Accordion defaultExpanded>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography sx={{ fontWeight: "bold", color: "#004d40" }}>
                        Model Hyperparameters - ARIMA
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                        {/* p parameter */}
                        <Box>
                          <Typography variant="body2" sx={{ mb: 1, fontWeight: "bold" }}>
                            p (AR order):
                          </Typography>
                          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, alignItems: "center" }}>
                            {arimaGridSearchParams.p.map((val, idx) => (
                              <Box key={idx} sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
                                <input
                                  type="number"
                                  value={val}
                                  onChange={(e) => {
                                    const newP = [...arimaGridSearchParams.p];
                                    newP[idx] = parseInt(e.target.value) || 0;
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, p: newP });
                                  }}
                                  style={{ width: "60px", padding: "4px" }}
                                />
                                <Button
                                  size="small"
                                  onClick={() => {
                                    const newP = arimaGridSearchParams.p.filter((_, i) => i !== idx);
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, p: newP });
                                  }}
                                  sx={{ minWidth: "auto", px: 1, fontSize: "0.75rem" }}
                                >
                                  Remove
                                </Button>
                              </Box>
                            ))}
                            <Button
                              size="small"
                              variant="outlined"
                              onClick={() => {
                                setArimaGridSearchParams({
                                  ...arimaGridSearchParams,
                                  p: [...arimaGridSearchParams.p, 0]
                                });
                              }}
                            >
                              Add number
                            </Button>
                          </Box>
                        </Box>

                        {/* d parameter */}
                        <Box>
                          <Typography variant="body2" sx={{ mb: 1, fontWeight: "bold" }}>
                            d (Differencing order):
                          </Typography>
                          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, alignItems: "center" }}>
                            {arimaGridSearchParams.d.map((val, idx) => (
                              <Box key={idx} sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
                                <input
                                  type="number"
                                  value={val}
                                  onChange={(e) => {
                                    const newD = [...arimaGridSearchParams.d];
                                    newD[idx] = parseInt(e.target.value) || 0;
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, d: newD });
                                  }}
                                  style={{ width: "60px", padding: "4px" }}
                                />
                                <Button
                                  size="small"
                                  onClick={() => {
                                    const newD = arimaGridSearchParams.d.filter((_, i) => i !== idx);
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, d: newD });
                                  }}
                                  sx={{ minWidth: "auto", px: 1, fontSize: "0.75rem" }}
                                >
                                  Remove
                                </Button>
                              </Box>
                            ))}
                            <Button
                              size="small"
                              variant="outlined"
                              onClick={() => {
                                setArimaGridSearchParams({
                                  ...arimaGridSearchParams,
                                  d: [...arimaGridSearchParams.d, 0]
                                });
                              }}
                            >
                              Add number
                            </Button>
                          </Box>
                        </Box>

                        {/* q parameter */}
                        <Box>
                          <Typography variant="body2" sx={{ mb: 1, fontWeight: "bold" }}>
                            q (MA order):
                          </Typography>
                          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, alignItems: "center" }}>
                            {arimaGridSearchParams.q.map((val, idx) => (
                              <Box key={idx} sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
                                <input
                                  type="number"
                                  value={val}
                                  onChange={(e) => {
                                    const newQ = [...arimaGridSearchParams.q];
                                    newQ[idx] = parseInt(e.target.value) || 0;
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, q: newQ });
                                  }}
                                  style={{ width: "60px", padding: "4px" }}
                                />
                                <Button
                                  size="small"
                                  onClick={() => {
                                    const newQ = arimaGridSearchParams.q.filter((_, i) => i !== idx);
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, q: newQ });
                                  }}
                                  sx={{ minWidth: "auto", px: 1, fontSize: "0.75rem" }}
                                >
                                  Remove
                                </Button>
                              </Box>
                            ))}
                            <Button
                              size="small"
                              variant="outlined"
                              onClick={() => {
                                setArimaGridSearchParams({
                                  ...arimaGridSearchParams,
                                  q: [...arimaGridSearchParams.q, 0]
                                });
                              }}
                            >
                              Add number
                            </Button>
                          </Box>
                        </Box>
                      </Box>
                    </AccordionDetails>
                  </Accordion>

                  {/* SARIMA Hyperparameters Accordion */}
                  <Accordion defaultExpanded sx={{ mt: 2 }}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography sx={{ fontWeight: "bold", color: "#004d40" }}>
                        Model Hyperparameters - SARIMA
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                        {/* Toggle for seasonal parameters */}
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={useSeasonalParamsHT}
                              onChange={(e) => setUseSeasonalParamsHT(e.target.checked)}
                              color="primary"
                            />
                          }
                          label="Use Seasonal Hyperparameters"
                        />

                        {/* P parameter (seasonal) */}
                        <Box>
                          <Typography variant="body2" sx={{ mb: 1, fontWeight: "bold", color: useSeasonalParamsHT ? "inherit" : "#ccc" }}>
                            P (Seasonal AR order):
                          </Typography>
                          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, alignItems: "center" }}>
                            {arimaGridSearchParams.P.map((val, idx) => (
                              <Box key={idx} sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
                                <input
                                  type="number"
                                  value={val}
                                  disabled={!useSeasonalParamsHT}
                                  onChange={(e) => {
                                    const newP = [...arimaGridSearchParams.P];
                                    newP[idx] = parseInt(e.target.value) || 0;
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, P: newP });
                                  }}
                                  style={{ width: "60px", padding: "4px", backgroundColor: useSeasonalParamsHT ? "white" : "#f0f0f0" }}
                                />
                                <Button
                                  size="small"
                                  disabled={!useSeasonalParamsHT}
                                  onClick={() => {
                                    const newP = arimaGridSearchParams.P.filter((_, i) => i !== idx);
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, P: newP });
                                  }}
                                  sx={{ minWidth: "auto", px: 1, fontSize: "0.75rem" }}
                                >
                                  Remove
                                </Button>
                              </Box>
                            ))}
                            <Button
                              size="small"
                              variant="outlined"
                              disabled={!useSeasonalParamsHT}
                              onClick={() => {
                                setArimaGridSearchParams({
                                  ...arimaGridSearchParams,
                                  P: [...arimaGridSearchParams.P, 0]
                                });
                              }}
                            >
                              Add number
                            </Button>
                          </Box>
                        </Box>

                        {/* D parameter (seasonal) */}
                        <Box>
                          <Typography variant="body2" sx={{ mb: 1, fontWeight: "bold", color: useSeasonalParamsHT ? "inherit" : "#ccc" }}>
                            D (Seasonal Differencing order):
                          </Typography>
                          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, alignItems: "center" }}>
                            {arimaGridSearchParams.D.map((val, idx) => (
                              <Box key={idx} sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
                                <input
                                  type="number"
                                  value={val}
                                  disabled={!useSeasonalParamsHT}
                                  onChange={(e) => {
                                    const newD = [...arimaGridSearchParams.D];
                                    newD[idx] = parseInt(e.target.value) || 0;
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, D: newD });
                                  }}
                                  style={{ width: "60px", padding: "4px", backgroundColor: useSeasonalParamsHT ? "white" : "#f0f0f0" }}
                                />
                                <Button
                                  size="small"
                                  disabled={!useSeasonalParamsHT}
                                  onClick={() => {
                                    const newD = arimaGridSearchParams.D.filter((_, i) => i !== idx);
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, D: newD });
                                  }}
                                  sx={{ minWidth: "auto", px: 1, fontSize: "0.75rem" }}
                                >
                                  Remove
                                </Button>
                              </Box>
                            ))}
                            <Button
                              size="small"
                              variant="outlined"
                              disabled={!useSeasonalParamsHT}
                              onClick={() => {
                                setArimaGridSearchParams({
                                  ...arimaGridSearchParams,
                                  D: [...arimaGridSearchParams.D, 0]
                                });
                              }}
                            >
                              Add number
                            </Button>
                          </Box>
                        </Box>

                        {/* Q parameter (seasonal) */}
                        <Box>
                          <Typography variant="body2" sx={{ mb: 1, fontWeight: "bold", color: useSeasonalParamsHT ? "inherit" : "#ccc" }}>
                            Q (Seasonal MA order):
                          </Typography>
                          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, alignItems: "center" }}>
                            {arimaGridSearchParams.Q.map((val, idx) => (
                              <Box key={idx} sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
                                <input
                                  type="number"
                                  value={val}
                                  disabled={!useSeasonalParamsHT}
                                  onChange={(e) => {
                                    const newQ = [...arimaGridSearchParams.Q];
                                    newQ[idx] = parseInt(e.target.value) || 0;
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, Q: newQ });
                                  }}
                                  style={{ width: "60px", padding: "4px", backgroundColor: useSeasonalParamsHT ? "white" : "#f0f0f0" }}
                                />
                                <Button
                                  size="small"
                                  disabled={!useSeasonalParamsHT}
                                  onClick={() => {
                                    const newQ = arimaGridSearchParams.Q.filter((_, i) => i !== idx);
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, Q: newQ });
                                  }}
                                  sx={{ minWidth: "auto", px: 1, fontSize: "0.75rem" }}
                                >
                                  Remove
                                </Button>
                              </Box>
                            ))}
                            <Button
                              size="small"
                              variant="outlined"
                              disabled={!useSeasonalParamsHT}
                              onClick={() => {
                                setArimaGridSearchParams({
                                  ...arimaGridSearchParams,
                                  Q: [...arimaGridSearchParams.Q, 0]
                                });
                              }}
                            >
                              Add number
                            </Button>
                          </Box>
                        </Box>

                        {/* S parameter (seasonal) */}
                        <Box>
                          <Typography variant="body2" sx={{ mb: 1, fontWeight: "bold", color: useSeasonalParamsHT ? "inherit" : "#ccc" }}>
                            S (Seasonal period):
                          </Typography>
                          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, alignItems: "center" }}>
                            {arimaGridSearchParams.s.map((val, idx) => (
                              <Box key={idx} sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
                                <input
                                  type="number"
                                  value={val === null ? "" : val}
                                  disabled={!useSeasonalParamsHT}
                                  placeholder="null"
                                  onChange={(e) => {
                                    const newS = [...arimaGridSearchParams.s];
                                    newS[idx] = e.target.value === "" ? null : parseInt(e.target.value) || 0;
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, s: newS });
                                  }}
                                  style={{ width: "60px", padding: "4px", backgroundColor: useSeasonalParamsHT ? "white" : "#f0f0f0" }}
                                />
                                <Button
                                  size="small"
                                  disabled={!useSeasonalParamsHT}
                                  onClick={() => {
                                    const newS = arimaGridSearchParams.s.filter((_, i) => i !== idx);
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, s: newS });
                                  }}
                                  sx={{ minWidth: "auto", px: 1, fontSize: "0.75rem" }}
                                >
                                  Remove
                                </Button>
                              </Box>
                            ))}
                            <Button
                              size="small"
                              variant="outlined"
                              disabled={!useSeasonalParamsHT}
                              onClick={() => {
                                setArimaGridSearchParams({
                                  ...arimaGridSearchParams,
                                  s: [...arimaGridSearchParams.s, 0]
                                });
                              }}
                            >
                              Add number
                            </Button>
                          </Box>
                          <FormControlLabel
                            control={
                              <Checkbox
                                disabled={!useSeasonalParamsHT}
                                onChange={(e) => {
                                  if (e.target.checked) {
                                    setArimaGridSearchParams({
                                      ...arimaGridSearchParams,
                                      s: [...arimaGridSearchParams.s, null]
                                    });
                                  }
                                }}
                                color="primary"
                              />
                            }
                            label="None (iterate P,D,Q as ARIMA)"
                            sx={{ mt: 1 }}
                          />
                        </Box>
                      </Box>
                    </AccordionDetails>
                  </Accordion>

                  {/* Model Configurations Accordion */}
                  <Accordion defaultExpanded sx={{ mt: 2 }}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography sx={{ fontWeight: "bold", color: "#004d40" }}>
                        Model Configurations
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                        {/* Trend checkboxes */}
                        <Box>
                          <Typography variant="body2" sx={{ mb: 1, fontWeight: "bold" }}>
                            Trend:
                          </Typography>
                          <FormGroup row>
                            <FormControlLabel
                              control={
                                <Checkbox
                                  checked={true}
                                  disabled={true}
                                  color="primary"
                                />
                              }
                              label="n (no trend)"
                            />
                            <FormControlLabel
                              control={
                                <Checkbox
                                  checked={arimaGridSearchParams.trend.includes("c")}
                                  onChange={(e) => {
                                    const newTrend = e.target.checked
                                      ? [...arimaGridSearchParams.trend, "c"]
                                      : arimaGridSearchParams.trend.filter(t => t !== "c");
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, trend: newTrend });
                                  }}
                                  color="primary"
                                />
                              }
                              label="c (constant trend)"
                            />
                            <FormControlLabel
                              control={
                                <Checkbox
                                  checked={arimaGridSearchParams.trend.includes("t")}
                                  onChange={(e) => {
                                    const newTrend = e.target.checked
                                      ? [...arimaGridSearchParams.trend, "t"]
                                      : arimaGridSearchParams.trend.filter(t => t !== "t");
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, trend: newTrend });
                                  }}
                                  color="primary"
                                />
                              }
                              label="t (linear trend)"
                            />
                            <FormControlLabel
                              control={
                                <Checkbox
                                  checked={arimaGridSearchParams.trend.includes("ct")}
                                  onChange={(e) => {
                                    const newTrend = e.target.checked
                                      ? [...arimaGridSearchParams.trend, "ct"]
                                      : arimaGridSearchParams.trend.filter(t => t !== "ct");
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, trend: newTrend });
                                  }}
                                  color="primary"
                                />
                              }
                              label="ct (constant + linear trend)"
                            />
                          </FormGroup>
                        </Box>

                        {/* Enforce stationarity checkboxes */}
                        <Box>
                          <Typography variant="body2" sx={{ mb: 1, fontWeight: "bold" }}>
                            Enforce stationarity:
                          </Typography>
                          <FormGroup row>
                            <FormControlLabel
                              control={
                                <Checkbox
                                  checked={arimaGridSearchParams.enforce_stationarity.includes(true)}
                                  onChange={(e) => {
                                    const newEnforce = e.target.checked
                                      ? [...arimaGridSearchParams.enforce_stationarity, true]
                                      : arimaGridSearchParams.enforce_stationarity.filter(t => t !== true);
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, enforce_stationarity: newEnforce });
                                  }}
                                  color="primary"
                                />
                              }
                              label="True"
                            />
                            <FormControlLabel
                              control={
                                <Checkbox
                                  checked={arimaGridSearchParams.enforce_stationarity.includes(false)}
                                  onChange={(e) => {
                                    const newEnforce = e.target.checked
                                      ? [...arimaGridSearchParams.enforce_stationarity, false]
                                      : arimaGridSearchParams.enforce_stationarity.filter(t => t !== false);
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, enforce_stationarity: newEnforce });
                                  }}
                                  color="primary"
                                />
                              }
                              label="False"
                            />
                          </FormGroup>
                        </Box>

                        {/* Enforce invertibility checkboxes */}
                        <Box>
                          <Typography variant="body2" sx={{ mb: 1, fontWeight: "bold" }}>
                            Enforce invertibility:
                          </Typography>
                          <FormGroup row>
                            <FormControlLabel
                              control={
                                <Checkbox
                                  checked={arimaGridSearchParams.enforce_invertibility.includes(true)}
                                  onChange={(e) => {
                                    const newEnforce = e.target.checked
                                      ? [...arimaGridSearchParams.enforce_invertibility, true]
                                      : arimaGridSearchParams.enforce_invertibility.filter(t => t !== true);
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, enforce_invertibility: newEnforce });
                                  }}
                                  color="primary"
                                />
                              }
                              label="True"
                            />
                            <FormControlLabel
                              control={
                                <Checkbox
                                  checked={arimaGridSearchParams.enforce_invertibility.includes(false)}
                                  onChange={(e) => {
                                    const newEnforce = e.target.checked
                                      ? [...arimaGridSearchParams.enforce_invertibility, false]
                                      : arimaGridSearchParams.enforce_invertibility.filter(t => t !== false);
                                    setArimaGridSearchParams({ ...arimaGridSearchParams, enforce_invertibility: newEnforce });
                                  }}
                                  color="primary"
                                />
                              }
                              label="False"
                            />
                          </FormGroup>
                        </Box>
                      </Box>
                    </AccordionDetails>
                  </Accordion>
                </Box>
              ) : (
                /* Random Search parameter ranges for ARIMA */
                <Box sx={{ mt: 2 }}>
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 2 }}>
                    Rangos de parámetros - ARIMA Random Search
                  </Typography>

                  <Typography variant="body2" sx={{ color: "#666", mb: 2 }}>
                    Define los rangos para la búsqueda aleatoria de parámetros ARIMA
                  </Typography>

                  {/* Basic ARIMA parameter ranges */}
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                    <TextField
                      label="Rango p (min)"
                      type="number"
                      value={arimaRandomRanges.p_range[0]}
                      onChange={(e) => setArimaRandomRanges(prev => ({
                        ...prev,
                        p_range: [parseInt(e.target.value) || 0, prev.p_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 5 }
                      }}
                      size="small"
                    />
                    <TextField
                      label="Rango p (max)"
                      type="number"
                      value={arimaRandomRanges.p_range[1]}
                      onChange={(e) => setArimaRandomRanges(prev => ({
                        ...prev,
                        p_range: [prev.p_range[0], parseInt(e.target.value) || 4]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 5 }
                      }}
                      size="small"
                    />
                    <TextField
                      label="Rango d (min)"
                      type="number"
                      value={arimaRandomRanges.d_range[0]}
                      onChange={(e) => setArimaRandomRanges(prev => ({
                        ...prev,
                        d_range: [parseInt(e.target.value) || 0, prev.d_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 3 }
                      }}
                      size="small"
                    />
                    <TextField
                      label="Rango d (max)"
                      type="number"
                      value={arimaRandomRanges.d_range[1]}
                      onChange={(e) => setArimaRandomRanges(prev => ({
                        ...prev,
                        d_range: [prev.d_range[0], parseInt(e.target.value) || 3]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 3 }
                      }}
                      size="small"
                    />
                    <TextField
                      label="Rango q (min)"
                      type="number"
                      value={arimaRandomRanges.q_range[0]}
                      onChange={(e) => setArimaRandomRanges(prev => ({
                        ...prev,
                        q_range: [parseInt(e.target.value) || 0, prev.q_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 5 }
                      }}
                      size="small"
                    />
                    <TextField
                      label="Rango q (max)"
                      type="number"
                      value={arimaRandomRanges.q_range[1]}
                      onChange={(e) => setArimaRandomRanges(prev => ({
                        ...prev,
                        q_range: [prev.q_range[0], parseInt(e.target.value) || 4]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 5 }
                      }}
                      size="small"
                    />
                  </Box>

                  {/* Seasonal parameters toggle and ranges */}
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={enableSeasonalParams}
                        onChange={(e) => setEnableSeasonalParams(e.target.checked)}
                        color="primary"
                      />
                    }
                    label="Incluir parámetros estacionales en Random Search"
                    sx={{ mb: 2 }}
                  />

                  {enableSeasonalParams && (
                    <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                      <TextField
                        label="Rango P estacional (min)"
                        type="number"
                        value={arimaRandomRanges.seasonal_P_range[0]}
                        onChange={(e) => setArimaRandomRanges(prev => ({
                          ...prev,
                          seasonal_P_range: [parseInt(e.target.value) || 0, prev.seasonal_P_range[1]]
                        }))}
                        slotProps={{
                          input: { min: 0, max: 3 }
                        }}
                        size="small"
                      />
                      <TextField
                        label="Rango P estacional (max)"
                        type="number"
                        value={arimaRandomRanges.seasonal_P_range[1]}
                        onChange={(e) => setArimaRandomRanges(prev => ({
                          ...prev,
                          seasonal_P_range: [prev.seasonal_P_range[0], parseInt(e.target.value) || 3]
                        }))}
                        slotProps={{
                          input: { min: 0, max: 3 }
                        }}
                        size="small"
                      />
                      <TextField
                        label="Rango D estacional (min)"
                        type="number"
                        value={arimaRandomRanges.seasonal_D_range[0]}
                        onChange={(e) => setArimaRandomRanges(prev => ({
                          ...prev,
                          seasonal_D_range: [parseInt(e.target.value) || 0, prev.seasonal_D_range[1]]
                        }))}
                        slotProps={{
                          input: { min: 0, max: 3 }
                        }}
                        size="small"
                      />
                      <TextField
                        label="Rango D estacional (max)"
                        type="number"
                        value={arimaRandomRanges.seasonal_D_range[1]}
                        onChange={(e) => setArimaRandomRanges(prev => ({
                          ...prev,
                          seasonal_D_range: [prev.seasonal_D_range[0], parseInt(e.target.value) || 3]
                        }))}
                        slotProps={{
                          input: { min: 0, max: 3 }
                        }}
                        size="small"
                      />
                      <TextField
                        label="Rango Q estacional (min)"
                        type="number"
                        value={arimaRandomRanges.seasonal_Q_range[0]}
                        onChange={(e) => setArimaRandomRanges(prev => ({
                          ...prev,
                          seasonal_Q_range: [parseInt(e.target.value) || 0, prev.seasonal_Q_range[1]]
                        }))}
                        slotProps={{
                          input: { min: 0, max: 3 }
                        }}
                        size="small"
                      />
                      <TextField
                        label="Rango Q estacional (max)"
                        type="number"
                        value={arimaRandomRanges.seasonal_Q_range[1]}
                        onChange={(e) => setArimaRandomRanges(prev => ({
                          ...prev,
                          seasonal_Q_range: [prev.seasonal_Q_range[0], parseInt(e.target.value) || 3]
                        }))}
                        slotProps={{
                          input: { min: 0, max: 3 }
                        }}
                        size="small"
                      />
                    </Box>
                  )}
                </Box>
              )}

              {/* Optimization Metric Dropdown - visible for Grid/Random/Bayesian Search */}
              {(optimizationMethod === "grid" || optimizationMethod === "random" || optimizationMethod === "bayesian") && (
                <FormControl fullWidth sx={{ mt: 2 }}>
                  <InputLabel id="optimization-metric-label">Optimization Metric</InputLabel>
                  <Select
                    labelId="optimization-metric-label"
                    value={optimizationMetric}
                    label="Optimization Metric"
                    onChange={(e) => setOptimizationMetric(e.target.value)}
                  >
                    <MenuItem value="val_rmse">Validation RMSE</MenuItem>
                    <MenuItem value="val_mae">Validation MAE</MenuItem>
                    <MenuItem value="val_mape">Validation MAPE</MenuItem>
                    <MenuItem value="test_rmse">Test RMSE</MenuItem>
                    <MenuItem value="test_mae">Test MAE</MenuItem>
                    <MenuItem value="test_mape">Test MAPE</MenuItem>
                  </Select>
                </FormControl>
              )}
            </>
          )}

          {/* Para XGBoost */}
          {algorithm === "xgboost" && (
            <>
              {useGridSearch && (
                <Box sx={{ mt: 2, mb: 3, p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 2 }}>
                    Grid Search Automático - XGBoost
                  </Typography>

                  <Typography variant="body2" sx={{ color: "#666", mb: 2 }}>
                    El sistema probará automáticamente 81 combinaciones de hiperparámetros con los siguientes valores:
                  </Typography>

                  <Box sx={{ pl: 2, mb: 2 }}>
                    <Typography variant="body2" sx={{ color: "#555", mb: 0.5 }}>
                      • <strong>n_estimators</strong>: 100, 200, 300 (número de árboles)
                    </Typography>
                    <Typography variant="body2" sx={{ color: "#555", mb: 0.5 }}>
                      • <strong>max_depth</strong>: 3, 5, 7 (profundidad máxima de cada árbol)
                    </Typography>
                    <Typography variant="body2" sx={{ color: "#555", mb: 0.5 }}>
                      • <strong>learning_rate</strong>: 0.01, 0.1, 0.2 (tasa de aprendizaje)
                    </Typography>
                    <Typography variant="body2" sx={{ color: "#555", mb: 0.5 }}>
                      • <strong>subsample</strong>: 0.8, 0.9, 1.0 (fracción de muestras por árbol)
                    </Typography>
                  </Box>

                  <Typography variant="body2" sx={{ color: "#666", fontStyle: "italic" }}>
                    El modelo con mejor rendimiento en validación (RMSE) será seleccionado automáticamente.
                  </Typography>
                </Box>
              )}
              {useRandomSearch && (
                <Box sx={{ mt: 2, mb: 3, p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 2 }}>
                    Rangos de hiperparámetros - XGBoost Random Search
                  </Typography>

                  <Typography variant="body2" sx={{ color: "#666", mb: 2 }}>
                    Define los rangos para la búsqueda aleatoria de hiperparámetros XGBoost
                  </Typography>

                  {/* XGBoost parameter ranges */}
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                    <TextField
                      label="n_estimators (min)"
                      type="number"
                      value={xgboostRandomRanges.n_estimators_range[0]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        n_estimators_range: [parseInt(e.target.value) || 50, prev.n_estimators_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 1, max: 2000 }
                      }}
                      size="small"
                      helperText="Número de árboles"
                    />
                    <TextField
                      label="n_estimators (max)"
                      type="number"
                      value={xgboostRandomRanges.n_estimators_range[1]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        n_estimators_range: [prev.n_estimators_range[0], parseInt(e.target.value) || 1000]
                      }))}
                      slotProps={{
                        input: { min: 1, max: 2000 }
                      }}
                      size="small"
                    />
                    <TextField
                      label="max_depth (min)"
                      type="number"
                      value={xgboostRandomRanges.max_depth_range[0]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        max_depth_range: [parseInt(e.target.value) || 1, prev.max_depth_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 1, max: 20 }
                      }}
                      size="small"
                      helperText="Profundidad máxima"
                    />
                    <TextField
                      label="max_depth (max)"
                      type="number"
                      value={xgboostRandomRanges.max_depth_range[1]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        max_depth_range: [prev.max_depth_range[0], parseInt(e.target.value) || 10]
                      }))}
                      slotProps={{
                        input: { min: 1, max: 20 }
                      }}
                      size="small"
                    />
                    <TextField
                      label="learning_rate (min)"
                      type="number"
                      step="0.001"
                      value={xgboostRandomRanges.learning_rate_range[0]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        learning_rate_range: [parseFloat(e.target.value) || 0.001, prev.learning_rate_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 0.001, max: 1, step: 0.001 }
                      }}
                      size="small"
                      helperText="Tasa de aprendizaje"
                    />
                    <TextField
                      label="learning_rate (max)"
                      type="number"
                      step="0.001"
                      value={xgboostRandomRanges.learning_rate_range[1]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        learning_rate_range: [prev.learning_rate_range[0], parseFloat(e.target.value) || 0.3]
                      }))}
                      slotProps={{
                        input: { min: 0.001, max: 1, step: 0.001 }
                      }}
                      size="small"
                    />
                    <TextField
                      label="subsample (min)"
                      type="number"
                      step="0.01"
                      value={xgboostRandomRanges.subsample_range[0]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        subsample_range: [parseFloat(e.target.value) || 0.1, prev.subsample_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 0.1, max: 1, step: 0.01 }
                      }}
                      size="small"
                      helperText="Fracción de muestras"
                    />
                    <TextField
                      label="subsample (max)"
                      type="number"
                      step="0.01"
                      value={xgboostRandomRanges.subsample_range[1]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        subsample_range: [prev.subsample_range[0], parseFloat(e.target.value) || 1.0]
                      }))}
                      slotProps={{
                        input: { min: 0.1, max: 1, step: 0.01 }
                      }}
                      size="small"
                    />
                    <TextField
                      label="colsample_bytree (min)"
                      type="number"
                      step="0.01"
                      value={xgboostRandomRanges.colsample_bytree_range[0]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        colsample_bytree_range: [parseFloat(e.target.value) || 0.1, prev.colsample_bytree_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 0.1, max: 1, step: 0.01 }
                      }}
                      size="small"
                      helperText="Fracción de características"
                    />
                    <TextField
                      label="colsample_bytree (max)"
                      type="number"
                      step="0.01"
                      value={xgboostRandomRanges.colsample_bytree_range[1]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        colsample_bytree_range: [prev.colsample_bytree_range[0], parseFloat(e.target.value) || 1.0]
                      }))}
                      slotProps={{
                        input: { min: 0.1, max: 1, step: 0.01 }
                      }}
                      size="small"
                    />
                    <TextField
                      label="gamma (min)"
                      type="number"
                      step="0.1"
                      value={xgboostRandomRanges.gamma_range[0]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        gamma_range: [parseFloat(e.target.value) || 0, prev.gamma_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 10, step: 0.1 }
                      }}
                      size="small"
                      helperText="Reducción mínima de pérdida"
                    />
                    <TextField
                      label="gamma (max)"
                      type="number"
                      step="0.1"
                      value={xgboostRandomRanges.gamma_range[1]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        gamma_range: [prev.gamma_range[0], parseFloat(e.target.value) || 5.0]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 10, step: 0.1 }
                      }}
                      size="small"
                    />
                    <TextField
                      label="min_child_weight (min)"
                      type="number"
                      value={xgboostRandomRanges.min_child_weight_range[0]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        min_child_weight_range: [parseInt(e.target.value) || 1, prev.min_child_weight_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 1, max: 20 }
                      }}
                      size="small"
                      helperText="Peso mínimo hijo"
                    />
                    <TextField
                      label="min_child_weight (max)"
                      type="number"
                      value={xgboostRandomRanges.min_child_weight_range[1]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        min_child_weight_range: [prev.min_child_weight_range[0], parseInt(e.target.value) || 10]
                      }))}
                      slotProps={{
                        input: { min: 1, max: 20 }
                      }}
                      size="small"
                    />
                    <TextField
                      label="reg_alpha (min)"
                      type="number"
                      step="0.01"
                      value={xgboostRandomRanges.reg_alpha_range[0]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        reg_alpha_range: [parseFloat(e.target.value) || 0, prev.reg_alpha_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 10, step: 0.01 }
                      }}
                      size="small"
                      helperText="Regularización L1"
                    />
                    <TextField
                      label="reg_alpha (max)"
                      type="number"
                      step="0.01"
                      value={xgboostRandomRanges.reg_alpha_range[1]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        reg_alpha_range: [prev.reg_alpha_range[0], parseFloat(e.target.value) || 1.0]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 10, step: 0.01 }
                      }}
                      size="small"
                    />
                    <TextField
                      label="reg_lambda (min)"
                      type="number"
                      step="0.01"
                      value={xgboostRandomRanges.reg_lambda_range[0]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        reg_lambda_range: [parseFloat(e.target.value) || 0, prev.reg_lambda_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 10, step: 0.01 }
                      }}
                      size="small"
                      helperText="Regularización L2"
                    />
                    <TextField
                      label="reg_lambda (max)"
                      type="number"
                      step="0.01"
                      value={xgboostRandomRanges.reg_lambda_range[1]}
                      onChange={(e) => setXgboostRandomRanges(prev => ({
                        ...prev,
                        reg_lambda_range: [prev.reg_lambda_range[0], parseFloat(e.target.value) || 1.0]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 10, step: 0.01 }
                      }}
                      size="small"
                    />
                  </Box>
                </Box>
              )}

              {/* XGBoost Bayesian Parameter Ranges */}
              {optimizationMethod === "bayesian" && (
                <Box sx={{ mt: 2 }}>
                  <Button
                    onClick={() => setShowBayesianParamRanges(!showBayesianParamRanges)}
                    variant="outlined"
                    size="small"
                    sx={{ mb: showBayesianParamRanges ? 2 : 0, textTransform: "none" }}
                  >
                    {showBayesianParamRanges ? "▼" : "▶"} Rangos de Parámetros (Opcional)
                  </Button>

                  {showBayesianParamRanges && (
                    <Box sx={{ p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#fff" }}>
                      <Typography variant="body2" sx={{ color: "#666", mb: 2 }}>
                        Configura los rangos de búsqueda Bayesiana para XGBoost
                      </Typography>

                      {/* Integer params */}
                      <Typography variant="body2" sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>
                        Parámetros enteros:
                      </Typography>
                      <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                        <TextField
                          label="n_estimators (min)"
                          type="number"
                          value={xgboostBayesianRanges.n_estimators.min}
                          onChange={(e) => setXgboostBayesianRanges(prev => ({
                            ...prev,
                            n_estimators: { ...prev.n_estimators, min: parseInt(e.target.value) || 50 }
                          }))}
                          slotProps={{ input: { min: 10, max: 1000 } }}
                          size="small"
                          helperText="Número de árboles"
                        />
                        <TextField
                          label="n_estimators (max)"
                          type="number"
                          value={xgboostBayesianRanges.n_estimators.max}
                          onChange={(e) => setXgboostBayesianRanges(prev => ({
                            ...prev,
                            n_estimators: { ...prev.n_estimators, max: parseInt(e.target.value) || 500 }
                          }))}
                          slotProps={{ input: { min: 10, max: 1000 } }}
                          size="small"
                        />
                        <TextField
                          label="max_depth (min)"
                          type="number"
                          value={xgboostBayesianRanges.max_depth.min}
                          onChange={(e) => setXgboostBayesianRanges(prev => ({
                            ...prev,
                            max_depth: { ...prev.max_depth, min: parseInt(e.target.value) || 3 }
                          }))}
                          slotProps={{ input: { min: 1, max: 15 } }}
                          size="small"
                          helperText="Profundidad máxima"
                        />
                        <TextField
                          label="max_depth (max)"
                          type="number"
                          value={xgboostBayesianRanges.max_depth.max}
                          onChange={(e) => setXgboostBayesianRanges(prev => ({
                            ...prev,
                            max_depth: { ...prev.max_depth, max: parseInt(e.target.value) || 10 }
                          }))}
                          slotProps={{ input: { min: 1, max: 15 } }}
                          size="small"
                        />
                        <TextField
                          label="min_child_weight (min)"
                          type="number"
                          value={xgboostBayesianRanges.min_child_weight.min}
                          onChange={(e) => setXgboostBayesianRanges(prev => ({
                            ...prev,
                            min_child_weight: { ...prev.min_child_weight, min: parseInt(e.target.value) || 1 }
                          }))}
                          slotProps={{ input: { min: 1, max: 20 } }}
                          size="small"
                        />
                        <TextField
                          label="min_child_weight (max)"
                          type="number"
                          value={xgboostBayesianRanges.min_child_weight.max}
                          onChange={(e) => setXgboostBayesianRanges(prev => ({
                            ...prev,
                            min_child_weight: { ...prev.min_child_weight, max: parseInt(e.target.value) || 10 }
                          }))}
                          slotProps={{ input: { min: 1, max: 20 } }}
                          size="small"
                        />
                      </Box>

                      {/* Float params */}
                      <Typography variant="body2" sx={{ fontWeight: "bold", color: "#004d40", mb: 1, mt: 2 }}>
                        Parámetros decimales:
                      </Typography>
                      <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 1 }}>
                        <TextField
                          label="learning_rate (min)"
                          type="number"
                          step="0.001"
                          value={xgboostBayesianRanges.learning_rate.min}
                          onChange={(e) => setXgboostBayesianRanges(prev => ({
                            ...prev,
                            learning_rate: { ...prev.learning_rate, min: parseFloat(e.target.value) || 0.001 }
                          }))}
                          slotProps={{ input: { min: 0.0001, max: 1, step: 0.001 } }}
                          size="small"
                        />
                        <TextField
                          label="learning_rate (max)"
                          type="number"
                          step="0.001"
                          value={xgboostBayesianRanges.learning_rate.max}
                          onChange={(e) => setXgboostBayesianRanges(prev => ({
                            ...prev,
                            learning_rate: { ...prev.learning_rate, max: parseFloat(e.target.value) || 0.1 }
                          }))}
                          slotProps={{ input: { min: 0.0001, max: 1, step: 0.001 } }}
                          size="small"
                        />
                      </Box>
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={xgboostBayesianRanges.learning_rate.log || false}
                            onChange={(e) => setXgboostBayesianRanges(prev => ({
                              ...prev,
                              learning_rate: { ...prev.learning_rate, log: e.target.checked }
                            }))}
                          />
                        }
                        label="Usar escala logarítmica (recomendado)"
                        sx={{ mb: 2 }}
                      />

                      <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                        <TextField
                          label="subsample (min)"
                          type="number"
                          step="0.1"
                          value={xgboostBayesianRanges.subsample.min}
                          onChange={(e) => setXgboostBayesianRanges(prev => ({
                            ...prev,
                            subsample: { ...prev.subsample, min: parseFloat(e.target.value) || 0.5 }
                          }))}
                          slotProps={{ input: { min: 0.1, max: 1.0, step: 0.1 } }}
                          size="small"
                        />
                        <TextField
                          label="subsample (max)"
                          type="number"
                          step="0.1"
                          value={xgboostBayesianRanges.subsample.max}
                          onChange={(e) => setXgboostBayesianRanges(prev => ({
                            ...prev,
                            subsample: { ...prev.subsample, max: parseFloat(e.target.value) || 1.0 }
                          }))}
                          slotProps={{ input: { min: 0.1, max: 1.0, step: 0.1 } }}
                          size="small"
                        />
                        <TextField
                          label="colsample_bytree (min)"
                          type="number"
                          step="0.1"
                          value={xgboostBayesianRanges.colsample_bytree.min}
                          onChange={(e) => setXgboostBayesianRanges(prev => ({
                            ...prev,
                            colsample_bytree: { ...prev.colsample_bytree, min: parseFloat(e.target.value) || 0.5 }
                          }))}
                          slotProps={{ input: { min: 0.1, max: 1.0, step: 0.1 } }}
                          size="small"
                        />
                        <TextField
                          label="colsample_bytree (max)"
                          type="number"
                          step="0.1"
                          value={xgboostBayesianRanges.colsample_bytree.max}
                          onChange={(e) => setXgboostBayesianRanges(prev => ({
                            ...prev,
                            colsample_bytree: { ...prev.colsample_bytree, max: parseFloat(e.target.value) || 1.0 }
                          }))}
                          slotProps={{ input: { min: 0.1, max: 1.0, step: 0.1 } }}
                          size="small"
                        />
                        <TextField
                          label="gamma (min)"
                          type="number"
                          step="0.1"
                          value={xgboostBayesianRanges.gamma.min}
                          onChange={(e) => setXgboostBayesianRanges(prev => ({
                            ...prev,
                            gamma: { ...prev.gamma, min: parseFloat(e.target.value) || 0.0 }
                          }))}
                          slotProps={{ input: { min: 0, max: 5, step: 0.1 } }}
                          size="small"
                        />
                        <TextField
                          label="gamma (max)"
                          type="number"
                          step="0.1"
                          value={xgboostBayesianRanges.gamma.max}
                          onChange={(e) => setXgboostBayesianRanges(prev => ({
                            ...prev,
                            gamma: { ...prev.gamma, max: parseFloat(e.target.value) || 1.0 }
                          }))}
                          slotProps={{ input: { min: 0, max: 5, step: 0.1 } }}
                          size="small"
                        />
                      </Box>
                    </Box>
                  )}
                </Box>
              )}

              {optimizationMethod === "manual" && (
                <Box sx={{ mt: 2, mb: 3, p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 2 }}>
                    Hiperparámetros - XGBoost Manual
                  </Typography>

                  <Typography variant="body2" sx={{ color: "#666", mb: 2 }}>
                    Configure los hiperparámetros manualmente para el modelo XGBoost
                  </Typography>

                  {/* n_estimators */}
                  <TextField
                    fullWidth
                    type="number"
                    label="n_estimators"
                    value={xgBoostParams.n_estimators}
                    onChange={(e) => setXgboostParams({ ...xgBoostParams, n_estimators: parseInt(e.target.value) || 0 })}
                    slotProps={{
                      input: {
                        min: 0
                      }
                    }}
                    helperText="Número de árboles (0 = automático, mínimo: 0)"
                    sx={{ mb: 2 }}
                  />

                  {/* max_depth */}
                  <TextField
                    fullWidth
                    type="number"
                    label="max_depth"
                    value={xgBoostParams.max_depth}
                    onChange={(e) => setXgboostParams({ ...xgBoostParams, max_depth: Math.max(1, parseInt(e.target.value) || 6) })}
                    slotProps={{
                      input: {
                        min: 1
                      }
                    }}
                    helperText="Profundidad máxima del árbol (mínimo: 1)"
                    sx={{ mb: 2 }}
                  />

                  {/* learning_rate with slider and text input */}
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mt: 2, mb: 1 }}>
                    Learning Rate:
                  </Typography>
                  <Box sx={{ display: "flex", gap: 2, alignItems: "center", mb: 2 }}>
                    <Slider
                      value={xgBoostParams.learning_rate}
                      onChange={(_, newValue) => setXgboostParams({ ...xgBoostParams, learning_rate: newValue })}
                      min={0.001}
                      max={0.999}
                      step={0.001}
                      valueLabelDisplay="auto"
                      sx={{ flex: 1 }}
                    />
                    <TextField
                      type="number"
                      value={xgBoostParams.learning_rate}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val) && val >= 0.001 && val <= 0.999) {
                          setXgboostParams({ ...xgBoostParams, learning_rate: val });
                        }
                      }}
                      slotProps={{
                        input: {
                          step: 0.001,
                          min: 0.001,
                          max: 0.999
                        }
                      }}
                      sx={{ width: "120px" }}
                    />
                  </Box>

                  {/* subsample with slider and text input */}
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mt: 2, mb: 1 }}>
                    Subsample:
                  </Typography>
                  <Box sx={{ display: "flex", gap: 2, alignItems: "center", mb: 2 }}>
                    <Slider
                      value={xgBoostParams.subsample}
                      onChange={(_, newValue) => setXgboostParams({ ...xgBoostParams, subsample: newValue })}
                      min={0.00}
                      max={1.00}
                      step={0.01}
                      valueLabelDisplay="auto"
                      sx={{ flex: 1 }}
                    />
                    <TextField
                      type="number"
                      value={xgBoostParams.subsample}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val) && val >= 0.00 && val <= 1.00) {
                          setXgboostParams({ ...xgBoostParams, subsample: val });
                        }
                      }}
                      slotProps={{
                        input: {
                          step: 0.01,
                          min: 0.00,
                          max: 1.00
                        }
                      }}
                      sx={{ width: "120px" }}
                    />
                  </Box>

                  {/* gamma */}
                  <TextField
                    fullWidth
                    type="number"
                    label="gamma"
                    value={xgBoostParams.gamma}
                    onChange={(e) => setXgboostParams({ ...xgBoostParams, gamma: Math.max(0.0, parseFloat(e.target.value) || 0.0) })}
                    slotProps={{
                      input: {
                        step: 0.01,
                        min: 0.0
                      }
                    }}
                    helperText="Reducción mínima de pérdida para dividir (mínimo: 0.0)"
                    sx={{ mb: 2 }}
                  />

                  {/* min_child_weight */}
                  <TextField
                    fullWidth
                    type="number"
                    label="min_child_weight"
                    value={xgBoostParams.min_child_weight}
                    onChange={(e) => setXgboostParams({ ...xgBoostParams, min_child_weight: Math.max(1, parseInt(e.target.value) || 1) })}
                    slotProps={{
                      input: {
                        min: 1
                      }
                    }}
                    helperText="Peso mínimo requerido en nodo hijo (mínimo: 1)"
                    sx={{ mb: 2 }}
                  />
                </Box>
              )}

              {/* Optimization Metric Dropdown - visible for Grid/Random/Bayesian Search */}
              {(useGridSearch || useRandomSearch || optimizationMethod === "bayesian") && (
                <FormControl fullWidth sx={{ mt: 2 }}>
                  <InputLabel id="optimization-metric-label-xgboost">Optimization Metric</InputLabel>
                  <Select
                    labelId="optimization-metric-label-xgboost"
                    value={optimizationMetric}
                    label="Optimization Metric"
                    onChange={(e) => setOptimizationMetric(e.target.value)}
                  >
                    <MenuItem value="val_rmse">Validation RMSE</MenuItem>
                    <MenuItem value="val_mae">Validation MAE</MenuItem>
                    <MenuItem value="val_mape">Validation MAPE</MenuItem>
                    <MenuItem value="test_rmse">Test RMSE</MenuItem>
                    <MenuItem value="test_mae">Test MAE</MenuItem>
                    <MenuItem value="test_mape">Test MAPE</MenuItem>
                  </Select>
                </FormControl>
              )}
            </>
          )}

          {/* Para LSTM */}
          {algorithm === "lstm" && (
            <>

              {/* CPU-Only Warning Banner - Persistent */}
              <Alert severity="info" sx={{ mb: 2, mt: 2 }}>
                ℹ️ LSTM Training Notice: CPU-only mode (no GPU acceleration). Training may take multiple minutes for typical configurations.
              </Alert>

              {/* Sequence Length - common to all optimization methods */}
              <Typography sx={{ fontWeight: "bold", color: "#004d40", mt: 2, mb: 1 }}>
                Sequence Length (Lookback Window):
              </Typography>
              <TextField
                fullWidth
                type="number"
                value={sequenceLength}
                onChange={(e) => setSequenceLength(Math.max(1, parseInt(e.target.value) || 10))}
                slotProps={{
                  input: {
                    min: 1,
                    max: 100
                  }
                }}
                helperText="Número de pasos temporales previos a considerar (típico: 5-20)"
                sx={{ mb: 2 }}
              />

              <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>
                Early Stopping Patience:
              </Typography>
              <TextField
                fullWidth
                type="number"
                value={earlyStoppingPatience}
                onChange={(e) => setEarlyStoppingPatience(Math.max(1, parseInt(e.target.value) || 20))}
                slotProps={{
                  input: {
                    min: 1,
                    max: 100
                  }
                }}
                helperText="Épocas sin mejora antes de detener entrenamiento (típico: 10-30)"
                sx={{ mb: 2 }}
              />

              {optimizationMethod === "manual" ? (
                <>
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mt: 2 }}>
                    Hiperparámetros - LSTM
                  </Typography>

                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mt: 2, mb: 1 }}>
                    LSTM Units (Architecture):
                  </Typography>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Arquitectura de red</InputLabel>
                    <Select
                      value={lstmManualParams.lstm_units}
                      onChange={(e) => setLstmManualParams({ ...lstmManualParams, lstm_units: e.target.value })}
                      label="Arquitectura de red"
                    >
                      <MenuItem value="[32]">32 unidades (1 capa)</MenuItem>
                      <MenuItem value="[64]">64 unidades (1 capa)</MenuItem>
                      <MenuItem value="[128]">128 unidades (1 capa)</MenuItem>
                      <MenuItem value="[64,32]">64 → 32 unidades (2 capas)</MenuItem>
                      <MenuItem value="[128,64]">128 → 64 unidades (2 capas)</MenuItem>
                    </Select>
                  </FormControl>

                  <Box sx={{ display: "flex", gap: "10px", mb: 2 }}>
                    <TextField
                      type="number"
                      label="Dropout Rate"
                      value={lstmManualParams.dropout_rate}
                      onChange={(e) => setLstmManualParams({ ...lstmManualParams, dropout_rate: e.target.value })}
                      slotProps={{
                        input: {
                          step: 0.05,
                          min: 0,
                          max: 0.8
                        }
                      }}
                      helperText="0.0 - 0.5 típico"
                      sx={{ width: "50%" }}
                    />
                    <TextField
                      type="number"
                      label="Recurrent Dropout"
                      value={lstmManualParams.recurrent_dropout_rate}
                      onChange={(e) => setLstmManualParams({ ...lstmManualParams, recurrent_dropout_rate: e.target.value })}
                      slotProps={{
                        input: {
                          step: 0.05,
                          min: 0,
                          max: 0.8
                        }
                      }}
                      helperText="0.0 - 0.5 típico"
                      sx={{ width: "50%" }}
                    />
                  </Box>

                  <Box sx={{ display: "flex", gap: "10px", mb: 2 }}>
                    <TextField
                      type="number"
                      label="Learning Rate"
                      value={lstmManualParams.learning_rate}
                      onChange={(e) => setLstmManualParams({ ...lstmManualParams, learning_rate: e.target.value })}
                      slotProps={{
                        input: {
                          step: 0.0001,
                          min: 0.0001,
                          max: 0.1
                        }
                      }}
                      helperText="0.0001 - 0.01 típico"
                      sx={{ width: "33%" }}
                    />
                    <TextField
                      type="number"
                      label="Batch Size"
                      value={lstmManualParams.batch_size}
                      onChange={(e) => setLstmManualParams({ ...lstmManualParams, batch_size: e.target.value })}
                      slotProps={{
                        input: {
                          min: 1,
                          max: 256
                        }
                      }}
                      helperText="16, 32, 64 típico"
                      sx={{ width: "33%" }}
                    />
                    <TextField
                      type="number"
                      label="Epochs"
                      value={lstmManualParams.epochs}
                      onChange={(e) => setLstmManualParams({ ...lstmManualParams, epochs: e.target.value })}
                      slotProps={{
                        input: {
                          min: 10,
                          max: 500
                        }
                      }}
                      helperText="50 - 300 típico"
                      sx={{ width: "33%" }}
                    />
                  </Box>
                </>
              ) : optimizationMethod === "grid" ? (
                /* Grid Search parameter options for LSTM (Phase 2A) */
                <Box sx={{ mt: 2, mb: 3, p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 2 }}>
                    Opciones de Grid Search - LSTM (Fase 2A)
                  </Typography>

                  <Typography variant="body2" sx={{ color: "#666", mb: 2 }}>
                    Define valores específicos para cada hiperparámetro (separados por comas).
                    Se probarán todas las combinaciones posibles.
                  </Typography>

                  <TextField
                    fullWidth
                    label="LSTM Units Options"
                    value={lstmGridOptions.lstm_units_options}
                    onChange={(e) => setLstmGridOptions({ ...lstmGridOptions, lstm_units_options: e.target.value })}
                    helperText="Ej: [64], [128], [64,32] (separar con comas)"
                    sx={{ mb: 2 }}
                  />

                  <TextField
                    fullWidth
                    label="Dropout Rate Options"
                    value={lstmGridOptions.dropout_rate_options}
                    onChange={(e) => setLstmGridOptions({ ...lstmGridOptions, dropout_rate_options: e.target.value })}
                    helperText="Ej: 0.2, 0.3, 0.4 (separar con comas)"
                    sx={{ mb: 2 }}
                  />

                  <TextField
                    fullWidth
                    label="Recurrent Dropout Options"
                    value={lstmGridOptions.recurrent_dropout_rate_options}
                    onChange={(e) => setLstmGridOptions({ ...lstmGridOptions, recurrent_dropout_rate_options: e.target.value })}
                    helperText="Ej: 0.2, 0.3 (separar con comas)"
                    sx={{ mb: 2 }}
                  />

                  <TextField
                    fullWidth
                    label="Learning Rate Options"
                    value={lstmGridOptions.learning_rate_options}
                    onChange={(e) => setLstmGridOptions({ ...lstmGridOptions, learning_rate_options: e.target.value })}
                    helperText="Ej: 0.001, 0.01 (separar con comas)"
                    sx={{ mb: 2 }}
                  />

                  <TextField
                    fullWidth
                    label="Batch Size Options"
                    value={lstmGridOptions.batch_size_options}
                    onChange={(e) => setLstmGridOptions({ ...lstmGridOptions, batch_size_options: e.target.value })}
                    helperText="Ej: 16, 32, 64 (separar con comas)"
                    sx={{ mb: 2 }}
                  />

                  <TextField
                    fullWidth
                    label="Epochs Options"
                    value={lstmGridOptions.epochs_options}
                    onChange={(e) => setLstmGridOptions({ ...lstmGridOptions, epochs_options: e.target.value })}
                    helperText="Ej: 50, 100, 150 (separar con comas)"
                    sx={{ mb: 2 }}
                  />

                  {/* Advanced Settings */}
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mt: 3, mb: 1 }}>
                    Configuración Avanzada:
                  </Typography>

                  <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                    <Tooltip
                      title="Tracks memory usage during training iterations. Logs initial memory, final memory, and increase in MB to MLflow. Useful for debugging memory leaks. Has minimal performance impact on training speed."
                      arrow
                      placement="right"
                    >
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={enableMemoryProfiling}
                            onChange={(e) => setEnableMemoryProfiling(e.target.checked)}
                          />
                        }
                        label="Habilitar Perfilado de Memoria (para debugging)"
                      />
                    </Tooltip>
                  </Box>

                  <TextField
                    fullWidth
                    type="number"
                    label="Grid Warning Threshold"
                    value={gridWarningThreshold}
                    onChange={(e) => setGridWarningThreshold(Math.max(1, parseInt(e.target.value) || 50))}
                    slotProps={{
                      input: {
                        min: 1,
                        max: 500
                      }
                    }}
                    helperText="Mostrar advertencia si las combinaciones exceden este número (default: 50)"
                    sx={{ mb: 2 }}
                  />

                  {/* Grid Search Combination Warning - Dismissible */}
                  {gridCombinationsCount !== null &&
                   gridCombinationsCount > gridWarningThreshold &&
                   !gridWarningDismissed && (
                    <Alert
                      severity="warning"
                      onClose={() => setGridWarningDismissed(true)}
                      sx={{ mb: 2 }}
                    >
                      ⚠️ {gridCombinationsCount} combinaciones detectadas (&gt;{gridWarningThreshold} threshold). Entrenamiento puede ser lento.
                    </Alert>
                  )}

                  {/* Show "?" if calculation failed */}
                  {gridCombinationsCount === null && (
                    <Alert severity="info" sx={{ mb: 2 }}>
                      ℹ️ No se puede calcular número de combinaciones (formato inválido en opciones).
                    </Alert>
                  )}

                  <Typography variant="caption" sx={{ color: "#FF6F00", display: "block", mt: 1 }}>
                    ⚠️ Número de combinaciones = producto de todas las opciones.
                    Ejemplo: 2×2×1×2×1×1 = 8 combinaciones (valor por defecto).
                  </Typography>
                </Box>
              ) : optimizationMethod === "random" ? (
                /* Random Search parameter ranges for LSTM */
                <Box sx={{ mt: 2, mb: 3, p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 2 }}>
                    Rangos de hiperparámetros - LSTM Random Search
                  </Typography>

                  <Typography variant="body2" sx={{ color: "#666", mb: 2 }}>
                    Define los rangos para la búsqueda aleatoria de hiperparámetros LSTM
                  </Typography>

                  {/* LSTM Units - categorical selection */}
                  <Typography variant="body2" sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>
                    Opciones de arquitectura (se elegirá aleatoriamente):
                  </Typography>
                  <Typography variant="caption" sx={{ color: "#666", mb: 2, display: "block" }}>
                    Opciones disponibles: [32], [64], [128], [64,32], [128,64]
                  </Typography>

                  {/* Dropout ranges */}
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                    <TextField
                      label="Dropout Rate (min)"
                      type="number"
                      step="0.05"
                      value={lstmRandomRanges.dropout_rate_range[0]}
                      onChange={(e) => setLstmRandomRanges(prev => ({
                        ...prev,
                        dropout_rate_range: [parseFloat(e.target.value) || 0, prev.dropout_rate_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 0.8, step: 0.05 }
                      }}
                      size="small"
                      helperText="Dropout después de LSTM"
                    />
                    <TextField
                      label="Dropout Rate (max)"
                      type="number"
                      step="0.05"
                      value={lstmRandomRanges.dropout_rate_range[1]}
                      onChange={(e) => setLstmRandomRanges(prev => ({
                        ...prev,
                        dropout_rate_range: [prev.dropout_rate_range[0], parseFloat(e.target.value) || 0.5]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 0.8, step: 0.05 }
                      }}
                      size="small"
                    />
                    <TextField
                      label="Recurrent Dropout (min)"
                      type="number"
                      step="0.05"
                      value={lstmRandomRanges.recurrent_dropout_rate_range[0]}
                      onChange={(e) => setLstmRandomRanges(prev => ({
                        ...prev,
                        recurrent_dropout_rate_range: [parseFloat(e.target.value) || 0, prev.recurrent_dropout_rate_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 0.8, step: 0.05 }
                      }}
                      size="small"
                      helperText="Dropout dentro de LSTM"
                    />
                    <TextField
                      label="Recurrent Dropout (max)"
                      type="number"
                      step="0.05"
                      value={lstmRandomRanges.recurrent_dropout_rate_range[1]}
                      onChange={(e) => setLstmRandomRanges(prev => ({
                        ...prev,
                        recurrent_dropout_rate_range: [prev.recurrent_dropout_rate_range[0], parseFloat(e.target.value) || 0.5]
                      }))}
                      slotProps={{
                        input: { min: 0, max: 0.8, step: 0.05 }
                      }}
                      size="small"
                    />
                    <TextField
                      label="Learning Rate (min)"
                      type="number"
                      step="0.0001"
                      value={lstmRandomRanges.learning_rate_range[0]}
                      onChange={(e) => setLstmRandomRanges(prev => ({
                        ...prev,
                        learning_rate_range: [parseFloat(e.target.value) || 0.0001, prev.learning_rate_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 0.0001, max: 0.1, step: 0.0001 }
                      }}
                      size="small"
                      helperText="Tasa de aprendizaje"
                    />
                    <TextField
                      label="Learning Rate (max)"
                      type="number"
                      step="0.001"
                      value={lstmRandomRanges.learning_rate_range[1]}
                      onChange={(e) => setLstmRandomRanges(prev => ({
                        ...prev,
                        learning_rate_range: [prev.learning_rate_range[0], parseFloat(e.target.value) || 0.01]
                      }))}
                      slotProps={{
                        input: { min: 0.0001, max: 0.1, step: 0.001 }
                      }}
                      size="small"
                    />
                  </Box>

                  {/* Learning Rate Distribution - Display Only */}
                  <TextField
                    fullWidth
                    label="Learning Rate Distribution"
                    value="log-uniform"
                    disabled
                    helperText="El learning rate se muestrea usando distribución log-uniforme (mejor para rangos exponenciales)"
                    sx={{ mb: 2 }}
                  />

                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                    <TextField
                      label="Epochs (min)"
                      type="number"
                      value={lstmRandomRanges.epochs_range[0]}
                      onChange={(e) => setLstmRandomRanges(prev => ({
                        ...prev,
                        epochs_range: [parseInt(e.target.value) || 50, prev.epochs_range[1]]
                      }))}
                      slotProps={{
                        input: { min: 10, max: 500 }
                      }}
                      size="small"
                      helperText="Número de épocas"
                    />
                    <TextField
                      label="Epochs (max)"
                      type="number"
                      value={lstmRandomRanges.epochs_range[1]}
                      onChange={(e) => setLstmRandomRanges(prev => ({
                        ...prev,
                        epochs_range: [prev.epochs_range[0], parseInt(e.target.value) || 300]
                      }))}
                      slotProps={{
                        input: { min: 10, max: 500 }
                      }}
                      size="small"
                    />
                  </Box>

                  <Typography variant="body2" sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>
                    Batch Size (opciones disponibles: 16, 32, 64)
                  </Typography>
                </Box>
              ) : (
                /* Bayesian Search param_ranges for LSTM (Phase 9) */
                <Box sx={{ mt: 2 }}>
                  <Button
                    onClick={() => setShowBayesianParamRanges(!showBayesianParamRanges)}
                    variant="outlined"
                    size="small"
                    sx={{ mb: showBayesianParamRanges ? 2 : 0, textTransform: "none" }}
                  >
                    {showBayesianParamRanges ? "▼" : "▶"} Rangos de Parámetros (Opcional)
                  </Button>

                  {showBayesianParamRanges && (
                    <Box sx={{ p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#fff" }}>
                      <Typography variant="body2" sx={{ color: "#666", mb: 2, fontStyle: "italic" }}>
                        Personaliza los rangos de búsqueda para cada parámetro. Si no se especifica, se usan valores predeterminados.
                      </Typography>

                      {/* Section 1: Categorical Parameters */}
                      <Typography variant="body2" sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>
                        Parámetros categóricos:
                      </Typography>

                      {/* lstm_units - Multi-select checkboxes */}
                      <FormControl component="fieldset" sx={{ mb: 2 }}>
                        <FormLabel component="legend">LSTM Units</FormLabel>
                        <FormGroup row>
                          {[32, 64, 128, 256].map(option => (
                            <FormControlLabel
                              key={option}
                              control={
                                <Checkbox
                                  checked={lstmBayesianRanges.lstm_units.choices.includes(option)}
                                  onChange={(e) => {
                                    const newChoices = e.target.checked
                                      ? [...lstmBayesianRanges.lstm_units.choices, option]
                                      : lstmBayesianRanges.lstm_units.choices.filter(c => c !== option);
                                    setLstmBayesianRanges(prev => ({
                                      ...prev,
                                      lstm_units: { choices: newChoices }
                                    }));
                                  }}
                                />
                              }
                              label={option}
                            />
                          ))}
                        </FormGroup>
                        <FormHelperText>Número de unidades LSTM por capa</FormHelperText>
                      </FormControl>

                      {/* batch_size - Multi-select checkboxes */}
                      <FormControl component="fieldset" sx={{ mb: 2 }}>
                        <FormLabel component="legend">Batch Size</FormLabel>
                        <FormGroup row>
                          {[16, 32, 64, 128].map(option => (
                            <FormControlLabel
                              key={option}
                              control={
                                <Checkbox
                                  checked={lstmBayesianRanges.batch_size.choices.includes(option)}
                                  onChange={(e) => {
                                    const newChoices = e.target.checked
                                      ? [...lstmBayesianRanges.batch_size.choices, option]
                                      : lstmBayesianRanges.batch_size.choices.filter(c => c !== option);
                                    setLstmBayesianRanges(prev => ({
                                      ...prev,
                                      batch_size: { choices: newChoices }
                                    }));
                                  }}
                                />
                              }
                              label={option}
                            />
                          ))}
                        </FormGroup>
                        <FormHelperText>Tamaño del lote para entrenamiento</FormHelperText>
                      </FormControl>

                      {/* Section 2: Numeric Integer Parameters */}
                      <Typography variant="body2" sx={{ fontWeight: "bold", color: "#004d40", mb: 1, mt: 2 }}>
                        Parámetros enteros:
                      </Typography>
                      <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                        <TextField
                          label="epochs (min)"
                          type="number"
                          value={lstmBayesianRanges.epochs.min}
                          onChange={(e) => setLstmBayesianRanges(prev => ({
                            ...prev,
                            epochs: { ...prev.epochs, min: parseInt(e.target.value) || 30 }
                          }))}
                          slotProps={{ input: { min: 10, max: 500 } }}
                          size="small"
                          helperText="Número de épocas de entrenamiento"
                        />
                        <TextField
                          label="epochs (max)"
                          type="number"
                          value={lstmBayesianRanges.epochs.max}
                          onChange={(e) => setLstmBayesianRanges(prev => ({
                            ...prev,
                            epochs: { ...prev.epochs, max: parseInt(e.target.value) || 100 }
                          }))}
                          slotProps={{ input: { min: 10, max: 500 } }}
                          size="small"
                        />

                        <TextField
                          label="time_steps (min)"
                          type="number"
                          value={lstmBayesianRanges.time_steps.min}
                          onChange={(e) => setLstmBayesianRanges(prev => ({
                            ...prev,
                            time_steps: { ...prev.time_steps, min: parseInt(e.target.value) || 5 }
                          }))}
                          slotProps={{ input: { min: 1, max: 100 } }}
                          size="small"
                          helperText="Pasos temporales para secuencias"
                        />
                        <TextField
                          label="time_steps (max)"
                          type="number"
                          value={lstmBayesianRanges.time_steps.max}
                          onChange={(e) => setLstmBayesianRanges(prev => ({
                            ...prev,
                            time_steps: { ...prev.time_steps, max: parseInt(e.target.value) || 30 }
                          }))}
                          slotProps={{ input: { min: 1, max: 100 } }}
                          size="small"
                        />
                      </Box>

                      {/* Section 3: Float Parameters */}
                      <Typography variant="body2" sx={{ fontWeight: "bold", color: "#004d40", mb: 1, mt: 2 }}>
                        Parámetros decimales:
                      </Typography>

                      {/* dropout_rate */}
                      <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 1 }}>
                        <TextField
                          label="dropout_rate (min)"
                          type="number"
                          step="0.01"
                          value={lstmBayesianRanges.dropout_rate.min}
                          onChange={(e) => setLstmBayesianRanges(prev => ({
                            ...prev,
                            dropout_rate: { ...prev.dropout_rate, min: parseFloat(e.target.value) || 0.1 }
                          }))}
                          slotProps={{ input: { min: 0, max: 1, step: 0.01 } }}
                          size="small"
                          helperText="Tasa de dropout"
                        />
                        <TextField
                          label="dropout_rate (max)"
                          type="number"
                          step="0.01"
                          value={lstmBayesianRanges.dropout_rate.max}
                          onChange={(e) => setLstmBayesianRanges(prev => ({
                            ...prev,
                            dropout_rate: { ...prev.dropout_rate, max: parseFloat(e.target.value) || 0.4 }
                          }))}
                          slotProps={{ input: { min: 0, max: 1, step: 0.01 } }}
                          size="small"
                        />
                      </Box>

                      {/* learning_rate with log scale checkbox */}
                      <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 1 }}>
                        <TextField
                          label="learning_rate (min)"
                          type="number"
                          step="0.0001"
                          value={lstmBayesianRanges.learning_rate.min}
                          onChange={(e) => setLstmBayesianRanges(prev => ({
                            ...prev,
                            learning_rate: { ...prev.learning_rate, min: parseFloat(e.target.value) || 0.0001 }
                          }))}
                          slotProps={{ input: { min: 0.00001, max: 1, step: 0.0001 } }}
                          size="small"
                          helperText="Tasa de aprendizaje"
                        />
                        <TextField
                          label="learning_rate (max)"
                          type="number"
                          step="0.0001"
                          value={lstmBayesianRanges.learning_rate.max}
                          onChange={(e) => setLstmBayesianRanges(prev => ({
                            ...prev,
                            learning_rate: { ...prev.learning_rate, max: parseFloat(e.target.value) || 0.01 }
                          }))}
                          slotProps={{ input: { min: 0.00001, max: 1, step: 0.0001 } }}
                          size="small"
                        />
                      </Box>
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={lstmBayesianRanges.learning_rate.log || false}
                            onChange={(e) => setLstmBayesianRanges(prev => ({
                              ...prev,
                              learning_rate: { ...prev.learning_rate, log: e.target.checked }
                            }))}
                          />
                        }
                        label="Usar escala logarítmica para learning_rate (recomendado)"
                        sx={{ mb: 2 }}
                      />

                      {/* recurrent_dropout_rate - informational (synced with dropout_rate on backend) */}
                      <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 1 }}>
                        <TextField
                          label="recurrent_dropout_rate (min)"
                          type="number"
                          step="0.01"
                          value={lstmBayesianRanges.recurrent_dropout_rate.min}
                          onChange={(e) => setLstmBayesianRanges(prev => ({
                            ...prev,
                            recurrent_dropout_rate: { ...prev.recurrent_dropout_rate, min: parseFloat(e.target.value) || 0.1 }
                          }))}
                          slotProps={{ input: { min: 0, max: 1, step: 0.01 } }}
                          size="small"
                          helperText="Dropout recurrente (sincronizado con dropout_rate)"
                        />
                        <TextField
                          label="recurrent_dropout_rate (max)"
                          type="number"
                          step="0.01"
                          value={lstmBayesianRanges.recurrent_dropout_rate.max}
                          onChange={(e) => setLstmBayesianRanges(prev => ({
                            ...prev,
                            recurrent_dropout_rate: { ...prev.recurrent_dropout_rate, max: parseFloat(e.target.value) || 0.4 }
                          }))}
                          slotProps={{ input: { min: 0, max: 1, step: 0.01 } }}
                          size="small"
                        />
                      </Box>
                      <Typography variant="caption" sx={{ color: "#666", mb: 2, display: "block", fontStyle: "italic" }}>
                        Nota: recurrent_dropout_rate se sincroniza automáticamente con dropout_rate en el backend
                      </Typography>

                    </Box>
                  )}
                </Box>
              )}

              {/* Optimization Metric Dropdown - visible for Grid/Random/Bayesian Search */}
              {(optimizationMethod === "grid" || optimizationMethod === "random" || optimizationMethod === "bayesian") && (
                <FormControl fullWidth sx={{ mt: 2 }}>
                  <InputLabel id="optimization-metric-label-lstm">Optimization Metric</InputLabel>
                  <Select
                    labelId="optimization-metric-label-lstm"
                    value={optimizationMetric}
                    label="Optimization Metric"
                    onChange={(e) => setOptimizationMetric(e.target.value)}
                  >
                    <MenuItem value="val_rmse">Validation RMSE</MenuItem>
                    <MenuItem value="val_mae">Validation MAE</MenuItem>
                    <MenuItem value="val_mape">Validation MAPE</MenuItem>
                    <MenuItem value="test_rmse">Test RMSE</MenuItem>
                    <MenuItem value="test_mae">Test MAE</MenuItem>
                    <MenuItem value="test_mape">Test MAPE</MenuItem>
                  </Select>
                </FormControl>
              )}

              {/* Feature Engineering Section for LSTM */}
            </>
          )}

          {/* ============ PatchTSMixer Section ============ */}
          {algorithm === "patchtsmixer" && (
            <>
              {/* Channel Selection Header */}
              <Typography
                variant="subtitle1"
                sx={{
                  fontWeight: "bold",
                  color: "#004d40",
                  mt: 3,
                  mb: 1
                }}
              >
                Selección de Canales de Series Temporales
                <Tooltip title="PatchTSMixer pronostica todos los canales seleccionados simultáneamente. Todas las variables seleccionadas se usan como entradas y salidas.">
                  <InfoIcon sx={{ ml: 1, fontSize: 18, color: "#00796b", cursor: "pointer" }} />
                </Tooltip>
              </Typography>

              {/* Info Banner */}
              <Box sx={{
                mt: 1,
                mb: 2,
                p: 2,
                backgroundColor: '#e3f2fd',
                borderRadius: 1,
                border: '1px solid #90caf9'
              }}>
                <Typography sx={{ color: '#1565c0', fontSize: '0.9rem' }}>
                  <strong>PatchTSMixer:</strong> Modelo transformer ligero para pronóstico de series temporales multivariadas.
                  Pronostica todos los canales simultáneamente usando un enfoque de parches (patches).
                </Typography>
              </Box>

              {/* Select All / Deselect All Buttons */}
              <Box sx={{ mb: 1 }}>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => {
                    const selectableColumns = columns.filter(col => col !== dateColumnName);
                    setPatchTSMixerChannels(selectableColumns);
                  }}
                  sx={{ mr: 1, color: "#00796b", borderColor: "#00796b" }}
                >
                  Seleccionar Todos
                </Button>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => setPatchTSMixerChannels([])}
                  sx={{ color: "#00796b", borderColor: "#00796b" }}
                >
                  Deseleccionar Todos
                </Button>
              </Box>

              {/* Channel Checkboxes */}
              <FormGroup row sx={{ mb: 2 }}>
                {columns
                  .filter(col => col !== dateColumnName)
                  .map(col => (
                    <FormControlLabel
                      key={col}
                      control={
                        <Checkbox
                          checked={patchTSMixerChannels.includes(col)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setPatchTSMixerChannels([...patchTSMixerChannels, col]);
                            } else {
                              setPatchTSMixerChannels(patchTSMixerChannels.filter(c => c !== col));
                            }
                          }}
                          sx={{ color: "#00796b", '&.Mui-checked': { color: "#00796b" } }}
                        />
                      }
                      label={col}
                    />
                  ))}
              </FormGroup>

              {/* Preset Dropdown */}
              <FormControl fullWidth sx={{ mt: 2, mb: 2 }}>
                <InputLabel id="patchtsmixer-preset-label">Cargar Configuración Preestablecida</InputLabel>
                <Select
                  labelId="patchtsmixer-preset-label"
                  value={patchTSMixerPreset}
                  onChange={(e) => loadPatchTSMixerPreset(e.target.value)}
                  label="Cargar Configuración Preestablecida"
                >
                  <MenuItem value="small">Small (Rápido, menos preciso)</MenuItem>
                  <MenuItem value="medium">Medium (Equilibrado)</MenuItem>
                  <MenuItem value="large">Large (Lento, más preciso)</MenuItem>
                </Select>
              </FormControl>

              {/* Essential Hyperparameters Section */}
              <Typography
                variant="subtitle1"
                sx={{
                  fontWeight: "bold",
                  color: "#004d40",
                  mt: 2,
                  mb: 1
                }}
              >
                Hiperparámetros Esenciales
              </Typography>
              <Grid container spacing={2}>
                {/* 1. Context Length */}
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Context Length"
                    value={patchTSMixerParams.context_length}
                    onChange={(e) => setPatchTSMixerParams({...patchTSMixerParams, context_length: e.target.value})}
                    type="number"
                    helperText="Historical window size (must be divisible by patch length)"
                  />
                </Grid>

                {/* 2. Patch Length */}
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Patch Length"
                    value={patchTSMixerParams.patch_length}
                    onChange={(e) => setPatchTSMixerParams({
                      ...patchTSMixerParams,
                      patch_length: e.target.value,
                      patch_stride: e.target.value  // Keep stride = length for non-overlapping
                    })}
                    type="number"
                    helperText="Size of each time series patch"
                  />
                </Grid>

                {/* 3. D Model (Hidden Dimension) */}
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Hidden Dimension (d_model)"
                    value={patchTSMixerParams.d_model}
                    onChange={(e) => setPatchTSMixerParams({...patchTSMixerParams, d_model: e.target.value})}
                    type="number"
                    helperText="Size of hidden layers (16, 32, or 64 typical)"
                  />
                </Grid>

                {/* 4. Num Layers */}
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Number of Layers"
                    value={patchTSMixerParams.num_layers}
                    onChange={(e) => setPatchTSMixerParams({...patchTSMixerParams, num_layers: e.target.value})}
                    type="number"
                    helperText="Number of mixer layers (4-12 typical)"
                  />
                </Grid>

                {/* 5. Dropout */}
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Dropout"
                    value={patchTSMixerParams.dropout}
                    onChange={(e) => setPatchTSMixerParams({...patchTSMixerParams, dropout: e.target.value})}
                    type="number"
                    inputProps={{ step: 0.1, min: 0, max: 1 }}
                    helperText="Regularization (0.0-0.5 typical)"
                  />
                </Grid>

                {/* 6. Learning Rate */}
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Learning Rate"
                    value={patchTSMixerParams.learning_rate}
                    onChange={(e) => setPatchTSMixerParams({...patchTSMixerParams, learning_rate: e.target.value})}
                    type="number"
                    inputProps={{ step: 0.0001, min: 0 }}
                    helperText="Optimizer learning rate (0.001 typical)"
                  />
                </Grid>

                {/* 7. Batch Size */}
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Batch Size"
                    value={patchTSMixerParams.batch_size}
                    onChange={(e) => setPatchTSMixerParams({...patchTSMixerParams, batch_size: e.target.value})}
                    type="number"
                    helperText="Training batch size (16, 32, or 64 typical)"
                  />
                </Grid>

                {/* 8. Epochs */}
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Epochs"
                    value={patchTSMixerParams.epochs}
                    onChange={(e) => setPatchTSMixerParams({...patchTSMixerParams, epochs: e.target.value})}
                    type="number"
                    helperText="Maximum training epochs"
                  />
                </Grid>

                {/* 9. Early Stopping Patience */}
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Early Stopping Patience"
                    value={patchTSMixerParams.early_stopping_patience}
                    onChange={(e) => setPatchTSMixerParams({...patchTSMixerParams, early_stopping_patience: e.target.value})}
                    type="number"
                    helperText="Epochs to wait before early stopping"
                  />
                </Grid>
              </Grid>

              {/* Advanced Toggle Button */}
              <Button
                variant="text"
                onClick={() => setShowPatchTSMixerAdvanced(!showPatchTSMixerAdvanced)}
                sx={{ mt: 2, color: "#00796b" }}
                endIcon={showPatchTSMixerAdvanced ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              >
                {showPatchTSMixerAdvanced ? "Ocultar" : "Mostrar"} Configuración Avanzada
              </Button>

              {/* Advanced Hyperparameters (Collapsible) */}
              <Collapse in={showPatchTSMixerAdvanced}>
                <Typography
                  variant="subtitle1"
                  sx={{
                    fontWeight: "bold",
                    color: "#004d40",
                    mt: 2,
                    mb: 1
                  }}
                >
                  Hiperparámetros Avanzados
                </Typography>
                <Grid container spacing={2}>
                  {/* 1. Expansion Factor */}
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Expansion Factor"
                      value={patchTSMixerAdvanced.expansion_factor}
                      onChange={(e) => setPatchTSMixerAdvanced({...patchTSMixerAdvanced, expansion_factor: e.target.value})}
                      type="number"
                      helperText="MLP expansion multiplier (2 or 4 typical)"
                    />
                  </Grid>

                  {/* 2. Head Dropout */}
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Head Dropout"
                      value={patchTSMixerAdvanced.head_dropout}
                      onChange={(e) => setPatchTSMixerAdvanced({...patchTSMixerAdvanced, head_dropout: e.target.value})}
                      type="number"
                      inputProps={{ step: 0.1, min: 0, max: 1 }}
                      helperText="Dropout for prediction head"
                    />
                  </Grid>

                  {/* 3. Mode */}
                  <Grid item xs={12} sm={6}>
                    <FormControl fullWidth>
                      <InputLabel>Mode</InputLabel>
                      <Select
                        value={patchTSMixerAdvanced.mode}
                        onChange={(e) => setPatchTSMixerAdvanced({...patchTSMixerAdvanced, mode: e.target.value})}
                        label="Mode"
                      >
                        <MenuItem value="common_channel">Common Channel</MenuItem>
                        <MenuItem value="mix_channel">Mix Channel</MenuItem>
                      </Select>
                      <FormHelperText>Channel mixing mode</FormHelperText>
                    </FormControl>
                  </Grid>

                  {/* 4. Scaling */}
                  <Grid item xs={12} sm={6}>
                    <FormControl fullWidth>
                      <InputLabel>Scaling</InputLabel>
                      <Select
                        value={patchTSMixerAdvanced.scaling}
                        onChange={(e) => setPatchTSMixerAdvanced({...patchTSMixerAdvanced, scaling: e.target.value})}
                        label="Scaling"
                      >
                        <MenuItem value="std">Standard (std)</MenuItem>
                        <MenuItem value="mean">Mean</MenuItem>
                        <MenuItem value="none">None</MenuItem>
                      </Select>
                      <FormHelperText>Per-window normalization</FormHelperText>
                    </FormControl>
                  </Grid>

                  {/* 5. Norm MLP */}
                  <Grid item xs={12} sm={6}>
                    <FormControl fullWidth>
                      <InputLabel>Normalization</InputLabel>
                      <Select
                        value={patchTSMixerAdvanced.norm_mlp}
                        onChange={(e) => setPatchTSMixerAdvanced({...patchTSMixerAdvanced, norm_mlp: e.target.value})}
                        label="Normalization"
                      >
                        <MenuItem value="LayerNorm">Layer Norm</MenuItem>
                        <MenuItem value="BatchNorm">Batch Norm</MenuItem>
                        <MenuItem value="none">None</MenuItem>
                      </Select>
                      <FormHelperText>MLP normalization type</FormHelperText>
                    </FormControl>
                  </Grid>

                  {/* 6. Loss */}
                  <Grid item xs={12} sm={6}>
                    <FormControl fullWidth>
                      <InputLabel>Loss Function</InputLabel>
                      <Select
                        value={patchTSMixerAdvanced.loss}
                        onChange={(e) => setPatchTSMixerAdvanced({...patchTSMixerAdvanced, loss: e.target.value})}
                        label="Loss Function"
                      >
                        <MenuItem value="mse">MSE (Mean Squared Error)</MenuItem>
                        <MenuItem value="mae">MAE (Mean Absolute Error)</MenuItem>
                      </Select>
                      <FormHelperText>Training loss function</FormHelperText>
                    </FormControl>
                  </Grid>

                  {/* 7. Distribution Output */}
                  <Grid item xs={12} sm={6}>
                    <FormControl fullWidth>
                      <InputLabel>Distribution Output</InputLabel>
                      <Select
                        value={patchTSMixerAdvanced.distribution_output}
                        onChange={(e) => setPatchTSMixerAdvanced({...patchTSMixerAdvanced, distribution_output: e.target.value})}
                        label="Distribution Output"
                      >
                        <MenuItem value="student_t">Student-t</MenuItem>
                        <MenuItem value="normal">Normal</MenuItem>
                        <MenuItem value="negative_binomial">Negative Binomial</MenuItem>
                        <MenuItem value="none">None (Point Forecast)</MenuItem>
                      </Select>
                      <FormHelperText>Probabilistic output distribution</FormHelperText>
                    </FormControl>
                  </Grid>

                  {/* 8. Gated Attention */}
                  <Grid item xs={12} sm={6}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={patchTSMixerAdvanced.gated_attn}
                          onChange={(e) => setPatchTSMixerAdvanced({...patchTSMixerAdvanced, gated_attn: e.target.checked})}
                        />
                      }
                      label="Gated Attention"
                    />
                    <FormHelperText>Enable lightweight gated attention (recommended)</FormHelperText>
                  </Grid>

                  {/* 9. Self Attention */}
                  <Grid item xs={12} sm={6}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={patchTSMixerAdvanced.self_attn}
                          onChange={(e) => setPatchTSMixerAdvanced({...patchTSMixerAdvanced, self_attn: e.target.checked})}
                        />
                      }
                      label="Self Attention"
                    />
                    <FormHelperText>Enable full self-attention (slower, more expressive)</FormHelperText>
                  </Grid>
                </Grid>
              </Collapse>

              {/* Validation Warnings - uses validationWarnings state from validateSelections() */}
              {validationWarnings.length > 0 && (
                <Box sx={{ mt: 2, p: 2, backgroundColor: '#fff3e0', borderRadius: 1, border: '1px solid #ffb74d' }}>
                  <Typography sx={{ fontWeight: 'bold', color: '#e65100', mb: 1 }}>
                    Advertencias de validación:
                  </Typography>
                  {validationWarnings.map((warning, idx) => (
                    <Typography key={idx} sx={{ color: '#e65100', fontSize: '0.9rem' }}>
                      • {warning}
                    </Typography>
                  ))}
                </Box>
              )}

              {/* Loading State - uses existing trainInProgress from AppContext */}
              {trainInProgress && (
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 2, p: 2, backgroundColor: '#e0f2f1', borderRadius: 1 }}>
                  <CircularProgress size={24} sx={{ mr: 2, color: "#00796b" }} />
                  <Typography sx={{ color: "#004d40" }}>Entrenando modelo PatchTSMixer...</Typography>
                </Box>
              )}

              {/* Result Display - shows after training completes successfully */}
              {!trainInProgress && trainStatus && trainStatus.includes("correctamente") && (
                <Box sx={{
                  mt: 2,
                  p: 2,
                  backgroundColor: '#e8f5e9',
                  borderRadius: 1,
                  border: '1px solid #81c784'
                }}>
                  <Typography sx={{ fontWeight: 'bold', color: '#2e7d32', mb: 1 }}>
                    Entrenamiento PatchTSMixer completado
                  </Typography>
                  <Typography sx={{ color: '#1b5e20', fontSize: '0.9rem', whiteSpace: 'pre-line' }}>
                    {trainStatus.replace("✅ Modelo entrenado correctamente.", "").trim()}
                  </Typography>
                  <Typography sx={{
                    color: '#555',
                    fontSize: '0.85rem',
                    fontStyle: 'italic',
                    mt: 1
                  }}>
                    Revisa los resultados en MLflow y en el directorio &apos;trained&apos;
                  </Typography>
                </Box>
              )}
            </>
          )}
        </Box>

        <Typography
          variant="subtitle1"
          sx={{
            fontWeight: "bold",
            color: "#004d40",
            mt: 2,
            mb: 2
          }}
        >
          Proporciones de división del dataset:
        </Typography>

        {/* Train Slider + Input */}
        <Box sx={{ mb: 3 }}>
          <Typography
            variant="body2"
            sx={{
              mb: 1,
              color: "#004d40",
              fontWeight: 500
            }}
          >
            Entrenamiento:
          </Typography>
          <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>
            <Slider
              value={splitRatios.train}
              onChange={handleSliderChange("train")}
              min={0}
              max={1}
              step={0.01}
              valueLabelDisplay="auto"
              valueLabelFormat={(value) => formatToComma(value)}
              sx={{
                width: "70%",
                color: "#00796b",
                '& .MuiSlider-thumb': {
                  '&:hover, &.Mui-focusVisible': {
                    boxShadow: '0px 0px 0px 8px rgba(0, 121, 107, 0.16)',
                  },
                },
              }}
            />
            <TextField
              value={inputDisplayValues.train}
              onChange={handleInputChange("train")}
              onFocus={handleInputFocus("train")}
              onKeyDown={handleInputKeyDown("train")}
              onBlur={handleInputBlur("train")}
              size="small"
              sx={{
                width: "30%",
                '& .MuiOutlinedInput-root': {
                  '& fieldset': {
                    borderColor: showValidation
                      ? (splitRatiosValid ? "#4caf50" : "#f44336")
                      : undefined,
                    borderWidth: showValidation ? 2 : 1,
                  }
                }
              }}
              inputProps={{
                inputMode: "decimal",
                style: { textAlign: "center" }
              }}
            />
          </Box>
        </Box>

        {/* Val Slider + Input */}
        <Box sx={{ mb: 3 }}>
          <Typography
            variant="body2"
            sx={{
              mb: 1,
              color: "#004d40",
              fontWeight: 500
            }}
          >
            Validación:
          </Typography>
          <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>
            <Slider
              value={splitRatios.val}
              onChange={handleSliderChange("val")}
              min={0}
              max={1}
              step={0.01}
              valueLabelDisplay="auto"
              valueLabelFormat={(value) => formatToComma(value)}
              sx={{
                width: "70%",
                color: "#00796b",
                '& .MuiSlider-thumb': {
                  '&:hover, &.Mui-focusVisible': {
                    boxShadow: '0px 0px 0px 8px rgba(0, 121, 107, 0.16)',
                  },
                },
              }}
            />
            <TextField
              value={inputDisplayValues.val}
              onChange={handleInputChange("val")}
              onFocus={handleInputFocus("val")}
              onKeyDown={handleInputKeyDown("val")}
              onBlur={handleInputBlur("val")}
              size="small"
              sx={{
                width: "30%",
                '& .MuiOutlinedInput-root': {
                  '& fieldset': {
                    borderColor: showValidation
                      ? (splitRatiosValid ? "#4caf50" : "#f44336")
                      : undefined,
                    borderWidth: showValidation ? 2 : 1,
                  }
                }
              }}
              inputProps={{
                inputMode: "decimal",
                style: { textAlign: "center" }
              }}
            />
          </Box>
        </Box>

        {/* Test Slider + Input */}
        <Box sx={{ mb: 3 }}>
          <Typography
            variant="body2"
            sx={{
              mb: 1,
              color: "#004d40",
              fontWeight: 500
            }}
          >
            Prueba:
          </Typography>
          <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>
            <Slider
              value={splitRatios.test}
              onChange={handleSliderChange("test")}
              min={0}
              max={1}
              step={0.01}
              valueLabelDisplay="auto"
              valueLabelFormat={(value) => formatToComma(value)}
              sx={{
                width: "70%",
                color: "#00796b",
                '& .MuiSlider-thumb': {
                  '&:hover, &.Mui-focusVisible': {
                    boxShadow: '0px 0px 0px 8px rgba(0, 121, 107, 0.16)',
                  },
                },
              }}
            />
            <TextField
              value={inputDisplayValues.test}
              onChange={handleInputChange("test")}
              onFocus={handleInputFocus("test")}
              onKeyDown={handleInputKeyDown("test")}
              onBlur={handleInputBlur("test")}
              size="small"
              sx={{
                width: "30%",
                '& .MuiOutlinedInput-root': {
                  '& fieldset': {
                    borderColor: showValidation
                      ? (splitRatiosValid ? "#4caf50" : "#f44336")
                      : undefined,
                    borderWidth: showValidation ? 2 : 1,
                  }
                }
              }}
              inputProps={{
                inputMode: "decimal",
                style: { textAlign: "center" }
              }}
            />
          </Box>
        </Box>

        {/* Validation Message */}
        {showValidation && (
          <Box sx={{
            mb: 2,
            p: 1.5,
            backgroundColor: "#fff",
            borderRadius: 1,
            borderLeft: `4px solid ${splitRatiosValid ? "#4caf50" : "#f44336"}`,
            display: "flex",
            alignItems: "center",
            gap: 1
          }}>
            {splitRatiosValid ? (
              <CheckCircleIcon sx={{ color: "#4caf50", fontSize: 20 }} />
            ) : (
              <ErrorIcon sx={{ color: "#f44336", fontSize: 20 }} />
            )}
            <Typography variant="body2" sx={{
              color: splitRatiosValid ? "#2e7d32" : "#c62828",
              fontWeight: 500
            }}>
              {validationMessage}
            </Typography>
          </Box>
        )}

        {/* Progress Bar - shown when training */}
        {trainInProgress && (
          <Box sx={{ mt: 2, mb: 2 }}>
            <ProgressBar
              message="Entrenando modelo y optimizando hiperparámetros..."
              variant="tealHarmony"
              showPercentage={false}
              useWebSocket={false}
            />
          </Box>
        )}

        <Button
          variant="contained"
          onClick={handleTrain}
          disabled={isDisabled}
          sx={{
            mt: trainInProgress ? 0 : 3,
            backgroundColor: "#00796b",
            "&:hover": { backgroundColor: "#004d40" },
            width: "100%",
            padding: "10px 20px",
            fontSize: "1rem",
            fontWeight: "bold",
          }}
        >
          {trainInProgress ? "Entrenando..." : "Entrenar Modelo"}
        </Button>

        {/* Current State Indicator */}
        <Box
          sx={{
            mt: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 1,
          }}
        >
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.75,
              color: getStateColor(getCurrentState().type),
              backgroundColor: `${getStateColor(getCurrentState().type)}15`,
              px: 2,
              py: 1,
              borderRadius: 2,
              border: `1px solid ${getStateColor(getCurrentState().type)}40`,
            }}
          >
            {getCurrentState().icon}
            <Typography
              variant="body2"
              sx={{
                fontWeight: 500,
                color: getStateColor(getCurrentState().type),
              }}
            >
              {getCurrentState().message}
            </Typography>
          </Box>
        </Box>

        {/* Additional status for success cases */}
        {trainStatus && trainStatus.includes("correctamente") && (
          <Typography
            variant="caption"
            sx={{
              mt: 1.5,
              display: 'block',
              color: "#555",
              textAlign: "center",
              fontStyle: 'italic',
            }}
          >
            Revisa los resultados en MLflow y en el directorio &apos;trained&apos;
          </Typography>
        )}
      </CardContent>

      {/* Info Modals */}
      <InfoModal
        open={showTargetInfo}
        onClose={() => setShowTargetInfo(false)}
        title="Variable de Salida (Target)"
        content={infoContent.variablesDeSalida}
      />

      <InfoModal
        open={showFeatureInfo}
        onClose={() => setShowFeatureInfo(false)}
        title="Variables de Entrada (Features)"
        content={infoContent.variablesDeEntrada}
      />
    </Card>
  );
};


export default TSTrainCard;
