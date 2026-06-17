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


import React, { useState, useContext } from "react";
import InfoModal from './InfoModal';
import ValidationSummary from './ValidationSummary';
import ProgressBar from './ProgressBar';
import {
  Card,
  CardContent,
  Typography,
  Button,
  CircularProgress,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Checkbox,
  FormGroup,
  TextField,
  Radio,
  RadioGroup,
  IconButton,
  Slider,
} from "@mui/material";
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Autorenew as AutorenewIcon,
} from "@mui/icons-material";
import axios from "../utils/axiosConfig";
import { AppContext } from "../AppContext";
import { variableSelectionStyles, infoContent, helperTextStrings } from "../styles/variableSelectionStyles";

const TrainCard = () => {
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

  const [algorithm, setAlgorithm] = useState("logistic");
  const [problemType, setProblemType] = useState("binary"); // Para MLP y XGBoost
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

  // Hiperparámetros para cada algoritmo (se muestran sólo si NO se usa Grid Search)
  const [logisticParams, setLogisticParams] = useState({
    regularization: "1.0",
    maxIter: "100",
    solver: "lbfgs",
  });

  const [mlpParams, setMlpParams] = useState({
    hidden_layer_sizes: "4",
    activation: "relu",
    solver: "adam",
    maxIter: "200",
  });

  const [xgboostParams, setXgboostParams] = useState({
    learning_rate: "0.1",
    n_estimators: "100",
    max_depth: "3",
    subsample: "1.0",
    colsample_bytree: "1.0",
  });

  // Optimization method: "manual", "grid", "random", "bayesian"
  const [optimizationMethod, setOptimizationMethod] = useState("manual");

  // Random Search parameters
  const [nRandomIterations, setNRandomIterations] = useState(100);

  // Bayesian Search UI state
  const [showBayesianAdvanced, setShowBayesianAdvanced] = useState(false);

  // Logistic Regression Random Search parameter ranges
  const [logisticRandomRanges, setLogisticRandomRanges] = useState({
    C_range: [0.001, 100.0],
    max_iter_range: [100, 1000],
    solver_options: ["lbfgs", "liblinear", "saga"],
    penalty_options: ["l2", "none"]
  });

  // MLP Random Search parameter ranges
  const [mlpRandomRanges, setMlpRandomRanges] = useState({
    hidden_layer_sizes_options: [[4], [10], [10, 5], [50], [100], [100, 50], [100, 50, 10]],
    activation_options: ["relu", "tanh", "logistic"],
    solver_options: ["adam", "sgd"],
    learning_rate_init_range: [0.0001, 0.1],
    max_iter_range: [200, 500]
  });

  // XGBoost Random Search parameter ranges
  const [xgboostRandomRanges, setXgboostRandomRanges] = useState({
    n_estimators_range: [50, 500],
    max_depth_range: [3, 10],
    learning_rate_range: [0.01, 0.3],
    subsample_range: [0.5, 1.0],
    colsample_bytree_range: [0.5, 1.0],
    gamma_range: [0.0, 5.0],
    min_child_weight_range: [1, 10],
    reg_alpha_range: [0.0, 1.0],
    reg_lambda_range: [0.0, 1.0]
  });

  // Logistic Regression Bayesian Search parameter space
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

  // MLP Bayesian Search parameter space
  const [mlpBayesianParams, setMlpBayesianParams] = useState({
    hidden_layer_sizes: {
      type: "categorical",
      choices: [[4], [10], [10, 5], [50], [100], [100, 50], [100, 50, 10]]
    },
    activation: {
      type: "categorical",
      choices: ["relu", "tanh", "logistic"]
    },
    solver: {
      type: "categorical",
      choices: ["adam", "sgd"]
    },
    learning_rate_init: {
      type: "real",
      distribution: "log-uniform",
      low: 0.0001,
      high: 0.1
    },
    max_iter: {
      type: "integer",
      low: 200,
      high: 500
    }
  });

  // XGBoost Bayesian Search parameter space
  const [xgboostBayesianParams, setXgboostBayesianParams] = useState({
    n_estimators: {
      type: "integer",
      low: 50,
      high: 500
    },
    max_depth: {
      type: "integer",
      low: 3,
      high: 10
    },
    learning_rate: {
      type: "real",
      distribution: "log-uniform",
      low: 0.01,
      high: 0.3
    },
    subsample: {
      type: "real",
      distribution: "uniform",
      low: 0.5,
      high: 1.0
    },
    colsample_bytree: {
      type: "real",
      distribution: "uniform",
      low: 0.5,
      high: 1.0
    },
    gamma: {
      type: "real",
      distribution: "uniform",
      low: 0.0,
      high: 5.0
    },
    min_child_weight: {
      type: "integer",
      low: 1,
      high: 10
    },
    reg_alpha: {
      type: "real",
      distribution: "uniform",
      low: 0.0,
      high: 1.0
    },
    reg_lambda: {
      type: "real",
      distribution: "uniform",
      low: 0.0,
      high: 1.0
    }
  });

  // Bayesian Search advanced configuration
  const [bayesianConfig, setBayesianConfig] = useState({
    n_trials: 50,                       // Number of optimization trials
    n_initial_points: 10,
    acq_func: "EI",
    random_state: null,
    max_memory_mb: null,
    timeout_seconds: null,
    convergence_tolerance: 0.001,
    convergence_patience: 5,
    save_gp_model: true
  });

  // Info modal states
  const [showTargetInfo, setShowTargetInfo] = useState(false);
  const [showFeatureInfo, setShowFeatureInfo] = useState(false);

  // Validation warnings
  const [validationWarnings, setValidationWarnings] = useState([]);

  // Eliminamos el estado local de status para usar el global trainStatus

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

  // Manejo del archivo CSV
  const handleFileChange = (event) => {
    setCsvFile(event.target.files[0]);
    setColumns([]);
    setInputFeatures([]);
    setTargetVariable("");
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
      setTargetVariable("");
      setInputFeatures([]);
    } else {
      setTargetVariable(column);
      const remainingColumns = columns.filter((col) => col !== column);
      const newFeatures = [...new Set([...inputFeatures, ...remainingColumns])];
      setInputFeatures(newFeatures);
    }
    validateSelections();
  };

  // Validation function
  const validateSelections = () => {
    const warnings = [];
    if (inputFeatures.length === 0 && targetVariable) {
      warnings.push("Debes seleccionar al menos 1 variable de entrada");
    }
    if (!targetVariable && inputFeatures.length > 0) {
      warnings.push("Debes seleccionar 1 variable de salida");
    }
    if (targetVariable && inputFeatures.includes(targetVariable)) {
      warnings.push("Una columna no puede ser entrada y salida simultáneamente");
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

  // Validation function for Random Search parameters
  const isRandomSearchParamsValid = () => {
    if (optimizationMethod !== "random") return true;

    // Validate number of iterations
    if (nRandomIterations < 1 || nRandomIterations > 1000) return false;

    // Validate algorithm-specific ranges
    if (algorithm === "logistic") {
      // Check that at least one solver and penalty option are selected
      if (logisticRandomRanges.solver_options.length === 0) return false;
      if (logisticRandomRanges.penalty_options.length === 0) return false;
      // Check that ranges are valid (min < max)
      if (logisticRandomRanges.C_range[0] >= logisticRandomRanges.C_range[1]) return false;
      if (logisticRandomRanges.max_iter_range[0] >= logisticRandomRanges.max_iter_range[1]) return false;
    } else if (algorithm === "mlp") {
      // Check that at least one option is selected for each categorical parameter
      if (mlpRandomRanges.hidden_layer_sizes_options.length === 0) return false;
      if (mlpRandomRanges.activation_options.length === 0) return false;
      if (mlpRandomRanges.solver_options.length === 0) return false;
      // Check that ranges are valid (min < max)
      if (mlpRandomRanges.learning_rate_init_range[0] >= mlpRandomRanges.learning_rate_init_range[1]) return false;
      if (mlpRandomRanges.max_iter_range[0] >= mlpRandomRanges.max_iter_range[1]) return false;
    } else if (algorithm === "xgboost") {
      // Check that all ranges are valid (min < max)
      if (xgboostRandomRanges.n_estimators_range[0] >= xgboostRandomRanges.n_estimators_range[1]) return false;
      if (xgboostRandomRanges.max_depth_range[0] >= xgboostRandomRanges.max_depth_range[1]) return false;
      if (xgboostRandomRanges.learning_rate_range[0] >= xgboostRandomRanges.learning_rate_range[1]) return false;
      if (xgboostRandomRanges.subsample_range[0] >= xgboostRandomRanges.subsample_range[1]) return false;
      if (xgboostRandomRanges.colsample_bytree_range[0] >= xgboostRandomRanges.colsample_bytree_range[1]) return false;
      if (xgboostRandomRanges.gamma_range[0] >= xgboostRandomRanges.gamma_range[1]) return false;
      if (xgboostRandomRanges.min_child_weight_range[0] >= xgboostRandomRanges.min_child_weight_range[1]) return false;
      if (xgboostRandomRanges.reg_alpha_range[0] >= xgboostRandomRanges.reg_alpha_range[1]) return false;
      if (xgboostRandomRanges.reg_lambda_range[0] >= xgboostRandomRanges.reg_lambda_range[1]) return false;
    }

    return true;
  };

  // Validation function for Bayesian Search parameters
  const isBayesianSearchParamsValid = () => {
    if (optimizationMethod !== "bayesian") return true;

    // Validate number of trials
    if (bayesianConfig.n_trials < 1 || bayesianConfig.n_trials > 200) return false;

    // Validate algorithm-specific parameter spaces
    if (algorithm === "logistic") {
      // Check that at least one solver and penalty option are selected
      if (logisticBayesianParams.solver.choices.length === 0) return false;
      if (logisticBayesianParams.penalty.choices.length === 0) return false;
      // Check that ranges are valid (min < max)
      if (logisticBayesianParams.C.low >= logisticBayesianParams.C.high) return false;
      if (logisticBayesianParams.max_iter.low >= logisticBayesianParams.max_iter.high) return false;
    } else if (algorithm === "mlp") {
      // Check that at least one option is selected for each categorical parameter
      if (mlpBayesianParams.hidden_layer_sizes.choices.length === 0) return false;
      if (mlpBayesianParams.activation.choices.length === 0) return false;
      if (mlpBayesianParams.solver.choices.length === 0) return false;
      // Check that ranges are valid (min < max)
      if (mlpBayesianParams.learning_rate_init.low >= mlpBayesianParams.learning_rate_init.high) return false;
      if (mlpBayesianParams.max_iter.low >= mlpBayesianParams.max_iter.high) return false;
    } else if (algorithm === "xgboost") {
      // Check that all ranges are valid (min < max)
      if (xgboostBayesianParams.n_estimators.low >= xgboostBayesianParams.n_estimators.high) return false;
      if (xgboostBayesianParams.max_depth.low >= xgboostBayesianParams.max_depth.high) return false;
      if (xgboostBayesianParams.learning_rate.low >= xgboostBayesianParams.learning_rate.high) return false;
      if (xgboostBayesianParams.subsample.low >= xgboostBayesianParams.subsample.high) return false;
      if (xgboostBayesianParams.colsample_bytree.low >= xgboostBayesianParams.colsample_bytree.high) return false;
      if (xgboostBayesianParams.gamma.low >= xgboostBayesianParams.gamma.high) return false;
      if (xgboostBayesianParams.min_child_weight.low >= xgboostBayesianParams.min_child_weight.high) return false;
      if (xgboostBayesianParams.reg_alpha.low >= xgboostBayesianParams.reg_alpha.high) return false;
      if (xgboostBayesianParams.reg_lambda.low >= xgboostBayesianParams.reg_lambda.high) return false;
    }

    return true;
  };

  // Validation function for log-uniform distribution constraints
  const validateLogUniformConstraints = () => {
    const errors = [];

    if (optimizationMethod !== "bayesian") return errors;

    if (algorithm === "logistic") {
      // Check C parameter
      if (logisticBayesianParams.C.distribution === "log-uniform") {
        const { low, high } = logisticBayesianParams.C;
        if (low <= 0 || high <= 0) {
          errors.push("Logistic C: distribución log-uniforme requiere valores estrictamente positivos (> 0)");
        }
      }
    } else if (algorithm === "mlp") {
      // Check learning_rate_init
      if (mlpBayesianParams.learning_rate_init.distribution === "log-uniform") {
        const { low, high } = mlpBayesianParams.learning_rate_init;
        if (low <= 0 || high <= 0) {
          errors.push("MLP learning_rate_init: distribución log-uniforme requiere valores estrictamente positivos (> 0)");
        }
      }
    } else if (algorithm === "xgboost") {
      // Check learning_rate
      if (xgboostBayesianParams.learning_rate.distribution === "log-uniform") {
        const { low, high } = xgboostBayesianParams.learning_rate;
        if (low <= 0 || high <= 0) {
          errors.push("XGBoost learning_rate: distribución log-uniforme requiere valores estrictamente positivos (> 0)");
        }
      }
    }

    return errors;
  };

  // Función para enviar la solicitud de entrenamiento
  const handleTrain = async () => {
    if (!csvFile) {
      setTrainStatus("⚠️ Por favor, selecciona un archivo CSV.");
      return;
    }
    if (!inputFeatures.length || !targetVariable) {
      setTrainStatus("⚠️ Selecciona al menos una variable de entrada y una variable target.");
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

    // Validate log-uniform distribution constraints for Bayesian search
    const logUniformErrors = validateLogUniformConstraints();
    if (logUniformErrors.length > 0) {
      setTrainStatus("error");
      alert("Errores de validación:\n\n" + logUniformErrors.join("\n"));
      return;
    }

    setTrainInProgress(true);
    setTrainStatus("🚀 Entrenando modelo, por favor espera...");

    // Seleccionar hiperparámetros según el algoritmo elegido
    let finalParams;
    if (algorithm === "logistic") {
      finalParams = { ...logisticParams };
    } else if (algorithm === "mlp") {
      finalParams = { ...mlpParams };
    } else if (algorithm === "xgboost") {
      finalParams = { ...xgboostParams };
    }

    // Preparar el payload con los datos de entrenamiento,
    // incluyendo la estrategia de optimización de hiperparámetros
    const payload = {
      model_name: modelName,
      input_features: inputFeatures,
      target_variable: targetVariable,
      experiment_dir: experimentDir,
      split_ratios: splitRatios,
      run_id: runId,
      algorithm: algorithm,
      params: finalParams,
      hyperparameter_search_strategy: optimizationMethod === "manual" ? "none" : optimizationMethod,
    };

    // Agregar parámetros de Random Search si aplica
    if (optimizationMethod === "random") {
      payload.n_random_iterations = nRandomIterations;

      // Seleccionar los rangos apropiados según el algoritmo
      if (algorithm === "logistic") {
        payload.random_search_params = logisticRandomRanges;
      } else if (algorithm === "mlp") {
        payload.random_search_params = mlpRandomRanges;
      } else if (algorithm === "xgboost") {
        payload.random_search_params = xgboostRandomRanges;
      }
    }

    // Agregar parámetros de Bayesian Search si aplica
    if (optimizationMethod === "bayesian") {
      // Seleccionar el espacio de parámetros apropiado según el algoritmo
      if (algorithm === "logistic") {
        payload.bayesian_search_params = logisticBayesianParams;
      } else if (algorithm === "mlp") {
        payload.bayesian_search_params = mlpBayesianParams;
      } else if (algorithm === "xgboost") {
        payload.bayesian_search_params = xgboostBayesianParams;
      }

      // Incluir configuración avanzada (filtrar valores null)
      const cleanBayesianConfig = {};
      Object.keys(bayesianConfig).forEach(key => {
        if (bayesianConfig[key] !== null) {
          cleanBayesianConfig[key] = bayesianConfig[key];
        }
      });
      payload.bayesian_config = cleanBayesianConfig;
    }

    // Para MLP y XGBoost se incluye el "Tipo de problema"
    if (algorithm === "mlp" || algorithm === "xgboost") {
      payload.problem_type = problemType;
    }

    const formData = new FormData();
    formData.append("file", csvFile);
    formData.append("data", JSON.stringify(payload));
    // Iterate over the entries and log them
    for (const pair of formData.entries()) {
      console.log(`${pair[0]}: ${pair[1]}`);
    }

    try {
      const response = await axios.post("/train-model/", formData);
      setTrainStatus(
        "✅ Modelo entrenado correctamente. Puedes revisar los resultados en MLflow y en el directorio 'trained' de tu experimento."
      );
      markStepDone("trainDone");
    } catch (error) {
      console.error("Error al entrenar el modelo:", error);
      setTrainStatus(
        `❌ Error durante el entrenamiento: ${error.response?.data?.error || "Error desconocido"}`
      );
    } finally {
      setTrainInProgress(false);
    }
  };

  // Se deshabilita la acción si:
  // - El proceso de entrenamiento está en curso.
  // - No hay experimento o run_id.
  // - No se han completado los pasos previos obligatorios (en este caso, la codificación debe estar completada).
  // - O ya se entrenó el modelo.
  // - Hay advertencias de validación (nuevo)
  // - Split ratios are invalid
  // - Random Search parameters are invalid
  // - Bayesian Search parameters are invalid
  const isDisabled =
    trainInProgress ||
    !experimentDir ||
    !runId ||
    !flow.encodeDone ||
    flow.trainDone ||
    !inputFeatures.length ||
    !targetVariable ||
    !modelName.trim() ||
    validationWarnings.length > 0 ||
    !splitRatiosValid ||
    !isRandomSearchParamsValid() ||
    !isBayesianSearchParamsValid();

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

        <Box sx={{ textAlign: "left", mt: 2 }}>
          <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>Algoritmo:</Typography>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Selecciona un algoritmo</InputLabel>
            <Select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
              label="Selecciona un algoritmo"
            >
              <MenuItem value="logistic">Regresión Logística</MenuItem>
              <MenuItem value="mlp">Redes Neuronales (MLP)</MenuItem>
              <MenuItem value="xgboost">XGBoost</MenuItem>
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

          {/* Optimization method selection */}
          <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}>
            Método de optimización:
          </Typography>
          <FormControl component="fieldset" sx={{ mb: 2 }}>
            {["manual", "grid", "random", "bayesian"].map((method) => (
              <FormControlLabel
                key={method}
                control={
                  <input
                    type="radio"
                    name="optimizationMethod"
                    checked={optimizationMethod === method}
                    onChange={() => setOptimizationMethod(method)}
                    style={{ marginRight: "8px", transform: "scale(1.2)" }}
                  />
                }
                label={
                  method === "manual" ? "Parámetros manuales" :
                  method === "grid" ? "Grid Search (búsqueda automática)" :
                  method === "random" ? "Random Search (búsqueda aleatoria)" :
                  "Bayesian Search (búsqueda bayesiana)"
                }
                sx={{ display: "block", padding: "5px 0" }}
              />
            ))}
          </FormControl>

          {/* Random Search configuration */}
          {optimizationMethod === "random" && (
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
                  n_trials: Math.max(1, Math.min(200, parseInt(e.target.value) || 50))
                })}
                slotProps={{
                  input: {
                    min: 1,
                    max: 200
                  }
                }}
                helperText="Entre 1 y 200 pruebas (recomendado: 30-100). Bayesian optimization es costoso."
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
                        <MenuItem value="EI">EI (Expected Improvement)</MenuItem>
                        <MenuItem value="PI">PI (Probability of Improvement)</MenuItem>
                        <MenuItem value="LCB">LCB (Lower Confidence Bound)</MenuItem>
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

                    {/* convergence_tolerance */}
                    <TextField
                      type="number"
                      label="Tolerancia de convergencia"
                      value={bayesianConfig.convergence_tolerance}
                      onChange={(e) => setBayesianConfig({
                        ...bayesianConfig,
                        convergence_tolerance: parseFloat(e.target.value) || 0.001
                      })}
                      slotProps={{ input: { step: 0.0001, min: 0.0001 } }}
                      helperText="Mejora mínima para considerar que hay progreso"
                      size="small"
                      sx={{ width: "100%", mb: 2 }}
                    />

                    {/* convergence_patience */}
                    <TextField
                      type="number"
                      label="Paciencia de convergencia"
                      value={bayesianConfig.convergence_patience}
                      onChange={(e) => setBayesianConfig({
                        ...bayesianConfig,
                        convergence_patience: Math.max(1, parseInt(e.target.value) || 5)
                      })}
                      slotProps={{ input: { min: 1 } }}
                      helperText="Iteraciones sin mejora antes de detener"
                      size="small"
                      sx={{ width: "100%", mb: 2 }}
                    />

                    {/* save_gp_model */}
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={bayesianConfig.save_gp_model}
                          onChange={(e) => setBayesianConfig({
                            ...bayesianConfig,
                            save_gp_model: e.target.checked
                          })}
                        />
                      }
                      label="Guardar modelo GP (Gaussian Process)"
                    />
                    <Typography variant="caption" sx={{ color: "#666", ml: 4, display: "block", mb: 1 }}>
                      Guardar el modelo de proceso gaussiano para análisis posterior
                    </Typography>
                  </Box>
                )}
              </Box>
            </Box>
          )}

          {/* Para Logistic Regression */}
          {optimizationMethod === "manual" && algorithm === "logistic" && (
            <>
              <Typography sx={{ fontWeight: "bold", color: "#004d40", mt: 2 }}>
                Hiperparámetros - Regresión Logística
              </Typography>
              <TextField
                fullWidth
                label="Regularización (C)"
                value={logisticParams.regularization}
                onChange={(e) =>
                  setLogisticParams({ ...logisticParams, regularization: e.target.value })
                }
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Máximo de iteraciones"
                value={logisticParams.maxIter}
                onChange={(e) =>
                  setLogisticParams({ ...logisticParams, maxIter: e.target.value })
                }
                sx={{ mb: 2 }}
              />
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Solver</InputLabel>
                <Select
                  value={logisticParams.solver}
                  onChange={(e) =>
                    setLogisticParams({ ...logisticParams, solver: e.target.value })
                  }
                  label="Solver"
                >
                  <MenuItem value="lbfgs">lbfgs</MenuItem>
                  <MenuItem value="liblinear">liblinear</MenuItem>
                  <MenuItem value="saga">saga</MenuItem>
                </Select>
              </FormControl>
            </>
          )}

          {/* Logistic Regression Random Search parameter ranges */}
          {optimizationMethod === "random" && algorithm === "logistic" && (
            <Box sx={{ mt: 2, mb: 3, p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
              <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 2 }}>
                Rangos de hiperparámetros - Regresión Logística Random Search
              </Typography>

              <Typography variant="body2" sx={{ color: "#666", mb: 2 }}>
                Define los rangos para la búsqueda aleatoria de hiperparámetros
              </Typography>

              {/* C Range */}
              <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                Rango de C (Regularización - escala logarítmica):
              </Typography>
              <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                <TextField
                  type="number"
                  label="C mínimo"
                  value={logisticRandomRanges.C_range[0]}
                  onChange={(e) => setLogisticRandomRanges({
                    ...logisticRandomRanges,
                    C_range: [parseFloat(e.target.value) || 0.001, logisticRandomRanges.C_range[1]]
                  })}
                  slotProps={{ input: { step: 0.001, min: 0.001 } }}
                  size="small"
                />
                <TextField
                  type="number"
                  label="C máximo"
                  value={logisticRandomRanges.C_range[1]}
                  onChange={(e) => setLogisticRandomRanges({
                    ...logisticRandomRanges,
                    C_range: [logisticRandomRanges.C_range[0], parseFloat(e.target.value) || 100.0]
                  })}
                  slotProps={{ input: { step: 0.1, min: 0.001 } }}
                  size="small"
                />
              </Box>

              {/* Max Iter Range */}
              <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                Rango de iteraciones máximas:
              </Typography>
              <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                <TextField
                  type="number"
                  label="Max iter mínimo"
                  value={logisticRandomRanges.max_iter_range[0]}
                  onChange={(e) => setLogisticRandomRanges({
                    ...logisticRandomRanges,
                    max_iter_range: [parseInt(e.target.value) || 100, logisticRandomRanges.max_iter_range[1]]
                  })}
                  slotProps={{ input: { step: 10, min: 50 } }}
                  size="small"
                />
                <TextField
                  type="number"
                  label="Max iter máximo"
                  value={logisticRandomRanges.max_iter_range[1]}
                  onChange={(e) => setLogisticRandomRanges({
                    ...logisticRandomRanges,
                    max_iter_range: [logisticRandomRanges.max_iter_range[0], parseInt(e.target.value) || 1000]
                  })}
                  slotProps={{ input: { step: 10, min: 50 } }}
                  size="small"
                />
              </Box>

              {/* Solver Options */}
              <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                Opciones de Solver (selecciona al menos una):
              </Typography>
              <FormGroup sx={{ mb: 2, ml: 1 }}>
                {["lbfgs", "liblinear", "saga"].map((solver) => (
                  <FormControlLabel
                    key={solver}
                    control={
                      <Checkbox
                        checked={logisticRandomRanges.solver_options.includes(solver)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setLogisticRandomRanges({
                              ...logisticRandomRanges,
                              solver_options: [...logisticRandomRanges.solver_options, solver]
                            });
                          } else {
                            setLogisticRandomRanges({
                              ...logisticRandomRanges,
                              solver_options: logisticRandomRanges.solver_options.filter(s => s !== solver)
                            });
                          }
                        }}
                      />
                    }
                    label={solver}
                  />
                ))}
              </FormGroup>

              {/* Penalty Options */}
              <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                Opciones de Penalización (selecciona al menos una):
              </Typography>
              <FormGroup sx={{ ml: 1 }}>
                {["l2", "none"].map((penalty) => (
                  <FormControlLabel
                    key={penalty}
                    control={
                      <Checkbox
                        checked={logisticRandomRanges.penalty_options.includes(penalty)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setLogisticRandomRanges({
                              ...logisticRandomRanges,
                              penalty_options: [...logisticRandomRanges.penalty_options, penalty]
                            });
                          } else {
                            setLogisticRandomRanges({
                              ...logisticRandomRanges,
                              penalty_options: logisticRandomRanges.penalty_options.filter(p => p !== penalty)
                            });
                          }
                        }}
                      />
                    }
                    label={penalty}
                  />
                ))}
              </FormGroup>
            </Box>
          )}

          {/* Logistic Regression Bayesian Search parameter space */}
          {optimizationMethod === "bayesian" && algorithm === "logistic" && (
            <Box sx={{ mt: 2, mb: 3, p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
              <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 2 }}>
                Espacio de parámetros - Regresión Logística Bayesian Search
              </Typography>

              <Typography variant="body2" sx={{ color: "#666", mb: 2 }}>
                Define el espacio de búsqueda para optimización Bayesiana
              </Typography>

              {/* C Parameter */}
              <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                C (Regularización):
              </Typography>
              <Box sx={{ mb: 2, p: 1.5, backgroundColor: "#fff", borderRadius: "4px" }}>
                <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                  <InputLabel>Distribución</InputLabel>
                  <Select
                    value={logisticBayesianParams.C.distribution}
                    onChange={(e) => setLogisticBayesianParams({
                      ...logisticBayesianParams,
                      C: { ...logisticBayesianParams.C, distribution: e.target.value }
                    })}
                    label="Distribución"
                  >
                    <MenuItem value="log-uniform">Log-uniforme (recomendado)</MenuItem>
                    <MenuItem value="uniform">Uniforme</MenuItem>
                  </Select>
                </FormControl>
                <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                  <TextField
                    type="number"
                    label="Valor mínimo"
                    value={logisticBayesianParams.C.low}
                    onChange={(e) => setLogisticBayesianParams({
                      ...logisticBayesianParams,
                      C: { ...logisticBayesianParams.C, low: parseFloat(e.target.value) || 0.001 }
                    })}
                    slotProps={{ input: { step: 0.001, min: 0.001 } }}
                    size="small"
                  />
                  <TextField
                    type="number"
                    label="Valor máximo"
                    value={logisticBayesianParams.C.high}
                    onChange={(e) => setLogisticBayesianParams({
                      ...logisticBayesianParams,
                      C: { ...logisticBayesianParams.C, high: parseFloat(e.target.value) || 100.0 }
                    })}
                    slotProps={{ input: { step: 0.1, min: 0.001 } }}
                    size="small"
                  />
                </Box>
              </Box>

              {/* max_iter Parameter */}
              <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                Max Iter (Iteraciones máximas):
              </Typography>
              <Box sx={{ mb: 2, p: 1.5, backgroundColor: "#fff", borderRadius: "4px" }}>
                <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                  <TextField
                    type="number"
                    label="Valor mínimo"
                    value={logisticBayesianParams.max_iter.low}
                    onChange={(e) => setLogisticBayesianParams({
                      ...logisticBayesianParams,
                      max_iter: { ...logisticBayesianParams.max_iter, low: parseInt(e.target.value) || 100 }
                    })}
                    slotProps={{ input: { step: 10, min: 50 } }}
                    size="small"
                  />
                  <TextField
                    type="number"
                    label="Valor máximo"
                    value={logisticBayesianParams.max_iter.high}
                    onChange={(e) => setLogisticBayesianParams({
                      ...logisticBayesianParams,
                      max_iter: { ...logisticBayesianParams.max_iter, high: parseInt(e.target.value) || 1000 }
                    })}
                    slotProps={{ input: { step: 10, min: 50 } }}
                    size="small"
                  />
                </Box>
              </Box>

              {/* Solver Options */}
              <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                Opciones de Solver (selecciona al menos una):
              </Typography>
              <FormGroup sx={{ mb: 2, ml: 1 }}>
                {["lbfgs", "liblinear", "saga"].map((solver) => (
                  <FormControlLabel
                    key={solver}
                    control={
                      <Checkbox
                        checked={logisticBayesianParams.solver.choices.includes(solver)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setLogisticBayesianParams({
                              ...logisticBayesianParams,
                              solver: { ...logisticBayesianParams.solver, choices: [...logisticBayesianParams.solver.choices, solver] }
                            });
                          } else {
                            setLogisticBayesianParams({
                              ...logisticBayesianParams,
                              solver: { ...logisticBayesianParams.solver, choices: logisticBayesianParams.solver.choices.filter(s => s !== solver) }
                            });
                          }
                        }}
                      />
                    }
                    label={solver}
                  />
                ))}
              </FormGroup>

              {/* Penalty Options */}
              <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                Opciones de Penalización (selecciona al menos una):
              </Typography>
              <FormGroup sx={{ ml: 1 }}>
                {["l2", "none"].map((penalty) => (
                  <FormControlLabel
                    key={penalty}
                    control={
                      <Checkbox
                        checked={logisticBayesianParams.penalty.choices.includes(penalty)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setLogisticBayesianParams({
                              ...logisticBayesianParams,
                              penalty: { ...logisticBayesianParams.penalty, choices: [...logisticBayesianParams.penalty.choices, penalty] }
                            });
                          } else {
                            setLogisticBayesianParams({
                              ...logisticBayesianParams,
                              penalty: { ...logisticBayesianParams.penalty, choices: logisticBayesianParams.penalty.choices.filter(p => p !== penalty) }
                            });
                          }
                        }}
                      />
                    }
                    label={penalty}
                  />
                ))}
              </FormGroup>
            </Box>
          )}

          {/* Para MLP */}
          {algorithm === "mlp" && (
            <>
              {optimizationMethod === "manual" && (
                <>
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mt: 2 }}>
                    Hiperparámetros - Redes Neuronales (MLP)
                  </Typography>
                  <TextField
                    fullWidth
                    label="Estructura (hidden_layer_sizes)"
                    placeholder="Ej: 4 o 10,50"
                    value={mlpParams.hidden_layer_sizes}
                    onChange={(e) =>
                      setMlpParams({ ...mlpParams, hidden_layer_sizes: e.target.value })
                    }
                    sx={{ mb: 2 }}
                  />
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Función de activación</InputLabel>
                    <Select
                      value={mlpParams.activation}
                      onChange={(e) =>
                        setMlpParams({ ...mlpParams, activation: e.target.value })
                      }
                      label="Función de activación"
                    >
                      <MenuItem value="relu">relu</MenuItem>
                      <MenuItem value="tanh">tanh</MenuItem>
                      <MenuItem value="logistic">logistic</MenuItem>
                    </Select>
                  </FormControl>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Solver</InputLabel>
                    <Select
                      value={mlpParams.solver}
                      onChange={(e) =>
                        setMlpParams({ ...mlpParams, solver: e.target.value })
                      }
                      label="Solver"
                    >
                      <MenuItem value="adam">adam</MenuItem>
                      <MenuItem value="sgd">sgd</MenuItem>
                    </Select>
                  </FormControl>
                  <TextField
                    fullWidth
                    label="Máximo de iteraciones"
                    value={mlpParams.maxIter}
                    onChange={(e) =>
                      setMlpParams({ ...mlpParams, maxIter: e.target.value })
                    }
                    sx={{ mb: 2 }}
                  />
                </>
              )}

              {/* MLP Random Search parameter ranges */}
              {optimizationMethod === "random" && (
                <Box sx={{ mt: 2, mb: 3, p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 2 }}>
                    Rangos de hiperparámetros - MLP Random Search
                  </Typography>

                  <Typography variant="body2" sx={{ color: "#666", mb: 2 }}>
                    Define los rangos para la búsqueda aleatoria de hiperparámetros MLP
                  </Typography>

                  {/* Hidden Layer Sizes Options */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Arquitecturas disponibles (selecciona al menos una):
                  </Typography>
                  <FormGroup sx={{ mb: 2, ml: 1 }}>
                    {[[4], [10], [10, 5], [50], [100], [100, 50], [100, 50, 10]].map((arch, idx) => {
                      const archStr = arch.join(',');
                      const archLabel = `(${archStr})`;
                      const isChecked = mlpRandomRanges.hidden_layer_sizes_options.some(
                        option => JSON.stringify(option) === JSON.stringify(arch)
                      );
                      return (
                        <FormControlLabel
                          key={idx}
                          control={
                            <Checkbox
                              checked={isChecked}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setMlpRandomRanges({
                                    ...mlpRandomRanges,
                                    hidden_layer_sizes_options: [...mlpRandomRanges.hidden_layer_sizes_options, arch]
                                  });
                                } else {
                                  setMlpRandomRanges({
                                    ...mlpRandomRanges,
                                    hidden_layer_sizes_options: mlpRandomRanges.hidden_layer_sizes_options.filter(
                                      option => JSON.stringify(option) !== JSON.stringify(arch)
                                    )
                                  });
                                }
                              }}
                            />
                          }
                          label={archLabel}
                        />
                      );
                    })}
                  </FormGroup>

                  {/* Activation Options */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Funciones de activación (selecciona al menos una):
                  </Typography>
                  <FormGroup sx={{ mb: 2, ml: 1 }}>
                    {["relu", "tanh", "logistic"].map((activation) => (
                      <FormControlLabel
                        key={activation}
                        control={
                          <Checkbox
                            checked={mlpRandomRanges.activation_options.includes(activation)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setMlpRandomRanges({
                                  ...mlpRandomRanges,
                                  activation_options: [...mlpRandomRanges.activation_options, activation]
                                });
                              } else {
                                setMlpRandomRanges({
                                  ...mlpRandomRanges,
                                  activation_options: mlpRandomRanges.activation_options.filter(a => a !== activation)
                                });
                              }
                            }}
                          />
                        }
                        label={activation}
                      />
                    ))}
                  </FormGroup>

                  {/* Solver Options */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Opciones de Solver (selecciona al menos una):
                  </Typography>
                  <FormGroup sx={{ mb: 2, ml: 1 }}>
                    {["adam", "sgd"].map((solver) => (
                      <FormControlLabel
                        key={solver}
                        control={
                          <Checkbox
                            checked={mlpRandomRanges.solver_options.includes(solver)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setMlpRandomRanges({
                                  ...mlpRandomRanges,
                                  solver_options: [...mlpRandomRanges.solver_options, solver]
                                });
                              } else {
                                setMlpRandomRanges({
                                  ...mlpRandomRanges,
                                  solver_options: mlpRandomRanges.solver_options.filter(s => s !== solver)
                                });
                              }
                            }}
                          />
                        }
                        label={solver}
                      />
                    ))}
                  </FormGroup>

                  {/* Learning Rate Init Range */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Rango de tasa de aprendizaje inicial (escala logarítmica):
                  </Typography>
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                    <TextField
                      type="number"
                      label="Learning rate mínimo"
                      value={mlpRandomRanges.learning_rate_init_range[0]}
                      onChange={(e) => setMlpRandomRanges({
                        ...mlpRandomRanges,
                        learning_rate_init_range: [parseFloat(e.target.value) || 0.0001, mlpRandomRanges.learning_rate_init_range[1]]
                      })}
                      slotProps={{ input: { step: 0.0001, min: 0.0001 } }}
                      size="small"
                    />
                    <TextField
                      type="number"
                      label="Learning rate máximo"
                      value={mlpRandomRanges.learning_rate_init_range[1]}
                      onChange={(e) => setMlpRandomRanges({
                        ...mlpRandomRanges,
                        learning_rate_init_range: [mlpRandomRanges.learning_rate_init_range[0], parseFloat(e.target.value) || 0.1]
                      })}
                      slotProps={{ input: { step: 0.001, min: 0.0001 } }}
                      size="small"
                    />
                  </Box>

                  {/* Max Iter Range */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Rango de iteraciones máximas:
                  </Typography>
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                    <TextField
                      type="number"
                      label="Max iter mínimo"
                      value={mlpRandomRanges.max_iter_range[0]}
                      onChange={(e) => setMlpRandomRanges({
                        ...mlpRandomRanges,
                        max_iter_range: [parseInt(e.target.value) || 200, mlpRandomRanges.max_iter_range[1]]
                      })}
                      slotProps={{ input: { step: 10, min: 50 } }}
                      size="small"
                    />
                    <TextField
                      type="number"
                      label="Max iter máximo"
                      value={mlpRandomRanges.max_iter_range[1]}
                      onChange={(e) => setMlpRandomRanges({
                        ...mlpRandomRanges,
                        max_iter_range: [mlpRandomRanges.max_iter_range[0], parseInt(e.target.value) || 500]
                      })}
                      slotProps={{ input: { step: 10, min: 50 } }}
                      size="small"
                    />
                  </Box>
                </Box>
              )}

              {/* MLP Bayesian Search parameter space */}
              {optimizationMethod === "bayesian" && (
                <Box sx={{ mt: 2, mb: 3, p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 2 }}>
                    Espacio de parámetros - MLP Bayesian Search
                  </Typography>

                  <Typography variant="body2" sx={{ color: "#666", mb: 2 }}>
                    Define el espacio de búsqueda para optimización Bayesiana
                  </Typography>

                  {/* Hidden Layer Sizes Options */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Arquitecturas disponibles (selecciona al menos una):
                  </Typography>
                  <FormGroup sx={{ mb: 2, ml: 1 }}>
                    {[[4], [10], [10, 5], [50], [100], [100, 50], [100, 50, 10]].map((arch, idx) => {
                      const archStr = arch.join(',');
                      const archLabel = `(${archStr})`;
                      const isChecked = mlpBayesianParams.hidden_layer_sizes.choices.some(
                        option => JSON.stringify(option) === JSON.stringify(arch)
                      );
                      return (
                        <FormControlLabel
                          key={idx}
                          control={
                            <Checkbox
                              checked={isChecked}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setMlpBayesianParams({
                                    ...mlpBayesianParams,
                                    hidden_layer_sizes: {
                                      ...mlpBayesianParams.hidden_layer_sizes,
                                      choices: [...mlpBayesianParams.hidden_layer_sizes.choices, arch]
                                    }
                                  });
                                } else {
                                  setMlpBayesianParams({
                                    ...mlpBayesianParams,
                                    hidden_layer_sizes: {
                                      ...mlpBayesianParams.hidden_layer_sizes,
                                      choices: mlpBayesianParams.hidden_layer_sizes.choices.filter(
                                        option => JSON.stringify(option) !== JSON.stringify(arch)
                                      )
                                    }
                                  });
                                }
                              }}
                            />
                          }
                          label={archLabel}
                        />
                      );
                    })}
                  </FormGroup>

                  {/* Activation Options */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Funciones de activación (selecciona al menos una):
                  </Typography>
                  <FormGroup sx={{ mb: 2, ml: 1 }}>
                    {["relu", "tanh", "logistic"].map((activation) => (
                      <FormControlLabel
                        key={activation}
                        control={
                          <Checkbox
                            checked={mlpBayesianParams.activation.choices.includes(activation)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setMlpBayesianParams({
                                  ...mlpBayesianParams,
                                  activation: { ...mlpBayesianParams.activation, choices: [...mlpBayesianParams.activation.choices, activation] }
                                });
                              } else {
                                setMlpBayesianParams({
                                  ...mlpBayesianParams,
                                  activation: { ...mlpBayesianParams.activation, choices: mlpBayesianParams.activation.choices.filter(a => a !== activation) }
                                });
                              }
                            }}
                          />
                        }
                        label={activation}
                      />
                    ))}
                  </FormGroup>

                  {/* Solver Options */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Opciones de Solver (selecciona al menos una):
                  </Typography>
                  <FormGroup sx={{ mb: 2, ml: 1 }}>
                    {["adam", "sgd"].map((solver) => (
                      <FormControlLabel
                        key={solver}
                        control={
                          <Checkbox
                            checked={mlpBayesianParams.solver.choices.includes(solver)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setMlpBayesianParams({
                                  ...mlpBayesianParams,
                                  solver: { ...mlpBayesianParams.solver, choices: [...mlpBayesianParams.solver.choices, solver] }
                                });
                              } else {
                                setMlpBayesianParams({
                                  ...mlpBayesianParams,
                                  solver: { ...mlpBayesianParams.solver, choices: mlpBayesianParams.solver.choices.filter(s => s !== solver) }
                                });
                              }
                            }}
                          />
                        }
                        label={solver}
                      />
                    ))}
                  </FormGroup>

                  {/* Learning Rate Init */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Learning Rate Init (Tasa de aprendizaje inicial):
                  </Typography>
                  <Box sx={{ mb: 2, p: 1.5, backgroundColor: "#fff", borderRadius: "4px" }}>
                    <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                      <InputLabel>Distribución</InputLabel>
                      <Select
                        value={mlpBayesianParams.learning_rate_init.distribution}
                        onChange={(e) => setMlpBayesianParams({
                          ...mlpBayesianParams,
                          learning_rate_init: { ...mlpBayesianParams.learning_rate_init, distribution: e.target.value }
                        })}
                        label="Distribución"
                      >
                        <MenuItem value="log-uniform">Log-uniforme (recomendado)</MenuItem>
                        <MenuItem value="uniform">Uniforme</MenuItem>
                      </Select>
                    </FormControl>
                    <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                      <TextField
                        type="number"
                        label="Valor mínimo"
                        value={mlpBayesianParams.learning_rate_init.low}
                        onChange={(e) => setMlpBayesianParams({
                          ...mlpBayesianParams,
                          learning_rate_init: { ...mlpBayesianParams.learning_rate_init, low: parseFloat(e.target.value) || 0.0001 }
                        })}
                        slotProps={{ input: { step: 0.0001, min: 0.0001 } }}
                        size="small"
                      />
                      <TextField
                        type="number"
                        label="Valor máximo"
                        value={mlpBayesianParams.learning_rate_init.high}
                        onChange={(e) => setMlpBayesianParams({
                          ...mlpBayesianParams,
                          learning_rate_init: { ...mlpBayesianParams.learning_rate_init, high: parseFloat(e.target.value) || 0.1 }
                        })}
                        slotProps={{ input: { step: 0.001, min: 0.0001 } }}
                        size="small"
                      />
                    </Box>
                  </Box>

                  {/* max_iter Parameter */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Max Iter (Iteraciones máximas):
                  </Typography>
                  <Box sx={{ mb: 2, p: 1.5, backgroundColor: "#fff", borderRadius: "4px" }}>
                    <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                      <TextField
                        type="number"
                        label="Valor mínimo"
                        value={mlpBayesianParams.max_iter.low}
                        onChange={(e) => setMlpBayesianParams({
                          ...mlpBayesianParams,
                          max_iter: { ...mlpBayesianParams.max_iter, low: parseInt(e.target.value) || 200 }
                        })}
                        slotProps={{ input: { step: 10, min: 50 } }}
                        size="small"
                      />
                      <TextField
                        type="number"
                        label="Valor máximo"
                        value={mlpBayesianParams.max_iter.high}
                        onChange={(e) => setMlpBayesianParams({
                          ...mlpBayesianParams,
                          max_iter: { ...mlpBayesianParams.max_iter, high: parseInt(e.target.value) || 500 }
                        })}
                        slotProps={{ input: { step: 10, min: 50 } }}
                        size="small"
                      />
                    </Box>
                  </Box>
                </Box>
              )}

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Tipo de problema</InputLabel>
                <Select
                  value={problemType}
                  onChange={(e) => setProblemType(e.target.value)}
                  label="Tipo de problema"
                >
                  <MenuItem value="binary">Binario</MenuItem>
                  <MenuItem value="multiclass">Multiclase</MenuItem>
                </Select>
              </FormControl>
            </>
          )}

          {/* Para XGBoost */}
          {algorithm === "xgboost" && (
            <>
              {optimizationMethod === "manual" && (
                <>
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mt: 2 }}>
                    Hiperparámetros - XGBoost
                  </Typography>
                  <TextField
                    fullWidth
                    label="Tasa de aprendizaje"
                    value={xgboostParams.learning_rate}
                    onChange={(e) =>
                      setXgboostParams({ ...xgboostParams, learning_rate: e.target.value })
                    }
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="Número de estimadores"
                    value={xgboostParams.n_estimators}
                    onChange={(e) =>
                      setXgboostParams({ ...xgboostParams, n_estimators: e.target.value })
                    }
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="Profundidad máxima"
                    value={xgboostParams.max_depth}
                    onChange={(e) =>
                      setXgboostParams({ ...xgboostParams, max_depth: e.target.value })
                    }
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="Submuestra (subsample)"
                    value={xgboostParams.subsample}
                    onChange={(e) =>
                      setXgboostParams({ ...xgboostParams, subsample: e.target.value })
                    }
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="Colsample por árbol"
                    value={xgboostParams.colsample_bytree}
                    onChange={(e) =>
                      setXgboostParams({ ...xgboostParams, colsample_bytree: e.target.value })
                    }
                    sx={{ mb: 2 }}
                  />
                </>
              )}

              {/* XGBoost Random Search parameter ranges */}
              {optimizationMethod === "random" && (
                <Box sx={{ mt: 2, mb: 3, p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 2 }}>
                    Rangos de hiperparámetros - XGBoost Random Search
                  </Typography>

                  <Typography variant="body2" sx={{ color: "#666", mb: 2 }}>
                    Define los rangos para la búsqueda aleatoria de hiperparámetros XGBoost
                  </Typography>

                  {/* N Estimators Range */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Rango de número de estimadores:
                  </Typography>
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                    <TextField
                      type="number"
                      label="N estimators mínimo"
                      value={xgboostRandomRanges.n_estimators_range[0]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        n_estimators_range: [parseInt(e.target.value) || 50, xgboostRandomRanges.n_estimators_range[1]]
                      })}
                      slotProps={{ input: { step: 10, min: 10 } }}
                      size="small"
                    />
                    <TextField
                      type="number"
                      label="N estimators máximo"
                      value={xgboostRandomRanges.n_estimators_range[1]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        n_estimators_range: [xgboostRandomRanges.n_estimators_range[0], parseInt(e.target.value) || 500]
                      })}
                      slotProps={{ input: { step: 10, min: 10 } }}
                      size="small"
                    />
                  </Box>

                  {/* Max Depth Range */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Rango de profundidad máxima:
                  </Typography>
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                    <TextField
                      type="number"
                      label="Max depth mínimo"
                      value={xgboostRandomRanges.max_depth_range[0]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        max_depth_range: [parseInt(e.target.value) || 3, xgboostRandomRanges.max_depth_range[1]]
                      })}
                      slotProps={{ input: { step: 1, min: 1 } }}
                      size="small"
                    />
                    <TextField
                      type="number"
                      label="Max depth máximo"
                      value={xgboostRandomRanges.max_depth_range[1]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        max_depth_range: [xgboostRandomRanges.max_depth_range[0], parseInt(e.target.value) || 10]
                      })}
                      slotProps={{ input: { step: 1, min: 1 } }}
                      size="small"
                    />
                  </Box>

                  {/* Learning Rate Range */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Rango de tasa de aprendizaje:
                  </Typography>
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                    <TextField
                      type="number"
                      label="Learning rate mínimo"
                      value={xgboostRandomRanges.learning_rate_range[0]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        learning_rate_range: [parseFloat(e.target.value) || 0.01, xgboostRandomRanges.learning_rate_range[1]]
                      })}
                      slotProps={{ input: { step: 0.01, min: 0.001 } }}
                      size="small"
                    />
                    <TextField
                      type="number"
                      label="Learning rate máximo"
                      value={xgboostRandomRanges.learning_rate_range[1]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        learning_rate_range: [xgboostRandomRanges.learning_rate_range[0], parseFloat(e.target.value) || 0.3]
                      })}
                      slotProps={{ input: { step: 0.01, min: 0.001 } }}
                      size="small"
                    />
                  </Box>

                  {/* Subsample Range */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Rango de subsample:
                  </Typography>
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                    <TextField
                      type="number"
                      label="Subsample mínimo"
                      value={xgboostRandomRanges.subsample_range[0]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        subsample_range: [parseFloat(e.target.value) || 0.5, xgboostRandomRanges.subsample_range[1]]
                      })}
                      slotProps={{ input: { step: 0.1, min: 0.1, max: 1.0 } }}
                      size="small"
                    />
                    <TextField
                      type="number"
                      label="Subsample máximo"
                      value={xgboostRandomRanges.subsample_range[1]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        subsample_range: [xgboostRandomRanges.subsample_range[0], parseFloat(e.target.value) || 1.0]
                      })}
                      slotProps={{ input: { step: 0.1, min: 0.1, max: 1.0 } }}
                      size="small"
                    />
                  </Box>

                  {/* Colsample by Tree Range */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Rango de colsample_bytree:
                  </Typography>
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                    <TextField
                      type="number"
                      label="Colsample mínimo"
                      value={xgboostRandomRanges.colsample_bytree_range[0]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        colsample_bytree_range: [parseFloat(e.target.value) || 0.5, xgboostRandomRanges.colsample_bytree_range[1]]
                      })}
                      slotProps={{ input: { step: 0.1, min: 0.1, max: 1.0 } }}
                      size="small"
                    />
                    <TextField
                      type="number"
                      label="Colsample máximo"
                      value={xgboostRandomRanges.colsample_bytree_range[1]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        colsample_bytree_range: [xgboostRandomRanges.colsample_bytree_range[0], parseFloat(e.target.value) || 1.0]
                      })}
                      slotProps={{ input: { step: 0.1, min: 0.1, max: 1.0 } }}
                      size="small"
                    />
                  </Box>

                  {/* Gamma Range */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Rango de gamma:
                  </Typography>
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                    <TextField
                      type="number"
                      label="Gamma mínimo"
                      value={xgboostRandomRanges.gamma_range[0]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        gamma_range: [parseFloat(e.target.value) || 0.0, xgboostRandomRanges.gamma_range[1]]
                      })}
                      slotProps={{ input: { step: 0.1, min: 0.0 } }}
                      size="small"
                    />
                    <TextField
                      type="number"
                      label="Gamma máximo"
                      value={xgboostRandomRanges.gamma_range[1]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        gamma_range: [xgboostRandomRanges.gamma_range[0], parseFloat(e.target.value) || 5.0]
                      })}
                      slotProps={{ input: { step: 0.1, min: 0.0 } }}
                      size="small"
                    />
                  </Box>

                  {/* Min Child Weight Range */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Rango de min_child_weight:
                  </Typography>
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                    <TextField
                      type="number"
                      label="Min child weight mínimo"
                      value={xgboostRandomRanges.min_child_weight_range[0]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        min_child_weight_range: [parseInt(e.target.value) || 1, xgboostRandomRanges.min_child_weight_range[1]]
                      })}
                      slotProps={{ input: { step: 1, min: 1 } }}
                      size="small"
                    />
                    <TextField
                      type="number"
                      label="Min child weight máximo"
                      value={xgboostRandomRanges.min_child_weight_range[1]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        min_child_weight_range: [xgboostRandomRanges.min_child_weight_range[0], parseInt(e.target.value) || 10]
                      })}
                      slotProps={{ input: { step: 1, min: 1 } }}
                      size="small"
                    />
                  </Box>

                  {/* Reg Alpha Range */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Rango de reg_alpha (regularización L1):
                  </Typography>
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2, mb: 2 }}>
                    <TextField
                      type="number"
                      label="Reg alpha mínimo"
                      value={xgboostRandomRanges.reg_alpha_range[0]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        reg_alpha_range: [parseFloat(e.target.value) || 0.0, xgboostRandomRanges.reg_alpha_range[1]]
                      })}
                      slotProps={{ input: { step: 0.1, min: 0.0 } }}
                      size="small"
                    />
                    <TextField
                      type="number"
                      label="Reg alpha máximo"
                      value={xgboostRandomRanges.reg_alpha_range[1]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        reg_alpha_range: [xgboostRandomRanges.reg_alpha_range[0], parseFloat(e.target.value) || 1.0]
                      })}
                      slotProps={{ input: { step: 0.1, min: 0.0 } }}
                      size="small"
                    />
                  </Box>

                  {/* Reg Lambda Range */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Rango de reg_lambda (regularización L2):
                  </Typography>
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                    <TextField
                      type="number"
                      label="Reg lambda mínimo"
                      value={xgboostRandomRanges.reg_lambda_range[0]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        reg_lambda_range: [parseFloat(e.target.value) || 0.0, xgboostRandomRanges.reg_lambda_range[1]]
                      })}
                      slotProps={{ input: { step: 0.1, min: 0.0 } }}
                      size="small"
                    />
                    <TextField
                      type="number"
                      label="Reg lambda máximo"
                      value={xgboostRandomRanges.reg_lambda_range[1]}
                      onChange={(e) => setXgboostRandomRanges({
                        ...xgboostRandomRanges,
                        reg_lambda_range: [xgboostRandomRanges.reg_lambda_range[0], parseFloat(e.target.value) || 1.0]
                      })}
                      slotProps={{ input: { step: 0.1, min: 0.0 } }}
                      size="small"
                    />
                  </Box>
                </Box>
              )}

              {/* XGBoost Bayesian Search parameter space */}
              {optimizationMethod === "bayesian" && (
                <Box sx={{ mt: 2, mb: 3, p: 2, border: "1px solid #b0bec5", borderRadius: "8px", backgroundColor: "#f9f9f9" }}>
                  <Typography sx={{ fontWeight: "bold", color: "#004d40", mb: 2 }}>
                    Espacio de parámetros - XGBoost Bayesian Search
                  </Typography>

                  <Typography variant="body2" sx={{ color: "#666", mb: 2 }}>
                    Define el espacio de búsqueda para optimización Bayesiana
                  </Typography>

                  {/* n_estimators Parameter */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    N Estimators (Número de estimadores):
                  </Typography>
                  <Box sx={{ mb: 2, p: 1.5, backgroundColor: "#fff", borderRadius: "4px" }}>
                    <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                      <TextField
                        type="number"
                        label="Valor mínimo"
                        value={xgboostBayesianParams.n_estimators.low}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          n_estimators: { ...xgboostBayesianParams.n_estimators, low: parseInt(e.target.value) || 50 }
                        })}
                        slotProps={{ input: { step: 10, min: 10 } }}
                        size="small"
                      />
                      <TextField
                        type="number"
                        label="Valor máximo"
                        value={xgboostBayesianParams.n_estimators.high}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          n_estimators: { ...xgboostBayesianParams.n_estimators, high: parseInt(e.target.value) || 500 }
                        })}
                        slotProps={{ input: { step: 10, min: 10 } }}
                        size="small"
                      />
                    </Box>
                  </Box>

                  {/* max_depth Parameter */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Max Depth (Profundidad máxima):
                  </Typography>
                  <Box sx={{ mb: 2, p: 1.5, backgroundColor: "#fff", borderRadius: "4px" }}>
                    <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                      <TextField
                        type="number"
                        label="Valor mínimo"
                        value={xgboostBayesianParams.max_depth.low}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          max_depth: { ...xgboostBayesianParams.max_depth, low: parseInt(e.target.value) || 3 }
                        })}
                        slotProps={{ input: { step: 1, min: 1 } }}
                        size="small"
                      />
                      <TextField
                        type="number"
                        label="Valor máximo"
                        value={xgboostBayesianParams.max_depth.high}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          max_depth: { ...xgboostBayesianParams.max_depth, high: parseInt(e.target.value) || 10 }
                        })}
                        slotProps={{ input: { step: 1, min: 1 } }}
                        size="small"
                      />
                    </Box>
                  </Box>

                  {/* learning_rate Parameter */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Learning Rate (Tasa de aprendizaje):
                  </Typography>
                  <Box sx={{ mb: 2, p: 1.5, backgroundColor: "#fff", borderRadius: "4px" }}>
                    <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                      <InputLabel>Distribución</InputLabel>
                      <Select
                        value={xgboostBayesianParams.learning_rate.distribution}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          learning_rate: { ...xgboostBayesianParams.learning_rate, distribution: e.target.value }
                        })}
                        label="Distribución"
                      >
                        <MenuItem value="log-uniform">Log-uniforme (recomendado)</MenuItem>
                        <MenuItem value="uniform">Uniforme</MenuItem>
                      </Select>
                    </FormControl>
                    <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                      <TextField
                        type="number"
                        label="Valor mínimo"
                        value={xgboostBayesianParams.learning_rate.low}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          learning_rate: { ...xgboostBayesianParams.learning_rate, low: parseFloat(e.target.value) || 0.01 }
                        })}
                        slotProps={{ input: { step: 0.01, min: 0.001 } }}
                        size="small"
                      />
                      <TextField
                        type="number"
                        label="Valor máximo"
                        value={xgboostBayesianParams.learning_rate.high}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          learning_rate: { ...xgboostBayesianParams.learning_rate, high: parseFloat(e.target.value) || 0.3 }
                        })}
                        slotProps={{ input: { step: 0.01, min: 0.001 } }}
                        size="small"
                      />
                    </Box>
                  </Box>

                  {/* subsample Parameter */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Subsample:
                  </Typography>
                  <Box sx={{ mb: 2, p: 1.5, backgroundColor: "#fff", borderRadius: "4px" }}>
                    <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                      <InputLabel>Distribución</InputLabel>
                      <Select
                        value={xgboostBayesianParams.subsample.distribution}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          subsample: { ...xgboostBayesianParams.subsample, distribution: e.target.value }
                        })}
                        label="Distribución"
                      >
                        <MenuItem value="uniform">Uniforme (recomendado)</MenuItem>
                        <MenuItem value="log-uniform">Log-uniforme</MenuItem>
                      </Select>
                    </FormControl>
                    <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                      <TextField
                        type="number"
                        label="Valor mínimo"
                        value={xgboostBayesianParams.subsample.low}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          subsample: { ...xgboostBayesianParams.subsample, low: parseFloat(e.target.value) || 0.5 }
                        })}
                        slotProps={{ input: { step: 0.1, min: 0.1, max: 1.0 } }}
                        size="small"
                      />
                      <TextField
                        type="number"
                        label="Valor máximo"
                        value={xgboostBayesianParams.subsample.high}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          subsample: { ...xgboostBayesianParams.subsample, high: parseFloat(e.target.value) || 1.0 }
                        })}
                        slotProps={{ input: { step: 0.1, min: 0.1, max: 1.0 } }}
                        size="small"
                      />
                    </Box>
                  </Box>

                  {/* colsample_bytree Parameter */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Colsample by Tree:
                  </Typography>
                  <Box sx={{ mb: 2, p: 1.5, backgroundColor: "#fff", borderRadius: "4px" }}>
                    <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                      <InputLabel>Distribución</InputLabel>
                      <Select
                        value={xgboostBayesianParams.colsample_bytree.distribution}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          colsample_bytree: { ...xgboostBayesianParams.colsample_bytree, distribution: e.target.value }
                        })}
                        label="Distribución"
                      >
                        <MenuItem value="uniform">Uniforme (recomendado)</MenuItem>
                        <MenuItem value="log-uniform">Log-uniforme</MenuItem>
                      </Select>
                    </FormControl>
                    <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                      <TextField
                        type="number"
                        label="Valor mínimo"
                        value={xgboostBayesianParams.colsample_bytree.low}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          colsample_bytree: { ...xgboostBayesianParams.colsample_bytree, low: parseFloat(e.target.value) || 0.5 }
                        })}
                        slotProps={{ input: { step: 0.1, min: 0.1, max: 1.0 } }}
                        size="small"
                      />
                      <TextField
                        type="number"
                        label="Valor máximo"
                        value={xgboostBayesianParams.colsample_bytree.high}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          colsample_bytree: { ...xgboostBayesianParams.colsample_bytree, high: parseFloat(e.target.value) || 1.0 }
                        })}
                        slotProps={{ input: { step: 0.1, min: 0.1, max: 1.0 } }}
                        size="small"
                      />
                    </Box>
                  </Box>

                  {/* gamma Parameter */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Gamma:
                  </Typography>
                  <Box sx={{ mb: 2, p: 1.5, backgroundColor: "#fff", borderRadius: "4px" }}>
                    <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                      <InputLabel>Distribución</InputLabel>
                      <Select
                        value={xgboostBayesianParams.gamma.distribution}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          gamma: { ...xgboostBayesianParams.gamma, distribution: e.target.value }
                        })}
                        label="Distribución"
                      >
                        <MenuItem value="uniform">Uniforme (recomendado)</MenuItem>
                        <MenuItem value="log-uniform">Log-uniforme</MenuItem>
                      </Select>
                    </FormControl>
                    <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                      <TextField
                        type="number"
                        label="Valor mínimo"
                        value={xgboostBayesianParams.gamma.low}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          gamma: { ...xgboostBayesianParams.gamma, low: parseFloat(e.target.value) || 0.0 }
                        })}
                        slotProps={{ input: { step: 0.1, min: 0.0 } }}
                        size="small"
                      />
                      <TextField
                        type="number"
                        label="Valor máximo"
                        value={xgboostBayesianParams.gamma.high}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          gamma: { ...xgboostBayesianParams.gamma, high: parseFloat(e.target.value) || 5.0 }
                        })}
                        slotProps={{ input: { step: 0.1, min: 0.0 } }}
                        size="small"
                      />
                    </Box>
                  </Box>

                  {/* min_child_weight Parameter */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Min Child Weight:
                  </Typography>
                  <Box sx={{ mb: 2, p: 1.5, backgroundColor: "#fff", borderRadius: "4px" }}>
                    <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                      <TextField
                        type="number"
                        label="Valor mínimo"
                        value={xgboostBayesianParams.min_child_weight.low}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          min_child_weight: { ...xgboostBayesianParams.min_child_weight, low: parseInt(e.target.value) || 1 }
                        })}
                        slotProps={{ input: { step: 1, min: 1 } }}
                        size="small"
                      />
                      <TextField
                        type="number"
                        label="Valor máximo"
                        value={xgboostBayesianParams.min_child_weight.high}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          min_child_weight: { ...xgboostBayesianParams.min_child_weight, high: parseInt(e.target.value) || 10 }
                        })}
                        slotProps={{ input: { step: 1, min: 1 } }}
                        size="small"
                      />
                    </Box>
                  </Box>

                  {/* reg_alpha Parameter */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Reg Alpha (Regularización L1):
                  </Typography>
                  <Box sx={{ mb: 2, p: 1.5, backgroundColor: "#fff", borderRadius: "4px" }}>
                    <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                      <InputLabel>Distribución</InputLabel>
                      <Select
                        value={xgboostBayesianParams.reg_alpha.distribution}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          reg_alpha: { ...xgboostBayesianParams.reg_alpha, distribution: e.target.value }
                        })}
                        label="Distribución"
                      >
                        <MenuItem value="uniform">Uniforme (recomendado)</MenuItem>
                        <MenuItem value="log-uniform">Log-uniforme</MenuItem>
                      </Select>
                    </FormControl>
                    <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                      <TextField
                        type="number"
                        label="Valor mínimo"
                        value={xgboostBayesianParams.reg_alpha.low}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          reg_alpha: { ...xgboostBayesianParams.reg_alpha, low: parseFloat(e.target.value) || 0.0 }
                        })}
                        slotProps={{ input: { step: 0.1, min: 0.0 } }}
                        size="small"
                      />
                      <TextField
                        type="number"
                        label="Valor máximo"
                        value={xgboostBayesianParams.reg_alpha.high}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          reg_alpha: { ...xgboostBayesianParams.reg_alpha, high: parseFloat(e.target.value) || 1.0 }
                        })}
                        slotProps={{ input: { step: 0.1, min: 0.0 } }}
                        size="small"
                      />
                    </Box>
                  </Box>

                  {/* reg_lambda Parameter */}
                  <Typography variant="body2" sx={{ fontWeight: 500, color: "#004d40", mb: 1 }}>
                    Reg Lambda (Regularización L2):
                  </Typography>
                  <Box sx={{ mb: 2, p: 1.5, backgroundColor: "#fff", borderRadius: "4px" }}>
                    <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                      <InputLabel>Distribución</InputLabel>
                      <Select
                        value={xgboostBayesianParams.reg_lambda.distribution}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          reg_lambda: { ...xgboostBayesianParams.reg_lambda, distribution: e.target.value }
                        })}
                        label="Distribución"
                      >
                        <MenuItem value="uniform">Uniforme (recomendado)</MenuItem>
                        <MenuItem value="log-uniform">Log-uniforme</MenuItem>
                      </Select>
                    </FormControl>
                    <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
                      <TextField
                        type="number"
                        label="Valor mínimo"
                        value={xgboostBayesianParams.reg_lambda.low}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          reg_lambda: { ...xgboostBayesianParams.reg_lambda, low: parseFloat(e.target.value) || 0.0 }
                        })}
                        slotProps={{ input: { step: 0.1, min: 0.0 } }}
                        size="small"
                      />
                      <TextField
                        type="number"
                        label="Valor máximo"
                        value={xgboostBayesianParams.reg_lambda.high}
                        onChange={(e) => setXgboostBayesianParams({
                          ...xgboostBayesianParams,
                          reg_lambda: { ...xgboostBayesianParams.reg_lambda, high: parseFloat(e.target.value) || 1.0 }
                        })}
                        slotProps={{ input: { step: 0.1, min: 0.0 } }}
                        size="small"
                      />
                    </Box>
                  </Box>
                </Box>
              )}

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Tipo de problema</InputLabel>
                <Select
                  value={problemType}
                  onChange={(e) => setProblemType(e.target.value)}
                  label="Tipo de problema"
                >
                  <MenuItem value="binary">Binario</MenuItem>
                  <MenuItem value="multiclass">Multiclase</MenuItem>
                </Select>
              </FormControl>
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
                {helperTextStrings.variablesDeEntrada}
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
                          disabled={col === targetVariable}
                        />
                      }
                      label={col}
                      sx={variableSelectionStyles.formControlLabel}
                    />
                  ))}
                </FormGroup>
              </Box>
            </Box>
          </>
        )}

        {/* Progress Bar - shown when training */}
        {trainInProgress && (
          <Box sx={{ mt: 2, mb: 2 }}>
            <ProgressBar
              message="Entrenando modelo y optimizando hiperparámetros..."
              variant="tealHarmony"
              showPercentage={false}
              useWebSocket={true}
              wsStep="training"
            />
          </Box>
        )}

        <Button
          variant="contained"
          onClick={handleTrain}
          disabled={isDisabled}
          sx={{
            mt: trainInProgress ? 0 : 2,
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
            Revisa los resultados en MLflow y en el directorio 'trained'
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




const styles = {
  card: {
    backgroundColor: "#e0f7fa",
    borderRadius: "12px",
    padding: "30px",
    textAlign: "center",
    boxShadow: "0 4px 12px rgba(0, 121, 107, 0.3)",
    margin: "20px",
    border: "1px solid #00796b",
  },
  title: {
    fontSize: "1.8rem",
    color: "#333",
    marginBottom: "20px",
    fontWeight: "bold",
  },
  description: {
    fontSize: "1.2rem",
    color: "#333",
    marginBottom: "20px",
  },
  label: {
    display: "block",
    marginBottom: "5px",
    color: "#333",
    fontWeight: "bold",
  },
  input: {
    width: "100%",
    padding: "10px",
    marginBottom: "15px",
    border: "1px solid #ccc",
    borderRadius: "8px",
  },
  select: {
    width: "100%",
    padding: "10px",
    marginBottom: "15px",
    border: "1px solid #ccc",
    borderRadius: "8px",
  },
  button: {
    marginTop: "20px",
    padding: "10px 20px",
    backgroundColor: "#00796b",
    color: "white",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
  },
  status: {
    marginTop: "15px",
    color: "#333",
  },
  subtitle: {
    fontSize: "1.2rem",
    color: "#2c3e50",
    fontWeight: "bold",
    marginTop: "15px",
  },
  scrollableColumnList: {
    maxHeight: "150px",
    overflowY: "auto",
    padding: "10px",
    border: "1px solid #b0bec5",
    backgroundColor: "#f5f5f5",
    borderRadius: "8px",
  },
  columnLabel: {
    marginLeft: "10px",
    fontSize: "1rem",
    color: "#212121",
  },
  columnItem: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: "10px",
    padding: "5px 10px",
  },
  hyperparameterSection: {
    marginTop: "20px",
  },
};

export default TrainCard;
