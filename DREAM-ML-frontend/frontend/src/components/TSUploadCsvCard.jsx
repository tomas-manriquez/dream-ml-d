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


import  React, { useState, useContext, useEffect } from "react";
import ProgressBar from './ProgressBar';
import InfoModal from './InfoModal';
import ValidationSummary from './ValidationSummary';
import {
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Checkbox,
  Box,
  Grid,
  FormGroup,
  Radio,
  RadioGroup,
  IconButton,
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

const TSUploadCsvCard = () => {
  const { 
    experimentDir, 
    setCleanedFilePath, 
    setRunId, 
    flow, 
    markStepDone,
    csvAnalyzeInProgress,
    setCsvAnalyzeInProgress,
    csvUploadCleaningInProgress,
    setCsvUploadCleaningInProgress,
    uploadStatus,
    setUploadStatus,
  } = useContext(AppContext);
  
  const [csvFile, setCsvFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [inputFeatures, setInputFeatures] = useState([]);
  const [targetVariable, setTargetVariable] = useState(""); // Changed to single selection
  const [eliminarDuplicados, setEliminarDuplicados] = useState(false);
  const [filtrarOutliers, setFiltrarOutliers] = useState(false);
  const [rellenoValoresNumericos, setRellenoValoresNumericos] = useState("dejar");
  const [valorImputacion, setValorImputacion] = useState("");
  const [advertencia, setAdvertencia] = useState("");
  const [edaFilePath, setEdaFilePath] = useState("");
  const [trainFilePath, setTrainFilePath] = useState("");
  const [dateStandardization, setDateStandardization] = useState("none");
  const [dateColumn, setDateColumn] = useState("");
  const [dateImputationStrategy, setDateImputationStrategy] = useState("mean_timedelta");
  const [datePreview, setDatePreview] = useState(null);
  const [dateFormatDetection, setDateFormatDetection] = useState(null);
  const [dateValidationErrors, setDateValidationErrors] = useState([]);
  const [showDatePreview, setShowDatePreview] = useState(false);
  const [datePreviewLoading, setDatePreviewLoading] = useState(false);

  // Info modal states
  const [showTargetInfo, setShowTargetInfo] = useState(false);
  const [showFeatureInfo, setShowFeatureInfo] = useState(false);

  // Validation warnings
  const [validationWarnings, setValidationWarnings] = useState([]);

  // Auto-select first column when date standardization is enabled and auto-trigger preview
  useEffect(() => {
    if (dateStandardization !== "none" && columns.length > 0 && !dateColumn) {
      setDateColumn(columns[0]);
    }

    // Auto-trigger preview when date column changes and standardization is enabled
    if (dateStandardization !== "none" && dateColumn && csvFile) {
      const timeoutId = setTimeout(() => {
        previewDateStandardization();
      }, 500); // Debounce to avoid excessive API calls

      return () => clearTimeout(timeoutId);
    } else {
      setShowDatePreview(false);
      setDatePreview(null);
      setDateFormatDetection(null);
    }
  }, [dateStandardization, dateColumn, csvFile]);

  // Maneja la selección del archivo CSV
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setCsvFile(file);
    setUploadStatus("Archivo seleccionado");
    setColumns([]);
    setAdvertencia("");
    setEdaFilePath("");
    setTrainFilePath("");
    setDateStandardization("none");
    setDateColumn("");

    // Reset new date-related state
    setDateImputationStrategy("mean_timedelta");
    setDatePreview(null);
    setDateFormatDetection(null);
    setDateValidationErrors([]);
    setShowDatePreview(false);
  };

  // Maneja el cambio de la opción de imputación
  const handleRellenoChange = (e) => {
    const value = e.target.value;
    setRellenoValoresNumericos(value);
    if (value === "dejar") {
      setAdvertencia(
        "Advertencia: Dejar valores faltantes puede generar errores en algoritmos de machine learning."
      );
    } else {
      setAdvertencia("");
    }
  };

  // Preview date standardization functionality
  const previewDateStandardization = async () => {
    if (!csvFile || !dateColumn || dateStandardization === "none") {
      return;
    }

    setDatePreviewLoading(true);
    setDateValidationErrors([]);

    const formData = new FormData();
    formData.append("file", csvFile);
    formData.append("date_column", dateColumn);
    formData.append("standardization_type", dateStandardization);

    try {
      const response = await axios.post("/ts/preview-date-standardization/", formData);
      setDateFormatDetection(response.data.format_detection);
      setDatePreview(response.data.preview_samples);
      setShowDatePreview(true);

      if (response.data.validation_warnings && response.data.validation_warnings.length > 0) {
        setDateValidationErrors(response.data.validation_warnings);
      }
    } catch (error) {
      console.error("Error previewing date standardization:", error);
      setDateValidationErrors(["Error al previsualizar la estandarización de fechas"]);
    } finally {
      setDatePreviewLoading(false);
    }
  };

  // Llama al endpoint para analizar el CSV y obtener las columnas
  const analyzeCsv = async () => {
    if (!csvFile) {
      setUploadStatus("Por favor, selecciona un archivo CSV primero.");
      return;
    }
    setCsvAnalyzeInProgress(true);
    setUploadStatus("Cargando columnas...");
    const formData = new FormData();
    formData.append("file", csvFile);
    try {
      const response = await axios.post("/ts/analyze-csv/", formData);
      if (response.data.columns && response.data.columns.length > 0) {
        setColumns(response.data.columns);
        setUploadStatus("Selecciona las variables de entrada y salida.");
        console.log(response.data.columns);
      } else {
        setUploadStatus("El archivo no contiene columnas válidas.");
      }
    } catch (error) {
      console.error("Error al analizar el CSV:", error);
      setUploadStatus("Error al analizar el archivo.");
    } finally {
      setCsvAnalyzeInProgress(false);
    }
  };

    const handleFeatureChange = (column) => {
    setInputFeatures((prev) =>
        prev.includes(column)
        ? prev.filter((item) => item !== column)
        : [...prev, column]
    );
    validateSelections();
    };

    const handleTargetChange = (column) => {
    if (targetVariable === column) {
        // If clicking the same target, deselect it
        setTargetVariable("");
        setInputFeatures([]);
    } else {
        // Set new target and auto-select remaining columns as features
        setTargetVariable(column);
        // Auto-select all other columns as input features (keeping previous selections + adding new ones)
        const remainingColumns = columns.filter((col) => col !== column);
        const newFeatures = [...new Set([...inputFeatures, ...remainingColumns])];
        setInputFeatures(newFeatures);
    }
    validateSelections();
    };

    // Validation function
    const validateSelections = () => {
    const warnings = [];

    // Check if at least 1 feature is selected
    if (inputFeatures.length === 0 && targetVariable) {
        warnings.push("Debes seleccionar al menos 1 variable de entrada");
    }

    // Check if target is selected
    if (!targetVariable && inputFeatures.length > 0) {
        warnings.push("Debes seleccionar 1 variable de salida");
    }

    // Check for overlap
    if (targetVariable && inputFeatures.includes(targetVariable)) {
        warnings.push("Una columna no puede ser entrada y salida simultáneamente");
    }

    setValidationWarnings(warnings);
    return warnings.length === 0;
    };

  // Helper function to determine current state
  const getCurrentState = () => {
    // Error state (highest priority)
    if (uploadStatus.includes("Error") || uploadStatus.includes("error")) {
      return {
        type: 'error',
        message: uploadStatus,
        icon: <ErrorIcon fontSize="small" />
      };
    }

    // Success state
    if (flow.cleaningDone) {
      return {
        type: 'success',
        message: 'Archivo procesado exitosamente',
        icon: <CheckCircleIcon fontSize="small" />
      };
    }

    // Processing states
    if (csvUploadCleaningInProgress) {
      return {
        type: 'processing',
        message: 'Subiendo y procesando archivo...',
        icon: <AutorenewIcon fontSize="small" sx={{ animation: 'spin 1s linear infinite', '@keyframes spin': { '0%': { transform: 'rotate(0deg)' }, '100%': { transform: 'rotate(360deg)' } } }} />
      };
    }

    if (csvAnalyzeInProgress) {
      return {
        type: 'processing',
        message: 'Analizando columnas del CSV...',
        icon: <AutorenewIcon fontSize="small" sx={{ animation: 'spin 1s linear infinite', '@keyframes spin': { '0%': { transform: 'rotate(0deg)' }, '100%': { transform: 'rotate(360deg)' } } }} />
      };
    }

    // Prerequisites check
    if (!experimentDir || !flow.configRemoteDvc) {
      return {
        type: 'warning',
        message: 'Requiere experimento creado y almacenamiento remoto configurado',
        icon: <InfoIcon fontSize="small" />
      };
    }

    // Columns loaded, ready to configure or upload
    if (columns.length > 0 && targetVariable && inputFeatures.length > 0) {
      return {
        type: 'info',
        message: 'Configuración completada - listo para subir',
        icon: <InfoIcon fontSize="small" />
      };
    }

    // File selected but not analyzed
    if (csvFile) {
      return {
        type: 'info',
        message: 'Archivo seleccionado - previsualiza columnas',
        icon: <InfoIcon fontSize="small" />
      };
    }

    // Initial state
    return {
      type: 'info',
      message: 'Listo para cargar archivo CSV',
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

  // Build optionalMethods object from form state
  const buildOptionalMethods = () => {
    const cleaningMethods = [];
    
    // Add drop_duplicates if enabled
    if (eliminarDuplicados) {
      cleaningMethods.push({
        method: "drop_duplicates", 
        params: { include: true }
      });
    }
    
    // Add filter_outliers if enabled
    if (filtrarOutliers) {
      cleaningMethods.push({
        method: "filter_outliers",
        params: {}
      });
    }
    
    // Add date standardization methods if enabled
    if (dateStandardization === "utc" && dateColumn) {
      cleaningMethods.push({
        method: "standardize_date_to_utc",
        params: {
          date_column: dateColumn,
          imputation_strategy: dateImputationStrategy
        }
      });
    } else if (dateStandardization === "retain_timezone" && dateColumn) {
      cleaningMethods.push({
        method: "standardize_date_retain_timezone",
        params: {
          date_column: dateColumn,
          imputation_strategy: dateImputationStrategy
        }
      });
    }
    
    // Calculate the value for imputation
    let imputationValue = null;
    if (rellenoValoresNumericos === "valor") {
      const parsedValue = parseFloat(valorImputacion);
      imputationValue = isNaN(parsedValue) ? null : parsedValue;
    }
    
    // Always add fill_missing_numeric_values 
    cleaningMethods.push({
      method: "fill_missing_numeric_values",
      params: {
        method: rellenoValoresNumericos,
        value: imputationValue
      }
    });

    return { cleaning_methods: cleaningMethods };
  };

  // Llama al endpoint para subir y limpiar el CSV
  const uploadAndCleanCsv = async () => {
    if (!inputFeatures.length || !targetVariable) {
      setUploadStatus("Por favor, selecciona variables de entrada y salida.");
      return;
    }
    if (rellenoValoresNumericos === "valor" && isNaN(Number(valorImputacion))) {
      setUploadStatus("Por favor, introduce un valor numérico válido para la imputación.");
      return;
    }
    if (dateStandardization !== "none" && !dateColumn) {
      setUploadStatus("Por favor, selecciona una columna de fecha para la estandarización.");
      return;
    }
    setCsvUploadCleaningInProgress(true);
    setUploadStatus("Subiendo y procesando el archivo...");
    const formData = new FormData();
    formData.append("file", csvFile);
    formData.append("experiment_dir", experimentDir);
    const optionalMethodsConfig = buildOptionalMethods();
    formData.append("optional_methods", JSON.stringify(optionalMethodsConfig));
    for (let pair of formData.entries()) {
        console.log(`${pair[0]}: ${pair[1]}`);
      }
    try {
      const response = await axios.post("/ts/upload-and-clean-csv/", formData);
      setUploadStatus(response.data.status || "Proceso completado");
      setCleanedFilePath(response.data.cleaned_file_path);
      setRunId(response.data.run_id);
      setEdaFilePath(response.data.processed_eda_path);
      setTrainFilePath(response.data.processed_train_path);
      // Marcar el paso de limpieza como completado
      markStepDone("cleaningDone");
    } catch (error) {
      console.error("Error al procesar el CSV:", error);
      setUploadStatus("Error al subir y procesar el archivo.");
    } finally {
      setCsvUploadCleaningInProgress(false);
    }
  };

  // Deshabilitar la acción si:
  // - No hay archivo CSV seleccionado.
  // - No se han seleccionado variables de entrada y salida.
  // - No existe el experimento.
  // - El almacenamiento remoto no está configurado (paso previo).
  // - O ya se completó la limpieza.
  // - Hay advertencias de validación (nuevo)
  const isDisabled =
    csvUploadCleaningInProgress ||
    !csvFile ||
    !experimentDir ||
    !targetVariable ||
    !inputFeatures.length ||
    !flow.configRemoteDvc ||
    flow.cleaningDone ||
    validationWarnings.length > 0;

  return (
    <Card
      sx={{
        backgroundColor: "#e0f7fa",
        borderRadius: 2,
        border: "1px solid #00796b",
        boxShadow: "0 4px 12px rgba(0, 121, 107, 0.3)",
        p: 3,
        m: 2,
      }}
    >
      <CardContent>
        {/* Título de la etapa */}
        <Typography variant="h5" sx={{ mb: 2, color: "#004d40", textAlign: "center" }}>
          1. Subir y limpiar archivo CSV
        </Typography>
        {/* Botón para seleccionar el archivo */}
        <Button variant="outlined" component="label" sx={{ mb: 2 }}
          disabled={csvUploadCleaningInProgress}
        >
          Seleccionar Archivo
          <input type="file" accept=".csv" hidden onChange={handleFileChange} />
        </Button>
        {/* Botón para previsualizar columnas */}
        <Button
          variant="contained"
          onClick={analyzeCsv}
          disabled={!csvFile || csvAnalyzeInProgress || csvUploadCleaningInProgress}
          sx={{
            mb: 2,
            backgroundColor: "#00796b",
            "&:hover": { backgroundColor: "#004d40" },
          }}
        >
          {csvAnalyzeInProgress ? "Cargando..." : "Previsualizar Columnas"}
        </Button>

        {/* Barra de progreso */}
        {(csvAnalyzeInProgress || csvUploadCleaningInProgress) && (
          <ProgressBar
            useWebSocket={true}
            wsStep="data_cleaning"
            variant="tealHarmony"
            showPercentage={false}
          />
        )}

        {/* Current State Indicator - Location A: Always visible when columns not loaded */}
        {columns.length === 0 && (
          <Box
            sx={{
              mt: 2,
              mb: 2,
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
        )}

        {/* Si se obtuvieron columnas, permitir seleccionar variables */}
        {columns.length > 0 && (
          <>
            {/* VARIABLES DE SALIDA - con info modal y helper text */}
            <Box sx={variableSelectionStyles.sectionContainer}>
              <Box sx={variableSelectionStyles.sectionTitleContainer}>
                <Typography variant="subtitle1" sx={variableSelectionStyles.sectionTitle}>
                  Variables de Salida (Target)
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

            {/* VALIDATION SUMMARY - between Variables de Salida and Variables de Entrada */}
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

            {/*Seleccion de metodos preprocesamiento */}
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={12} sm={6}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={eliminarDuplicados}
                      onChange={(e) => setEliminarDuplicados(e.target.checked)}
                    />
                  }
                  label="Eliminar duplicados"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={filtrarOutliers}
                      onChange={(e) => setFiltrarOutliers(e.target.checked)}
                    />
                  }
                  label="Filtrar outliers"
                />
              </Grid>
            </Grid>

            {/* Date standardization section */}
            <Box sx={{ mb: 2, textAlign: "left" }}>
              <Typography variant="body2" sx={{ color: "#004d40", fontWeight: "500", mb: 1 }}>
                Estandarización de fechas:
              </Typography>
              <FormControl component="fieldset" sx={{ mb: 2 }}>
                <RadioGroup
                  value={dateStandardization}
                  onChange={(e) => setDateStandardization(e.target.value)}
                >
                  <FormControlLabel value="none" control={<Radio />} label="No estandarizar fechas" />
                  <FormControlLabel value="utc" control={<Radio />} label="Convertir a UTC" />
                  <FormControlLabel value="retain_timezone" control={<Radio />} label="Conservar zona horaria original" />
                </RadioGroup>
              </FormControl>
              
              {dateStandardization !== "none" && (
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel id="date-column-select-label">Columna de fecha</InputLabel>
                  <Select
                    labelId="date-column-select-label"
                    value={dateColumn}
                    label="Columna de fecha"
                    onChange={(e) => setDateColumn(e.target.value)}
                  >
                    {columns.map((col) => (
                      <MenuItem key={col} value={col}>{col}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              )}

              {/* Date Format Detection & Preview */}
              {dateStandardization !== "none" && dateColumn && (
                <Box sx={{ mb: 2, p: 2, backgroundColor: "#f5f5f5", borderRadius: 1 }}>
                  <Typography variant="subtitle2" sx={{ color: "#004d40", fontWeight: "bold", mb: 1 }}>
                    Detección de Formato y Vista Previa
                  </Typography>

                  {datePreviewLoading && (
                    <Typography variant="body2" sx={{ color: "#666", fontStyle: "italic" }}>
                      Analizando formato de fechas...
                    </Typography>
                  )}

                  {dateFormatDetection && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" sx={{ color: "#004d40" }}>
                        <strong>Formato detectado:</strong> {dateFormatDetection.detected_format || "Formato mixto"}
                      </Typography>
                      <Typography variant="body2" sx={{ color: "#004d40" }}>
                        <strong>Zona horaria:</strong> {dateFormatDetection.timezone_info || "Sin zona horaria"}
                      </Typography>
                      {dateFormatDetection.parsing_success_rate && (
                        <Typography variant="body2" sx={{ color: "#004d40" }}>
                          <strong>Tasa de éxito:</strong> {(dateFormatDetection.parsing_success_rate * 100).toFixed(1)}%
                        </Typography>
                      )}
                    </Box>
                  )}

                  {datePreview && (
                    <Box>
                      <Typography variant="body2" sx={{ color: "#004d40", fontWeight: "bold", mb: 1 }}>
                        Vista previa de transformación:
                      </Typography>
                      <Box sx={{ maxHeight: 150, overflowY: "auto", border: "1px solid #ddd", borderRadius: 1, p: 1 }}>
                        {datePreview.map((sample, index) => (
                          <Box key={index} sx={{ mb: 1, fontSize: "0.875rem" }}>
                            <span style={{ color: "#666" }}>{sample.original}</span>
                            <span style={{ margin: "0 8px", color: "#004d40" }}>→</span>
                            <span style={{ color: "#004d40", fontWeight: "500" }}>{sample.transformed || "Error"}</span>
                          </Box>
                        ))}
                      </Box>
                    </Box>
                  )}
                </Box>
              )}

              {/* Date Imputation Strategy */}
              {dateStandardization !== "none" && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" sx={{ color: "#004d40", fontWeight: "500", mb: 1 }}>
                    Estrategia para fechas inválidas:
                  </Typography>
                  <FormControl component="fieldset">
                    <RadioGroup
                      value={dateImputationStrategy}
                      onChange={(e) => setDateImputationStrategy(e.target.value)}
                    >
                      <FormControlLabel
                        value="mean_timedelta"
                        control={<Radio />}
                        label="Calcular intervalo promedio"
                      />
                      <FormControlLabel
                        value="leave_as_is"
                        control={<Radio />}
                        label="Dejar fechas inválidas sin cambios"
                      />
                      <FormControlLabel
                        value="auto_detected"
                        control={<Radio />}
                        label="Usar intervalo detectado automáticamente"
                      />
                    </RadioGroup>
                  </FormControl>
                </Box>
              )}

              {/* Date Validation Errors */}
              {dateValidationErrors.length > 0 && (
                <Box sx={{ mb: 2, p: 2, backgroundColor: "#fff3e0", border: "1px solid #ff9800", borderRadius: 1 }}>
                  <Typography variant="subtitle2" sx={{ color: "#e65100", fontWeight: "bold", mb: 1 }}>
                    Advertencias de Validación de Fechas:
                  </Typography>
                  {dateValidationErrors.map((error, index) => (
                    <Typography key={index} variant="body2" sx={{ color: "#e65100", mb: 0.5 }}>
                      • {error}
                    </Typography>
                  ))}
                </Box>
              )}
            </Box>

            <Box sx={{ mb: 2, textAlign: "left" }}>
              <Typography variant="body2" sx={{ color: "#004d40", fontWeight: "500", mb: 1 }}>
                Relleno de valores numéricos:
              </Typography>
              <FormControl fullWidth>
                <InputLabel id="relleno-select-label">Selecciona una opción</InputLabel>
                <Select
                  labelId="relleno-select-label"
                  value={rellenoValoresNumericos}
                  label="Selecciona una opción"
                  onChange={handleRellenoChange}
                >
                  <MenuItem value="media">Imputar con la media</MenuItem>
                  <MenuItem value="valor">Imputar con un valor</MenuItem>
                  <MenuItem value="dejar">Dejar valores faltantes</MenuItem>
                  <MenuItem value="eliminar">Eliminar filas con valores nulos</MenuItem>
                </Select>
              </FormControl>
              {rellenoValoresNumericos === "valor" && (
                <TextField
                  type="number"
                  placeholder="Introduce un valor"
                  value={valorImputacion}
                  onChange={(e) => setValorImputacion(e.target.value)}
                  fullWidth
                  sx={{ mt: 1 }}
                />
              )}
            </Box>
            {advertencia && (
              <Typography variant="body2" sx={{ color: "#ff5722", fontWeight: "500", mt: 1 }}>
                {advertencia}
              </Typography>
            )}
            <Button
              variant="contained"
              onClick={uploadAndCleanCsv}
              disabled={isDisabled}
              sx={{
                backgroundColor: "#00796b",
                "&:hover": { backgroundColor: "#004d40" },
                mt: 2,
                mb: 2,
              }}
            >
              {csvUploadCleaningInProgress
                ? "Procesando..."
                : flow.cleaningDone
                ? "Limpieza Completada"
                : "Subir y limpiar CSV"}
            </Button>

            {/* Current State Indicator - Location B: Shows state after configuration */}
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

            {/* File path details - shown on success */}
            {edaFilePath && (
              <Typography variant="body2" sx={{ mt: 2, color: "#004d40", fontWeight: "500" }}>
                Archivo EDA generado: {edaFilePath}
              </Typography>
            )}
            {trainFilePath && (
              <Typography variant="body2" sx={{ mt: 2, color: "#004d40", fontWeight: "500" }}>
                Archivo de entrenamiento generado: {trainFilePath}
              </Typography>
            )}
          </>
        )}
      </CardContent>

      {/* Info Modals */}
      <InfoModal
        open={showTargetInfo}
        onClose={() => setShowTargetInfo(false)}
        title="Variables de Salida (Target)"
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

export default TSUploadCsvCard;