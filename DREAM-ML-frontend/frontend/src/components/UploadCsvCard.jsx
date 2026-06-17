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

const UploadCsvCard = () => {
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

  // Info modal states
  const [showTargetInfo, setShowTargetInfo] = useState(false);
  const [showFeatureInfo, setShowFeatureInfo] = useState(false);

  // Validation warnings
  const [validationWarnings, setValidationWarnings] = useState([]);

  // Maneja la selección del archivo CSV
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setCsvFile(file);
    setUploadStatus("Archivo seleccionado");
    setColumns([]);
    setAdvertencia("");
    setEdaFilePath("");
    setTrainFilePath("");
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
      const response = await axios.post("/analyze-csv/", formData);
      if (response.data.columns && response.data.columns.length > 0) {
        setColumns(response.data.columns);
        setUploadStatus("Selecciona las variables de entrada y salida.");
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

  // Alterna la selección de una columna para las variables de entrada
  const handleFeatureChange = (column) => {
    setInputFeatures((prev) =>
      prev.includes(column)
        ? prev.filter((item) => item !== column)
        : [...prev, column]
    );
    validateSelections();
  };

  // Selección de variable de salida (radio button - single selection)
  // Con auto-selección de features
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

    // Check for overlap (should not happen with current logic, but defensive)
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
    setCsvUploadCleaningInProgress(true);
    setUploadStatus("Subiendo y procesando el archivo...");
    const formData = new FormData();
    formData.append("file", csvFile);
    formData.append("experiment_dir", experimentDir);
    formData.append("input_features", inputFeatures.join(","));
    formData.append("target_variables", targetVariable); // Single target now
    formData.append("eliminar_duplicados", eliminarDuplicados);
    formData.append("filtrar_outliers", filtrarOutliers);
    formData.append("relleno_valores_numericos", rellenoValoresNumericos);
    if (rellenoValoresNumericos === "valor") {
      formData.append("valor_imputacion", valorImputacion);
    }
    try {
      console.log(formData);
      const response = await axios.post("/upload-and-clean-csv/", formData);
      console.log(response)
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
        {/* Descripción */}
        <Typography variant="body2" sx={{ mb: 2, color: "#004d40", textAlign: "center" }}>
          En esta etapa se analizan los datos del archivo CSV para previsualizar sus columnas.
          Luego, podrás seleccionar las variables de entrada y salida y aplicar opciones de limpieza.
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

        {/* Current State Indicator - Always visible */}
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
            {/* VARIABLES DE SALIDA PRIMERO - con info modal y helper text */}
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
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                width: "100%",
                py: 1.5,
                fontSize: "1.1rem",
                fontWeight: "bold",
              }}
            >
              {csvUploadCleaningInProgress
                ? "Procesando..."
                : flow.cleaningDone
                ? "Limpieza Completada"
                : "Subir y limpiar CSV"}
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

            {/* File path details - shown on success */}
            {edaFilePath && (
              <Typography variant="body2" sx={{ mt: 2, color: "#004d40", fontWeight: "500", textAlign: "center" }}>
                Archivo EDA generado: {edaFilePath}
              </Typography>
            )}
            {trainFilePath && (
              <Typography variant="body2" sx={{ mt: 2, color: "#004d40", fontWeight: "500", textAlign: "center" }}>
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

export default UploadCsvCard;
