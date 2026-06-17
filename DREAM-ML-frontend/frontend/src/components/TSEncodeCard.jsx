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
  Box,
  Checkbox,
  FormControlLabel,
  FormGroup,
  TextField,
  Select,
  MenuItem,
  InputLabel,
  FormControl,
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

const TSEncodeCard = () => {
  const { 
    experimentDir, 
    runId, 
    flow, 
    markStepDone,
    encodeCardAnalyzeInProgress,
    setEncodeCardAnalyzeInProgress,
    encodeCardEncodeInProgress,
    setEncodeCardEncodeInProgress,
    encodeStatus,
    setEncodeStatus,
  } = useContext(AppContext);

  const [csvFile, setCsvFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [inputFeatures, setInputFeatures] = useState([]);
  const [targetVariable, setTargetVariable] = useState(""); // Changed to single selection
  const [encodeTargetOHE, setEncodeTargetOHE] = useState(false);
  const [encodeTargetLabel, setEncodeTargetLabel] = useState(false);
  const [trainFilePath, setTrainFilePath] = useState("");

  // Time Series Parameters
  const [lagPeriods, setLagPeriods] = useState(0);
  const [lagNanHandling, setLagNanHandling] = useState("leave_as_is");
  const [dateColumn, setDateColumn] = useState("");

  // Info modal states
  const [showTargetInfo, setShowTargetInfo] = useState(false);
  const [showFeatureInfo, setShowFeatureInfo] = useState(false);

  // Validation warnings
  const [validationWarnings, setValidationWarnings] = useState([]);

  // Progress bar variant (consistent with TSEdaCard)
  const progressVariant = 'tealHarmony';

  // Helper function to determine current state
  const getCurrentState = () => {
    // Error state (highest priority)
    if (encodeStatus.includes("❌") || encodeStatus.includes("Error")) {
      return {
        type: 'error',
        message: encodeStatus.replace("❌", "").trim(),
        icon: <ErrorIcon fontSize="small" />
      };
    }

    // Success state
    if (flow.encodeDone) {
      return {
        type: 'success',
        message: 'Codificación completada exitosamente',
        icon: <CheckCircleIcon fontSize="small" />
      };
    }

    // Processing states
    if (encodeCardEncodeInProgress) {
      return {
        type: 'processing',
        message: 'Codificando dataset con parámetros de series temporales...',
        icon: <AutorenewIcon fontSize="small" sx={{ animation: 'spin 1s linear infinite', '@keyframes spin': { '0%': { transform: 'rotate(0deg)' }, '100%': { transform: 'rotate(360deg)' } } }} />
      };
    }

    if (encodeCardAnalyzeInProgress) {
      return {
        type: 'processing',
        message: 'Analizando columnas del CSV...',
        icon: <AutorenewIcon fontSize="small" sx={{ animation: 'spin 1s linear infinite', '@keyframes spin': { '0%': { transform: 'rotate(0deg)' }, '100%': { transform: 'rotate(360deg)' } } }} />
      };
    }

    // Prerequisites check
    if (!experimentDir || !runId || !flow.cleaningDone || !flow.edaDone) {
      return {
        type: 'warning',
        message: 'Requiere experimento creado y pasos previos completados',
        icon: <InfoIcon fontSize="small" />
      };
    }

    // Ready to encode (columns loaded, variables selected)
    if (columns.length > 0 && targetVariable && inputFeatures.length > 0) {
      return {
        type: 'info',
        message: 'Configuración completada - listo para codificar',
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
      message: 'Listo para codificar datos',
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

  const handleFileChange = (event) => {
    setCsvFile(event.target.files[0]);
    setEncodeStatus("📂 Archivo seleccionado.");
    setColumns([]);
    setTrainFilePath("");
    setEncodeTargetOHE(false);
    setEncodeTargetLabel(false);

    // Reset Time Series Parameters
    setLagPeriods(0);
    setLagNanHandling("leave_as_is");
    setDateColumn("");
  };

  const analyzeCsv = async () => {
    if (!csvFile) {
      setEncodeStatus("⚠️ Por favor, selecciona un archivo CSV primero.");
      return;
    }

    setEncodeCardAnalyzeInProgress(true);
    setEncodeStatus("📊 Cargando columnas...");

    const formData = new FormData();
    formData.append("file", csvFile);

    try {
      const response = await axios.post("/analyze-csv/", formData);
      setColumns(response.data.columns);
      setEncodeStatus("✅ Columnas cargadas. Selecciona variables.");
    } catch (error) {
      console.error("Error al analizar el archivo:", error);
      setEncodeStatus("❌ Error al analizar el archivo.");
    } finally {
      setEncodeCardAnalyzeInProgress(false);
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

  const handleOheChange = () => {
    setEncodeTargetOHE(!encodeTargetOHE);
    if (encodeTargetLabel) setEncodeTargetLabel(false);
  };

  const handleLabelChange = () => {
    setEncodeTargetLabel(!encodeTargetLabel);
    if (encodeTargetOHE) setEncodeTargetOHE(false);
  };

  const encodeCsv = async () => {
    if (!csvFile) {
      setEncodeStatus("⚠️ Por favor, selecciona un archivo CSV.");
      return;
    }
    if (!inputFeatures.length || !targetVariable) {
      setEncodeStatus("⚠️ Selecciona al menos una variable de entrada y una de salida.");
      return;
    }
    if (!runId || !experimentDir) {
      setEncodeStatus("❌ Error: Configuración incompleta.");
      return;
    }

    setEncodeCardEncodeInProgress(true);
    setEncodeStatus("🔄 Codificando archivo...");

    const formData = new FormData();
    formData.append("file", csvFile);
    formData.append("experiment_dir", experimentDir);
    formData.append("input_features", inputFeatures.join(","));
    formData.append("target_variables", targetVariable); // Single target now
    formData.append("run_id", runId);
    formData.append("encode_target_ohe", encodeTargetOHE ? "True" : "False");
    formData.append("encode_target_label", encodeTargetLabel ? "True" : "False");

    // Time Series Parameters
    formData.append("lag_periods", lagPeriods.toString());
    formData.append("lag_nan_handling", lagNanHandling);
    formData.append("date_column", dateColumn);

    try {
      const response = await axios.post("ts/encode-csv/", formData);

      if (response.data.status === "Archivo CSV codificado correctamente.") {
        setTrainFilePath(response.data.processed_train_path);
        setEncodeStatus("✅ Archivo codificado exitosamente.");
        // Marcar el paso de codificación como completado
        markStepDone("encodeDone");
      } else {
        setEncodeStatus(`❌ Error: ${response.data.error || "Error desconocido"}`);
      }
    } catch (error) {
      console.error("Error al codificar el archivo:", error);
      setEncodeStatus("❌ Error al procesar el archivo.");
    } finally {
      setEncodeCardEncodeInProgress(false);
    }
  };

  // Se deshabilita la acción si:
  // - Se está cargando (ya sea análisis o codificación).
  // - No existe el experimento o el run_id.
  // - No se han completado los pasos previos (limpieza y EDA).
  // - O ya se ejecutó la codificación.
  // - Hay advertencias de validación (nuevo)
  const isDisabled =
    encodeCardAnalyzeInProgress ||
    encodeCardEncodeInProgress ||
    !experimentDir ||
    !runId ||
    !flow.cleaningDone ||
    !flow.edaDone ||
    flow.encodeDone ||
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
        <Typography
          variant="h5"
          sx={{ mb: 2, color: "#004d40", textAlign: "center", fontWeight: "bold" }}
        >
          3. Codificación de Datos
        </Typography>

        <Typography
          variant="body1"
          sx={{ mb: 2, color: "#004d40", textAlign: "center" }}
        >
          Selecciona un CSV, define las variables y elige el método de codificación para targets.
        </Typography>

        {/* Bloqueamos el botón de seleccionar archivo mientras se esté procesando */}
        <Button
          variant="outlined"
          component="label"
          sx={{ mb: 2 }}
          disabled={encodeCardAnalyzeInProgress || encodeCardEncodeInProgress}
        >
          Seleccionar Archivo
          <input type="file" accept=".csv" hidden onChange={handleFileChange} />
        </Button>

        {/* Progress Bar - shown when analyzing */}
        {encodeCardAnalyzeInProgress && (
          <Box sx={{ mt: 2, mb: 2 }}>
            <ProgressBar
              message="Analizando columnas del CSV..."
              variant={progressVariant}
              showPercentage={false}
            />
          </Box>
        )}

        <Button
          variant="contained"
          onClick={analyzeCsv}
          disabled={!csvFile || encodeCardAnalyzeInProgress || encodeCardEncodeInProgress}
          sx={{
            backgroundColor: "#00796b",
            "&:hover": { backgroundColor: "#004d40" },
            width: "100%",
            py: 1.5,
            fontSize: "1.1rem",
            fontWeight: "bold",
            mt: encodeCardAnalyzeInProgress ? 0 : 2,
          }}
        >
          {encodeCardAnalyzeInProgress ? "Cargando..." : "Previsualizar Columnas"}
        </Button>

        {/* Current State Indicator - shown when no columns loaded */}
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

            <Box sx={{ mt: 3 }}>
              <Typography variant="body1" sx={{ fontWeight: "bold", color: "#004d40", mb: 2 }}>
                Configuración de Series Temporales:
              </Typography>

              <TextField
                label="Períodos de Lag"
                type="number"
                value={lagPeriods}
                onChange={(e) => setLagPeriods(Math.max(0, parseInt(e.target.value) || 0))}
                slotProps={{ htmlInput: { min: 0, step: 1 } }}
                fullWidth
                sx={{ mb: 2 }}
                helperText="Número de períodos pasados a incluir como características (0 = sin lag)"
              />

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Manejo de valores NaN</InputLabel>
                <Select
                  value={lagNanHandling}
                  label="Manejo de valores NaN"
                  onChange={(e) => setLagNanHandling(e.target.value)}
                >
                  <MenuItem value="leave_as_is">Dejar como está</MenuItem>
                  <MenuItem value="drop">Eliminar filas con NaN</MenuItem>
                  <MenuItem value="forward_fill">Rellenar hacia adelante</MenuItem>
                </Select>
              </FormControl>

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Columna de Fecha (opcional)</InputLabel>
                <Select
                  value={dateColumn}
                  label="Columna de Fecha (opcional)"
                  onChange={(e) => setDateColumn(e.target.value)}
                >
                  <MenuItem value="">Automático (primera columna)</MenuItem>
                  {columns.map((col) => (
                    <MenuItem key={col} value={col}>
                      {col}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>

            {/* Progress Bar - shown when encoding */}
            {encodeCardEncodeInProgress && (
              <Box sx={{ mt: 2, mb: 2 }}>
                <ProgressBar
                  message="Codificando dataset con parámetros de series temporales..."
                  variant={progressVariant}
                  showPercentage={false}
                />
              </Box>
            )}

            <Button
              variant="contained"
              onClick={encodeCsv}
              disabled={isDisabled}
              sx={{
                mt: encodeCardEncodeInProgress ? 0 : 3,
                backgroundColor: "#00796b",
                "&:hover": { backgroundColor: "#004d40" },
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                width: "100%",
                py: 1.5,
                fontSize: "1.1rem",
                fontWeight: "bold",
              }}
            >
              {encodeCardEncodeInProgress
                ? "Procesando..."
                : flow.encodeDone
                ? "Codificación Completada"
                : "Codificar Dataset"}
            </Button>

            {/* Current State Indicator - shown after columns loaded */}
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

export default TSEncodeCard;
