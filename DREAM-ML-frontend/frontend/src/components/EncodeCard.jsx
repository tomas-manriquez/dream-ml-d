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

const EncodeCard = () => {
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

  // Info modal states
  const [showTargetInfo, setShowTargetInfo] = useState(false);
  const [showFeatureInfo, setShowFeatureInfo] = useState(false);

  // Validation warnings
  const [validationWarnings, setValidationWarnings] = useState([]);

  const handleFileChange = (event) => {
    setCsvFile(event.target.files[0]);
    setEncodeStatus("📂 Archivo seleccionado.");
    setColumns([]);
    setTrainFilePath("");
    setEncodeTargetOHE(false);
    setEncodeTargetLabel(false);
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

  // Helper function to determine current state
  const getCurrentState = () => {
    // Error state (highest priority)
    if (encodeStatus && (encodeStatus.includes("Error") || encodeStatus.includes("error") || encodeStatus.includes("❌"))) {
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
        message: 'Archivo codificado exitosamente',
        icon: <CheckCircleIcon fontSize="small" />
      };
    }

    // Processing states
    if (encodeCardEncodeInProgress) {
      return {
        type: 'processing',
        message: 'Codificando archivo...',
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
        message: 'Requiere experimento creado, run_id, limpieza y EDA completados',
        icon: <InfoIcon fontSize="small" />
      };
    }

    // Ready to encode (columns loaded and configured)
    if (columns.length > 0 && targetVariable && inputFeatures.length > 0) {
      return {
        type: 'info',
        message: 'Listo para codificar dataset',
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
      message: 'Listo para seleccionar archivo CSV',
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

    try {
      const response = await axios.post("/encode-csv/", formData);

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

        <Button
          variant="contained"
          onClick={analyzeCsv}
          disabled={!csvFile || encodeCardAnalyzeInProgress || encodeCardEncodeInProgress}
          sx={{
            mb: 2,
            backgroundColor: "#00796b",
            "&:hover": { backgroundColor: "#004d40" },
            width: "100%",
          }}
        >
          {encodeCardAnalyzeInProgress ? "Cargando..." : "Previsualizar Columnas"}
        </Button>

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
              <Typography variant="body1" sx={{ fontWeight: "bold", color: "#004d40" }}>
                Opciones de Codificación para Targets (opcional):
              </Typography>
              <FormControlLabel
                control={<Checkbox checked={encodeTargetOHE} onChange={handleOheChange} />}
                label="One-Hot Encoding"
              />
              <FormControlLabel
                control={<Checkbox checked={encodeTargetLabel} onChange={handleLabelChange} />}
                label="Label Encoding"
              />
            </Box>

            <Button
              variant="contained"
              onClick={encodeCsv}
              disabled={isDisabled}
              sx={{
                mt: 3,
                backgroundColor: "#00796b",
                "&:hover": { backgroundColor: "#004d40" },
                width: "100%",
                py: 1.5,
                fontSize: "1.1rem",
                fontWeight: "bold",
              }}
            >
              {encodeCardEncodeInProgress
                ? "Codificando..."
                : flow.encodeDone
                ? "Codificación Completada"
                : "Codificar Dataset"}
            </Button>

            {/* Progress Bar - shown when processing */}
            {(encodeCardAnalyzeInProgress || encodeCardEncodeInProgress) && (
              <Box sx={{ mt: 2, mb: 2 }}>
                <ProgressBar
                  message={
                    encodeCardAnalyzeInProgress
                      ? "Analizando columnas del CSV..."
                      : "Codificando archivo..."
                  }
                  variant="tealHarmony"
                  showPercentage={false}
                />
              </Box>
            )}

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

            {/* Additional info for success case */}
            {trainFilePath && flow.encodeDone && (
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
                Archivo procesado: {trainFilePath}
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

export default EncodeCard;
