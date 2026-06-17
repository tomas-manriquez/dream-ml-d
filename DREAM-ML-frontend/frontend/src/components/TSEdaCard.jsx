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
import {
  Card,
  CardContent,
  Typography,
  Button,
  Box,
} from "@mui/material";
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Autorenew as AutorenewIcon,
} from "@mui/icons-material";
import axios from "../utils/axiosConfig";
import { AppContext } from "../AppContext";
import ProgressBar from "./ProgressBar";

const TSEdaCard = () => {
  const { experimentDir, runId, flow, markStepDone, edaGenerationInProgress, setEdaGenerationInProgress } = useContext(AppContext);
  const [status, setStatus] = useState("");

  // You can change this to 'modernMinimal' or 'enhancedMaterial' to try different designs
  const progressVariant = 'tealHarmony';

  const datasetType = "eda";

  // Helper function to determine current state
  const getCurrentState = () => {
    if (!experimentDir || !runId) {
      return {
        type: 'warning',
        message: 'Requiere experimento creado y run_id',
        icon: <InfoIcon fontSize="small" />
      };
    }
    if (!flow.cleaningDone) {
      return {
        type: 'warning',
        message: 'Esperando completar limpieza de datos',
        icon: <InfoIcon fontSize="small" />
      };
    }
    if (flow.edaDone) {
      return {
        type: 'success',
        message: 'EDA completado exitosamente',
        icon: <CheckCircleIcon fontSize="small" />
      };
    }
    if (edaGenerationInProgress) {
      return {
        type: 'processing',
        message: 'Generando reporte YData Profiling ',
        icon: <AutorenewIcon fontSize="small" sx={{ animation: 'spin 1s linear infinite', '@keyframes spin': { '0%': { transform: 'rotate(0deg)' }, '100%': { transform: 'rotate(360deg)' } } }} />
      };
    }
    if (status.includes("❌") || status.includes("Error")) {
      return {
        type: 'error',
        message: status.replace("❌", "").trim(),
        icon: <ErrorIcon fontSize="small" />
      };
    }
    return {
      type: 'info',
      message: 'Listo para generar reportes EDA',
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

  const handleGenerateEda = async () => {
    if (!experimentDir || !runId) {
      setStatus("Por favor, asegúrate de tener un experimento creado y un run_id.");
      return;
    }
    // Evitar múltiples ejecuciones si ya está en curso
    if (edaGenerationInProgress) return;

    setEdaGenerationInProgress(true);
    setStatus("Generando reportes EDA...");
    try {
      const response = await axios.post("ts/generate-eda/", {
        experiment_dir: experimentDir,
        run_id: runId,
        dataset_type: datasetType,
      });

      if (response.data.success) {
        setStatus(
          "✅ EDA generado correctamente. Puedes ver los reportes en la carpeta 'eda_reports' de tu experimento. También han sido guardados en artifacts en MLflow."
        );
        // Marcar el paso EDA como completado
        markStepDone("edaDone");
      } else {
        setStatus(`❌ Error al generar los reportes: ${response.data.error}`);
      }
    } catch (error) {
      console.error("Error al generar EDA:", error);
      setStatus(
        `❌ Error al generar EDA: ${error.response?.data?.error || error.message}`
      );
    } finally {
      setEdaGenerationInProgress(false);
    }
  };

  // Se deshabilita si:
  // - El proceso EDA está en curso.
  // - No hay experimento o run_id.
  // - La limpieza no se completó o ya se ejecutó el EDA.
  const isDisabled = edaGenerationInProgress || !experimentDir || !runId || !flow.cleaningDone || flow.edaDone;

  const currentState = getCurrentState();

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
          2. Generación de Reportes EDA
        </Typography>

        <Typography
          variant="body1"
          sx={{ mb: 2, color: "#004d40", textAlign: "center" }}
        >
          Analiza automáticamente tus datos con <strong>YData Profiling</strong>.
          Se generará el EDA del archivo <strong>processed_eda.csv</strong>.
        </Typography>

        {/* Progress Bar - shown when processing */}
        {edaGenerationInProgress && (
          <Box sx={{ mt: 2, mb: 2 }}>
            <ProgressBar
              message="Generando reportes YData Profiling "
              variant={progressVariant}
              showPercentage={false}
            />
          </Box>
        )}

        <Button
          variant="contained"
          onClick={handleGenerateEda}
          disabled={isDisabled}
          sx={{
            backgroundColor: "#00796b",
            "&:hover": { backgroundColor: "#004d40" },
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: "100%",
            py: 1.5,
            fontSize: "1.1rem",
            fontWeight: "bold",
            mt: edaGenerationInProgress ? 0 : 2,
          }}
        >
          {flow.edaDone ? "EDA Generado" : "Generar Reportes"}
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
              color: getStateColor(currentState.type),
              backgroundColor: `${getStateColor(currentState.type)}15`,
              px: 2,
              py: 1,
              borderRadius: 2,
              border: `1px solid ${getStateColor(currentState.type)}40`,
            }}
          >
            {currentState.icon}
            <Typography
              variant="body2"
              sx={{
                fontWeight: 500,
                color: getStateColor(currentState.type),
              }}
            >
              {currentState.message}
            </Typography>
          </Box>
        </Box>

        {/* Additional status message for success cases */}
        {status && status.includes("correctamente") && (
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
            Los reportes están disponibles en &apos;eda_reports&apos; y en MLflow artifacts
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default TSEdaCard;
