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

const EdaManualCard = () => {
  const { experimentDir, runId, jupyterStartingInProgress, setJupyterStartingInProgress } = useContext(AppContext);
  const [status, setStatus] = useState("");
  const [notebookUrl, setNotebookUrl] = useState("");

  // Helper function to determine current state
  const getCurrentState = () => {
    // Prerequisites check
    if (!experimentDir) {
      return {
        type: 'warning',
        message: 'Requiere experimento creado',
        icon: <InfoIcon fontSize="small" />
      };
    }

    // Processing state
    if (jupyterStartingInProgress) {
      return {
        type: 'processing',
        message: 'Iniciando Jupyter Notebook...',
        icon: <AutorenewIcon fontSize="small" sx={{ animation: 'spin 1s linear infinite', '@keyframes spin': { '0%': { transform: 'rotate(0deg)' }, '100%': { transform: 'rotate(360deg)' } } }} />
      };
    }

    // Success state
    if (notebookUrl && !status.includes("Error")) {
      return {
        type: 'success',
        message: 'Jupyter Notebook iniciado exitosamente',
        icon: <CheckCircleIcon fontSize="small" />
      };
    }

    // Error state
    if (status.includes("Error") || status.includes("error")) {
      return {
        type: 'error',
        message: status.replace("❌", "").replace("Error:", "").trim(),
        icon: <ErrorIcon fontSize="small" />
      };
    }

    // Idle/Ready state
    return {
      type: 'info',
      message: 'Listo para iniciar EDA manual',
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

  const handleStartJupyter = async () => {
    if (!experimentDir) {
      setStatus("Error: Selecciona un directorio de experimento válido");
      return;
    }
    // Evitar múltiples ejecuciones si ya está en curso
    if (jupyterStartingInProgress) return;

    setJupyterStartingInProgress(true);
    setStatus("");

    console.log("experimentDir: ", experimentDir);
    console.log("run_id: ", runId);
    try {
      const response = await axios.post("/start-jupyter/", {
        experiment_dir: experimentDir,
        run_id: runId, // Enviar el run_id al backend
      });
    console.log("recieved response from backend...");
    console.log(response.data);

      if (response.data.success) {
        setNotebookUrl(response.data.notebook_url);
        setStatus("");
      } else {
        setStatus(`Error: ${response.data.error}`);
      }
    } catch (error) {
      console.error("Error al iniciar Jupyter Notebook:", error);
      setStatus(`Error: ${error.response?.data?.error || error.message}`);
    } finally {
      setJupyterStartingInProgress(false);
    }
  };

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
          2.5. EDA Manual (Opcional)
        </Typography>

        <Typography
          variant="body1"
          sx={{ mb: 2, color: "#004d40", textAlign: "center" }}
        >
          Abre un <strong>Jupyter Notebook</strong> preconfigurado con guías para análisis exploratorio manual.
          Los resultados serán guardados en tu experimento y en MLflow.
        </Typography>

        {/* Progress Bar - shown when processing */}
        {jupyterStartingInProgress && (
          <Box sx={{ mt: 2, mb: 2 }}>
            <ProgressBar
              message="Iniciando Jupyter Notebook..."
              variant="tealHarmony"
              showPercentage={false}
            />
          </Box>
        )}

        <Button
          variant="contained"
          onClick={handleStartJupyter}
          disabled={jupyterStartingInProgress || !experimentDir}
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
            mt: jupyterStartingInProgress ? 0 : 2,
          }}
        >
          Abrir EDA Manual
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

        {/* Notebook URL link */}
        {notebookUrl && (
          <Typography
            component="a"
            href={notebookUrl}
            target="_blank"
            rel="noopener noreferrer"
            sx={{
              display: "block",
              mt: 2,
              fontSize: "1rem",
              color: "#00796b",
              fontWeight: "bold",
              textDecoration: "underline",
              textAlign: "center",
            }}
          >
            📓 Abrir Notebook Manual
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default EdaManualCard;
