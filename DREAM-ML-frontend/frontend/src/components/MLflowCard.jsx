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
  CircularProgress, 
  Alert, 
  Collapse, 
  IconButton, 
  Box 
} from "@mui/material";
import CloseIcon from '@mui/icons-material/Close';
import axios from "../utils/axiosConfig";
import { AppContext } from "../AppContext";

const MLflowCard = () => {
  // Directorio base fijo desde el contenedor
  const baseDir = import.meta.env.VITE_EXPERIMENTS_DIR || "/app/experimentos";

  const [mlflowStatus, setMlflowStatus] = useState({
    message: "No inicializado",
    isError: false,
    details: null
  });
  const [showAlert, setShowAlert] = useState(false);

  // Usamos el flag global para controlar el inicio de MLflow
  const { mlflowStartingInProgress, setMlflowStartingInProgress } = useContext(AppContext);

  // Función para iniciar MLflow
  const startMLflow = async () => {
    if (mlflowStartingInProgress) return; // Evitar múltiples ejecuciones

    setShowAlert(true);
    setMlflowStatus({
      message: "Iniciando servidor MLflow...",
      isError: false,
      details: null
    });
    setMlflowStartingInProgress(true);
    
    try {
      // Se envía el valor baseDir, que es fijo
      const response = await axios.post("/start-mlflow/", { directory_path: baseDir });
      
      setMlflowStatus({
        message: "Servidor MLflow en ejecución",
        isError: false,
        details: response.data
      });

      // Abrir MLflow en una nueva pestaña
      window.open("http://127.0.0.1:5000/", "_blank");
    } catch (error) {
      const errorMessage = error.response?.data?.status || "Error al iniciar MLflow";
      
      setMlflowStatus({
        message: errorMessage,
        isError: true,
        details: error.response?.data || null
      });
    } finally {
      setMlflowStartingInProgress(false);
    }
  };

  const buttonText = mlflowStatus.details 
    ? "Reiniciar MLflow" 
    : mlflowStartingInProgress 
      ? "Iniciando..." 
      : "Iniciar MLflow";

  return (
    <Card
      sx={{
        backgroundColor: "#e0f7fa",
        borderRadius: 2,
        border: "1px solid #00796b",
        boxShadow: "0 4px 12px rgba(0, 121, 107, 0.3)",
        p: 3,
        maxWidth: 600,
        width: "90%",
        m: "20px auto",
      }}
    >
      <CardContent sx={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
        <Typography variant="h5" sx={{ mb: 1, color: "#004d40" }}>
          MLflow Server
        </Typography>
        
        <Typography
          variant="body2"
          sx={{ 
            mb: 2, 
            color: "#004d40", 
            textAlign: "center", 
            maxWidth: 500 
          }}
        >
          Inicia el servidor MLflow para visualizar y comparar los experimentos registrados.
        </Typography>

        {/* Estado actual */}
        <Typography variant="h6" sx={{ mb: 2, color: "#004d40", fontWeight: 'bold' }}>
          Estado: {mlflowStatus.message}
        </Typography>

        {/* Alertas para mostrar estado */}
        <Collapse in={showAlert} sx={{ width: '100%', mb: 2 }}>
          <Alert
            severity={mlflowStatus.isError ? "error" : mlflowStartingInProgress ? "info" : "success"}
            action={
              <IconButton
                aria-label="close"
                color="inherit"
                size="small"
                onClick={() => setShowAlert(false)}
              >
                <CloseIcon fontSize="inherit" />
              </IconButton>
            }
            sx={{ width: '100%' }}
          >
            {mlflowStatus.message}
          </Alert>
        </Collapse>

        {/* Directorio base */}
        <Box sx={{ 
          width: '100%', 
          mb: 2,
          p: 1.5,
          backgroundColor: '#ffffff',
          borderRadius: 1,
          border: '1px solid #b2dfdb'
        }}>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
            Directorio Base:
          </Typography>
          <Typography 
            variant="body2" 
            sx={{ 
              color: "#004d40", 
              wordBreak: "break-all",
              fontWeight: 'medium'
            }}
          >
            {baseDir}
          </Typography>
        </Box>

        <Button
          variant="contained"
          onClick={startMLflow}
          disabled={mlflowStartingInProgress}
          sx={{
            backgroundColor: mlflowStatus.details 
              ? "#4caf50" 
              : mlflowStartingInProgress 
                ? "#cfd8dc" 
                : "#00796b",
            "&:hover": { 
              backgroundColor: mlflowStatus.details 
                ? "#388e3c" 
                : "#004d40" 
            },
            fontSize: "1.1rem",
            px: 3,
            py: 1.5,
            mb: 2,
            minWidth: 200
          }}
          startIcon={mlflowStartingInProgress && <CircularProgress size={24} color="inherit" />}
        >
          {buttonText}
        </Button>

        {mlflowStatus.details && (
          <Box 
            sx={{ 
              width: '100%', 
              mt: 2, 
              p: 2, 
              backgroundColor: '#ffffff', 
              borderRadius: 1,
              border: '1px solid #b2dfdb'
            }}
          >
            <Typography variant="subtitle2" sx={{ mb: 1, color: "#00796b", fontWeight: 'bold' }}>
              Detalles del servidor:
            </Typography>
            
            <Box sx={{ mb: 1 }}>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                Backend URI:
              </Typography>
              <Typography variant="body2" sx={{ wordBreak: "break-all", color: "#004d40" }}>
                {mlflowStatus.details.backend_store_uri}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 1 }}>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                Artifact Store:
              </Typography>
              <Typography variant="body2" sx={{ wordBreak: "break-all", color: "#004d40" }}>
                {mlflowStatus.details.artifact_store}
              </Typography>
            </Box>
            
            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                Logs:
              </Typography>
              <Typography variant="body2" sx={{ wordBreak: "break-all", color: "#004d40" }}>
                stdout: {mlflowStatus.details.log_stdout}
                <br />
                stderr: {mlflowStatus.details.log_stderr}
              </Typography>
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default MLflowCard;