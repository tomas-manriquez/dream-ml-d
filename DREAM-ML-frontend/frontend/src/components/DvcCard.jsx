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

const DvcCard = () => {
  const { 
    experimentDir, 
    flow, 
    markStepDone, 
    dvcInitializationInProgress, 
    setDvcInitializationInProgress 
  } = useContext(AppContext);
  
  const [dvcStatus, setDvcStatus] = useState({
    message: "No inicializado",
    isError: false,
    details: null
  });
  const [showAlert, setShowAlert] = useState(false);
  const [lastDirectory, setLastDirectory] = useState(null);

  // Resetear estado cuando cambia el directorio
  useEffect(() => {
    if (experimentDir && experimentDir !== lastDirectory) {
      setDvcStatus({
        message: "No inicializado",
        isError: false,
        details: null
      });
      setLastDirectory(experimentDir);
    }
  }, [experimentDir, lastDirectory]);

  const initializeDvc = async () => {
    if (!experimentDir) {
      setDvcStatus({
        message: "Por favor, crea un experimento primero",
        isError: true,
        details: null
      });
      setShowAlert(true);
      return;
    }
    
    if (dvcInitializationInProgress) return;

    setDvcInitializationInProgress(true);
    setDvcStatus({
      message: "Inicializando DVC y Git...",
      isError: false,
      details: null
    });
    setShowAlert(true);
    
    try {
      const response = await axios.post("/init-dvc/", { 
        experiment_dir: experimentDir 
      });
      
      const statusMessage = response.data?.status || "DVC y Git inicializados correctamente";
      
      setDvcStatus({
        message: statusMessage,
        isError: false,
        details: response.data
      });
      
      markStepDone("initDvcGit");
    } catch (error) {
      let errorMessage = "Error al inicializar DVC y Git";
      let errorDetails = null;
      
      if (error.response?.data?.status) {
        errorMessage = error.response.data.status;
        errorDetails = error.response.data;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      setDvcStatus({
        message: errorMessage,
        isError: true,
        details: errorDetails
      });
      console.error("Error al inicializar DVC y Git:", error);
    } finally {
      setDvcInitializationInProgress(false);
    }
  };

  const isDisabled = dvcInitializationInProgress || !experimentDir || flow.initDvcGit;
  const buttonText = flow.initDvcGit 
    ? "DVC Iniciado" 
    : dvcInitializationInProgress 
      ? "Inicializando..." 
      : "Inicializar DVC";

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
      <CardContent
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <Typography variant="h5" sx={{ mb: 1, color: "#004d40" }}>
          Inicializar DVC y Git
        </Typography>
        
        <Typography
          variant="body2"
          sx={{
            mb: 2,
            color: "#004d40",
            textAlign: "center",
            maxWidth: 500,
          }}
        >
          Configura DVC y Git en el directorio del experimento para el versionado de datos y modelos.
          <br />
          <strong>NOTA:</strong> Requiere que primero hayas creado un experimento.
        </Typography>

        {/* Alertas para mostrar estado */}
        <Collapse in={showAlert} sx={{ width: '100%', mb: 2 }}>
          <Alert
            severity={dvcStatus.isError ? "error" : dvcInitializationInProgress ? "info" : "success"}
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
          >
            {dvcStatus.message}
            {dvcStatus.details && (
              <div style={{ marginTop: 8, fontSize: '0.8rem' }}>
                <pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>
                  {JSON.stringify(dvcStatus.details, null, 2)}
                </pre>
              </div>
            )}
          </Alert>
        </Collapse>

        <Box sx={{ 
          width: '100%', 
          mb: 2,
          p: 1,
          bgcolor: 'background.default',
          borderRadius: 1,
          overflow: 'hidden',
          textOverflow: 'ellipsis'
        }}>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
            Directorio del Experimento:
          </Typography>
          <Typography variant="body2" sx={{ 
            whiteSpace: 'nowrap', 
            overflow: 'hidden',
            textOverflow: 'ellipsis'
          }}>
            {experimentDir || "No configurado"}
          </Typography>
        </Box>

        <Button
          variant="contained"
          onClick={initializeDvc}
          disabled={isDisabled}
          sx={{
            backgroundColor: flow.initDvcGit 
              ? "#4caf50" 
              : dvcInitializationInProgress 
                ? "#cfd8dc" 
                : "#00796b",
            "&:hover": { 
              backgroundColor: flow.initDvcGit 
                ? "#388e3c" 
                : "#004d40" 
            },
            mb: 2,
            px: 3,
            py: 1.5,
            fontSize: "1.1rem",
            minWidth: 200,
          }}
          startIcon={dvcInitializationInProgress && <CircularProgress size={24} color="inherit" />}
        >
          {buttonText}
        </Button>
        
        {dvcStatus.details?.experiment_dir && (
          <div style={{ width: '100%', marginTop: 2 }}>
            <Typography variant="body2" color="textSecondary" sx={{ fontWeight: 'bold' }}>
              Detalles de Configuración:
            </Typography>
            <Typography variant="body2" color="textSecondary" sx={{ wordBreak: "break-all" }}>
              Directorio: {dvcStatus.details.experiment_dir}
            </Typography>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default DvcCard;