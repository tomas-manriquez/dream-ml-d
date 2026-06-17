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

const DvcRemoteCard = () => {
  const { 
    experimentDir, 
    flow, 
    markStepDone, 
    dvcRemoteConfigInProgress, 
    setDvcRemoteConfigInProgress 
  } = useContext(AppContext);
  
  const [remoteStatus, setRemoteStatus] = useState({
    message: "No configurado",
    isError: false,
    details: null
  });
  const [showAlert, setShowAlert] = useState(false);

  // Función para configurar el almacenamiento remoto de DVC
  const configureDvcRemote = async () => {
    if (!experimentDir) {
      setRemoteStatus({
        message: "Por favor, crea un experimento primero",
        isError: true,
        details: null
      });
      setShowAlert(true);
      return;
    }
    
    if (dvcRemoteConfigInProgress) return;

    setDvcRemoteConfigInProgress(true);
    setRemoteStatus({
      message: "Configurando almacenamiento remoto DVC...",
      isError: false,
      details: null
    });
    setShowAlert(true);
    
    try {
      const response = await axios.post("/configure-dvc-remote/", {
        experiment_dir: experimentDir,
      });
      
      const statusMessage = response.data?.status || "Almacenamiento remoto de DVC configurado correctamente";
      
      setRemoteStatus({
        message: statusMessage,
        isError: false,
        details: response.data
      });
      
      markStepDone("configRemoteDvc");
    } catch (error) {
      let errorMessage = "Error al configurar el almacenamiento remoto de DVC";
      let errorDetails = null;
      
      if (error.response?.data?.status) {
        errorMessage = error.response.data.status;
        errorDetails = error.response.data;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      setRemoteStatus({
        message: errorMessage,
        isError: true,
        details: errorDetails
      });
    } finally {
      setDvcRemoteConfigInProgress(false);
    }
  };

  // Determinar estado del botón
  const isDisabled = dvcRemoteConfigInProgress || !experimentDir || !flow.initDvcGit || flow.configRemoteDvc;
  const buttonText = flow.configRemoteDvc 
    ? "Remoto Configurado" 
    : dvcRemoteConfigInProgress 
      ? "Configurando..." 
      : "Configurar Almacenamiento Remoto";

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
          Configurar Almacenamiento Remoto de DVC
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
          Configura un almacenamiento compartido para versionar datos y modelos con DVC.
          <br />
          <strong>NOTA:</strong> Requiere que primero hayas inicializado DVC y Git.
        </Typography>

        {/* Alertas para mostrar estado */}
        <Collapse in={showAlert} sx={{ width: '100%', mb: 2 }}>
          <Alert
            severity={remoteStatus.isError ? "error" : dvcRemoteConfigInProgress ? "info" : "success"}
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
            {remoteStatus.message}
            {remoteStatus.details && (
              <Box sx={{ mt: 1, fontSize: '0.8rem' }}>
                <pre style={{ 
                  whiteSpace: 'pre-wrap', 
                  margin: 0,
                  fontFamily: 'monospace'
                }}>
                  {JSON.stringify(remoteStatus.details, null, 2)}
                </pre>
              </Box>
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
          onClick={configureDvcRemote}
          disabled={isDisabled}
          sx={{
            backgroundColor: flow.configRemoteDvc 
              ? "#4caf50" 
              : dvcRemoteConfigInProgress 
                ? "#cfd8dc" 
                : "#00796b",
            "&:hover": { 
              backgroundColor: flow.configRemoteDvc 
                ? "#388e3c" 
                : "#004d40" 
            },
            mb: 2,
            px: 3,
            py: 1.5,
            fontSize: "1.1rem",
            minWidth: 250,
          }}
          startIcon={dvcRemoteConfigInProgress && <CircularProgress size={24} color="inherit" />}
        >
          {buttonText}
        </Button>
        
        {remoteStatus.details?.remote_path && (
          <Box sx={{ width: '100%', mt: 1 }}>
            <Typography variant="body2" color="textSecondary" sx={{ fontWeight: 'bold' }}>
              Ruta del Remoto:
            </Typography>
            <Typography variant="body2" color="textSecondary" sx={{ wordBreak: "break-all" }}>
              {remoteStatus.details.remote_path}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default DvcRemoteCard;