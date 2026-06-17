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


// CreateExperimentCard.jsx
import React, { useState, useContext } from "react";
import { 
  Card,
  CardContent,
  Typography,
  Button,
  CircularProgress,
  Alert,
  Collapse,
  IconButton
} from "@mui/material";
import CloseIcon from '@mui/icons-material/Close';
import axios from "../utils/axiosConfig";
import { AppContext } from "../AppContext";

const CreateExperimentCard = () => {
  const { 
    setExperimentDir, 
    setMlflowExperimentId, 
    markStepDone, 
    resetFlow,
    flow,
    experimentCreationInProgress,
    setExperimentCreationInProgress,
    setExperimentDetails
  } = useContext(AppContext);
  
  const [experimentStatus, setExperimentStatus] = useState({
    message: "No iniciado",
    isError: false,
    details: null
  });
  const [showAlert, setShowAlert] = useState(false);

  const createExperiment = async () => {
    if (experimentCreationInProgress) return;

    setExperimentCreationInProgress(true);
    setExperimentStatus({
      message: "Creando experimento...",
      isError: false,
      details: null
    });
    
    try {
      resetFlow();
      
      const response = await axios.post("/create-experiment/");
      console.log("Respuesta del backend:", response.data);

      const { details, status } = response.data;

      if (!details || !details.experiment_dir || !details.mlflow_experiment_id) {
        throw new Error("Datos incompletos recibidos del backend");
      }

      // Actualizar el contexto global con todos los detalles
      setExperimentDir(details.experiment_dir);
      setMlflowExperimentId(details.mlflow_experiment_id);
      setExperimentDetails(details);

      setExperimentStatus({
        message: status || "Experimento creado correctamente",
        isError: false,
        details
      });
      
      markStepDone("experimentCreated");
      setShowAlert(true);
    } catch (error) {
      console.error("Error al crear el experimento:", error);
      
      // Manejar diferentes tipos de errores del backend
      let errorMessage = "Error al crear el experimento";
      let errorDetails = null;
      
      if (error.response?.data) {
        errorMessage = error.response.data.status || errorMessage;
        errorDetails = error.response.data.details || error.response.data;
      } else {
        errorMessage = error.message || errorMessage;
      }

      setExperimentStatus({
        message: errorMessage,
        isError: true,
        details: errorDetails
      });
      setShowAlert(true);
    } finally {
      setExperimentCreationInProgress(false);
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
          Crear Experimento
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
          Crea un nuevo espacio de trabajo para tu proyecto con seguimiento completo de MLflow.
          <br />
          <strong>NOTA:</strong> Cada nuevo experimento reinicia el entorno de MLflow.
        </Typography>

        {/* Alertas para mostrar estado */}
        <Collapse in={showAlert} sx={{ width: '100%', mb: 2 }}>
          <Alert
            severity={experimentStatus.isError ? "error" : "success"}
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
            {experimentStatus.message}
            {experimentStatus.details && (
              <div style={{ marginTop: 8, fontSize: '0.8rem' }}>
                <pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>
                  {JSON.stringify(experimentStatus.details, null, 2)}
                </pre>
              </div>
            )}
          </Alert>
        </Collapse>

        <Button
          variant="contained"
          onClick={createExperiment}
          disabled={experimentCreationInProgress}
          sx={{
            backgroundColor: experimentCreationInProgress ? "#cfd8dc" : "#00796b",
            "&:hover": { backgroundColor: "#004d40" },
            mb: 2,
            px: 3,
            py: 1.5,
            fontSize: "1.1rem",
          }}
        >
          {experimentCreationInProgress ? (
            <CircularProgress size={24} color="inherit" />
          ) : flow.experimentCreated ? (
            "Reiniciar Experimento"
          ) : (
            "Crear Nuevo Experimento"
          )}
        </Button>
        
        {experimentStatus.details?.experiment_dir && (
          <div style={{ width: '100%', marginTop: 2 }}>
            <Typography variant="body2" color="textSecondary" sx={{ fontWeight: 'bold' }}>
              Detalles del Experimento:
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Nombre: {experimentStatus.details.experiment_name}
            </Typography>
            <Typography variant="body2" color="textSecondary" sx={{ wordBreak: "break-all" }}>
              Directorio: {experimentStatus.details.experiment_dir}
            </Typography>
            <Typography variant="body2" color="textSecondary" sx={{ wordBreak: "break-all" }}>
              ID MLflow: {experimentStatus.details.mlflow_experiment_id}
            </Typography>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default CreateExperimentCard;