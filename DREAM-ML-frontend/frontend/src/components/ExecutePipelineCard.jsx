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


import React, { useState, useEffect, useRef, useContext, useMemo } from "react";
import axios from "../utils/axiosConfig";
import {
  Box,
  Card,
  Typography,
  Button,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  useMediaQuery,
  useTheme,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Switch,
  FormControlLabel,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

// React Flow imports
import ReactFlow, {
  ReactFlowProvider,
  Background,
  Controls,
} from "reactflow";
import "reactflow/dist/style.css";
import { AppContext } from "../AppContext";

const ExecutePipelineCard = () => {
  const baseDir = "/app/experimentos";
  const [pipelineFile, setPipelineFile] = useState(null);
  const [pipelineConfig, setPipelineConfig] = useState(null); // JSON parseado
  const [executionLog, setExecutionLog] = useState([]); // Ej. [{ step: "data_cleaning", status: "PENDING", details: null }, ...]
  const [selectedStep, setSelectedStep] = useState(null); // Para mostrar detalles dinámicos
  const [openGridDialog, setOpenGridDialog] = useState(false);
  const [isTimeSeries, setIsTimeSeries] = useState(false); // false = classification, true = time series forecast

  const theme = useTheme();
  const isLargeScreen = useMediaQuery(theme.breakpoints.up("md"));

  const {
    pipelineExecutionInProgress,
    setPipelineExecutionInProgress,
    pipelineRunning,
    setPipelineRunning,
    pipelineStatusMessage,
    setPipelineStatusMessage,
  } = useContext(AppContext);

  // Referencia para el WebSocket
  const wsRef = useRef(null);
  // Referencia para la instancia de React Flow
  const reactFlowInstanceRef = useRef(null);

  // Función de mapeo para convertir nombres de paso a claves de respuesta
  const mapStepToResponseKey = (stepName) => {
    if (!stepName) return null;
    const name = stepName.toLowerCase();
    if (name.includes("cleaning")) return "data_cleaning";
    if (name.includes("eda")) return "eda";
    if (name.includes("encoding")) return "data_encoding";
    if (name.startsWith("train")) return "training";
    return null;
  };

  // Conexión WebSocket: se activa cuando pipelineRunning es true.
  useEffect(() => {
    if (pipelineRunning) {
      // Construct WebSocket URL properly
      const wsBaseUrl = import.meta.env.VITE_WS_URL || "ws://localhost:8000";
      const wsUrl = `${wsBaseUrl}/ws/progreso/`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("WebSocket conectado");
      };

      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          if (data.step && data.status) {
            console.log("Mensaje WS recibido:", data);
            setExecutionLog((prevLogs) =>
              prevLogs.map((logEntry) => {
                const mappedStep = mapStepToResponseKey(logEntry.step);
                if (
                  mappedStep === data.step ||
                  (mappedStep === "training" &&
                    (data.step === "train" || data.step === "training"))
                ) {
                  return { ...logEntry, status: data.status };
                }
                return logEntry;
              })
            );
          }
        } catch (error) {
          console.error("Error parseando mensaje WS:", error);
        }
      };

      ws.onerror = (e) => {
        console.error("WebSocket error", e);
      };

      ws.onclose = () => {
        console.log("WebSocket desconectado");
      };

      return () => {
        if (wsRef.current) {
          wsRef.current.close();
          wsRef.current = null;
        }
      };
    }
  }, [pipelineRunning]);

  // Inicializa executionLog con todos los pasos en "PENDING"
  const initializeExecutionLog = () => {
    if (pipelineConfig) {
      const groupedSteps = getGroupedSteps();
      const initialLog = groupedSteps.map((step) => ({
        step: step.step ? step.step : "Sin nombre",
        status: "PENDING",
        details: null,
      }));
      setExecutionLog(initialLog);
    }
  };

  // Manejo de carga de archivo
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setPipelineFile(file);
    setPipelineConfig(null);
    setExecutionLog([]);
    setPipelineStatusMessage("");
    setSelectedStep(null);
  };

  // Parseo del JSON del pipeline_config
  const handleLoadPipeline = () => {
    if (!pipelineFile) {
      setPipelineStatusMessage("Por favor, selecciona un archivo pipeline_config.json.");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const jsonContent = JSON.parse(e.target.result);
        setPipelineConfig(jsonContent);
        setPipelineStatusMessage("¡pipeline_config cargado con éxito!");
        const groupedSteps = getGroupedSteps();
        const initialLog = groupedSteps.map((step) => ({
          step: step.step ? step.step : "Sin nombre",
          status: "PENDING",
          details: null,
        }));
        setExecutionLog(initialLog);
        const trainingStep = jsonContent.steps.find(
          (step) =>
            step.step &&
            step.step.toLowerCase().startsWith("train") &&
            step.grid_search &&
            step.grid_search.use_grid_search === true
        );
        if (trainingStep) {
          setOpenGridDialog(true);
        }
      } catch (error) {
        setPipelineStatusMessage("Error al parsear el JSON del pipeline_config.");
      }
    };
    reader.readAsText(pipelineFile);
  };

  // Agrupa los pasos; si hay varios de entrenamiento, conserva solo el último.
  const getGroupedSteps = () => {
    if (!pipelineConfig) return [];
    const steps = pipelineConfig.steps;
    const trainingSteps = steps.filter(
      (step) => step.step && step.step.toLowerCase().startsWith("train")
    );
    const nonTrainingSteps = steps.filter(
      (step) => !step.step || !step.step.toLowerCase().startsWith("train")
    );
    if (trainingSteps.length > 0) {
      return [...nonTrainingSteps, trainingSteps[trainingSteps.length - 1]];
    }
    return steps;
  };

  // Genera nodos y edges para React Flow con espaciado fijo
  const getFlowElements = () => {
    const groupedSteps = getGroupedSteps();
    const spacing = 180;
    const marginLeft = 20;
    const containerHeight = 500;
    const nodes = groupedSteps.map((step, index) => {
      const stepName = step.step ? step.step : "Sin nombre";
      const log = executionLog.find((item) => item.step === stepName);
      let nodeStyle = {
        padding: 10,
        border: "1px solid #ccc",
        borderRadius: 8,
        background: "#fff",
        width: 160,
        cursor: "pointer",
        transition: "box-shadow 0.3s",
      };
      if (log) {
        if (log.status === "OK") {
          nodeStyle = { ...nodeStyle, background: "#C8E6C9", border: "2px solid #388E3C" };
        } else if (log.status === "ERROR") {
          nodeStyle = { ...nodeStyle, background: "#FFCDD2", border: "2px solid #D32F2F" };
        }
      }
      const x = marginLeft + index * spacing;
      const y = containerHeight / 2 - 50;
      return {
        id: `${index}`,
        data: {
          label: (
            <Box
              onClick={() => setSelectedStep(step)}
              sx={{
                textAlign: "center",
                "&:hover": { boxShadow: "0px 4px 10px rgba(0,0,0,0.2)" },
              }}
            >
              <Typography variant="subtitle1" sx={{ fontWeight: "bold" }}>
                {stepName}
              </Typography>
              {log && (
                <Typography variant="caption" sx={{ color: "#555" }}>
                  {log.status}
                </Typography>
              )}
            </Box>
          ),
        },
        position: { x, y },
        style: nodeStyle,
      };
    });

    const edges = [];
    for (let i = 1; i < nodes.length; i++) {
      edges.push({
        id: `e${i - 1}-${i}`,
        source: `${i - 1}`,
        target: `${i}`,
        animated: executionLog[i - 1]?.status === "OK",
        style: { stroke: "#004d40", strokeWidth: 2 },
      });
    }
    return { nodes, edges };
  };

  // Memorizar los elementos del flujo para evitar recálculos
  const flowElements = useMemo(() => getFlowElements(), [pipelineConfig, executionLog]);

  // Efecto para reajustar el diagrama al cargar el pipeline
  useEffect(() => {
    if (reactFlowInstanceRef.current) {
      reactFlowInstanceRef.current.fitView();
    }
  }, [pipelineConfig]);

  // Función para ejecutar el pipeline (llamada al backend)
  const handleRunPipeline = async () => {
    if (!pipelineConfig) {
      setPipelineStatusMessage("Por favor, carga primero el archivo pipeline_config.");
      return;
    }
    if (pipelineExecutionInProgress) return;

    setPipelineExecutionInProgress(true);
    setPipelineStatusMessage("Iniciando ejecución del pipeline...");
    initializeExecutionLog();
    setSelectedStep(null);
    setPipelineRunning(true);
  
    try {
      // Select endpoint based on experiment type
      const endpoint = isTimeSeries ? "/ts/run-pipeline" : "/run-pipeline/";
      const response = await axios.post(endpoint, {
        base_dir: baseDir,
        pipeline_config: pipelineConfig,
      });
  
      if (response.data.status) {
        const groupedSteps = getGroupedSteps();
        setExecutionLog((prevLogs) =>
          prevLogs.map((logEntry) => {
            const responseKey = mapStepToResponseKey(logEntry.step);
            if (response.data[responseKey]) {
              return { ...logEntry, status: "OK", details: response.data[responseKey] };
            }
            return logEntry;
          })
        );
        setPipelineStatusMessage("Pipeline ejecutado correctamente.");
      } else {
        setPipelineStatusMessage("Pipeline no se ejecutó correctamente: " + (response.data.error || ""));
      }
    } catch (error) {
      console.error(error);
      setPipelineStatusMessage("Error al ejecutar el pipeline.");
    } finally {
      setPipelineExecutionInProgress(false);
      setPipelineRunning(false);
    }
  };

  // Calcula el progreso en porcentaje
  const computeProgress = () => {
    if (!pipelineConfig) return 0;
    const totalSteps = getGroupedSteps().length;
    const completedSteps = executionLog.filter((step) => step.status === "OK").length;
    return Math.round((completedSteps / totalSteps) * 100);
  };

  const renderProgressBar = () => {
    const progress = computeProgress();
    return (
      <Box sx={{ width: "100%", my: 2 }}>
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{ height: 10, borderRadius: 5 }}
        />
        <Typography
          variant="body2"
          color="textSecondary"
          align="center"
          sx={{ mt: 1, fontWeight: 500 }}
        >
          {progress}% completado
        </Typography>
      </Box>
    );
  };

  const renderStepDetailsPanel = () => {
    if (!pipelineConfig) return null;
    const groupedSteps = getGroupedSteps();
    return (
      <Box sx={{ mt: 2, pr: 1 }}>
        {groupedSteps.map((step, idx) => {
          const stepName = step.step ? step.step : "Sin nombre";
          return (
            <Accordion
              key={idx}
              defaultExpanded={false}
              TransitionProps={{ timeout: 300 }}
              sx={{ mb: 1, background: "#f5f5f5" }}
            >
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1" sx={{ fontWeight: "bold" }}>
                  {stepName}
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <List dense>
                  {Object.entries(step).map(([key, value]) => (
                    <ListItem key={key}>
                      <ListItemText
                        primary={key}
                        secondary={
                          typeof value === "object"
                            ? JSON.stringify(value, null, 2)
                            : value?.toString()
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </AccordionDetails>
            </Accordion>
          );
        })}
      </Box>
    );
  };

  const renderPipelineDiagram = () => {
    if (!pipelineConfig) return null;
    return (
      <Box
        sx={{
          height: 500,
          width: "100%",
          border: "1px solid #ccc",
          borderRadius: 2,
          mb: 2,
          boxShadow: 2,
          overflow: "hidden",
          // Se define la variable CSS --scale-factor para React Flow
          "--scale-factor": 1,
        }}
      >
        <ReactFlowProvider>
          <ReactFlow
            onInit={(instance) => {
              reactFlowInstanceRef.current = instance;
            }}
            nodes={flowElements.nodes}
            edges={flowElements.edges}
            fitView
          >
            <Background color="#aaa" gap={16} />
            <Controls />
          </ReactFlow>
        </ReactFlowProvider>
      </Box>
    );
  };

  // Funciones para el diálogo de grid search
  const handleApplyGridSearch = () => {
    setOpenGridDialog(false);
  };

  const handleUseExistingParams = () => {
    if (pipelineConfig && pipelineConfig.steps) {
      const updatedSteps = pipelineConfig.steps.map((step) => {
        if (
          step.step &&
          step.step.toLowerCase().startsWith("train") &&
          step.grid_search &&
          step.grid_search.use_grid_search === true
        ) {
          return { ...step, grid_search: null };
        }
        return step;
      });
      setPipelineConfig({ ...pipelineConfig, steps: updatedSteps });
    }
    setOpenGridDialog(false);
  };

  return (
    <Card
      sx={{
        bgcolor: "#e0f7fa",
        p: 3,
        maxWidth: 600,
        width: "90%",
        mx: "auto",
        my: 4,
        boxShadow: 4,
        borderRadius: 3,
      }}
    >
      <Typography
        variant="h4"
        sx={{ color: "#004d40", mb: 3, textAlign: "center" }}
      >
        Ejecución Automática del Pipeline
      </Typography>

      {/* Sección de inputs */}
      <Box sx={{ mb: 2 }}>
        <Typography
          variant="subtitle1"
          sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}
        >
          Ruta base (base_dir):
        </Typography>
        <input
          type="text"
          value={baseDir}
          readOnly
          style={{
            width: "100%",
            padding: "12px",
            borderRadius: "8px",
            border: "1px solid #ccc",
            fontSize: "1rem",
            backgroundColor: "#f0f0f0",
          }}
          placeholder="/app/experimentos"
        />
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography
          variant="subtitle1"
          sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}
        >
          Archivo pipeline_config.json:
        </Typography>
        <Box sx={{ display: "flex", alignItems: "center" }}>
          <input
            type="file"
            accept=".json"
            onChange={handleFileChange}
            style={{ marginRight: "10px" }}
          />
          <Button variant="contained" color="secondary" onClick={handleLoadPipeline}>
            Cargar Pipeline
          </Button>
        </Box>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography
          variant="subtitle1"
          sx={{ fontWeight: "bold", color: "#004d40", mb: 1 }}
        >
          Tipo de experimento:
        </Typography>
        <FormControlLabel
          control={
            <Switch
              checked={isTimeSeries}
              onChange={(e) => setIsTimeSeries(e.target.checked)}
              color="primary"
            />
          }
          label={isTimeSeries ? "Pronóstico de Series Temporales" : "Clasificación Tabular"}
        />
      </Box>

      {pipelineConfig && (
        <Box>
          {renderPipelineDiagram()}
          {renderProgressBar()}
          {renderStepDetailsPanel()}
        </Box>
      )}

      <Box
        sx={{
          width: "100%",
          mt: 3,
          display: "flex",
          justifyContent: "center",
        }}
      >
        <Button
          variant="contained"
          color="primary"
          onClick={handleRunPipeline}
          disabled={pipelineExecutionInProgress}
          fullWidth
          sx={{
            fontSize: "1rem",
            py: 1.5,
            whiteSpace: "normal",
          }}
        >
          {pipelineExecutionInProgress ? "Ejecutando Pipeline..." : "Ejecutar Pipeline"}
        </Button>
      </Box>

      {pipelineStatusMessage && (
        <Typography
          variant="body1"
          sx={{
            mt: 3,
            color: "#004d40",
            fontWeight: "bold",
            textAlign: "center",
          }}
        >
          {pipelineStatusMessage}
        </Typography>
      )}

      <Dialog open={openGridDialog} onClose={() => setOpenGridDialog(false)}>
        <DialogTitle>Confirmar configuración de entrenamiento</DialogTitle>
        <DialogContent>
          <Typography variant="body1">
            Se detectó que en el pipeline se optimizaron los hiperparámetros mediante grid
            search. ¿Deseas volver a aplicar grid search o usar los hiperparámetros ya definidos en el archivo?
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleApplyGridSearch} color="primary">
            Volver a aplicar grid search
          </Button>
          <Button onClick={handleUseExistingParams} color="primary">
            Usar hiperparámetros actuales
          </Button>
        </DialogActions>
      </Dialog>
    </Card>
  );
};

export default ExecutePipelineCard;
