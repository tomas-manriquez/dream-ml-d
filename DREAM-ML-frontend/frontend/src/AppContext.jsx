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


// AppContext.jsx
import React, { createContext, useState } from "react";

export const AppContext = createContext();

const initialFlow = {
  experimentCreated: false,    // Se completó "Crear Experimento"
  initDvcGit: false,           // Se inicializó DVC y Git
  configRemoteDvc: false,      // Se configuró el almacenamiento remoto DVC
  cleaningDone: false,         // Se completó la limpieza
  edaDone: false,              // Se generó el EDA
  encodeDone: false,           // Se realizó la codificación
  trainDone: false,            // Se completó el entrenamiento
  edaManualDone: false,        // Se ejecutó el EDA manual (opcional)
  summaryGenerated: false,     // Se generó el resumen (opcional)
  mlflowStarted: false,        // Se inició MLflow (opcional)

  csvUploadCleaningInProgress: false,
  uploadStatus: "",
  

};

export const AppProvider = ({ children }) => {
  // Variables ya existentes:
  const [experimentDetails, setExperimentDetails] = useState(null);
  const [directoryPath, setDirectoryPath] = useState(""); 
  const [experimentDir, setExperimentDir] = useState(null); 
  const [mlflowExperimentId, setMlflowExperimentId] = useState(null);
  const [cleanedFilePath, setCleanedFilePath] = useState(null);
  const [runId, setRunId] = useState(null);
  const [flow, setFlow] = useState(initialFlow);
  const [experimentCreationInProgress, setExperimentCreationInProgress] = useState(false);
  const [dvcInitializationInProgress, setDvcInitializationInProgress] = useState(false);
  const [dvcRemoteConfigInProgress, setDvcRemoteConfigInProgress] = useState(false);
  const [mlflowStartingInProgress, setMlflowStartingInProgress] = useState(false);
  const [csvAnalyzeInProgress, setCsvAnalyzeInProgress] = useState(false);
  const [csvUploadCleaningInProgress, setCsvUploadCleaningInProgress] = useState(false);
  const [edaGenerationInProgress, setEdaGenerationInProgress] = useState(false);
  const [jupyterStartingInProgress, setJupyterStartingInProgress] = useState(false);
  const [encodeCardAnalyzeInProgress, setEncodeCardAnalyzeInProgress] = useState(false);
  const [encodeCardEncodeInProgress, setEncodeCardEncodeInProgress] = useState(false);
  const [trainInProgress, setTrainInProgress] = useState(false);
  const [summaryInProgress, setSummaryInProgress] = useState(false);
  const [pipelineExecutionInProgress, setPipelineExecutionInProgress] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(initialFlow.uploadStatus);
  const [encodeStatus, setEncodeStatus] = useState("");
  const [trainStatus, setTrainStatus] = useState("");
  const [pipelineStatusMessage, setPipelineStatusMessage] = useState("");
  const markStepDone = (stepName) => {
    setFlow((prev) => ({ ...prev, [stepName]: true }));
  };

  // Función para reiniciar el flujo a su estado inicial:
  const resetFlow = () => {
    setFlow(initialFlow);
  };

  const setBaseDirectory = (path) => {
    setDirectoryPath(path);
  };

  return (
    <AppContext.Provider
      value={{
        directoryPath,
        setDirectoryPath,
        experimentDir,
        setExperimentDir,
        mlflowExperimentId,
        setMlflowExperimentId,
        cleanedFilePath,
        setCleanedFilePath,
        runId,
        setRunId,
        setBaseDirectory,
        flow,
        markStepDone,
        resetFlow,
        //flag global para creación del experimento
        experimentCreationInProgress,
        setExperimentCreationInProgress,
        dvcInitializationInProgress,
        setDvcInitializationInProgress,
        dvcRemoteConfigInProgress,
        setDvcRemoteConfigInProgress,
        mlflowStartingInProgress,
        setMlflowStartingInProgress,
        csvAnalyzeInProgress,
        setCsvAnalyzeInProgress,
        csvUploadCleaningInProgress,
        setCsvUploadCleaningInProgress,
        edaGenerationInProgress,
        setEdaGenerationInProgress,
        jupyterStartingInProgress,
        setJupyterStartingInProgress,
        encodeCardAnalyzeInProgress,
        setEncodeCardAnalyzeInProgress,
        encodeCardEncodeInProgress,
        setEncodeCardEncodeInProgress,
        trainInProgress,
        setTrainInProgress,
        summaryInProgress,
        setSummaryInProgress,
        pipelineExecutionInProgress,
        setPipelineExecutionInProgress,
        pipelineRunning,
        setPipelineRunning,
        uploadStatus,
        setUploadStatus,
        encodeStatus,
        setEncodeStatus,
        trainStatus,
        setTrainStatus,
        pipelineStatusMessage,
        setPipelineStatusMessage,
        experimentDetails,
        setExperimentDetails
      }}
    >
      {children}
    </AppContext.Provider>
  );
};
