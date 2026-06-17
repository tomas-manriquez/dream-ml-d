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
import { Card, CardContent, Typography, Button } from "@mui/material";
import axios from "../utils/axiosConfig";
import { AppContext } from "../AppContext"; // Contexto global

const ExperimentSummaryCard = () => {
  const { experimentDir, summaryInProgress, setSummaryInProgress } = useContext(AppContext);
  const [summaryStatus, setSummaryStatus] = useState("Resumen no generado");

  // Función para solicitar la generación del resumen y descargar el PDF resultante
  const generateSummary = async () => {
    if (!experimentDir) {
      setSummaryStatus("Por favor, asegúrate de tener un experimento creado primero");
      return;
    }
    // Evita múltiples ejecuciones si el proceso ya está en curso
    if (summaryInProgress) return;
    
    setSummaryInProgress(true);
    try {
      // Se llama al endpoint get_experiment_summary pasando el directorio del experimento
      const response = await axios.get(
        `/get-experiment-summary?directory_path=${encodeURIComponent(experimentDir)}`,
        { responseType: "blob" }
      );
      // Se crea un objeto Blob y se genera una URL para descargar el PDF
      const url = window.URL.createObjectURL(new Blob([response.data], { type: "application/pdf" }));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "experiment_summary.pdf");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setSummaryStatus("Resumen descargado exitosamente");
    } catch (error) {
      console.error(error);
      setSummaryStatus("Error al obtener el resumen");
    } finally {
      setSummaryInProgress(false);
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
        {/* Título de la tarjeta */}
        <Typography variant="h5" sx={{ mb: 1, color: "#004d40" }}>
          Obtener Resumen del Experimento
        </Typography>
        {/* Breve descripción de la funcionalidad */}
        <Typography variant="body2" sx={{ mb: 2, color: "#004d40", textAlign: "center", maxWidth: 500 }}>
          Esta tarjeta genera un informe en PDF que resume los parámetros, resultados y análisis de
          cada paso del experimento a partir del pipeline_config.json.
        </Typography>
        {/* Mostrar el estado actual */}
        <Typography variant="h5" sx={{ mb: 2, color: "#004d40" }}>
          Estado: {summaryStatus}
        </Typography>
        <Typography variant="body1" sx={{ mb: 2, color: "#004d40" }}>
          Directorio del Experimento: {experimentDir || "No configurado"}
        </Typography>
        <Button
          variant="contained"
          onClick={generateSummary}
          disabled={summaryInProgress || !experimentDir}
          sx={{
            backgroundColor: "#00796b",
            "&:hover": { backgroundColor: "#004d40" },
            fontSize: "1.1rem",
            px: 3,
            py: 1.5,
          }}
        >
          {summaryInProgress ? "Generando..." : "Generar Resumen"}
        </Button>
      </CardContent>
    </Card>
  );
};

export default ExperimentSummaryCard;
