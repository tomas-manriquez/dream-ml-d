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


import React, { useEffect, useState, useContext } from "react";
import axios from "../utils/axiosConfig";
import { AppContext } from "../AppContext";

const PipelineConfigCard = () => {
  const { directoryPath } = useContext(AppContext); // Obtener el valor desde el contexto
  const [pipelineConfig, setPipelineConfig] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchPipelineConfig = async () => {
    if (!directoryPath) {
      setError("Por favor, ingresa la ruta del directorio base.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const response = await axios.get("/get-pipeline-config/", {
        params: { directory_path: directoryPath },
      });
      setPipelineConfig(response.data.steps || []);
    } catch (error) {
      setError("Error al cargar la configuración del pipeline.");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  // Llamar a fetchPipelineConfig cada vez que cambie el directoryPath
  useEffect(() => {
    fetchPipelineConfig();
  }, [directoryPath]);

  return (
    <div style={styles.card}>
      <h2 style={styles.title}>Configuración del Pipeline</h2>
      <button style={styles.button} onClick={fetchPipelineConfig} disabled={loading}>
        {loading ? "Cargando..." : "Actualizar Configuración"}
      </button>
      {loading ? (
        <p style={styles.status}>Cargando configuraciones...</p>
      ) : error ? (
        <p style={styles.error}>{error}</p>
      ) : (
        <ul style={styles.list}>
          {pipelineConfig.length > 0 ? (
            pipelineConfig.map((step, index) => (
              <li key={index} style={styles.step}>
                <strong>Paso {index + 1}: </strong> {step.step}
                <br />
                <strong>Archivo crudo:</strong> {step.raw_file_path || "N/A"}
                <br />
                <strong>Archivo limpio:</strong> {step.cleaned_file_path || "N/A"}
              </li>
            ))
          ) : (
            <p style={styles.status}>No se encontraron configuraciones.</p>
          )}
        </ul>
      )}
    </div>
  );
};

const styles = {
  card: {
    backgroundColor: "#e0f7fa", // Fondo azul claro
    borderRadius: "12px",
    padding: "30px",
    textAlign: "center",
    boxShadow: "0 4px 12px rgba(0, 121, 107, 0.3)", // Sombra verde suave
    margin: "20px auto",
    border: "1px solid #00796b", 
    width: "90%",
    maxWidth: "600px",
  },
  title: {
    fontSize: "1.8rem",
    color: "#004d40", // Verde oscuro
    marginBottom: "20px",
    fontWeight: "bold",
  },
  button: {
    padding: "10px 20px",
    margin: "10px 0",
    backgroundColor: "#00796b", // Verde oscuro
    color: "#ffffff", // Texto blanco
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "1rem",
    transition: "background-color 0.3s ease",
    fontWeight: "bold",
  },
  error: {
    color: "#ff0000",
    fontSize: "1rem",
    margin: "10px 0",
  },
  status: {
    fontSize: "1rem",
    color: "#004d40", // Verde oscuro
    margin: "10px 0",
  },
  list: {
    textAlign: "left",
    marginTop: "20px",
    padding: "0",
    listStyleType: "none",
  },
  step: {
    marginBottom: "15px",
    fontSize: "1rem",
    color: "#004d40", // Verde oscuro
  },
};

export default PipelineConfigCard;
