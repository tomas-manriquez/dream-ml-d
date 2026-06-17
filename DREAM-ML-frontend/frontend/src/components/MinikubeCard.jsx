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


import React, { useState } from "react";
import axios from "../utils/axiosConfig";

const MinikubeCard = () => {
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  const handleStartMinikube = async () => {
    setLoading(true);
    setStatus("Iniciando Minikube...");
    try {
      const response = await axios.post("/start-minikube/");
      setStatus(response.data.status + "\n" + response.data.output); // Mostrar salida detallada
    } catch (error) {
      if (error.response) {
        setStatus("Error al iniciar Minikube: " + error.response.data.error);
      } else {
        setStatus("Error inesperado al iniciar Minikube.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleStopMinikube = async () => {
    setLoading(true);
    setStatus("Deteniendo Minikube...");
    try {
      const response = await axios.post("/stop-minikube/");
      setStatus(response.data.status);
    } catch (error) {
      setStatus("Error al detener Minikube.");
    } finally {
      setLoading(false);
    }
  };

  const handleGetStatus = async () => {
    setLoading(true);
    setStatus("Obteniendo estado de Minikube...");
    try {
      const response = await axios.get("/minikube-status/");
      setStatus(JSON.stringify(response.data.minikube_status, null, 2));
    } catch (error) {
      if (error.response) {
        setStatus("Error al obtener estado de Minikube: " + error.response.data.error);
      } else {
        setStatus("Error inesperado al obtener estado de Minikube.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.card}>
      <h2 style={styles.title}>Controlar Minikube</h2>
      <button onClick={handleStartMinikube} disabled={loading} style={styles.button}>
        Iniciar Minikube
      </button>
      <button onClick={handleStopMinikube} disabled={loading} style={styles.button}>
        Detener Minikube
      </button>
      <button onClick={handleGetStatus} disabled={loading} style={styles.button}>
        Verificar Estado
      </button>
      <pre style={styles.status}>{status}</pre>
    </div>
  );
};

const styles = {
  card: {
    padding: "20px",
    border: "1px solid #00796b", 
    borderRadius: "8px",
    maxWidth: "400px",
    margin: "20px auto",
    textAlign: "center",
    backgroundColor: "#e0f7fa", 
    boxShadow: "0 4px 8px rgba(0, 121, 107, 0.3)", // Sombra suave verde
  },
  title: {
    color: "#004d40", // Verde oscuro
    marginBottom: "15px",
    fontSize: "1.5rem",
  },
  button: {
    padding: "10px 20px",
    margin: "10px",
    backgroundColor: "#00796b", 
    color: "#fff",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
    fontSize: "1rem",
    transition: "background-color 0.3s",
  },
  buttonHover: {
    backgroundColor: "#004d40", 
  },
  status: {
    marginTop: "15px",
    fontSize: "0.9rem",
    color: "#004d40",
    textAlign: "left",
    whiteSpace: "pre-wrap",
    backgroundColor: "#ffffff", 
    padding: "10px",
    borderRadius: "4px",
    border: "1px solid #00796b",
  },
};

export default MinikubeCard;
