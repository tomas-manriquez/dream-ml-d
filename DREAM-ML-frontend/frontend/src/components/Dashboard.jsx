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
import {
  AppBar,
  Tabs,
  Tab,
  Box,
  Typography,
  Grid,
  Toolbar,
  useTheme,
  Button,
} from "@mui/material";
import {
  LinkedIn as LinkedInIcon,
  GitHub as GitHubIcon,
  Email as EmailIcon,
  Phone as PhoneIcon,
} from "@mui/icons-material";

import MLflowCard from "./MLflowCard";
import DvcCard from "./DvcCard";
import DvcRemoteCard from "./DvcRemoteCard";
import TSUploadCsvCard from "./TSUploadCsvCard";
import UploadCsvCard from "./UploadCsvCard";
import TSEncodeCard from "./TSEncodeCard";
import EncodeCard from "./EncodeCard";
import PipelineConfigCard from "./PipelineConfigCard";
import ExecutePipelineCard from "./ExecutePipelineCard";
import EdaCard from "./EdaCard";
import TSEdaCard from "./TSEdaCard";
import EdaManualCard from "./EdaManualCard";
import TSTrainCard from "./TSTrainCard";
import TrainCard from "./TrainCard"; // Tarjeta de entrenamiento
import CreateExperimentCard from "./Create"; // Tarjeta para crear experimentos
import ExperimentSummaryCard from "./Report"; // Tarjeta para generar y descargar el resumen en PDF
import TSExperimentSummaryCard from "./TSExperimentSummaryCard";

import { AppProvider } from "../AppContext";

const Dashboard = () => {
  const [currentTab, setCurrentTab] = useState(0);
  const theme = useTheme();

  const handleChangeTab = (event, newValue) => {
    setCurrentTab(newValue);
  };

  // TabPanel que oculta el contenido sin desmontarlo
  const TabPanel = ({ children, value, index }) => {
    return (
      <Box
        role="tabpanel"
        sx={{
          display: value !== index ? "none" : "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "flex-start",
          width: "100%",
          p: 2,
          backgroundColor: "transparent",
          borderRadius: 2,
          boxShadow: 1,
          mt: 2,
        }}
      >
        {children}
      </Box>
    );
  };

  return (
    <AppProvider>
      <Box
        sx={{
          height: "100vh",
          width: "100vw",
          background: "linear-gradient(135deg, #e0f7fa 0%, #f1f8e9 100%)",
          display: "flex",
          flexDirection: "column",
          overflowX: "hidden",
          boxSizing: "border-box",
        }}
      >
        {/* AppBar con logo, título y Tabs en una misma fila */}
        <AppBar position="static" sx={{ backgroundColor: "#00796b" }}>
  <Toolbar
    sx={{
      padding: theme.spacing(0.5),
      minHeight: "auto",
      display: "flex",
      flexDirection: "row",
      alignItems: "center",
    }}
  >
    <Box sx={{ display: "flex", alignItems: "center" }}>
      <img
        src="/logo.png"
        alt="DREAM ML Logo"
        style={{
          height: "80px", // tamaño logo
          marginRight: theme.spacing(2),
        }}
      />
      <Typography variant="h6" sx={{ fontWeight: "bold", mr: 2 }}>
        Dashboard DREAM ML
      </Typography>
    </Box>
    <Box
      sx={{
        flexGrow: 1,
        display: "flex",
        justifyContent: "flex-start", // Alinea a la izquierda
        ml: theme.spacing(30), // margen izquierdo para separar del título
      }}
    >
      <Tabs
        value={currentTab}
        onChange={handleChangeTab}
        variant="scrollable"
        textColor="inherit"
        indicatorColor="secondary"
        sx={{
          "& .MuiTab-root": {
            fontWeight: "bold",
            fontSize: "1rem",
            color: "#000000",
          },
          "& .Mui-selected": {
            color: "#ffffff",
          },
          minHeight: "auto",
        }}
      >
        <Tab label="Inicio" />
        <Tab label="Inicializar Sistemas" />
        <Tab label="Experimento - Clasificación" />
        <Tab label="Experimento - Series de Tiempo"/>
        <Tab label="Ejecución del Pipeline" />
      </Tabs>
    </Box>
  </Toolbar>
</AppBar>



        {/* Área de contenido scrollable */}
        <Box
          sx={{
            height: "calc(100vh - 100px - 56px)", 
            overflowY: "auto",
            overflowX: "hidden",
            px: 2,
            boxSizing: "border-box",
          }}
        >
          {/* Panel 0: INICIO */}
          <TabPanel value={currentTab} index={0}>
            <Typography
              variant="h6"
              align="center"
              sx={{ color: "#004d40", mb: 2 }}
            >
              Bienvenido a DREAM ML
            </Typography>
            <Typography
              variant="body1"
              align="justify"
              sx={{
                color: "#004d40",
                maxWidth: "800px",
                lineHeight: "1.6",
                mb: 3,
              }}
            >
              DREAM ML es un sistema enfocado en problemas de clasificación que
              centraliza todas las etapas clave desde la ingesta y limpieza de
              datos, hasta el entrenamiento y validación de tus modelos. Está
              diseñado para entornos académicos, facilitando la reproducibilidad y
              documentación mediante:
            </Typography>
            <Typography
              component="ul"
              variant="body1"
              align="justify"
              sx={{
                color: "#004d40",
                maxWidth: "800px",
                lineHeight: "1.6",
                pl: 4,
                mb: 3,
              }}
            >
              <li>
                <strong>Versionado de datos y modelos con DVC:</strong> garantiza
                la trazabilidad de cada conjunto de datos y modelo entrenado.
              </li>
              <li>
                <strong>Registro de experimentos con MLflow:</strong> almacena
                parámetros, métricas y artefactos de cada ejecución (<em>run</em>),
                facilitando la comparación de resultados.
              </li>
              <li>
                <strong>Automatización o flujo manual:</strong> elige entre ejecutar
                cada paso de forma independiente o reproducir todo el proceso a través
                de un pipeline secuencial.
              </li>
              <li>
                <strong>EDA (Análisis Exploratorio) automático y manual:</strong>
                genera reportes con herramientas como Ydata Profiling y Sweetviz, o
                profundiza con tu propio Jupyter Notebook.
              </li>
              <li>
                <strong>Codificación de datos para clasificación:</strong> soporta
                LabelEncoder y OneHotEncoder, adaptándose a modelos como Regresión
                Logística, MLP o XGBoost.
              </li>
              <li>
                <strong>Reportes de validación y resultados:</strong> obtén un PDF
                consolidado con métricas (Accuracy, F1, ROC-AUC, etc.) y detalles del
                proceso.
              </li>
            </Typography>
            <Typography
              variant="body1"
              align="justify"
              sx={{ color: "#004d40", maxWidth: "800px", lineHeight: "1.6" }}
            >
              Con DREAM ML puedes organizar tus experimentos de clasificación de
              manera transparente. Explora las pestañas superiores para inicializar
              los sistemas (Git, DVC y MLflow), cargar tus datos, generar análisis
              exploratorios, codificar los atributos y entrenar tus modelos. Asegura
              la reproducibilidad y mantén un registro detallado de cada paso en tus
              experimentos de clasificación.
            </Typography>
            <Button
              variant="contained"
              color="primary"
              href="/manual.pdf"
              download
              sx={{
                mt: 2,
                backgroundColor: "#00796b",
                "&:hover": { backgroundColor: "#004d40" },
              }}
            >
              Descargar Manual de Usuario
            </Button>
          </TabPanel>

          {/* Panel 1: INICIALIZAR SISTEMAS */}
          <TabPanel value={currentTab} index={1}>
            <Box sx={{ maxWidth: "1200px", width: "100%", mx: "auto", px: 2 }}>
              <Grid container spacing={4} justifyContent="center">
                <Grid item xs={12} sm={6} md={4}>
                  <CreateExperimentCard />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <DvcCard />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <DvcRemoteCard />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <MLflowCard />
                </Grid>
              </Grid>
            </Box>
          </TabPanel>

          {/* Panel 2: FLUJO DEL EXPERIMENTO  - CLASIFICACION*/}
          <TabPanel value={currentTab} index={2}>
            <Box sx={{ maxWidth: "1200px", width: "100%", mx: "auto", px: 2 }}>
              <Grid container spacing={4} justifyContent="center">
                <Grid item xs={12} sm={6} md={4}>
                  <UploadCsvCard />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <EdaCard />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <EdaManualCard />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <EncodeCard />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <TrainCard />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <ExperimentSummaryCard />
                </Grid>
              </Grid>
            </Box>
          </TabPanel>

          {/* Panel 3: FLUJO DEL EXPERIMENTO - SERIE DE TIEMPO*/}
          <TabPanel value={currentTab} index={3}>
            <Box sx={{ maxWidth: "1200px", width: "100%", mx: "auto", px: 2 }}>
              <Grid container spacing={4} justifyContent="center">
                <TSUploadCsvCard/>
              </Grid>
              <Grid item xs={12} sm={6} md={4}>
                  <TSEdaCard />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <TSEncodeCard />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <TSTrainCard />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <TSExperimentSummaryCard />
                </Grid>
            </Box>
          </TabPanel>

          {/* Panel 4: EJECUCIÓN DEL PIPELINE */}
          <TabPanel value={currentTab} index={4}>
            <Box sx={{ maxWidth: "1200px", width: "100%", mx: "auto", px: 2 }}>
              <Grid container spacing={4} justifyContent="center">
                <Grid item xs={12} sm={6} md={6}>
                  <ExecutePipelineCard />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <MLflowCard />
                </Grid>
              </Grid>
            </Box>
          </TabPanel>
        </Box>

        {/* Footer reorganizado en 2 columnas */}
        <Box
          component="footer"
          sx={{
            backgroundColor: "#00796b",
            color: "#ffffff",
            textAlign: "center",
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "center",
            gap: 1,
            boxSizing: "border-box",
            px: 2,
            py: 1,
          }}
        >
          <Typography variant="body2" sx={{ width: "100%", mb: 1 }}>
            © {new Date().getFullYear()} DREAM ML
          </Typography>
          {/* Columna 1: Contacto de Leonardo */}
          <Box
            sx={{
              display: "flex",
              flexBasis: "45%",
              justifyContent: "center",
              alignItems: "center",
              gap: 1,
            }}
          >
            <EmailIcon fontSize="small" />
            <Typography variant="body2">
              <strong>Correo:</strong> leonardo.espinoza.o@usach.cl
            </Typography>
                        <PhoneIcon fontSize="small" />
            <Typography variant="body2">
              <strong>Celular:</strong> +56 9 7371 1546
            </Typography>
          </Box>

          <Box
            sx={{
              display: "flex",
              flexBasis: "45%",
              justifyContent: "center",
              alignItems: "center",
              gap: 1,
            }}
          >
            <EmailIcon fontSize="small" />
            <Typography variant="body2">
              <strong>Correo:</strong> tomas.manriquez@usach.cl 
            </Typography>
                        <PhoneIcon fontSize="small" />
            <Typography variant="body2">
              <strong>Celular:</strong> +56 9 6776 6586
            </Typography>
          </Box>

          
          {/* Columna 2: Contacto de Tomas */}
          
          <Box
            sx={{
              display: "flex",
              flexBasis: "45%",
              justifyContent: "center",
              alignItems: "center",
              gap: 1,
            }}
          >
            <LinkedInIcon fontSize="small" />
            <a
              href="https://www.linkedin.com/in/leonardo-espinoza-ortiz-311229263/"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "#ffffff", textDecoration: "none" }}
            >
              linkedin.com/in/leonardo-espinoza-ortiz-311229263/
            </a>
                        <GitHubIcon fontSize="small" />
            <a
              href="https://github.com/lespinozaortiz"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "#ffffff", textDecoration: "none" }}
            >
              github.com/lespinozaortiz
            </a>
          </Box>

          <Box
            sx={{
              display: "flex",
              flexBasis: "45%",
              justifyContent: "center",
              alignItems: "center",
              gap: 1,
            }}
          >
            <LinkedInIcon fontSize="small" />
            <a
              href="www.linkedin.com/in/tomas-manriquez-789234372/"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "#ffffff", textDecoration: "none" }}
            >
              linkedin.com/in/tomas-manriquez-789234372/
            </a>
                        <GitHubIcon fontSize="small" />
            <a
              href="https://github.com/tomas-manriquez"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "#ffffff", textDecoration: "none" }}
            >
              github.com/tomas-manriquez
            </a>
          </Box>

        </Box>
      </Box>
    </AppProvider>
  );
};

export default Dashboard;
