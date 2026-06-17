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


import React, { useEffect, useState } from 'react';
import { LinearProgress, Typography, Box } from '@mui/material';

/**
 * Enhanced ProgressBar Component
 *
 * @param {Object} props
 * @param {number} props.progress - Progress value (0-100). If not provided, shows indeterminate.
 * @param {string} props.message - Message to display with progress
 * @param {string} props.variant - Style variant: 'tealHarmony' | 'modernMinimal' | 'enhancedMaterial'
 * @param {boolean} props.useWebSocket - If true, connects to WebSocket for real-time progress
 * @param {string} props.wsStep - WebSocket step to listen for (e.g., 'eda', 'data_cleaning')
 * @param {boolean} props.showPercentage - Whether to show percentage (default: true)
 */
const ProgressBar = ({
  progress: externalProgress,
  message: externalMessage,
  variant = 'tealHarmony',
  useWebSocket = false,
  wsStep = null,
  showPercentage = true
}) => {
  const [progress, setProgress] = useState(externalProgress || 0);
  const [stage, setStage] = useState(externalMessage || "");

  // WebSocket connection (legacy support for UploadCsvCard)
  useEffect(() => {
    if (!useWebSocket) return;

    const socket = new WebSocket(`ws://${import.meta.env.VITE_WS_URL}/ws/progreso/`);
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // If wsStep is specified, only update for matching step
      if (wsStep && data.step !== wsStep) return;

      if (data.progress !== undefined) {
        setProgress(data.progress);
        // Mapear el porcentaje a un mensaje de etapa
        if (data.progress < 20) {
          setStage("Guardando archivo CSV");
        } else if (data.progress < 40) {
          setStage("Versionando archivo crudo");
        } else if (data.progress < 50) {
          setStage("Subiendo archivo crudo");
        } else if (data.progress < 70) {
          setStage("Aplicando limpieza de datos");
        } else if (data.progress < 90) {
          setStage("Versionando archivo procesado");
        } else if (data.progress < 100) {
          setStage("Finalizando...");
        } else {
          setStage("Proceso completado");
        }
      }
    };
    socket.onclose = () => console.log("WebSocket cerrado");
    return () => socket.close();
  }, [useWebSocket, wsStep]);

  // Update from external props
  useEffect(() => {
    if (externalProgress !== undefined) {
      setProgress(externalProgress);
    }
  }, [externalProgress]);

  useEffect(() => {
    if (externalMessage !== undefined) {
      setStage(externalMessage);
    }
  }, [externalMessage]);

  const displayMessage = stage || externalMessage || "";
  const displayProgress = progress || externalProgress;
  const isIndeterminate = displayProgress === undefined || displayProgress === null;

  // Variant styles
  const getVariantStyles = () => {
    switch (variant) {
      case 'tealHarmony':
        return {
          container: { width: '100%', mb: 2 },
          progressBar: {
            height: 8,
            borderRadius: 4,
            backgroundColor: '#b2dfdb',
            '& .MuiLinearProgress-bar': {
              borderRadius: 4,
              background: 'linear-gradient(90deg, #00796b 0%, #004d40 100%)',
            }
          },
          text: {
            color: '#004d40',
            fontWeight: 500,
            fontSize: '0.875rem',
            mt: 1,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }
        };

      case 'modernMinimal':
        return {
          container: { width: '100%', mb: 2 },
          progressBar: {
            height: 6,
            borderRadius: 3,
            backgroundColor: '#e0e0e0',
            boxShadow: '0 1px 2px rgba(0,0,0,0.1)',
            '& .MuiLinearProgress-bar': {
              borderRadius: 3,
              backgroundColor: '#00796b',
            }
          },
          text: {
            color: '#555',
            fontWeight: 400,
            fontSize: '0.813rem',
            mt: 0.75,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }
        };

      case 'enhancedMaterial':
        return {
          container: { width: '100%', mb: 2 },
          progressBar: {
            height: 10,
            borderRadius: 5,
            backgroundColor: '#cfd8dc',
            boxShadow: '0 2px 4px rgba(0,0,0,0.12)',
            '& .MuiLinearProgress-bar': {
              borderRadius: 5,
              backgroundColor: '#00796b',
              boxShadow: '0 0 8px rgba(0, 121, 107, 0.4)',
            }
          },
          text: {
            color: '#004d40',
            fontWeight: 600,
            fontSize: '0.875rem',
            mt: 1.25,
            display: 'flex',
            flexDirection: 'column',
            gap: 0.5
          }
        };

      default:
        return getVariantStyles.call(this, 'tealHarmony');
    }
  };

  const styles = getVariantStyles();

  return (
    <Box sx={styles.container}>
      <LinearProgress
        variant={isIndeterminate ? "indeterminate" : "determinate"}
        value={displayProgress}
        sx={styles.progressBar}
      />
      {(displayMessage || showPercentage) && (
        <Box sx={styles.text}>
          {variant === 'enhancedMaterial' ? (
            <>
              {showPercentage && !isIndeterminate && (
                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                  {Math.round(displayProgress)}%
                </Typography>
              )}
              {displayMessage && (
                <Typography variant="caption">
                  {displayMessage}
                </Typography>
              )}
            </>
          ) : (
            <>
              <Typography variant="caption" sx={{ flex: 1 }}>
                {displayMessage}
              </Typography>
              {showPercentage && !isIndeterminate && (
                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                  {Math.round(displayProgress)}%
                </Typography>
              )}
            </>
          )}
        </Box>
      )}
    </Box>
  );
};

export default ProgressBar;
