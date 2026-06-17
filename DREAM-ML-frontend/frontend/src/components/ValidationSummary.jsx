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


import React from "react";
import { Box, Typography } from "@mui/material";
import WarningIcon from "@mui/icons-material/Warning";

const ValidationSummary = ({ warnings }) => {
  if (!warnings || warnings.length === 0) {
    return null;
  }

  return (
    <Box
      sx={{
        backgroundColor: "#fff3e0",
        border: "2px solid #ff6f00",
        borderRadius: "2px",
        padding: "16px",
        marginTop: "16px",
        marginBottom: "16px",
        fontFamily: "'Roboto Mono', 'Courier New', monospace",
      }}
    >
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          marginBottom: "12px",
        }}
      >
        <WarningIcon
          sx={{
            color: "#ff6f00",
            marginRight: "8px",
            fontSize: "1.5rem",
          }}
        />
        <Typography
          variant="subtitle1"
          sx={{
            fontFamily: "'Roboto Mono', 'Courier New', monospace",
            fontWeight: "bold",
            color: "#e65100",
            fontSize: "0.95rem",
          }}
        >
          ADVERTENCIAS DE VALIDACIÓN
        </Typography>
      </Box>

      {warnings.map((warning, index) => (
        <Typography
          key={index}
          variant="body2"
          sx={{
            fontFamily: "'Roboto Mono', 'Courier New', monospace",
            color: "#e65100",
            marginLeft: "32px",
            marginBottom: "4px",
            fontSize: "0.85rem",
            "&::before": {
              content: '"▸ "',
              fontWeight: "bold",
            },
          }}
        >
          {warning}
        </Typography>
      ))}
    </Box>
  );
};

export default ValidationSummary;
