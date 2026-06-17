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


/**
 * Brutalist + Technical Blueprint Fusion Design System
 * - Monospace fonts for technical feel
 * - Sharp corners (2-4px border-radius)
 * - High contrast black/white with teal/cyan accents
 * - Asymmetric, measured layouts
 */

export const variableSelectionStyles = {
  // Section container - asymmetric offset
  sectionContainer: {
    marginBottom: "24px",
    marginLeft: "8px", // Asymmetric offset
  },

  // Section title with info button
  sectionTitleContainer: {
    display: "flex",
    alignItems: "center",
    marginBottom: "8px",
    gap: "8px",
  },

  sectionTitle: {
    fontFamily: "'Roboto Mono', 'Courier New', monospace",
    fontWeight: "bold",
    color: "#004d40",
    fontSize: "1rem",
    letterSpacing: "0.5px",
    textTransform: "uppercase",
  },

  // Info icon button
  infoButton: {
    padding: "4px",
    color: "#00796b",
    border: "2px solid #00796b",
    borderRadius: "2px",
    width: "28px",
    height: "28px",
    "&:hover": {
      backgroundColor: "#00796b",
      color: "#ffffff",
    },
  },

  // Helper text below title
  helperText: {
    fontFamily: "'Roboto Mono', 'Courier New', monospace",
    fontSize: "0.8rem",
    color: "#00796b",
    marginBottom: "12px",
    marginLeft: "4px",
    fontStyle: "italic",
  },

  // Variable selection box - sharp corners, technical feel
  variableBox: {
    maxHeight: "150px",
    overflowY: "auto",
    border: "2px solid #004d40",
    borderRadius: "2px",
    padding: "12px",
    backgroundColor: "#f5f5f5",
    fontFamily: "'Roboto Mono', 'Courier New', monospace",
  },

  // Form control label
  formControlLabel: {
    fontFamily: "'Roboto Mono', 'Courier New', monospace",
    fontSize: "0.9rem",
    color: "#212121",
    "&:hover": {
      backgroundColor: "#e0f7fa",
    },
  },

  // Radio button style
  radioButton: {
    color: "#00796b",
    "&.Mui-checked": {
      color: "#004d40",
    },
  },

  // Checkbox style
  checkbox: {
    color: "#00796b",
    "&.Mui-checked": {
      color: "#004d40",
    },
  },
};

// Info modal content in Spanish (technical ML terminology)
export const infoContent = {
  variablesDeSalida: `La variable de salida, también conocida como variable objetivo o target, es la variable dependiente que el modelo de machine learning intentará predecir.

En terminología técnica:
  • Variable dependiente (dependent variable)
  • Variable de respuesta (response variable)
  • Variable objetivo (target variable)
  • Label (en clasificación)

Ejemplo práctico:
Si estás prediciendo si un cliente comprará un producto, la columna "compró" (sí/no) sería tu variable de salida.

Restricción: Debes seleccionar exactamente 1 variable de salida.`,

  variablesDeEntrada: `Las variables de entrada, también conocidas como características o features, son las variables independientes que el modelo utiliza para hacer predicciones.

En terminología técnica:
  • Variables independientes (independent variables)
  • Predictores (predictors)
  • Características (features)
  • Atributos (attributes)

Ejemplo práctico:
Para predecir ventas, podrías usar: edad del cliente, ingreso anual, historial de compras, ubicación geográfica, etc.

Restricciones:
  • Mínimo 1 variable de entrada requerida
  • Una columna no puede ser entrada y salida simultáneamente`,
};

// Helper text strings
export const helperTextStrings = {
  variablesDeSalida: "Selecciona la variable que el modelo debe predecir",
  variablesDeEntrada: "Selecciona las características que el modelo usará para hacer predicciones (mínimo 1 requerida)",
};
