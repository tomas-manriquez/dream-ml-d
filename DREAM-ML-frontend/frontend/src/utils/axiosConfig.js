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


// src/utils/axiosConfig.js
import axios from 'axios';

// Si VITE_API_URL está definida (por ejemplo, "http://backend:8000"), se usará para construir la URL base.
// De lo contrario, se usará la ruta relativa '/api'.
const baseURL = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : '/api';

const axiosInstance = axios.create({
  baseURL,
});

axiosInstance.interceptors.response.use(
  response => response,
  error => {
    if (error.response) {
      // El servidor respondió con un código de estado fuera del rango 2xx
      return Promise.reject({
        message: error.response.data?.status || 'Error en la solicitud',
        response: error.response
      });
    } else if (error.request) {
      // La solicitud fue hecha pero no se recibió respuesta
      return Promise.reject({ message: 'No se recibió respuesta del servidor' });
    } else {
      // Algo sucedió al configurar la solicitud
      return Promise.reject({ message: error.message });
    }
  }
);

export default axiosInstance;
