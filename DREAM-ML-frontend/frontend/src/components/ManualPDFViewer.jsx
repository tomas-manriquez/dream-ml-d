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


// ManualPDFViewer.jsx
import React from 'react';
import {
  Viewer,
  Worker,
  SpecialZoomLevel,
} from '@react-pdf-viewer/core';
import { defaultLayoutPlugin } from '@react-pdf-viewer/default-layout';

// Estilos de react-pdf-viewer
import '@react-pdf-viewer/core/lib/styles/index.css';
import '@react-pdf-viewer/default-layout/lib/styles/index.css';


import workerFile from 'pdfjs-dist/build/pdf.worker.min.js?url';

export default function ManualPDFViewer() {
  
  const pdfFilePath = '/manual.pdf';

  // Plugin que incluye barra de herramientas, miniaturas, etc.
  const defaultLayoutPluginInstance = defaultLayoutPlugin();

  return (
    <div
      style={{
        width: '100%',
        height: '80vh',
        margin: '0 auto',
        border: '1px solid #ccc',
        borderRadius: 8,
        overflow: 'hidden',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      }}
    >
      {/**/}
      <Worker workerUrl={workerFile}>
        <Viewer
          fileUrl={pdfFilePath}
          defaultScale={SpecialZoomLevel.PageWidth}
          plugins={[defaultLayoutPluginInstance]}
        />
      </Worker>
    </div>
  );
}
