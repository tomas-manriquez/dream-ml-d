// vite.config.js
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  // Carga las variables de entorno del directorio actual
  const env = loadEnv(mode, process.cwd(), '');
  return {
    plugins: [react()],
    server: {
      host:'0.0.0.0',
      proxy: {
        '/api': {
          
          target: env.VITE_API_URL || 'http://0.0.0.0:8000',
          changeOrigin: true,
          secure: false,
        },
      },
    },
  };
});
