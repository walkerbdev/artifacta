import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

export default defineConfig({
  root: path.resolve(__dirname, 'artifacta_ui'),
  publicDir: path.resolve(__dirname, 'public'),
  plugins: [react()],
  css: {
    preprocessorOptions: {
      scss: {
        api: 'modern-compiler'
      }
    }
  },
  server: {
    port: 5173,
    open: true
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true
  },

  // Path aliases for clean imports
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@/app': path.resolve(__dirname, './src/app'),
      '@/ml': path.resolve(__dirname, './src/ml'),
      '@/core': path.resolve(__dirname, './src/core'),
      '@/config': path.resolve(__dirname, './src/config'),
      '/src': path.resolve(__dirname, './src')
    }
  }
});
