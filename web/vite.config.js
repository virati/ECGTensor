import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  // main.js loads its data with top-level await
  build: { target: 'es2022', chunkSizeWarningLimit: 800 },
});
