import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const rootDir = path.resolve(__dirname, "../..");

export default defineConfig({
  base: "./",
  root: path.resolve(__dirname, "renderer"),
  plugins: [react()],
  resolve: {
    alias: {
      "@voice-agent/audio": path.resolve(rootDir, "packages/audio/index.ts"),
      "@voice-agent/speech": path.resolve(rootDir, "packages/speech/index.ts"),
      "@voice-agent/interaction": path.resolve(rootDir, "packages/interaction/index.ts"),
      "@voice-agent/agent": path.resolve(rootDir, "packages/agent/index.ts"),
      "@voice-agent/protocol": path.resolve(rootDir, "packages/protocol/index.ts"),
      "@voice-agent/metrics": path.resolve(rootDir, "packages/metrics/index.ts"),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
  },
  build: {
    outDir: path.resolve(__dirname, "dist/renderer"),
    emptyOutDir: true,
  },
});
