import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  resolve: {
    alias: {
      "@voice-agent/audio": path.resolve(__dirname, "packages/audio/index.ts"),
      "@voice-agent/speech": path.resolve(__dirname, "packages/speech/index.ts"),
      "@voice-agent/interaction": path.resolve(__dirname, "packages/interaction/index.ts"),
      "@voice-agent/agent": path.resolve(__dirname, "packages/agent/index.ts"),
      "@voice-agent/protocol": path.resolve(__dirname, "packages/protocol/index.ts"),
      "@voice-agent/metrics": path.resolve(__dirname, "packages/metrics/index.ts"),
    },
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts", "packages/**/*.test.ts", "apps/**/*.test.ts"],
  },
});
