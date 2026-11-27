/**
 * Electron main process entry & local Python speech backend supervisor.
 * Handles health polling, automatic restart with backoff, and clean shutdown.
 */

import { spawn, type ChildProcess } from "node:child_process";

export interface BackendSupervisorConfig {
  pythonPath?: string;
  host?: string;
  port?: number;
  maxRestarts?: number;
}

export class BackendSupervisor {
  private readonly pythonPath: string;
  private readonly host: string;
  private readonly port: number;
  private readonly maxRestarts: number;
  private proc: ChildProcess | null = null;
  private stopping = false;
  private restartCount = 0;

  constructor(config: BackendSupervisorConfig = {}) {
    this.pythonPath = config.pythonPath ?? ".venv/Scripts/python.exe";
    this.host = config.host ?? process.env.VOICE_AGENT_WS_HOST ?? "127.0.0.1";
    this.port = config.port ?? Number(process.env.VOICE_AGENT_WS_PORT ?? 8765);
    this.maxRestarts = config.maxRestarts ?? 5;
  }

  get baseUrl(): string {
    return `http://${this.host}:${this.port}`;
  }

  get wsUrl(): string {
    return `ws://${this.host}:${this.port}/ws/v1/voice`;
  }

  start(): void {
    this.stopping = false;
    this.spawnServer();
  }

  async waitForHealthy(timeoutMs = 15000): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      try {
        const res = await fetch(`${this.baseUrl}/health`);
        if (res.ok) return true;
      } catch {
        // Server still starting.
      }
      await new Promise((r) => setTimeout(r, 250));
    }
    return false;
  }

  stop(): void {
    this.stopping = true;
    if (this.proc && !this.proc.killed) {
      this.proc.kill();
      this.proc = null;
    }
  }

  private spawnServer(): void {
    this.proc = spawn(
      this.pythonPath,
      ["-m", "uvicorn", "inference.server:app", "--host", this.host, "--port", String(this.port)],
      {
        stdio: "inherit",
        windowsHide: true,
        env: { ...process.env, VOICE_AGENT_WS_HOST: this.host, VOICE_AGENT_WS_PORT: String(this.port) },
      },
    );

    this.proc.on("exit", () => {
      if (this.stopping || this.restartCount >= this.maxRestarts) return;
      this.restartCount += 1;
      const backoffMs = Math.min(5000, 500 * 2 ** (this.restartCount - 1));
      setTimeout(() => {
        if (!this.stopping) this.spawnServer();
      }, backoffMs);
    });
  }
}

export { VoiceAgentOrchestrator } from "./orchestrator.js";
