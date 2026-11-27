import { spawn, type ChildProcess } from "node:child_process";
import net from "node:net";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { VoiceAgentOrchestrator } from "../../apps/desktop/main/orchestrator.js";

function findFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, "127.0.0.1", () => {
      const addr = srv.address();
      const port = typeof addr === "object" && addr ? addr.port : 0;
      srv.close(() => resolve(port));
    });
    srv.on("error", reject);
  });
}

async function waitForHealth(port: number, timeoutMs = 15000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/health`);
      if (res.ok) return;
    } catch {
      // booting
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`Python inference server failed to start on port ${port}`);
}

describe("end-to-end TypeScript <-> Python Local WebSocket voice pipeline", () => {
  let serverProc: ChildProcess | null = null;
  let port = 0;

  beforeAll(async () => {
    port = await findFreePort();
    const pythonExe = path.resolve(".venv/Scripts/python.exe");
    const stderrChunks: string[] = [];
    serverProc = spawn(
      pythonExe,
      ["-m", "uvicorn", "inference.server:app", "--host", "127.0.0.1", "--port", String(port)],
      { stdio: ["ignore", "ignore", "pipe"], windowsHide: true },
    );
    serverProc.stderr?.on("data", (d: Buffer) => stderrChunks.push(d.toString("utf8")));
    try {
      await waitForHealth(port, 35000);
    } catch (err) {
      throw new Error(`${err instanceof Error ? err.message : String(err)}\nstderr: ${stderrChunks.join("")}`);
    }
  }, 40000);

  afterAll(() => {
    if (serverProc && !serverProc.killed) {
      serverProc.kill();
    }
  });

  it("streams mic PCM, receives live VAD/Turn predictions, synthesizes TTS audio, and supports barge-in cancel", async () => {
    const orch = new VoiceAgentOrchestrator({
      sessionId: "e2e_live_sess",
      wsUrl: `ws://127.0.0.1:${port}/ws/v1/voice`,
      allowedRoots: [path.resolve(".")],
      llmProvider: async function* () {
        yield { type: "delta", text: "Hello from VoiceAgent local pipeline. Everything is operational." };
      },
    });

    const receivedTypes: string[] = [];
    orch.speechClient.onEvent((e) => receivedTypes.push(e.type));

    await orch.connectBackend();
    expect(orch.getSnapshot().backendConnected).toBe(true);

    const healthRes = await fetch(`http://127.0.0.1:${port}/health`);
    const healthJson = (await healthRes.json()) as { models?: { vad?: { backend?: string } } };
    expect(healthJson.models?.vad?.backend).toBeTruthy();

    // Stream 10 frames (200ms) of 16kHz PCM -> triggers live streaming ASR feature extraction & Turn Adapter prediction.
    const voiced = new Int16Array(3200);
    for (let i = 0; i < voiced.length; i += 1) {
      voiced[i] = Math.round(Math.sin((2 * Math.PI * 220 * i) / 16000) * 12000);
    }
    orch.capture.ingestPcm16k(voiced, 1000);

    // Submit a turn and verify live streaming TTS chunks arrive into PlaybackQueue.
    const reply = await orch.submitTurn("Hello VoiceAgent");
    expect(reply).toContain("Everything is operational");

    // Wait briefly for streaming TTS binary frames from Python backend.
    const deadline = Date.now() + 5000;
    while (orch.playback.getMode() === "idle" && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 50));
    }

    expect(receivedTypes).toContain("turn.prediction");
    expect(receivedTypes).toContain("tts.chunk");
    expect(orch.playback.getMode()).toBe("playing");

    // Pull audible samples -> triggers playback.ack and marks audible latency.
    const slice = orch.playback.pullNextAudio(1200);
    expect(slice).not.toBeNull();
    expect(orch.metrics.summarize().e2eAudibleLatencyMs.count).toBeGreaterThanOrEqual(1);

    // Simulate active user interruption during playback -> immediately stops playback & sends generation.cancel.
    orch.interruption.onPotentialUserSpeech();
    const res = orch.interruption.resolve({
      event: "INTERRUPTION",
      confidence: 0.95,
      source: "e2e_test",
      reason: "user barge-in",
    });
    expect(res.resolution).toBe("cancelled");
    expect(orch.playback.getMode()).toBe("idle");

    orch.disconnect();
  }, 15000);
});
