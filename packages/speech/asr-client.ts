/**
 * Local WebSocket client bridging TypeScript audio/interaction engine with the
 * local Python speech inference server (127.0.0.1 only by default, PRD §8.3).
 */

import WebSocket from "ws";
import {
  type BinaryAudioFrame,
  createEvent,
  decodeAudioFrame,
  encodeAudioFrame,
  type EventEnvelope,
  type EventPayloadMap,
  parseEventEnvelope,
  type ProtocolEventType,
  StreamKind,
} from "@voice-agent/protocol";

export interface SpeechClientConfig {
  url?: string;
  allowRemoteHost?: boolean;
  reconnectBaseMs?: number;
  reconnectMaxMs?: number;
}

export class SpeechWsClient {
  private readonly url: string;
  private readonly reconnectBaseMs: number;
  private readonly reconnectMaxMs: number;
  private ws: WebSocket | null = null;
  private closedByUser = false;
  private reconnectAttempt = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private audioSeq = 0;

  private readonly eventListeners = new Set<(event: EventEnvelope) => void>();
  private readonly audioListeners = new Set<(frame: BinaryAudioFrame) => void>();
  private readonly statusListeners = new Set<(connected: boolean) => void>();

  constructor(config: SpeechClientConfig = {}) {
    const rawUrl = config.url ?? "ws://127.0.0.1:8765/ws/v1/voice";
    const parsed = new URL(rawUrl);
    const isLocal = parsed.hostname === "127.0.0.1" || parsed.hostname === "localhost" || parsed.hostname === "::1";
    if (!isLocal && !config.allowRemoteHost) {
      throw new Error(`Security policy: speech backend must bind to localhost, got "${parsed.hostname}"`);
    }
    this.url = rawUrl;
    this.reconnectBaseMs = config.reconnectBaseMs ?? 200;
    this.reconnectMaxMs = config.reconnectMaxMs ?? 2000;
  }

  onEvent(listener: (event: EventEnvelope) => void): () => void {
    this.eventListeners.add(listener);
    return () => this.eventListeners.delete(listener);
  }

  onBinaryAudio(listener: (frame: BinaryAudioFrame) => void): () => void {
    this.audioListeners.add(listener);
    return () => this.audioListeners.delete(listener);
  }

  onConnectionChange(listener: (connected: boolean) => void): () => void {
    this.statusListeners.add(listener);
    return () => this.statusListeners.delete(listener);
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  async connect(): Promise<void> {
    this.closedByUser = false;
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;

      const onOpen = (): void => {
        this.reconnectAttempt = 0;
        for (const cb of this.statusListeners) cb(true);
        resolve();
      };

      ws.once("open", onOpen);
      ws.once("error", (err) => {
        if (ws.readyState !== WebSocket.OPEN) reject(err);
      });

      ws.on("message", (data, isBinary) => {
        try {
          if (isBinary) {
            const bytes =
              data instanceof ArrayBuffer
                ? new Uint8Array(data)
                : Buffer.isBuffer(data)
                  ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
                  : new Uint8Array(Buffer.concat(data as Buffer[]));
            const frame = decodeAudioFrame(bytes);
            for (const cb of this.audioListeners) cb(frame);
          } else {
            const text = typeof data === "string" ? data : data.toString("utf8");
            const envelope = parseEventEnvelope(text);
            for (const cb of this.eventListeners) cb(envelope);
          }
        } catch (err) {
          const fallback = createEvent(
            "error",
            { sessionId: "client" },
            {
              module: "speech-client",
              code: "PROTOCOL_DECODE_ERROR",
              message: err instanceof Error ? err.message : String(err),
              recoverable: true,
              fallback_action: "none",
            },
          );
          for (const cb of this.eventListeners) cb(fallback);
        }
      });

      ws.on("close", () => {
        for (const cb of this.statusListeners) cb(false);
        if (!this.closedByUser) {
          this.scheduleReconnect();
        }
      });
    });
  }

  disconnect(): void {
    this.closedByUser = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  sendEvent<T extends ProtocolEventType>(
    type: T,
    ids: { sessionId: string; turnId?: string | null; generationId?: string | null },
    payload: EventPayloadMap[T],
  ): boolean {
    if (!this.isConnected()) return false;
    const envelope = createEvent(type, ids, payload);
    this.ws!.send(JSON.stringify(envelope));
    return true;
  }

  sendMicAudio(pcm16k: Int16Array, ptsMs: number): boolean {
    if (!this.isConnected()) return false;
    this.audioSeq += 1;
    const encoded = encodeAudioFrame({
      streamKind: StreamKind.MicInput,
      seq: this.audioSeq,
      ptsMs,
      sampleRate: 16000,
      generationTag: 0,
      pcm16: pcm16k,
    });
    this.ws!.send(encoded);
    return true;
  }

  private scheduleReconnect(): void {
    const delay = Math.min(this.reconnectMaxMs, this.reconnectBaseMs * 2 ** this.reconnectAttempt);
    this.reconnectAttempt += 1;
    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(() => {
        // Error handled by close listener scheduling the next attempt.
      });
    }, delay);
  }
}
