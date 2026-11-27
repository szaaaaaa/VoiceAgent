/**
 * Streaming TTS audio playback queue with Duck / Pause / Resume / Cancel controls,
 * exact character-level playback cursor tracking, and AEC reference fanout (FR-33, FR-34).
 */

import { int16ToFloat32, resampleMonoFloat32, float32ToInt16, TARGET_ASR_SAMPLE_RATE } from "./resampler.js";

export type PlaybackMode = "playing" | "ducked" | "paused" | "idle";

export interface TtsAudioChunk {
  readonly turnId: string;
  readonly generationId: string;
  readonly generationTag: number;
  readonly sentenceIndex: number;
  readonly chunkIndex: number;
  readonly charStart: number;
  readonly charEnd: number;
  readonly sampleRate: number;
  readonly pcm16: Int16Array;
}

export interface PlaybackCursor {
  readonly turnId: string;
  readonly generationId: string;
  readonly playedDurationMs: number;
  readonly playedCharIndex: number;
}

export interface PlaybackAckEvent extends PlaybackCursor {
  readonly playbackState: PlaybackMode;
}

export interface PlaybackController {
  enqueueChunk(chunk: TtsAudioChunk): void;
  duck(targetGain?: number): void;
  pause(): PlaybackCursor;
  resume(): void;
  cancel(generationId?: string): PlaybackCursor;
  getCurrentCursor(): PlaybackCursor;
  getMode(): PlaybackMode;
  getGain(): number;
  getQueuedDurationMs(): number;
}

interface QueuedChunk {
  chunk: TtsAudioChunk;
  sampleOffset: number;
}

export class PlaybackQueue implements PlaybackController {
  private readonly queue: QueuedChunk[] = [];
  private readonly maxQueuedMs: number;
  private mode: PlaybackMode = "idle";
  private gain = 1.0;
  private activeTurnId = "";
  private activeGenerationId = "";
  private playedDurationMs = 0;
  private playedCharIndex = 0;
  private readonly cancelledGenerations = new Set<string>();
  private readonly ackListeners = new Set<(ack: PlaybackAckEvent) => void>();
  private readonly aecRefListeners = new Set<(pcm16k: Int16Array) => void>();
  private selectedOutputDeviceId = "default";

  constructor(maxQueuedMs = 30000) {
    this.maxQueuedMs = maxQueuedMs;
  }

  selectOutputDevice(deviceId: string): void {
    this.selectedOutputDeviceId = deviceId;
  }

  getOutputDeviceId(): string {
    return this.selectedOutputDeviceId;
  }

  onPlaybackAck(listener: (ack: PlaybackAckEvent) => void): () => void {
    this.ackListeners.add(listener);
    return () => this.ackListeners.delete(listener);
  }

  onAecReference(listener: (pcm16k: Int16Array) => void): () => void {
    this.aecRefListeners.add(listener);
    return () => this.aecRefListeners.delete(listener);
  }

  getMode(): PlaybackMode {
    return this.mode;
  }

  getGain(): number {
    return this.gain;
  }

  getQueuedDurationMs(): number {
    let total = 0;
    for (const item of this.queue) {
      const rem = item.chunk.pcm16.length - item.sampleOffset;
      total += (rem / item.chunk.sampleRate) * 1000;
    }
    return total;
  }

  getCurrentCursor(): PlaybackCursor {
    return {
      turnId: this.activeTurnId,
      generationId: this.activeGenerationId,
      playedDurationMs: this.playedDurationMs,
      playedCharIndex: this.playedCharIndex,
    };
  }

  enqueueChunk(chunk: TtsAudioChunk): void {
    // Reject late audio chunks from cancelled generations immediately.
    if (this.cancelledGenerations.has(chunk.generationId)) return;

    if (chunk.generationId !== this.activeGenerationId) {
      this.activeTurnId = chunk.turnId;
      this.activeGenerationId = chunk.generationId;
      this.playedDurationMs = 0;
      this.playedCharIndex = 0;
    }

    // Prevent unbounded playback queue growth under stalled hardware output.
    while (this.getQueuedDurationMs() > this.maxQueuedMs && this.queue.length > 0) {
      this.queue.shift();
    }

    this.queue.push({ chunk, sampleOffset: 0 });
    if (this.mode === "idle") {
      this.mode = "playing";
      this.gain = 1.0;
    }
  }

  duck(targetGain = 0.15): void {
    if (this.mode === "idle" || this.mode === "paused") return;
    this.gain = Math.max(0, Math.min(1, targetGain));
    this.mode = "ducked";
    this.emitAck();
  }

  pause(): PlaybackCursor {
    if (this.mode === "playing" || this.mode === "ducked") {
      this.mode = "paused";
      this.emitAck();
    }
    return this.getCurrentCursor();
  }

  resume(): void {
    if (this.queue.length === 0) {
      this.mode = "idle";
      this.gain = 1.0;
      this.emitAck();
      return;
    }
    if (this.mode === "paused" || this.mode === "ducked") {
      this.gain = 1.0;
      this.mode = "playing";
      this.emitAck();
    }
  }

  cancel(generationId?: string): PlaybackCursor {
    const cursor = this.getCurrentCursor();
    const targetId = generationId ?? this.activeGenerationId;
    if (targetId) {
      this.cancelledGenerations.add(targetId);
      if (this.cancelledGenerations.size > 256) {
        const oldest = this.cancelledGenerations.values().next().value;
        if (oldest !== undefined) this.cancelledGenerations.delete(oldest);
      }
    }

    if (!generationId || generationId === this.activeGenerationId) {
      this.queue.length = 0;
      this.mode = "idle";
      this.gain = 1.0;
    } else {
      for (let i = this.queue.length - 1; i >= 0; i -= 1) {
        if (this.queue[i]!.chunk.generationId === generationId) {
          this.queue.splice(i, 1);
        }
      }
      if (this.queue.length === 0) {
        this.mode = "idle";
        this.gain = 1.0;
      }
    }

    this.emitAck();
    return cursor;
  }

  /**
   * Pulls the next audio slice for the speaker output device (or test harness),
   * updating the exact audible playback cursor and feeding AEC reference.
   */
  pullNextAudio(maxSamples: number): { pcm16: Int16Array; sampleRate: number } | null {
    if (this.mode === "paused" || this.mode === "idle" || this.queue.length === 0) {
      return null;
    }
    const head = this.queue[0]!;
    const remaining = head.chunk.pcm16.length - head.sampleOffset;
    const take = Math.min(maxSamples, remaining);
    const slice = new Int16Array(take);

    for (let i = 0; i < take; i += 1) {
      slice[i] = Math.round(head.chunk.pcm16[head.sampleOffset + i]! * this.gain);
    }

    head.sampleOffset += take;
    const stepMs = (take / head.chunk.sampleRate) * 1000;
    this.playedDurationMs += stepMs;

    const progressRatio = head.chunk.pcm16.length > 0 ? head.sampleOffset / head.chunk.pcm16.length : 1;
    const charSpan = Math.max(0, head.chunk.charEnd - head.chunk.charStart);
    const reachedChar = Math.round(head.chunk.charStart + charSpan * progressRatio);
    this.playedCharIndex = Math.max(this.playedCharIndex, reachedChar);

    if (head.sampleOffset >= head.chunk.pcm16.length) {
      this.queue.shift();
      if (this.queue.length === 0) {
        this.mode = "idle";
        this.gain = 1.0;
      }
    }

    if (this.aecRefListeners.size > 0) {
      const ref16k =
        head.chunk.sampleRate === TARGET_ASR_SAMPLE_RATE
          ? slice
          : float32ToInt16(
              resampleMonoFloat32(int16ToFloat32(slice), head.chunk.sampleRate, TARGET_ASR_SAMPLE_RATE),
            );
      for (const listener of this.aecRefListeners) {
        listener(ref16k);
      }
    }

    this.emitAck();
    return { pcm16: slice, sampleRate: head.chunk.sampleRate };
  }

  private emitAck(): void {
    const ack: PlaybackAckEvent = {
      ...this.getCurrentCursor(),
      playbackState: this.mode,
    };
    for (const listener of this.ackListeners) {
      listener(ack);
    }
  }
}
