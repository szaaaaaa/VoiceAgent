/**
 * Real browser/Electron Web Audio API capture (with hardware AEC/NS option)
 * and low-latency 24kHz streaming PCM16 player supporting Duck / Pause / Resume / Cancel.
 */

import { computeRmsDb, toMono16kPcm16 } from "@voice-agent/audio";

export interface AudioDeviceOption {
  deviceId: string;
  label: string;
  kind: "audioinput" | "audiooutput";
}

export async function listAudioDevices(): Promise<AudioDeviceOption[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices
    .filter((d): d is MediaDeviceInfo & { kind: "audioinput" | "audiooutput" } =>
      d.kind === "audioinput" || d.kind === "audiooutput",
    )
    .map((d, idx) => ({
      deviceId: d.deviceId,
      label: d.label || `${d.kind === "audioinput" ? "Microphone" : "Speaker"} ${idx + 1}`,
      kind: d.kind,
    }));
}

export class BrowserMicCapture {
  private stream: MediaStream | null = null;
  private ctx: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private pending = new Int16Array(0);
  private muted = false;
  private readonly frameSamples = 320; // 20ms @ 16kHz

  async start(
    onFrame: (pcm16k: Int16Array, ptsMs: number, rmsDb: number) => void,
    options: { deviceId?: string; aecEnabled?: boolean; noiseSuppression?: boolean } = {},
  ): Promise<void> {
    this.stop();
    const audioConstraints: MediaTrackConstraints = {
      channelCount: 1,
      echoCancellation: options.aecEnabled ?? true,
      noiseSuppression: options.noiseSuppression ?? true,
      autoGainControl: true,
      ...(options.deviceId && options.deviceId !== "default"
        ? { deviceId: { exact: options.deviceId } }
        : {}),
    };
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
    this.ctx = new AudioContext();
    this.source = this.ctx.createMediaStreamSource(this.stream);

    // shortcut: ScriptProcessorNode avoids bundling a separate AudioWorklet worker URL in Vite dev/Electron;
    // upgrade to dedicated AudioWorkletNode if main-thread UI stalls add >15ms audio jitter.
    this.processor = this.ctx.createScriptProcessor(1024, 1, 1);
    const sr = this.ctx.sampleRate;
    const frameMs = 20;

    this.processor.onaudioprocess = (ev) => {
      const input = ev.inputBuffer.getChannelData(0);
      const pcm16k = toMono16kPcm16(input, sr, 1);
      const merged = new Int16Array(this.pending.length + pcm16k.length);
      merged.set(this.pending, 0);
      merged.set(pcm16k, this.pending.length);

      let offset = 0;
      const basePts = performance.now();
      let emitted = 0;
      while (offset + this.frameSamples <= merged.length) {
        const slice = merged.slice(offset, offset + this.frameSamples);
        if (this.muted) slice.fill(0);
        onFrame(slice, basePts + emitted * frameMs, computeRmsDb(slice));
        offset += this.frameSamples;
        emitted += 1;
      }
      this.pending = merged.slice(offset);
    };

    this.source.connect(this.processor);
    this.processor.connect(this.ctx.destination);
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
  }

  stop(): void {
    this.processor?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    void this.ctx?.close();
    this.processor = null;
    this.source = null;
    this.stream = null;
    this.ctx = null;
    this.pending = new Int16Array(0);
  }
}

interface ScheduledNodeMeta {
  node: AudioBufferSourceNode;
  startTime: number;
  durationSec: number;
  charStart: number;
  charEnd: number;
  generationTag: number;
}

export class BrowserStreamPlayer {
  private ctx: AudioContext | null = null;
  private gainNode: GainNode | null = null;
  private nextStartTime = 0;
  private scheduled: ScheduledNodeMeta[] = [];
  private activeTag = 0;
  private mode: "idle" | "playing" | "ducked" | "paused" = "idle";
  private playedCharIndex = 0;
  private playedDurationMs = 0;
  private onStateChange?: (state: {
    mode: "idle" | "playing" | "ducked" | "paused";
    playedCharIndex: number;
    playedDurationMs: number;
  }) => void;

  setListener(
    cb: (state: {
      mode: "idle" | "playing" | "ducked" | "paused";
      playedCharIndex: number;
      playedDurationMs: number;
    }) => void,
  ): void {
    this.onStateChange = cb;
  }

  private ensureContext(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.gainNode = this.ctx.createGain();
      this.gainNode.gain.value = 1.0;
      this.gainNode.connect(this.ctx.destination);
      this.nextStartTime = this.ctx.currentTime;
    }
    if (this.ctx.state === "suspended" && this.mode !== "paused") {
      void this.ctx.resume();
    }
    return this.ctx;
  }

  enqueuePcm16(params: {
    pcm16: Int16Array;
    sampleRate: number;
    generationTag: number;
    charStart: number;
    charEnd: number;
  }): void {
    const ctx = this.ensureContext();
    if (params.generationTag !== this.activeTag) {
      this.activeTag = params.generationTag;
      this.playedCharIndex = 0;
      this.playedDurationMs = 0;
    }
    const buf = ctx.createBuffer(1, params.pcm16.length, params.sampleRate);
    const channel = buf.getChannelData(0);
    for (let i = 0; i < params.pcm16.length; i += 1) {
      channel[i] = params.pcm16[i]! / 32768;
    }

    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.gainNode!);

    const startAt = Math.max(ctx.currentTime + 0.015, this.nextStartTime);
    const meta: ScheduledNodeMeta = {
      node: src,
      startTime: startAt,
      durationSec: buf.duration,
      charStart: params.charStart,
      charEnd: params.charEnd,
      generationTag: params.generationTag,
    };
    this.scheduled.push(meta);
    this.nextStartTime = startAt + buf.duration;

    if (this.mode === "idle") {
      this.mode = "playing";
      this.emitState();
    }

    src.onended = () => {
      const idx = this.scheduled.indexOf(meta);
      if (idx >= 0) this.scheduled.splice(idx, 1);
      if (meta.generationTag === this.activeTag) {
        this.playedCharIndex = Math.max(this.playedCharIndex, meta.charEnd);
        this.playedDurationMs += meta.durationSec * 1000;
      }
      if (this.scheduled.length === 0 && this.mode !== "paused") {
        this.mode = "idle";
      }
      this.emitState();
    };

    src.start(startAt);
  }

  duck(targetGain = 0.15): void {
    if (!this.ctx || !this.gainNode || this.mode === "idle" || this.mode === "paused") return;
    this.gainNode.gain.setTargetAtTime(targetGain, this.ctx.currentTime, 0.015);
    this.mode = "ducked";
    this.emitState();
  }

  pause(): { playedCharIndex: number; playedDurationMs: number } {
    if (this.ctx && (this.mode === "playing" || this.mode === "ducked")) {
      void this.ctx.suspend();
      this.mode = "paused";
      this.updateCursorEstimate();
      this.emitState();
    }
    return { playedCharIndex: this.playedCharIndex, playedDurationMs: this.playedDurationMs };
  }

  resume(): void {
    if (!this.ctx || !this.gainNode) return;
    if (this.scheduled.length === 0) {
      this.mode = "idle";
      this.emitState();
      return;
    }
    this.gainNode.gain.setTargetAtTime(1.0, this.ctx.currentTime, 0.02);
    void this.ctx.resume();
    this.mode = "playing";
    this.emitState();
  }

  cancel(): { playedCharIndex: number; playedDurationMs: number } {
    this.updateCursorEstimate();
    const cursor = { playedCharIndex: this.playedCharIndex, playedDurationMs: this.playedDurationMs };
    for (const item of this.scheduled) {
      try {
        item.node.onended = null;
        item.node.stop();
      } catch {
        // Already ended.
      }
    }
    this.scheduled = [];
    if (this.ctx && this.gainNode) {
      this.gainNode.gain.setValueAtTime(1.0, this.ctx.currentTime);
      this.nextStartTime = this.ctx.currentTime;
      if (this.ctx.state === "suspended") void this.ctx.resume();
    }
    this.mode = "idle";
    this.emitState();
    return cursor;
  }

  getMode(): "idle" | "playing" | "ducked" | "paused" {
    return this.mode;
  }

  private updateCursorEstimate(): void {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    for (const item of this.scheduled) {
      if (now >= item.startTime && now <= item.startTime + item.durationSec) {
        const frac = (now - item.startTime) / Math.max(0.001, item.durationSec);
        const est = Math.round(item.charStart + (item.charEnd - item.charStart) * frac);
        this.playedCharIndex = Math.max(this.playedCharIndex, est);
      }
    }
  }

  private emitState(): void {
    this.onStateChange?.({
      mode: this.mode,
      playedCharIndex: this.playedCharIndex,
      playedDurationMs: Math.round(this.playedDurationMs),
    });
  }
}
