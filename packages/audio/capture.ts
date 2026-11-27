/**
 * Continuous 16kHz audio capture pipeline with pre-roll, ring buffer,
 * lightweight NLMS acoustic echo cancellation (AEC), and noise gate (FR-01~FR-07).
 */

import { computeRmsDb, float32ToInt16, int16ToFloat32, TARGET_ASR_SAMPLE_RATE, toMono16kPcm16 } from "./resampler.js";
import { Int16RingBuffer, PreRollBuffer } from "./ring-buffer.js";

export interface AudioFrame {
  readonly pcm16: Int16Array;
  readonly sampleRate: 16000;
  readonly ptsMs: number;
  readonly rmsDb: number;
}

export interface AudioDeviceDescriptor {
  deviceId: string;
  label: string;
  kind: "audioinput" | "audiooutput";
}

export interface AecProcessor {
  process(micPcm16k: Int16Array, refPcm16k: Int16Array): Int16Array;
  reset(): void;
}

export interface AudioCaptureConfig {
  frameDurationMs?: number;
  prerollMs?: number;
  ringBufferSeconds?: number;
  aecEnabled?: boolean;
  aecProcessor?: AecProcessor;
  noiseSuppressionEnabled?: boolean;
  noiseGateDb?: number;
}

/**
 * Lightweight normalized LMS adaptive filter for acoustic echo cancellation.
 * shortcut: 64-tap time-domain NLMS handles short desktop echo paths; inject WebRTC APM (AEC3)
 * via `AudioCaptureConfig.aecProcessor` for high-volume reverberant loudspeaker setups.
 */
export class AdaptiveAecFilter implements AecProcessor {
  private readonly weights: Float32Array;
  private readonly refHistory: Float32Array;
  private readonly stepSize: number;

  constructor(taps = 64, stepSize = 0.25) {
    this.weights = new Float32Array(taps);
    this.refHistory = new Float32Array(taps);
    this.stepSize = stepSize;
  }

  process(micPcm: Int16Array, refPcm: Int16Array): Int16Array {
    const mic = int16ToFloat32(micPcm);
    const ref = int16ToFloat32(refPcm);
    const out = new Float32Array(mic.length);
    const taps = this.weights.length;

    for (let i = 0; i < mic.length; i += 1) {
      // Shift reference delay line by one sample.
      for (let k = taps - 1; k > 0; k -= 1) {
        this.refHistory[k] = this.refHistory[k - 1]!;
      }
      this.refHistory[0] = i < ref.length ? ref[i]! : 0;

      let echoEst = 0;
      let energy = 1e-4;
      for (let k = 0; k < taps; k += 1) {
        const r = this.refHistory[k]!;
        echoEst += this.weights[k]! * r;
        energy += r * r;
      }

      const err = mic[i]! - echoEst;
      out[i] = Math.max(-1, Math.min(1, err));

      // Only adapt when reference signal has real energy to avoid drifting on silence.
      if (energy > 1e-3) {
        const gain = (this.stepSize * err) / energy;
        for (let k = 0; k < taps; k += 1) {
          this.weights[k] = Math.max(-2, Math.min(2, this.weights[k]! + gain * this.refHistory[k]!));
        }
      }
    }
    return float32ToInt16(out);
  }

  reset(): void {
    this.weights.fill(0);
    this.refHistory.fill(0);
  }
}

export class AudioCapturePipeline {
  readonly frameSamples: number;
  readonly ringBuffer: Int16RingBuffer;
  readonly preRoll: PreRollBuffer;
  private readonly aec: AecProcessor;
  private readonly aecRefQueue: Int16Array[] = [];
  private aecRefSamples = 0;
  private readonly maxAecRefSamples = TARGET_ASR_SAMPLE_RATE * 2;
  private pending: Int16Array<ArrayBufferLike> = new Int16Array(0);
  private muted = false;
  private selectedInputDeviceId = "default";
  private aecEnabled: boolean;
  private noiseSuppressionEnabled: boolean;
  private readonly noiseGateDb: number;
  private readonly listeners = new Set<(frame: AudioFrame) => void>();

  constructor(config: AudioCaptureConfig = {}) {
    const frameDurationMs = config.frameDurationMs ?? 20;
    this.frameSamples = Math.round((TARGET_ASR_SAMPLE_RATE * frameDurationMs) / 1000);
    this.preRoll = new PreRollBuffer(config.prerollMs ?? 300, TARGET_ASR_SAMPLE_RATE);
    this.ringBuffer = new Int16RingBuffer(
      TARGET_ASR_SAMPLE_RATE * (config.ringBufferSeconds ?? 30),
    );
    this.aec = config.aecProcessor ?? new AdaptiveAecFilter();
    this.aecEnabled = config.aecEnabled ?? true;
    this.noiseSuppressionEnabled = config.noiseSuppressionEnabled ?? true;
    this.noiseGateDb = config.noiseGateDb ?? -58;
  }

  onFrame(listener: (frame: AudioFrame) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
  }

  isMuted(): boolean {
    return this.muted;
  }

  selectInputDevice(deviceId: string): void {
    this.selectedInputDeviceId = deviceId;
    this.pending = new Int16Array(0);
    this.aec.reset();
  }

  getInputDeviceId(): string {
    return this.selectedInputDeviceId;
  }

  setAecEnabled(enabled: boolean): void {
    this.aecEnabled = enabled;
    if (!enabled) this.aec.reset();
  }

  /** Receives actual speaker output resampled to 16kHz for echo cancellation reference. */
  pushAecReference(refPcm16k: Int16Array): void {
    if (!this.aecEnabled || refPcm16k.length === 0) return;
    this.aecRefQueue.push(refPcm16k);
    this.aecRefSamples += refPcm16k.length;
    while (this.aecRefSamples > this.maxAecRefSamples && this.aecRefQueue.length > 0) {
      const dropped = this.aecRefQueue.shift()!;
      this.aecRefSamples -= dropped.length;
    }
  }

  /** Ingests arbitrary microphone or file PCM chunks, emitting exact 20ms 16kHz frames. */
  ingestFloat32(
    interleaved: Float32Array,
    sourceSampleRate: number,
    channels = 1,
    basePtsMs = performance.now(),
  ): AudioFrame[] {
    const pcm16 = toMono16kPcm16(interleaved, sourceSampleRate, channels);
    return this.ingestPcm16k(pcm16, basePtsMs);
  }

  ingestPcm16k(pcm16k: Int16Array, basePtsMs = performance.now()): AudioFrame[] {
    if (pcm16k.length === 0) return [];
    const combined = new Int16Array(this.pending.length + pcm16k.length);
    combined.set(this.pending, 0);
    combined.set(pcm16k, this.pending.length);

    const emitted: AudioFrame[] = [];
    let offset = 0;
    const frameMs = (this.frameSamples / TARGET_ASR_SAMPLE_RATE) * 1000;

    while (offset + this.frameSamples <= combined.length) {
      let slice: Int16Array = combined.slice(offset, offset + this.frameSamples);
      if (this.muted) {
        slice.fill(0);
      } else {
        if (this.aecEnabled && this.aecRefSamples > 0) {
          const refSlice = this.consumeAecRef(this.frameSamples);
          slice = this.aec.process(slice, refSlice);
        }
        if (this.noiseSuppressionEnabled && computeRmsDb(slice) < this.noiseGateDb) {
          // Soft attenuation (-18dB) preserves continuous ambient floor without abrupt zero-cuts.
          for (let i = 0; i < slice.length; i += 1) {
            slice[i] = Math.round(slice[i]! * 0.125);
          }
        }
      }

      this.preRoll.push(slice);
      this.ringBuffer.push(slice);

      const frame: AudioFrame = {
        pcm16: slice,
        sampleRate: TARGET_ASR_SAMPLE_RATE,
        ptsMs: basePtsMs + emitted.length * frameMs,
        rmsDb: computeRmsDb(slice),
      };
      emitted.push(frame);
      for (const listener of this.listeners) {
        listener(frame);
      }
      offset += this.frameSamples;
    }

    this.pending = combined.slice(offset);
    return emitted;
  }

  private consumeAecRef(count: number): Int16Array {
    const out = new Int16Array(count);
    let filled = 0;
    while (filled < count && this.aecRefQueue.length > 0) {
      const head = this.aecRefQueue[0]!;
      const take = Math.min(count - filled, head.length);
      out.set(head.subarray(0, take), filled);
      filled += take;
      this.aecRefSamples -= take;
      if (take === head.length) {
        this.aecRefQueue.shift();
      } else {
        this.aecRefQueue[0] = head.subarray(take);
      }
    }
    return out;
  }
}
