/**
 * Hysteresis VAD state tracker for Silero VAD probabilities (PRD §4.3).
 * Tracks speech/silence boundaries without unilaterally committing dialogue turns.
 */

export interface VadConfig {
  thresholdStart?: number;
  thresholdEnd?: number;
  minSpeechDurationMs?: number;
  minSilenceDurationMs?: number;
}

export interface VadTransition {
  event: "vad.start" | "vad.end";
  ptsMs: number;
  vadProb: number;
  speechDurationMs: number;
}

export interface VadStateSnapshot {
  inSpeech: boolean;
  lastProb: number;
  speechDurationMs: number;
  silenceDurationMs: number;
}

export class VadStateMachine {
  private readonly thresholdStart: number;
  private readonly thresholdEnd: number;
  private readonly minSpeechDurationMs: number;
  private readonly minSilenceDurationMs: number;

  private inSpeech = false;
  private candidateSpeechStartMs: number | null = null;
  private confirmedSpeechStartMs: number | null = null;
  private candidateSilenceStartMs: number | null = null;
  private lastSilenceStartMs: number | null = null;
  private lastProb = 0;
  private lastPtsMs = 0;

  constructor(config: VadConfig = {}) {
    this.thresholdStart = config.thresholdStart ?? 0.5;
    this.thresholdEnd = config.thresholdEnd ?? 0.35;
    this.minSpeechDurationMs = config.minSpeechDurationMs ?? 120;
    this.minSilenceDurationMs = config.minSilenceDurationMs ?? 200;
  }

  update(prob: number, ptsMs: number): VadTransition | null {
    this.lastProb = prob;
    this.lastPtsMs = ptsMs;

    if (!this.inSpeech) {
      if (prob >= this.thresholdStart) {
        if (this.candidateSpeechStartMs === null) {
          this.candidateSpeechStartMs = ptsMs;
        }
        if (ptsMs - this.candidateSpeechStartMs >= this.minSpeechDurationMs) {
          this.inSpeech = true;
          this.confirmedSpeechStartMs = this.candidateSpeechStartMs;
          this.candidateSilenceStartMs = null;
          this.lastSilenceStartMs = null;
          return {
            event: "vad.start",
            ptsMs: this.confirmedSpeechStartMs,
            vadProb: prob,
            speechDurationMs: ptsMs - this.confirmedSpeechStartMs,
          };
        }
      } else {
        this.candidateSpeechStartMs = null;
        if (this.lastSilenceStartMs === null) {
          this.lastSilenceStartMs = ptsMs;
        }
      }
      return null;
    }

    // Currently in speech.
    if (prob <= this.thresholdEnd) {
      if (this.candidateSilenceStartMs === null) {
        this.candidateSilenceStartMs = ptsMs;
      }
      if (ptsMs - this.candidateSilenceStartMs >= this.minSilenceDurationMs) {
        const speechDurationMs =
          this.confirmedSpeechStartMs !== null
            ? Math.max(0, this.candidateSilenceStartMs - this.confirmedSpeechStartMs)
            : 0;
        const silenceStart = this.candidateSilenceStartMs;
        this.inSpeech = false;
        this.candidateSpeechStartMs = null;
        this.confirmedSpeechStartMs = null;
        this.candidateSilenceStartMs = null;
        this.lastSilenceStartMs = silenceStart;
        return {
          event: "vad.end",
          ptsMs: silenceStart,
          vadProb: prob,
          speechDurationMs,
        };
      }
    } else {
      this.candidateSilenceStartMs = null;
    }

    return null;
  }

  snapshot(): VadStateSnapshot {
    return {
      inSpeech: this.inSpeech,
      lastProb: this.lastProb,
      speechDurationMs:
        this.inSpeech && this.confirmedSpeechStartMs !== null
          ? Math.max(0, this.lastPtsMs - this.confirmedSpeechStartMs)
          : 0,
      silenceDurationMs:
        !this.inSpeech && this.lastSilenceStartMs !== null
          ? Math.max(0, this.lastPtsMs - this.lastSilenceStartMs)
          : 0,
    };
  }

  reset(): void {
    this.inSpeech = false;
    this.candidateSpeechStartMs = null;
    this.confirmedSpeechStartMs = null;
    this.candidateSilenceStartMs = null;
    this.lastSilenceStartMs = null;
    this.lastProb = 0;
  }
}
