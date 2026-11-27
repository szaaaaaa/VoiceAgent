/**
 * Hybrid dialogue turn-taking & full-duplex interaction classifier (FR-15~FR-20).
 * Combines VAD evidence, Causal Turn Adapter probabilities, assistant playback state,
 * ASR transcript cues, and conservative silence timeout fallback.
 */

export type TurnStrategy = "fixed_silence" | "vad_only" | "smart_turn" | "adapter" | "hybrid";

export type TurnEventKind = "EOT" | "HOLD" | "BACKCHANNEL" | "INTERRUPTION" | "NOISE";

export interface TurnAdapterProbabilities {
  pCompletion: number;
  pFloorTaking: number;
  pNonFloorFeedback: number;
}

export interface TurnObservation {
  vadInSpeech: boolean;
  speechDurationMs: number;
  silenceDurationMs: number;
  assistantSpeaking: boolean;
  transcript: string;
  adapter?: TurnAdapterProbabilities | null;
  rmsDb?: number;
}

export interface TurnDecision {
  event: TurnEventKind;
  confidence: number;
  source: string;
  reason: string;
}

export interface TurnManagerConfig {
  strategy?: TurnStrategy;
  eotThreshold?: number;
  interruptionThreshold?: number;
  backchannelThreshold?: number;
  backchannelMaxDurationMs?: number;
  conservativeSilenceTimeoutMs?: number;
  fixedSilenceMs?: number;
  minSpeechForInterruptMs?: number;
}

const BACKCHANNEL_PHRASES = new Set([
  "ok",
  "okay",
  "yeah",
  "yes",
  "yep",
  "uh-huh",
  "uh huh",
  "mhm",
  "mm",
  "mm-hmm",
  "right",
  "sure",
  "got it",
  "i see",
  "cool",
  "嗯",
  "哦",
  "好",
  "好的",
  "明白",
  "行",
  "对",
]);

const INTERRUPT_CUES = [
  "wait",
  "stop",
  "hold on",
  "hang on",
  "actually",
  "never mind",
  "cancel",
  "change that",
  "no no",
  "等一下",
  "等等",
  "停",
  "先别说",
  "改一下",
  "换个",
];

const HOLD_TRAILING_WORDS = new Set([
  "um",
  "uh",
  "er",
  "and",
  "or",
  "but",
  "so",
  "because",
  "like",
  "well",
  "the",
  "a",
  "to",
  "of",
  "that",
  "嗯",
  "那个",
  "然后",
  "就是",
  "但是",
]);

function normalizeText(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/[.,!?;:。！？，、；：]+$/g, "")
    .replace(/\s+/g, " ");
}

function looksIncompleteHold(transcript: string): boolean {
  const trimmed = transcript.trim();
  if (!trimmed) return false;
  if (/(\.\.\.|……|[,，、\-—])$/.test(trimmed)) return true;
  const words = normalizeText(trimmed).split(" ").filter(Boolean);
  const last = words[words.length - 1];
  return last !== undefined && HOLD_TRAILING_WORDS.has(last);
}

function hasInterruptCue(transcript: string): boolean {
  const norm = normalizeText(transcript);
  if (!norm) return false;
  return INTERRUPT_CUES.some((cue) => norm === cue || norm.startsWith(`${cue} `) || norm.includes(cue));
}

export class HybridTurnManager {
  private strategy: TurnStrategy;
  private readonly eotThreshold: number;
  private readonly interruptionThreshold: number;
  private readonly backchannelThreshold: number;
  private readonly backchannelMaxDurationMs: number;
  private readonly conservativeSilenceTimeoutMs: number;
  private readonly fixedSilenceMs: number;
  private readonly minSpeechForInterruptMs: number;

  constructor(config: TurnManagerConfig = {}) {
    this.strategy = config.strategy ?? "hybrid";
    this.eotThreshold = config.eotThreshold ?? 0.72;
    this.interruptionThreshold = config.interruptionThreshold ?? 0.68;
    this.backchannelThreshold = config.backchannelThreshold ?? 0.6;
    this.backchannelMaxDurationMs = config.backchannelMaxDurationMs ?? 600;
    this.conservativeSilenceTimeoutMs = config.conservativeSilenceTimeoutMs ?? 1100;
    this.fixedSilenceMs = config.fixedSilenceMs ?? 700;
    this.minSpeechForInterruptMs = config.minSpeechForInterruptMs ?? 180;
  }

  setStrategy(strategy: TurnStrategy): void {
    this.strategy = strategy;
  }

  getStrategy(): TurnStrategy {
    return this.strategy;
  }

  evaluate(obs: TurnObservation): TurnDecision {
    if (obs.assistantSpeaking) {
      return this.evaluateDuringPlayback(obs);
    }
    return this.evaluateUserTurn(obs);
  }

  private evaluateDuringPlayback(obs: TurnObservation): TurnDecision {
    const norm = normalizeText(obs.transcript);
    const adapter = obs.adapter;

    // Very short burst with no recognizable speech and low floor-taking probability -> ambient noise.
    if (
      !norm &&
      obs.speechDurationMs < this.minSpeechForInterruptMs &&
      (!adapter || adapter.pFloorTaking < this.interruptionThreshold)
    ) {
      return {
        event: "NOISE",
        confidence: 0.85,
        source: this.strategy,
        reason: "Short non-verbal burst during playback",
      };
    }

    // Explicit barge-in lexical cue immediately takes the floor.
    if (hasInterruptCue(norm)) {
      return {
        event: "INTERRUPTION",
        confidence: 0.95,
        source: "lexical_cue",
        reason: `Explicit interruption phrase detected: "${norm}"`,
      };
    }

    // Short backchannel phrase ("ok", "uh-huh", "嗯") preserves the current reply.
    if (
      norm &&
      BACKCHANNEL_PHRASES.has(norm) &&
      obs.speechDurationMs <= this.backchannelMaxDurationMs &&
      (!adapter || adapter.pFloorTaking < this.interruptionThreshold)
    ) {
      return {
        event: "BACKCHANNEL",
        confidence: adapter ? Math.max(0.75, adapter.pNonFloorFeedback) : 0.8,
        source: adapter ? "hybrid_backchannel" : "lexical_backchannel",
        reason: `Short acknowledgement "${norm}"`,
      };
    }

    if (adapter && (this.strategy === "adapter" || this.strategy === "hybrid")) {
      if (adapter.pFloorTaking >= this.interruptionThreshold) {
        return {
          event: "INTERRUPTION",
          confidence: adapter.pFloorTaking,
          source: "turn_adapter",
          reason: `Adapter floor-taking probability ${adapter.pFloorTaking.toFixed(2)} >= ${this.interruptionThreshold}`,
        };
      }
      if (adapter.pNonFloorFeedback >= this.backchannelThreshold && obs.speechDurationMs <= this.backchannelMaxDurationMs) {
        return {
          event: "BACKCHANNEL",
          confidence: adapter.pNonFloorFeedback,
          source: "turn_adapter",
          reason: `Adapter non-floor feedback ${adapter.pNonFloorFeedback.toFixed(2)}`,
        };
      }
    }

    // Sustained intelligible speech during playback indicates a genuine interruption.
    const wordCount = norm ? norm.split(" ").filter(Boolean).length : 0;
    if (wordCount >= 3 || (obs.speechDurationMs > this.backchannelMaxDurationMs && norm.length > 0)) {
      return {
        event: "INTERRUPTION",
        confidence: 0.82,
        source: "sustained_speech",
        reason: `Sustained user utterance (${obs.speechDurationMs}ms, ${wordCount} words) during playback`,
      };
    }

    return {
      event: "NOISE",
      confidence: 0.65,
      source: this.strategy,
      reason: "Insufficient evidence for floor-taking interruption",
    };
  }

  private evaluateUserTurn(obs: TurnObservation): TurnDecision {
    const norm = normalizeText(obs.transcript);

    if (!norm && !obs.vadInSpeech) {
      return {
        event: "NOISE",
        confidence: 0.9,
        source: this.strategy,
        reason: "Empty transcript with no active speech",
      };
    }

    if (obs.vadInSpeech) {
      return {
        event: "HOLD",
        confidence: 0.9,
        source: "vad_active",
        reason: "User is still actively speaking",
      };
    }

    // Baseline strategies for controlled TurnBench / ablation experiments (TURN-A0 ~ A5).
    if (this.strategy === "fixed_silence" || this.strategy === "vad_only") {
      if (obs.silenceDurationMs >= this.fixedSilenceMs) {
        return {
          event: "EOT",
          confidence: 0.8,
          source: this.strategy,
          reason: `Silence ${obs.silenceDurationMs}ms >= ${this.fixedSilenceMs}ms`,
        };
      }
      return {
        event: "HOLD",
        confidence: 0.7,
        source: this.strategy,
        reason: "Waiting for fixed silence window",
      };
    }

    const incomplete = looksIncompleteHold(obs.transcript);

    // Conservative silence timeout fallback prevents indefinite waiting (FR-20).
    if (obs.silenceDurationMs >= this.conservativeSilenceTimeoutMs) {
      return {
        event: "EOT",
        confidence: 0.75,
        source: "conservative_timeout",
        reason: `Conservative silence timeout reached (${obs.silenceDurationMs}ms)`,
      };
    }

    if (this.strategy === "smart_turn") {
      const targetSilence = incomplete ? this.conservativeSilenceTimeoutMs : 450;
      return obs.silenceDurationMs >= targetSilence
        ? { event: "EOT", confidence: 0.82, source: "smart_turn", reason: "Syntactic & pause heuristic satisfied" }
        : { event: "HOLD", confidence: 0.78, source: "smart_turn", reason: "Mid-sentence hold detected" };
    }

    const adapter = obs.adapter;
    if (adapter && (this.strategy === "adapter" || this.strategy === "hybrid")) {
      const effectiveThreshold = this.strategy === "hybrid" && incomplete ? Math.min(0.92, this.eotThreshold + 0.14) : this.eotThreshold;
      if (adapter.pCompletion >= effectiveThreshold && obs.silenceDurationMs >= 160) {
        return {
          event: "EOT",
          confidence: adapter.pCompletion,
          source: this.strategy === "hybrid" ? "hybrid_adapter" : "turn_adapter",
          reason: `pCompletion=${adapter.pCompletion.toFixed(2)} >= ${effectiveThreshold.toFixed(2)}`,
        };
      }
      return {
        event: "HOLD",
        confidence: 1 - adapter.pCompletion,
        source: this.strategy === "hybrid" ? "hybrid_adapter" : "turn_adapter",
        reason: incomplete ? "Mid-utterance hesitation pause" : "Adapter predicts user may continue",
      };
    }

    // Hybrid fallback when adapter is not loaded yet.
    const requiredSilence = incomplete ? this.conservativeSilenceTimeoutMs : this.fixedSilenceMs;
    if (obs.silenceDurationMs >= requiredSilence) {
      return {
        event: "EOT",
        confidence: 0.78,
        source: "hybrid_fallback",
        reason: `Fallback silence threshold met (${obs.silenceDurationMs}ms)`,
      };
    }
    return {
      event: "HOLD",
      confidence: 0.72,
      source: "hybrid_fallback",
      reason: incomplete ? "Holding for unfinished clause" : "Awaiting end-of-turn pause",
    };
  }
}
