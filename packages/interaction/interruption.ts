/**
 * Two-stage full-duplex barge-in / interruption controller (PRD §4.5).
 * Stage 1 (≤250ms): Duck or Pause playback immediately on suspected user speech.
 * Stage 2 (≤500ms): Resume on Noise/Backchannel, or Cancel playback + Abort generation on Interruption.
 */

import type { PlaybackController, PlaybackCursor } from "@voice-agent/audio";
import type { MetricsCollector } from "@voice-agent/metrics";
import type { GenerationTracker } from "@voice-agent/protocol";
import type { TurnDecision } from "./turn-manager.js";

export type InterruptionResolution = "resumed" | "cancelled" | "ignored";

export interface InterruptionHooks {
  onAbortGeneration: (generationId: string, generationTag: number, cursor: PlaybackCursor, reason: string) => void;
}

export class InterruptionController {
  private readonly duckMode: "duck" | "pause";
  private readonly duckGain: number;
  private suspectedActive = false;

  constructor(
    private readonly playback: PlaybackController,
    private readonly tracker: GenerationTracker,
    private readonly metrics: MetricsCollector,
    private readonly hooks: InterruptionHooks,
    options: { duckMode?: "duck" | "pause"; duckGain?: number } = {},
  ) {
    this.duckMode = options.duckMode ?? "duck";
    this.duckGain = options.duckGain ?? 0.15;
  }

  isSuspectedSpeechActive(): boolean {
    return this.suspectedActive;
  }

  /** Stage 1: React immediately to VAD speech start while assistant is speaking. */
  onPotentialUserSpeech(nowMs = performance.now()): boolean {
    const mode = this.playback.getMode();
    if (mode !== "playing" && mode !== "ducked") {
      return false;
    }
    if (this.suspectedActive) return true;
    this.suspectedActive = true;

    const turnId = this.tracker.ids.turnId;
    if (turnId) {
      this.metrics.mark(turnId, { interruptSuspectedMs: nowMs });
    }

    if (this.duckMode === "pause") {
      this.playback.pause();
    } else {
      this.playback.duck(this.duckGain);
    }

    if (turnId) {
      this.metrics.mark(turnId, { playbackPausedMs: performance.now() });
    }
    return true;
  }

  /** Stage 2: Resolve suspected barge-in once TurnManager classifies Noise/Backchannel/Interruption. */
  resolve(decision: TurnDecision, nowMs = performance.now()): {
    resolution: InterruptionResolution;
    cursor?: PlaybackCursor;
  } {
    const mode = this.playback.getMode();
    const hasActivePlayback = mode === "playing" || mode === "ducked" || mode === "paused";

    if (!this.suspectedActive && !hasActivePlayback) {
      return { resolution: "ignored" };
    }

    if (decision.event === "NOISE" || decision.event === "BACKCHANNEL") {
      this.suspectedActive = false;
      this.playback.resume();
      return { resolution: "resumed" };
    }

    if (decision.event === "INTERRUPTION") {
      this.suspectedActive = false;
      const turnId = this.tracker.ids.turnId;
      if (turnId) {
        this.metrics.mark(turnId, { interruptConfirmedMs: nowMs });
      }

      const activeGen = this.tracker.cancelCurrentGeneration();
      const cursor = this.playback.cancel(activeGen?.generationId);

      if (activeGen) {
        this.hooks.onAbortGeneration(
          activeGen.generationId,
          activeGen.generationTag,
          cursor,
          decision.reason,
        );
      }

      if (turnId) {
        this.metrics.mark(turnId, { replyStoppedMs: performance.now() });
      }
      return { resolution: "cancelled", cursor };
    }

    return { resolution: "ignored" };
  }
}
