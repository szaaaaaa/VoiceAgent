/**
 * Speculative generation coordinator (PRD §4.8, P2).
 * Enforces strict isolation: speculative drafts can NEVER play audio, commit history,
 * or invoke external tools before the final transcript confirms match.
 */

export interface SpeculativeCandidate {
  readonly turnId: string;
  readonly generationId: string;
  readonly promptTranscript: string;
  readonly abortController: AbortController;
  readonly allowExternalTools: false;
  readonly bufferedDeltas: string[];
  committed: boolean;
  discarded: boolean;
}

export function tokenOverlapSimilarity(a: string, b: string): number {
  const norm = (s: string): string[] =>
    s
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s]/gu, " ")
      .split(/\s+/)
      .filter(Boolean);
  const ta = norm(a);
  const tb = norm(b);
  if (ta.length === 0 && tb.length === 0) return 1;
  if (ta.length === 0 || tb.length === 0) return 0;

  const setA = new Set(ta);
  const setB = new Set(tb);
  let overlap = 0;
  for (const w of setA) {
    if (setB.has(w)) overlap += 1;
  }
  return (2 * overlap) / (setA.size + setB.size);
}

export class SpeculativeGenerationController {
  private candidate: SpeculativeCandidate | null = null;
  private readonly minSimilarity: number;
  private readonly triggerProbability: number;

  constructor(options: { triggerProbability?: number; minSimilarity?: number } = {}) {
    this.triggerProbability = options.triggerProbability ?? 0.58;
    this.minSimilarity = options.minSimilarity ?? 0.85;
  }

  shouldTrigger(pCompletion: number, partialTranscript: string): boolean {
    return (
      this.candidate === null &&
      pCompletion >= this.triggerProbability &&
      partialTranscript.trim().length >= 4
    );
  }

  startCandidate(turnId: string, generationId: string, partialTranscript: string): SpeculativeCandidate {
    if (this.candidate) {
      this.discard("replaced_by_new_candidate");
    }
    this.candidate = {
      turnId,
      generationId,
      promptTranscript: partialTranscript.trim(),
      abortController: new AbortController(),
      allowExternalTools: false,
      bufferedDeltas: [],
      committed: false,
      discarded: false,
    };
    return this.candidate;
  }

  bufferDelta(generationId: string, delta: string): boolean {
    if (!this.candidate || this.candidate.generationId !== generationId || this.candidate.discarded) {
      return false;
    }
    this.candidate.bufferedDeltas.push(delta);
    return true;
  }

  confirmFinalTranscript(finalTranscript: string): {
    hit: boolean;
    generationId?: string;
    deltas: string[];
  } {
    if (!this.candidate || this.candidate.discarded) {
      return { hit: false, deltas: [] };
    }
    const sim = tokenOverlapSimilarity(this.candidate.promptTranscript, finalTranscript);
    if (sim >= this.minSimilarity) {
      const active = this.candidate;
      active.committed = true;
      this.candidate = null;
      return { hit: true, generationId: active.generationId, deltas: [...active.bufferedDeltas] };
    }
    this.discard("final_transcript_diverged");
    return { hit: false, deltas: [] };
  }

  discard(reason = "discarded"): void {
    if (!this.candidate) return;
    this.candidate.discarded = true;
    this.candidate.abortController.abort(reason);
    this.candidate = null;
  }

  getActiveCandidate(): Readonly<SpeculativeCandidate> | null {
    return this.candidate;
  }
}
