/**
 * Speakable text sanitizer & adaptive sentence buffer for streaming TTS (PRD §4.7, FR-32).
 * Guarantees code blocks, raw tool JSON, and internal reasoning never enter TTS playback.
 */

export interface SentenceSegment {
  readonly sentenceIndex: number;
  readonly text: string;
  readonly charStart: number;
  readonly charEnd: number;
}

export interface SentenceBufferConfig {
  firstClauseMinChars?: number;
  sentenceMinChars?: number;
  maxBufferChars?: number;
}

/** Strips non-speakable artifacts from LLM output while keeping natural prose intact. */
export function sanitizeForSpeech(input: string): string {
  return input
    .replace(/<think>[\s\S]*?<\/think>/gi, " ")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`]+`/g, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/\{[\s\S]*?"(?:tool|command|path|args)"[\s\S]*?\}/g, " ")
    .replace(/[*_~#>|]/g, " ")
    .replace(/\s+/g, " ");
}

/**
 * Incrementally consumes streamed LLM text deltas, suppresses fenced code / <think> blocks
 * across chunk boundaries, and emits low-latency speakable clauses with exact character offsets.
 */
export class SentenceBuffer {
  private readonly firstClauseMinChars: number;
  private readonly sentenceMinChars: number;
  private readonly maxBufferChars: number;

  private rawPending = "";
  private speakablePending = "";
  private inCodeFence = false;
  private inThinkTag = false;
  private sentenceIndex = 0;
  private emittedChars = 0;

  constructor(config: SentenceBufferConfig = {}) {
    this.firstClauseMinChars = config.firstClauseMinChars ?? 12;
    this.sentenceMinChars = config.sentenceMinChars ?? 24;
    this.maxBufferChars = config.maxBufferChars ?? 220;
  }

  pushDelta(rawDelta: string): SentenceSegment[] {
    if (!rawDelta) return [];
    this.rawPending += rawDelta;
    this.consumeRawIntoSpeakable();
    return this.extractReadySegments(false);
  }

  flush(): SentenceSegment[] {
    this.consumeRawIntoSpeakable(true);
    return this.extractReadySegments(true);
  }

  reset(): void {
    this.rawPending = "";
    this.speakablePending = "";
    this.inCodeFence = false;
    this.inThinkTag = false;
    this.sentenceIndex = 0;
    this.emittedChars = 0;
  }

  private consumeRawIntoSpeakable(forceEnd = false): void {
    while (this.rawPending.length > 0) {
      if (this.inCodeFence) {
        const closeIdx = this.rawPending.indexOf("```");
        if (closeIdx === -1) {
          this.rawPending = forceEnd ? "" : this.rawPending.slice(-2);
          return;
        }
        this.rawPending = this.rawPending.slice(closeIdx + 3);
        this.inCodeFence = false;
        continue;
      }

      if (this.inThinkTag) {
        const closeIdx = this.rawPending.toLowerCase().indexOf("</think>");
        if (closeIdx === -1) {
          this.rawPending = forceEnd ? "" : this.rawPending.slice(-7);
          return;
        }
        this.rawPending = this.rawPending.slice(closeIdx + 8);
        this.inThinkTag = false;
        continue;
      }

      const fenceIdx = this.rawPending.indexOf("```");
      const thinkIdx = this.rawPending.toLowerCase().indexOf("<think>");
      let nextSpecial = -1;
      let kind: "fence" | "think" | null = null;

      if (fenceIdx !== -1 && (thinkIdx === -1 || fenceIdx < thinkIdx)) {
        nextSpecial = fenceIdx;
        kind = "fence";
      } else if (thinkIdx !== -1) {
        nextSpecial = thinkIdx;
        kind = "think";
      }

      if (nextSpecial !== -1) {
        const before = this.rawPending.slice(0, nextSpecial);
        this.appendClean(before);
        this.rawPending = this.rawPending.slice(nextSpecial + (kind === "fence" ? 3 : 7));
        if (kind === "fence") this.inCodeFence = true;
        else this.inThinkTag = true;
        continue;
      }

      // Hold back the last 6 chars unless flushing, in case a tag/fence straddles chunk boundaries.
      const keepTail = forceEnd ? 0 : Math.min(6, this.rawPending.length);
      const safeLen = this.rawPending.length - keepTail;
      if (safeLen <= 0) return;
      this.appendClean(this.rawPending.slice(0, safeLen));
      this.rawPending = this.rawPending.slice(safeLen);
    }
  }

  private appendClean(text: string): void {
    const cleaned = sanitizeForSpeech(text);
    if (!cleaned) return;
    if (this.speakablePending.endsWith(" ") && cleaned.startsWith(" ")) {
      this.speakablePending += cleaned.slice(1);
    } else {
      this.speakablePending += cleaned;
    }
  }

  private extractReadySegments(flushAll: boolean): SentenceSegment[] {
    const out: SentenceSegment[] = [];

    while (this.speakablePending.trim().length > 0) {
      const isFirst = this.sentenceIndex === 0;
      const minChars = isFirst ? this.firstClauseMinChars : this.sentenceMinChars;
      const boundaryRegex = isFirst ? /[.!?。！？,，;:；：\n]/ : /[.!?。！？\n]/;

      let cutIdx = -1;
      if (this.speakablePending.length >= minChars) {
        const searchSlice = this.speakablePending.slice(minChars - 1);
        const match = boundaryRegex.exec(searchSlice);
        if (match) {
          cutIdx = minChars - 1 + match.index + 1;
        } else if (this.speakablePending.length >= this.maxBufferChars) {
          const lastSpace = this.speakablePending.lastIndexOf(" ", this.maxBufferChars);
          cutIdx = lastSpace > minChars ? lastSpace + 1 : this.maxBufferChars;
        }
      }

      if (cutIdx === -1) {
        if (!flushAll) break;
        cutIdx = this.speakablePending.length;
      }

      const segmentText = this.speakablePending.slice(0, cutIdx).trim();
      this.speakablePending = this.speakablePending.slice(cutIdx).trimStart();

      if (segmentText.length > 0) {
        const charStart = this.emittedChars;
        const charEnd = charStart + segmentText.length;
        this.emittedChars = charEnd + 1; // +1 accounts for inter-sentence space.
        out.push({
          sentenceIndex: this.sentenceIndex,
          text: segmentText,
          charStart,
          charEnd,
        });
        this.sentenceIndex += 1;
      }
    }

    return out;
  }
}
