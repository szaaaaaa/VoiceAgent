/**
 * Playback-aware conversation history store (PRD §7.4).
 * Separately tracks user transcripts, full assistant replies, actually played audible prefixes,
 * executed tool calls/results (never dropped on interruption), and uncommitted speculative drafts.
 */

export type ToolExecutionStatus = "running" | "ok" | "error" | "denied" | "cancelled";

export interface ExecutedToolRecord {
  toolCallId: string;
  toolName: string;
  argsSummary: string;
  status: ToolExecutionStatus;
  sideEffectCommitted: boolean;
  result: string;
}

export interface ConversationTurnRecord {
  turnId: string;
  userTranscript: string;
  assistantFullReply: string;
  assistantSpeakableReply: string;
  assistantAudibleReply: string;
  assistantUnheardSuffix: string;
  interrupted: boolean;
  toolCalls: ExecutedToolRecord[];
}

export interface TurnContextSnapshot {
  turnId: string;
  heardPrefix: string;
  unheardSuffix: string;
  committedTools: ExecutedToolRecord[];
  cancelledOrDeniedTools: ExecutedToolRecord[];
}

export interface AgentHistoryMessage {
  role: "user" | "assistant" | "tool";
  content: string;
  toolName?: string;
  toolCallId?: string;
}

export class PlaybackAwareHistory {
  private readonly turns: ConversationTurnRecord[] = [];
  private readonly turnIndex = new Map<string, ConversationTurnRecord>();
  private speculativeDraft: { turnId: string; generationId: string; text: string } | null = null;

  recordUserTurn(turnId: string, userTranscript: string): ConversationTurnRecord {
    const existing = this.turnIndex.get(turnId);
    if (existing) {
      existing.userTranscript = userTranscript;
      return existing;
    }
    const record: ConversationTurnRecord = {
      turnId,
      userTranscript,
      assistantFullReply: "",
      assistantSpeakableReply: "",
      assistantAudibleReply: "",
      assistantUnheardSuffix: "",
      interrupted: false,
      toolCalls: [],
    };
    this.turns.push(record);
    this.turnIndex.set(turnId, record);
    return record;
  }

  recordAssistantReply(turnId: string, fullReply: string, speakableReply: string): void {
    const turn = this.turnIndex.get(turnId);
    if (!turn) return;
    turn.assistantFullReply = fullReply;
    turn.assistantSpeakableReply = speakableReply;
    if (!turn.interrupted) {
      turn.assistantAudibleReply = speakableReply;
    }
  }

  /** Preserves every tool lifecycle transition so barge-in never misreports side effects. */
  recordToolExecution(
    turnId: string,
    tool: Omit<ExecutedToolRecord, "sideEffectCommitted"> & { sideEffectCommitted?: boolean },
  ): void {
    const turn = this.turnIndex.get(turnId);
    if (!turn) return;
    const committed = tool.sideEffectCommitted ?? tool.status === "ok";
    const idx = turn.toolCalls.findIndex((t) => t.toolCallId === tool.toolCallId);
    const entry: ExecutedToolRecord = { ...tool, sideEffectCommitted: committed };
    if (idx >= 0) {
      turn.toolCalls[idx] = entry;
    } else {
      turn.toolCalls.push(entry);
    }
  }

  /** Truncates audible text to what the user heard while preserving unheard suffix & tool ledger. */
  markInterrupted(turnId: string, playedCharIndex: number): void {
    const turn = this.turnIndex.get(turnId);
    if (!turn) return;
    turn.interrupted = true;
    const clamped = Math.max(0, Math.min(turn.assistantSpeakableReply.length, playedCharIndex));
    turn.assistantAudibleReply = turn.assistantSpeakableReply.slice(0, clamped).trim();
    turn.assistantUnheardSuffix = turn.assistantSpeakableReply.slice(clamped).trim();
    for (const t of turn.toolCalls) {
      if (t.status === "running") {
        t.status = "cancelled";
        t.result = "Cancelled due to user interruption prior to completion";
      }
    }
  }

  /** Inspects the synchronized 3-way state (playback cursor + dialogue text + tool side effects). */
  getTurnSyncSnapshot(turnId: string): TurnContextSnapshot | undefined {
    const turn = this.turnIndex.get(turnId);
    if (!turn) return undefined;
    return {
      turnId: turn.turnId,
      heardPrefix: turn.assistantAudibleReply,
      unheardSuffix: turn.assistantUnheardSuffix,
      committedTools: turn.toolCalls.filter((t) => t.sideEffectCommitted),
      cancelledOrDeniedTools: turn.toolCalls.filter((t) => !t.sideEffectCommitted),
    };
  }

  setSpeculativeDraft(turnId: string, generationId: string, text: string): void {
    this.speculativeDraft = { turnId, generationId, text };
  }

  getSpeculativeDraft(): { turnId: string; generationId: string; text: string } | null {
    return this.speculativeDraft;
  }

  clearSpeculativeDraft(): void {
    this.speculativeDraft = null;
  }

  getTurns(): readonly ConversationTurnRecord[] {
    return this.turns;
  }

  /**
   * Produces LLM-ready history where interrupted replies accurately reflect the audible prefix
   * heard by the user while preserving 100% of executed tool calls and their side-effect results.
   */
  toAgentMessages(): AgentHistoryMessage[] {
    const messages: AgentHistoryMessage[] = [];
    for (const turn of this.turns) {
      messages.push({ role: "user", content: turn.userTranscript });
      for (const tool of turn.toolCalls) {
        messages.push({
          role: "tool",
          toolCallId: tool.toolCallId,
          toolName: tool.toolName,
          content: `[tool:${tool.toolName} status=${tool.status}] ${tool.result}`,
        });
      }
      if (turn.interrupted) {
        const heard = turn.assistantAudibleReply || "(no audio heard before interruption)";
        messages.push({
          role: "assistant",
          content: `${heard} [interrupted by user before finishing reply]`,
        });
      } else if (turn.assistantFullReply) {
        messages.push({ role: "assistant", content: turn.assistantFullReply });
      }
    }
    return messages;
  }
}
