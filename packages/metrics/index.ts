/**
 * End-to-end latency, RTF, interruption timing, and VRAM telemetry collector.
 * Strictly separates cold vs hot turns and compute vs queue time (PRD §8.1).
 */

export interface TurnLatencyRecord {
  turnId: string;
  coldStart: boolean;
  usedTool: boolean;
  speechEndMs?: number;
  asrFinalMs?: number;
  turnCommitMs?: number;
  llmFirstTokenMs?: number;
  ttsFirstChunkMs?: number;
  audiblePlayMs?: number;
  interruptSuspectedMs?: number;
  playbackPausedMs?: number;
  interruptConfirmedMs?: number;
  replyStoppedMs?: number;
  asrComputeMs?: number;
  ttsComputeMs?: number;
  queueWaitMs?: number;
  vramUsedMb?: number;
}

export interface PercentileStats {
  count: number;
  p50: number | null;
  p95: number | null;
  max: number | null;
}

export interface MetricsSummary {
  totalTurns: number;
  coldTurns: number;
  hotTurns: number;
  e2eAudibleLatencyMs: PercentileStats;
  hotE2eAudibleLatencyMs: PercentileStats;
  interruptPauseLatencyMs: PercentileStats;
  interruptStopLatencyMs: PercentileStats;
  peakVramMb: number | null;
}

export function computePercentiles(values: number[]): PercentileStats {
  if (values.length === 0) {
    return { count: 0, p50: null, p95: null, max: null };
  }
  const sorted = [...values].sort((a, b) => a - b);
  const pick = (p: number): number => {
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
    return sorted[idx]!;
  };
  return {
    count: sorted.length,
    p50: pick(0.5),
    p95: pick(0.95),
    max: sorted[sorted.length - 1]!,
  };
}

export class MetricsCollector {
  private readonly records = new Map<string, TurnLatencyRecord>();
  private readonly order: string[] = [];
  private readonly maxHistory: number;
  private peakVramMb: number | null = null;
  private seenFirstTurn = false;

  constructor(maxHistory = 1000) {
    this.maxHistory = maxHistory;
  }

  startTurn(turnId: string, usedTool = false): TurnLatencyRecord {
    const record: TurnLatencyRecord = {
      turnId,
      coldStart: !this.seenFirstTurn,
      usedTool,
    };
    this.seenFirstTurn = true;
    this.records.set(turnId, record);
    this.order.push(turnId);
    if (this.order.length > this.maxHistory) {
      const evicted = this.order.shift();
      if (evicted !== undefined) this.records.delete(evicted);
    }
    return record;
  }

  mark(turnId: string, patch: Partial<Omit<TurnLatencyRecord, "turnId">>): TurnLatencyRecord {
    const record = this.records.get(turnId) ?? this.startTurn(turnId);
    Object.assign(record, patch);
    if (patch.vramUsedMb !== undefined) {
      this.peakVramMb =
        this.peakVramMb === null ? patch.vramUsedMb : Math.max(this.peakVramMb, patch.vramUsedMb);
    }
    return record;
  }

  getTurn(turnId: string): TurnLatencyRecord | undefined {
    return this.records.get(turnId);
  }

  summarize(): MetricsSummary {
    const all = Array.from(this.records.values());
    const e2eAll: number[] = [];
    const e2eHot: number[] = [];
    const pauseLatencies: number[] = [];
    const stopLatencies: number[] = [];
    let coldTurns = 0;

    for (const r of all) {
      if (r.coldStart) coldTurns += 1;
      if (r.speechEndMs !== undefined && r.audiblePlayMs !== undefined && r.audiblePlayMs >= r.speechEndMs) {
        const dt = r.audiblePlayMs - r.speechEndMs;
        e2eAll.push(dt);
        if (!r.coldStart) e2eHot.push(dt);
      }
      if (
        r.interruptSuspectedMs !== undefined &&
        r.playbackPausedMs !== undefined &&
        r.playbackPausedMs >= r.interruptSuspectedMs
      ) {
        pauseLatencies.push(r.playbackPausedMs - r.interruptSuspectedMs);
      }
      if (
        r.interruptConfirmedMs !== undefined &&
        r.replyStoppedMs !== undefined &&
        r.replyStoppedMs >= r.interruptConfirmedMs
      ) {
        stopLatencies.push(r.replyStoppedMs - r.interruptConfirmedMs);
      }
    }

    return {
      totalTurns: all.length,
      coldTurns,
      hotTurns: all.length - coldTurns,
      e2eAudibleLatencyMs: computePercentiles(e2eAll),
      hotE2eAudibleLatencyMs: computePercentiles(e2eHot),
      interruptPauseLatencyMs: computePercentiles(pauseLatencies),
      interruptStopLatencyMs: computePercentiles(stopLatencies),
      peakVramMb: this.peakVramMb,
    };
  }

  exportJsonl(): string {
    return Array.from(this.records.values())
      .map((r) => JSON.stringify(r))
      .join("\n");
  }
}
