/**
 * Local WebSocket protocol contracts, 24-byte binary PCM frame codec,
 * and three-level (session / turn / generation) event lifecycle guard.
 */

export const BINARY_MAGIC = 0x5641; // "VA"
export const PROTOCOL_VERSION = 1;
export const SUPPORTED_PROTOCOL_VERSIONS = [1, 2] as const;
export const BINARY_HEADER_BYTES = 24;

export type ClientRole = "controller" | "observer";
export type AudioMode = "local_device" | "ws_pcm";

export enum StreamKind {
  MicInput = 1,
  TtsOutput = 2,
  AecReference = 3,
}

export interface BinaryAudioFrame {
  streamKind: StreamKind;
  seq: number;
  ptsMs: number;
  sampleRate: number;
  generationTag: number;
  pcm16: Int16Array;
}

export type ProtocolEventType =
  | "session.init"
  | "session.reset"
  | "model.switch"
  | "audio.control"
  | "audio.level"
  | "vad.start"
  | "vad.end"
  | "vad.state"
  | "asr.partial"
  | "asr.final"
  | "turn.prediction"
  | "turn.commit"
  | "agent.delta"
  | "tool.start"
  | "tool.end"
  | "approval.request"
  | "approval.response"
  | "tts.synthesize"
  | "tts.chunk"
  | "playback.ack"
  | "playback.cursor"
  | "generation.cancel"
  | "telemetry.gpu"
  | "error";

export interface EventPayloadMap {
  "session.init": {
    sample_rate: number;
    role?: ClientRole;
    audio_mode?: AudioMode;
    input_device?: number | string | null;
    output_device?: number | string | null;
    turn_strategy?: string;
    tts_engine?: string;
    asr_checkpoint?: string | null;
    tts_checkpoint?: string | null;
    adapter_checkpoint?: string | null;
  };
  "session.reset": { reason?: string };
  "model.switch": {
    target: "asr" | "tts" | "turn_adapter";
    checkpoint_path: string | null;
    checkpoint_id?: string;
  };
  "audio.control": {
    action: "start_mic" | "stop_mic" | "ptt_down" | "ptt_up" | "duck" | "pause" | "resume";
    duck_gain?: number;
    input_device?: number | string | null;
    output_device?: number | string | null;
  };
  "audio.level": {
    mic_rms_db: number;
    spk_rms_db: number;
    aec_active: boolean;
  };
  "vad.start": {
    vad_prob: number;
    speech_start_pts_ms: number;
    preroll_included_ms: number;
  };
  "vad.end": {
    vad_prob: number;
    silence_start_pts_ms: number;
    speech_duration_ms: number;
  };
  "vad.state": {
    is_speech: boolean;
    vad_prob: number;
    pts_ms: number;
    event: "start" | "ongoing" | "end";
  };
  "asr.partial": {
    text: string;
    stability: number;
    audio_pts_ms: number;
    compute_ms: number;
  };
  "asr.final": {
    text: string;
    audio_start_pts_ms: number;
    audio_end_pts_ms: number;
    compute_ms: number;
    model_id?: string;
  };
  "turn.prediction": {
    p_completion: number;
    p_floor_taking: number;
    p_non_floor_feedback: number;
    assistant_speaking: boolean;
    adapter_version: string;
    recommended_event?: string;
    step_ms?: number;
  };
  "turn.commit": {
    final_transcript: string;
    decision_source: string;
    decision_latency_ms: number;
    speculative_hit: boolean;
  };
  "agent.delta": {
    text_delta: string;
    speakable_delta: string;
    is_speculative: boolean;
  };
  "tool.start": {
    tool_call_id: string;
    tool_name: string;
    args_summary: string;
    risk_level: "readonly" | "write" | "high_risk";
  };
  "tool.end": {
    tool_call_id: string;
    status: "ok" | "error" | "denied";
    duration_ms: number;
    result_preview: string;
  };
  "approval.request": {
    approval_id: string;
    tool_name: string;
    target_path: string | null;
    command: string | null;
    impact_summary: string;
    expires_in_ms: number;
  };
  "approval.response": {
    approval_id: string;
    approved: boolean;
    source: "ui" | "voice" | "timeout" | "policy";
    reason: string;
  };
  "tts.synthesize": {
    sentence_index: number;
    text: string;
    char_start: number;
    char_end: number;
    generation_tag: number;
    speaker_profile: string;
  };
  "tts.chunk": {
    sentence_index: number;
    chunk_index: number;
    char_start: number;
    char_end: number;
    duration_ms: number;
    ttfa_ms: number;
    generation_tag: number;
    sample_rate: number;
  };
  "playback.ack": {
    playback_state: "playing" | "ducked" | "paused" | "idle";
    played_char_index: number;
    played_duration_ms: number;
  };
  "playback.cursor": {
    mode: "playing" | "ducked" | "paused" | "idle";
    played_char_index: number;
    played_duration_ms: number;
    generation_id: string | null;
    turn_id: string | null;
  };
  "generation.cancel": {
    target_generation_id: string;
    generation_tag: number;
    reason: string;
    audible_cursor_chars: number;
  };
  "telemetry.gpu": {
    device: string;
    allocated_mb: number;
    reserved_mb: number;
    total_mb: number;
    active_models: Record<string, unknown>;
  };
  error: {
    module: string;
    code: string;
    message: string;
    recoverable: boolean;
    fallback_action: "none" | "retry" | "text_only" | "lightweight_model";
  };
}

export interface RegistryCheckpointEntry {
  checkpoint_id: string;
  model_type: "asr" | "turn_adapter" | "tts";
  path: string | null;
  label: string;
  experiment_id: string;
  metrics: Record<string, number>;
  promoted: boolean;
  builtin?: boolean;
  notes?: string;
}

export interface RegistrySnapshotResponse {
  active: Record<string, string | null>;
  active_ids: Record<string, string>;
  turn_strategy: string;
  tts_engine: string;
  checkpoints: RegistryCheckpointEntry[];
  models: Record<string, unknown>;
}

export interface AbAsrCandidateResult {
  checkpoint_id: string;
  label: string;
  transcript: string;
  compute_ms: number;
  rtf: number;
  wer: number | null;
  cer: number | null;
}

export interface AbTtsCandidateResult {
  slot_id: string;
  display_label: string;
  checkpoint_id: string | null;
  wav_base64: string;
  sample_rate: number;
  duration_ms: number;
  ttfa_ms: number;
  rtf: number;
}

export interface ExperimentReportItem {
  id: string;
  title: string;
  markdown_file: string | null;
  manifest_file: string | null;
  markdown: string;
  manifest: Record<string, unknown> | null;
}

export interface EventEnvelope<T extends ProtocolEventType = ProtocolEventType> {
  v: 1 | 2;
  type: T;
  event_id: string;
  timestamp_ms: number;
  session_id: string;
  turn_id: string | null;
  generation_id: string | null;
  payload: EventPayloadMap[T];
}

let eventCounter = 0;

export function createEvent<T extends ProtocolEventType>(
  type: T,
  ids: { sessionId: string; turnId?: string | null; generationId?: string | null },
  payload: EventPayloadMap[T],
  timestampMs = Date.now(),
): EventEnvelope<T> {
  eventCounter += 1;
  return {
    v: PROTOCOL_VERSION,
    type,
    event_id: `evt_${timestampMs.toString(36)}_${eventCounter.toString(36)}`,
    timestamp_ms: timestampMs,
    session_id: ids.sessionId,
    turn_id: ids.turnId ?? null,
    generation_id: ids.generationId ?? null,
    payload,
  };
}

export function parseEventEnvelope(raw: string): EventEnvelope {
  const parsed: unknown = JSON.parse(raw);
  if (typeof parsed !== "object" || parsed === null) {
    throw new Error("Protocol error: envelope must be a JSON object");
  }
  const obj = parsed as Record<string, unknown>;
  if (
    (obj.v !== 1 && obj.v !== 2) ||
    typeof obj.type !== "string" ||
    typeof obj.session_id !== "string"
  ) {
    throw new Error("Protocol error: invalid version, type, or session_id");
  }
  if (typeof obj.payload !== "object" || obj.payload === null) {
    throw new Error("Protocol error: payload must be an object");
  }
  return obj as unknown as EventEnvelope;
}

export function encodeAudioFrame(frame: BinaryAudioFrame): Uint8Array {
  const pcmBytes = frame.pcm16.byteLength;
  const out = new Uint8Array(BINARY_HEADER_BYTES + pcmBytes);
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
  view.setUint16(0, BINARY_MAGIC, true);
  view.setUint8(2, PROTOCOL_VERSION);
  view.setUint8(3, frame.streamKind);
  view.setUint32(4, frame.seq >>> 0, true);
  view.setFloat64(8, frame.ptsMs, true);
  view.setUint32(16, frame.sampleRate >>> 0, true);
  view.setUint32(20, frame.generationTag >>> 0, true);

  // Copy samples explicitly in little-endian to stay portable across host endianness.
  for (let i = 0; i < frame.pcm16.length; i += 1) {
    view.setInt16(BINARY_HEADER_BYTES + i * 2, frame.pcm16[i]!, true);
  }
  return out;
}

export function decodeAudioFrame(buffer: ArrayBuffer | Uint8Array): BinaryAudioFrame {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  if (bytes.byteLength < BINARY_HEADER_BYTES) {
    throw new Error(`Protocol error: binary frame too short (${bytes.byteLength} bytes)`);
  }
  const payloadBytes = bytes.byteLength - BINARY_HEADER_BYTES;
  if (payloadBytes % 2 !== 0) {
    throw new Error(`Protocol error: PCM16 payload byte length must be even (${payloadBytes})`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = view.getUint16(0, true);
  if (magic !== BINARY_MAGIC) {
    throw new Error(`Protocol error: invalid magic 0x${magic.toString(16)}`);
  }
  const version = view.getUint8(2);
  if (version !== 1 && version !== 2) {
    throw new Error(`Protocol error: unsupported version ${version}`);
  }
  const streamKind = view.getUint8(3) as StreamKind;
  if (
    streamKind !== StreamKind.MicInput &&
    streamKind !== StreamKind.TtsOutput &&
    streamKind !== StreamKind.AecReference
  ) {
    throw new Error(`Protocol error: unknown stream kind ${streamKind}`);
  }
  const seq = view.getUint32(4, true);
  const ptsMs = view.getFloat64(8, true);
  const sampleRate = view.getUint32(16, true);
  const generationTag = view.getUint32(20, true);
  const sampleCount = payloadBytes / 2;
  const pcm16 = new Int16Array(sampleCount);
  for (let i = 0; i < sampleCount; i += 1) {
    pcm16[i] = view.getInt16(BINARY_HEADER_BYTES + i * 2, true);
  }
  return { streamKind, seq, ptsMs, sampleRate, generationTag, pcm16 };
}

/**
 * Tracks active session / turn / generation IDs so stale audio chunks or deltas
 * from a cancelled generation can never leak into a new turn.
 */
export class GenerationTracker {
  private sessionId: string;
  private turnSeq = 0;
  private genSeq = 0;
  private currentTurnId: string | null = null;
  private currentGenerationId: string | null = null;
  private currentGenerationTag = 0;
  private readonly cancelledIds = new Set<string>();
  private readonly cancelledTags = new Set<number>();
  private readonly maxCancelledHistory: number;

  constructor(sessionId: string, maxCancelledHistory = 256) {
    this.sessionId = sessionId;
    this.maxCancelledHistory = maxCancelledHistory;
  }

  get ids(): {
    sessionId: string;
    turnId: string | null;
    generationId: string | null;
    generationTag: number;
  } {
    return {
      sessionId: this.sessionId,
      turnId: this.currentTurnId,
      generationId: this.currentGenerationId,
      generationTag: this.currentGenerationTag,
    };
  }

  resetSession(newSessionId: string): void {
    this.sessionId = newSessionId;
    this.turnSeq = 0;
    this.genSeq = 0;
    this.currentTurnId = null;
    this.currentGenerationId = null;
    this.currentGenerationTag = 0;
    this.cancelledIds.clear();
    this.cancelledTags.clear();
  }

  startTurn(): string {
    this.turnSeq += 1;
    this.currentTurnId = `turn_${String(this.turnSeq).padStart(4, "0")}`;
    this.currentGenerationId = null;
    this.currentGenerationTag = 0;
    return this.currentTurnId;
  }

  startGeneration(): { turnId: string; generationId: string; generationTag: number } {
    const turnId = this.currentTurnId ?? this.startTurn();
    this.genSeq += 1;
    const generationTag = this.genSeq;
    const generationId = `${turnId}_gen_${generationTag}`;
    this.currentGenerationId = generationId;
    this.currentGenerationTag = generationTag;
    return { turnId, generationId, generationTag };
  }

  cancelCurrentGeneration(): { generationId: string; generationTag: number } | null {
    if (!this.currentGenerationId) return null;
    const target = {
      generationId: this.currentGenerationId,
      generationTag: this.currentGenerationTag,
    };
    this.recordCancelled(target.generationId, target.generationTag);
    this.currentGenerationId = null;
    this.currentGenerationTag = 0;
    return target;
  }

  cancelGeneration(generationId: string, generationTag?: number): void {
    this.recordCancelled(generationId, generationTag);
    if (this.currentGenerationId === generationId) {
      this.currentGenerationId = null;
      this.currentGenerationTag = 0;
    }
  }

  isEventActive(envelope: Pick<EventEnvelope, "session_id" | "turn_id" | "generation_id">): boolean {
    if (envelope.session_id !== this.sessionId) return false;
    if (envelope.generation_id !== null) {
      if (this.cancelledIds.has(envelope.generation_id)) return false;
      if (this.currentGenerationId !== null && envelope.generation_id !== this.currentGenerationId) {
        return false;
      }
    }
    if (envelope.turn_id !== null && this.currentTurnId !== null && envelope.turn_id !== this.currentTurnId) {
      return false;
    }
    return true;
  }

  isGenerationTagActive(generationTag: number): boolean {
    if (generationTag === 0) return true;
    if (this.cancelledTags.has(generationTag)) return false;
    return this.currentGenerationTag === generationTag;
  }

  private recordCancelled(generationId: string, generationTag?: number): void {
    this.cancelledIds.add(generationId);
    if (this.cancelledIds.size > this.maxCancelledHistory) {
      const oldest = this.cancelledIds.values().next().value;
      if (oldest !== undefined) this.cancelledIds.delete(oldest);
    }
    if (generationTag !== undefined && generationTag > 0) {
      this.cancelledTags.add(generationTag);
      if (this.cancelledTags.size > this.maxCancelledHistory) {
        const oldestTag = this.cancelledTags.values().next().value;
        if (oldestTag !== undefined) this.cancelledTags.delete(oldestTag);
      }
    }
  }
}
