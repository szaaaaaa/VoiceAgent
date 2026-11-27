/**
 * Stateful full-duplex voice controller bridging Pi Coding Agent hooks with the
 * local Python VoiceAgent speech kernel (`ws://127.0.0.1:8765/ws/v1/voice`).
 */

import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  GenerationTracker,
  type EventEnvelope,
  type EventPayloadMap,
  type ProtocolEventType,
} from "@voice-agent/protocol";
import { SentenceBuffer, SpeechWsClient } from "@voice-agent/speech";
import {
  HybridTurnManager,
  PlaybackAwareHistory,
  type TurnAdapterProbabilities,
  type TurnContextSnapshot,
  type TurnStrategy,
} from "@voice-agent/interaction";
import { MetricsCollector } from "@voice-agent/metrics";

import type {
  PiAgentSettledEvent,
  PiBeforeAgentStartEvent,
  PiBeforeAgentStartResult,
  PiExtensionAPI,
  PiExtensionContext,
  PiMessageUpdateEvent,
  PiToolCallEvent,
  PiToolCallResult,
  PiToolExecutionEndEvent,
  PiTurnEndEvent,
} from "./pi-host-types.js";

export type VoiceInputMode = "handsfree" | "ptt";
export type VoiceTtsEngine = "qwen3" | "fast";

export interface VoiceTransport {
  isConnected(): boolean;
  connect(): Promise<void>;
  disconnect(): void;
  onEvent(listener: (event: EventEnvelope) => void): () => void;
  onConnectionChange(listener: (connected: boolean) => void): () => void;
  sendEvent<T extends ProtocolEventType>(
    type: T,
    ids: { sessionId: string; turnId?: string | null; generationId?: string | null },
    payload: EventPayloadMap[T],
  ): boolean;
}

export interface PiVoiceControllerOptions {
  wsUrl?: string;
  apiBase?: string;
  sessionId?: string;
  autoSpawnBackend?: boolean;
  turnStrategy?: TurnStrategy;
  ttsEngine?: VoiceTtsEngine;
  speakerProfile?: string;
  voiceApproveTools?: boolean;
  transport?: VoiceTransport;
}

const DESTRUCTIVE_SHELL_PATTERNS = [
  /\brm\s+-[^\n]*[rf]/i,
  /\bremove-item\b[^\n]*-recurse/i,
  /\bdel\s+\/[fsq]/i,
  /\bgit\s+reset\s+--hard/i,
  /\bgit\s+clean\s+-[^\n]*f/i,
  /\bformat\s+[a-z]:/i,
  /\bshutdown\b/i,
];

function isHighRiskToolCall(event: PiToolCallEvent): { risky: boolean; summary: string } {
  if (event.toolName === "bash" || event.toolName === "powershell") {
    const cmd = String(event.input.command ?? "");
    if (DESTRUCTIVE_SHELL_PATTERNS.some((rx) => rx.test(cmd))) {
      return { risky: true, summary: `${event.toolName}: ${cmd.slice(0, 140)}` };
    }
  }
  return { risky: false, summary: "" };
}

function resolveRepoRoot(): string {
  const here = path.dirname(fileURLToPath(import.meta.url));
  return path.resolve(here, "../../..");
}

export class PiVoiceSessionController {
  private readonly pi: PiExtensionAPI;
  private readonly wsUrl: string;
  private readonly apiBase: string;
  private readonly autoSpawnBackend: boolean;
  private readonly transport: VoiceTransport;
  private readonly tracker: GenerationTracker;
  private readonly sentenceBuffer: SentenceBuffer;
  private readonly turnManager: HybridTurnManager;
  readonly history: PlaybackAwareHistory;
  readonly metrics: MetricsCollector;

  private ctx: PiExtensionContext | null = null;
  private backendProc: ChildProcess | null = null;
  private unsubEvents: (() => void) | null = null;
  private unsubConn: (() => void) | null = null;

  private enabled = false;
  private mode: VoiceInputMode = "handsfree";
  private pttActive = false;
  private ttsEngine: VoiceTtsEngine;
  private speakerProfile: string;
  private voiceApproveTools: boolean;

  private inSpeech = false;
  private speechStartMs = 0;
  private silenceStartMs = 0;
  private partialTranscript = "";
  private lastAdapterProb: TurnAdapterProbabilities | null = null;

  private assistantSpeaking = false;
  private assistantDucked = false;
  private playbackCharIndex = 0;
  private activeAssistantRaw = "";
  private activeAssistantSpeakable = "";
  private activeGenerationId: string | null = null;
  private activeGenerationTag = 0;
  private pendingInterruption: TurnContextSnapshot | null = null;
  private bargeInAwaitingNewUtterance = false;
  private lastSubmittedText = "";
  private lastSubmittedAtMs = 0;

  constructor(pi: PiExtensionAPI, options: PiVoiceControllerOptions = {}) {
    this.pi = pi;
    const host = process.env.VOICE_AGENT_WS_HOST || "127.0.0.1";
    const port = process.env.VOICE_AGENT_WS_PORT || "8765";
    this.wsUrl = options.wsUrl ?? `ws://${host}:${port}/ws/v1/voice`;
    this.apiBase = options.apiBase ?? `http://${host}:${port}`;
    this.autoSpawnBackend =
      options.autoSpawnBackend ?? process.env.VOICE_AGENT_SKIP_BACKEND_SPAWN !== "1";

    this.ttsEngine = options.ttsEngine ?? "qwen3";
    this.speakerProfile = options.speakerProfile ?? "default";
    this.voiceApproveTools = options.voiceApproveTools ?? false;

    const sessionId = options.sessionId ?? `pi_voice_${Date.now().toString(36)}`;
    this.tracker = new GenerationTracker(sessionId);
    this.sentenceBuffer = new SentenceBuffer({ firstClauseMinChars: 10, sentenceMinChars: 22 });
    this.turnManager = new HybridTurnManager({ strategy: options.turnStrategy ?? "hybrid" });
    this.history = new PlaybackAwareHistory();
    this.metrics = new MetricsCollector();

    this.transport = options.transport ?? new SpeechWsClient({ url: this.wsUrl });
    this.unsubEvents = this.transport.onEvent((ev) => this.handleBackendEvent(ev));
    this.unsubConn = this.transport.onConnectionChange((connected) => {
      if (!connected && this.enabled) {
        this.updateStatus("⚠ 语音内核重连中");
      } else if (connected && this.enabled) {
        this.sendSessionInit();
        this.refreshStatus();
      }
    });
  }

  setContext(ctx: PiExtensionContext): void {
    this.ctx = ctx;
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  getStateSummary(): Record<string, unknown> {
    const perf = this.metrics.summarize();
    return {
      enabled: this.enabled,
      connected: this.transport.isConnected(),
      mode: this.mode,
      ttsEngine: this.ttsEngine,
      speakerProfile: this.speakerProfile,
      turnStrategy: this.turnManager.getStrategy(),
      voiceApproveTools: this.voiceApproveTools,
      assistantSpeaking: this.assistantSpeaking,
      playbackCharIndex: this.playbackCharIndex,
      e2eP50Ms: perf.e2eAudibleLatencyMs.p50,
    };
  }

  async enable(ctx: PiExtensionContext, mode?: VoiceInputMode): Promise<boolean> {
    this.ctx = ctx;
    if (mode) this.mode = mode;

    if (!this.transport.isConnected()) {
      const ok = await this.ensureBackendAndConnect();
      if (!ok) {
        ctx.ui.notify(
          `VoiceAgent 本地内核 (${this.wsUrl}) 未连接，请先启动 inference.server。`,
          "error",
        );
        this.updateStatus(undefined);
        return false;
      }
    }

    this.enabled = true;
    this.sendSessionInit();
    if (this.mode === "handsfree") {
      this.sendAudioControl("start_mic");
    } else {
      this.pttActive = false;
      this.sendAudioControl("stop_mic");
    }
    this.refreshStatus();
    return true;
  }

  disable(ctx?: PiExtensionContext): void {
    if (ctx) this.ctx = ctx;
    if (!this.enabled && !this.transport.isConnected()) {
      this.updateStatus(undefined);
      return;
    }
    this.enabled = false;
    this.pttActive = false;
    this.cancelActiveSpeech("voice_disabled");
    this.sendAudioControl("stop_mic");
    this.updateStatus(undefined);
  }

  async toggleShortcut(ctx: PiExtensionContext): Promise<void> {
    this.ctx = ctx;
    if (!this.enabled) {
      await this.enable(ctx);
      return;
    }
    if (this.mode === "ptt") {
      this.pttActive = !this.pttActive;
      this.sendAudioControl(this.pttActive ? "ptt_down" : "ptt_up");
      this.refreshStatus();
      return;
    }
    this.disable(ctx);
  }

  setMode(mode: VoiceInputMode): void {
    this.mode = mode;
    if (!this.enabled) return;
    if (mode === "handsfree") {
      this.pttActive = false;
      this.sendAudioControl("start_mic");
    } else {
      this.pttActive = false;
      this.sendAudioControl("stop_mic");
    }
    this.refreshStatus();
  }

  setTtsEngine(engine: VoiceTtsEngine): void {
    this.ttsEngine = engine;
    if (this.transport.isConnected()) {
      this.sendSessionInit();
    }
    this.refreshStatus();
  }

  setSpeakerProfile(profile: string): void {
    this.speakerProfile = profile.trim() || "default";
    this.refreshStatus();
  }

  setVoiceApproveTools(enabled: boolean): void {
    this.voiceApproveTools = enabled;
    this.refreshStatus();
  }

  async handleCommand(argsRaw: string, ctx: PiExtensionContext): Promise<void> {
    this.ctx = ctx;
    const parts = argsRaw.trim().split(/\s+/).filter(Boolean);
    const sub = (parts[0] ?? (this.enabled ? "off" : "on")).toLowerCase();

    if (sub === "on" || sub === "handsfree") {
      const ok = await this.enable(ctx, "handsfree");
      if (ok) ctx.ui.notify("VoiceAgent 免提全双工语音已开启（Alt+V 快速开关）。", "info");
      return;
    }
    if (sub === "off") {
      this.disable(ctx);
      ctx.ui.notify("VoiceAgent 语音模式已关闭。", "info");
      return;
    }
    if (sub === "ptt") {
      const ok = await this.enable(ctx, "ptt");
      if (ok) ctx.ui.notify("VoiceAgent 已切换为按键说话模式（按 Alt+V 开始/结束发言）。", "info");
      return;
    }
    if (sub === "engine") {
      const target = (parts[1] ?? "").toLowerCase();
      if (target !== "qwen3" && target !== "fast") {
        ctx.ui.notify("用法：/voice engine <qwen3|fast>", "warning");
        return;
      }
      this.setTtsEngine(target);
      ctx.ui.notify(`TTS 引擎已切换为：${target}`, "info");
      return;
    }
    if (sub === "speaker") {
      const spk = parts.slice(1).join(" ").trim();
      if (!spk) {
        ctx.ui.notify(`当前音色：${this.speakerProfile}。用法：/voice speaker <name>`, "info");
        return;
      }
      this.setSpeakerProfile(spk);
      ctx.ui.notify(`TTS 音色已切换为：${this.speakerProfile}`, "info");
      return;
    }
    if (sub === "approve") {
      const next = parts[1] ? parts[1].toLowerCase() === "on" : !this.voiceApproveTools;
      this.setVoiceApproveTools(next);
      ctx.ui.notify(`语音高危工具确认辅助：${next ? "已开启" : "已关闭"}`, "info");
      return;
    }
    if (sub === "status") {
      const s = this.getStateSummary();
      ctx.ui.notify(
        `VoiceAgent 状态 | 启用:${s.enabled} 连接:${s.connected} 模式:${s.mode} TTS:${s.ttsEngine} 音色:${s.speakerProfile} 策略:${s.turnStrategy}`,
        "info",
      );
      return;
    }

    ctx.ui.notify(
      "用法：/voice [on|off|handsfree|ptt|engine <qwen3|fast>|speaker <id>|approve <on|off>|status]",
      "warning",
    );
  }

  onBeforeAgentStart(_event: PiBeforeAgentStartEvent, ctx: PiExtensionContext): PiBeforeAgentStartResult | void {
    this.ctx = ctx;
    if (!this.enabled) return;

    this.prepareAssistantGeneration();

    const snap = this.pendingInterruption;
    if (!snap) return;
    this.pendingInterruption = null;

    const heard = snap.heardPrefix || "(尚未播放任何语音即被打断)";
    const unheard = snap.unheardSuffix || "(无)";
    const toolNote =
      snap.committedTools.length > 0
        ? `已完成且保持有效的工具调用：${snap.committedTools.map((t) => `${t.toolName}(${t.status})`).join(", ")}。`
        : "此前无额外工具副作用。";

    return {
      message: {
        customType: "voice-interruption-context",
        display: false,
        content: [
          "[Voice Playback State]",
          "上一条助手回复在语音播报中途被用户主动打断：",
          `- 用户实际已听到前缀："${heard}"`,
          `- 用户未听到的剩余内容："${unheard}"`,
          `- ${toolNote}`,
          "请直接基于用户的新指令自然衔接回复，不要重复已播报过的开场白。",
        ].join("\n"),
        details: snap,
      },
    };
  }

  onMessageUpdate(event: PiMessageUpdateEvent, ctx: PiExtensionContext): void {
    this.ctx = ctx;
    if (!this.enabled) return;
    // Strictly consume only user-facing text deltas; thinking_delta and toolcall_delta never enter TTS.
    if (event.assistantMessageEvent.type !== "text_delta") return;
    const delta = event.assistantMessageEvent.delta ?? "";
    if (!delta) return;

    if (!this.activeGenerationId) {
      this.prepareAssistantGeneration();
    }
    const turnId = this.tracker.ids.turnId;
    if (turnId && this.metrics.getTurn(turnId)?.llmFirstTokenMs === undefined) {
      this.metrics.mark(turnId, { llmFirstTokenMs: performance.now() });
    }

    this.activeAssistantRaw += delta;
    const segments = this.sentenceBuffer.pushDelta(delta);
    for (const seg of segments) {
      this.dispatchTtsSegment(seg.sentenceIndex, seg.text, seg.charStart, seg.charEnd);
    }
  }

  onToolExecutionEnd(event: PiToolExecutionEndEvent): void {
    const turnId = this.tracker.ids.turnId;
    if (!turnId) return;
    this.history.recordToolExecution(turnId, {
      toolCallId: event.toolCallId,
      toolName: event.toolName,
      argsSummary: "",
      status: event.isError ? "error" : "ok",
      sideEffectCommitted: !event.isError,
      result: typeof event.result === "string" ? event.result.slice(0, 200) : JSON.stringify(event.result ?? "").slice(0, 200),
    });
  }

  async onToolCall(event: PiToolCallEvent, ctx: PiExtensionContext): Promise<PiToolCallResult | void> {
    this.ctx = ctx;
    if (!this.enabled || !this.voiceApproveTools) return;
    const check = isHighRiskToolCall(event);
    if (!check.risky) return;

    // Announce brief warning before blocking confirmation dialog (fail-closed on timeout).
    this.speakSystemPrompt("检测到高风险命令操作，请在终端确认是否执行。");
    const approved = await ctx.ui.confirm(
      `VoiceAgent 安全确认：${event.toolName}`,
      `即将执行高风险操作：${check.summary}\n（15 秒内未确认将自动拒绝）`,
      { timeout: 15000 },
    );
    if (!approved) {
      return {
        block: true,
        reason: "Blocked by VoiceAgent fail-closed voice confirmation policy.",
      };
    }
  }

  onTurnEnd(event: PiTurnEndEvent, ctx: PiExtensionContext): void {
    this.ctx = ctx;
    if (!this.enabled) return;
    if (event.outcome === "aborted") {
      this.sentenceBuffer.reset();
      return;
    }
    this.flushRemainingSpeech();
  }

  onAgentSettled(event: PiAgentSettledEvent, ctx: PiExtensionContext): void {
    this.ctx = ctx;
    if (!this.enabled) return;
    if (event.aborted) {
      this.sentenceBuffer.reset();
    } else {
      this.flushRemainingSpeech();
    }
    this.refreshStatus();
  }

  shutdown(): void {
    this.disable();
    this.unsubEvents?.();
    this.unsubConn?.();
    this.unsubEvents = null;
    this.unsubConn = null;
    this.transport.disconnect();
    if (this.backendProc && !this.backendProc.killed) {
      this.backendProc.kill();
      this.backendProc = null;
    }
  }

  private prepareAssistantGeneration(): void {
    this.sentenceBuffer.reset();
    this.activeAssistantRaw = "";
    this.activeAssistantSpeakable = "";
    this.playbackCharIndex = 0;
    const gen = this.tracker.startGeneration();
    this.activeGenerationId = gen.generationId;
    this.activeGenerationTag = gen.generationTag;
  }

  private flushRemainingSpeech(): void {
    const trailing = this.sentenceBuffer.flush();
    for (const seg of trailing) {
      this.dispatchTtsSegment(seg.sentenceIndex, seg.text, seg.charStart, seg.charEnd);
    }
    const turnId = this.tracker.ids.turnId;
    if (turnId && this.activeAssistantRaw) {
      this.history.recordAssistantReply(turnId, this.activeAssistantRaw, this.activeAssistantSpeakable);
    }
  }

  private dispatchTtsSegment(
    sentenceIndex: number,
    text: string,
    charStart: number,
    charEnd: number,
  ): void {
    if (!this.activeGenerationId || !this.transport.isConnected()) return;
    this.activeAssistantSpeakable = this.activeAssistantSpeakable
      ? `${this.activeAssistantSpeakable} ${text}`
      : text;
    const turnId = this.tracker.ids.turnId;
    if (turnId) {
      this.history.recordAssistantReply(turnId, this.activeAssistantRaw, this.activeAssistantSpeakable);
    }
    this.transport.sendEvent(
      "tts.synthesize",
      {
        sessionId: this.tracker.ids.sessionId,
        turnId,
        generationId: this.activeGenerationId,
      },
      {
        sentence_index: sentenceIndex,
        text,
        char_start: charStart,
        char_end: charEnd,
        generation_tag: this.activeGenerationTag,
        speaker_profile: this.speakerProfile,
      },
    );
  }

  private speakSystemPrompt(text: string): void {
    if (!this.transport.isConnected()) return;
    this.transport.sendEvent(
      "tts.synthesize",
      {
        sessionId: this.tracker.ids.sessionId,
        turnId: this.tracker.ids.turnId,
        generationId: null,
      },
      {
        sentence_index: 0,
        text,
        char_start: 0,
        char_end: text.length,
        generation_tag: 0,
        speaker_profile: this.speakerProfile,
      },
    );
  }

  private cancelActiveSpeech(reason: string): number {
    const heardChars = this.playbackCharIndex;
    const cancelled = this.tracker.cancelCurrentGeneration();
    if (cancelled && this.transport.isConnected()) {
      this.transport.sendEvent(
        "generation.cancel",
        {
          sessionId: this.tracker.ids.sessionId,
          turnId: this.tracker.ids.turnId,
          generationId: cancelled.generationId,
        },
        {
          target_generation_id: cancelled.generationId,
          generation_tag: cancelled.generationTag,
          reason,
          audible_cursor_chars: heardChars,
        },
      );
    }
    this.sentenceBuffer.reset();
    this.activeGenerationId = null;
    this.activeGenerationTag = 0;
    this.assistantSpeaking = false;
    this.assistantDucked = false;
    return heardChars;
  }

  private handleBackendEvent(event: EventEnvelope): void {
    if (!this.enabled) return;

    if (event.type === "playback.cursor") {
      const p = event.payload as EventPayloadMap["playback.cursor"];
      this.playbackCharIndex = p.played_char_index;
      this.assistantSpeaking = p.mode === "playing" || p.mode === "ducked";
      this.assistantDucked = p.mode === "ducked";
      const turnId = this.tracker.ids.turnId;
      if (turnId && p.mode === "playing" && this.metrics.getTurn(turnId)?.audiblePlayMs === undefined) {
        this.metrics.mark(turnId, { audiblePlayMs: performance.now() });
      }
      this.refreshStatus();
      return;
    }

    if (event.type === "tts.chunk") {
      const turnId = this.tracker.ids.turnId;
      if (turnId && this.metrics.getTurn(turnId)?.ttsFirstChunkMs === undefined) {
        this.metrics.mark(turnId, { ttsFirstChunkMs: performance.now() });
      }
      return;
    }

    if (event.type === "vad.start") {
      const p = event.payload as EventPayloadMap["vad.start"];
      this.inSpeech = true;
      this.speechStartMs = p.speech_start_pts_ms;
      this.silenceStartMs = 0;
      if (this.assistantSpeaking && !this.assistantDucked) {
        // Stage-1 low-latency barge-in avoidance: duck speaker immediately on suspected user speech.
        this.assistantDucked = true;
        this.sendAudioControl("duck", 0.15);
      }
      this.refreshStatus();
      return;
    }

    if (event.type === "vad.end") {
      const p = event.payload as EventPayloadMap["vad.end"];
      this.inSpeech = false;
      this.silenceStartMs = p.silence_start_pts_ms;
      this.evaluatePlaybackOrTurn(p.speech_duration_ms, 240);
      this.refreshStatus();
      return;
    }

    if (event.type === "asr.partial") {
      const p = event.payload as EventPayloadMap["asr.partial"];
      this.partialTranscript = p.text;
      if (this.assistantSpeaking) {
        const speechDur = Math.max(0, p.audio_pts_ms - this.speechStartMs);
        this.evaluatePlaybackOrTurn(speechDur, 0);
      }
      return;
    }

    if (event.type === "turn.prediction") {
      const p = event.payload as EventPayloadMap["turn.prediction"];
      this.lastAdapterProb = {
        pCompletion: p.p_completion,
        pFloorTaking: p.p_floor_taking,
        pNonFloorFeedback: p.p_non_floor_feedback,
      };
      if (this.assistantSpeaking || this.assistantDucked) {
        this.evaluatePlaybackOrTurn(240, this.inSpeech ? 0 : 240);
      }
      return;
    }

    if (event.type === "asr.final") {
      const p = event.payload as EventPayloadMap["asr.final"];
      const finalText = p.text.trim();
      this.partialTranscript = "";
      if (!finalText) return;

      const speechDur = Math.max(0, p.audio_end_pts_ms - p.audio_start_pts_ms);
      if (this.assistantSpeaking || this.assistantDucked) {
        const decision = this.turnManager.evaluate({
          vadInSpeech: false,
          speechDurationMs: speechDur,
          silenceDurationMs: 260,
          assistantSpeaking: true,
          transcript: finalText,
          adapter: this.lastAdapterProb,
        });
        if (
          decision.event === "BACKCHANNEL" ||
          decision.event === "NOISE" ||
          decision.action === "resume_playback"
        ) {
          this.assistantDucked = false;
          this.sendAudioControl("resume");
          this.refreshStatus();
          return;
        }
        this.triggerBargeInInterrupt(decision.reason);
      }

      this.submitFinalUserTranscript(finalText);
    }
  }

  private evaluatePlaybackOrTurn(speechDurationMs: number, silenceDurationMs: number): void {
    if (!this.assistantSpeaking && !this.assistantDucked) return;
    const decision = this.turnManager.evaluate({
      vadInSpeech: this.inSpeech,
      speechDurationMs,
      silenceDurationMs,
      assistantSpeaking: true,
      transcript: this.partialTranscript,
      adapter: this.lastAdapterProb,
    });
    if (
      decision.event === "INTERRUPTION" ||
      decision.action === "cancel_and_regenerate" ||
      decision.action === "cancel_and_listen"
    ) {
      this.triggerBargeInInterrupt(decision.reason);
    } else if (
      !this.inSpeech &&
      (decision.event === "BACKCHANNEL" ||
        decision.event === "NOISE" ||
        decision.action === "resume_playback")
    ) {
      this.assistantDucked = false;
      this.sendAudioControl("resume");
    }
  }

  private triggerBargeInInterrupt(reason: string): void {
    const prevTurnId = this.tracker.ids.turnId;
    const heardChars = this.cancelActiveSpeech(reason);
    if (prevTurnId) {
      this.history.markInterrupted(prevTurnId, heardChars);
      this.pendingInterruption = this.history.getTurnSyncSnapshot(prevTurnId) ?? null;
    }
    this.bargeInAwaitingNewUtterance = true;
    if (this.ctx && !this.ctx.isIdle()) {
      this.ctx.abort();
    }
    this.refreshStatus();
  }

  private submitFinalUserTranscript(transcript: string): void {
    const now = performance.now();
    if (transcript === this.lastSubmittedText && now - this.lastSubmittedAtMs < 600) {
      return;
    }
    this.lastSubmittedText = transcript;
    this.lastSubmittedAtMs = now;

    const newTurnId = this.tracker.startTurn();
    this.metrics.startTurn(newTurnId);
    this.metrics.mark(newTurnId, { speechEndMs: now, turnCommitMs: now });
    this.history.recordUserTurn(newTurnId, transcript);

    const streaming = this.ctx ? !this.ctx.isIdle() : false;
    const shouldSteer = streaming || this.bargeInAwaitingNewUtterance;
    this.bargeInAwaitingNewUtterance = false;

    if (shouldSteer && streaming) {
      this.pi.sendUserMessage(transcript, { deliverAs: "steer" });
    } else {
      this.pi.sendUserMessage(transcript);
    }
    this.refreshStatus();
  }

  private sendSessionInit(): void {
    this.transport.sendEvent(
      "session.init",
      { sessionId: this.tracker.ids.sessionId },
      {
        sample_rate: 16000,
        role: "controller",
        audio_mode: "local_device",
        turn_strategy: this.turnManager.getStrategy(),
        tts_engine: this.ttsEngine,
      },
    );
  }

  private sendAudioControl(
    action: EventPayloadMap["audio.control"]["action"],
    duckGain?: number,
  ): void {
    this.transport.sendEvent(
      "audio.control",
      { sessionId: this.tracker.ids.sessionId, turnId: this.tracker.ids.turnId },
      { action, ...(duckGain !== undefined ? { duck_gain: duckGain } : {}) },
    );
  }

  private refreshStatus(): void {
    if (!this.enabled) {
      this.updateStatus(undefined);
      return;
    }
    if (!this.transport.isConnected()) {
      this.updateStatus("⚠ Voice 离线");
      return;
    }
    if (this.assistantDucked) {
      this.updateStatus("⏸ Voice 避让中");
    } else if (this.assistantSpeaking) {
      this.updateStatus(`🔊 Voice 播报中 (${this.ttsEngine})`);
    } else if (this.mode === "ptt") {
      this.updateStatus(this.pttActive ? "🎙 Voice PTT 录音中" : "🎙 Voice PTT (Alt+V)");
    } else if (this.inSpeech) {
      this.updateStatus("🎙 Voice 识别中...");
    } else {
      this.updateStatus(`🎙 Voice 就绪 (${this.ttsEngine})`);
    }
  }

  private updateStatus(text: string | undefined): void {
    if (this.ctx?.hasUI) {
      this.ctx.ui.setStatus("voice", text);
    }
  }

  private async ensureBackendAndConnect(): Promise<boolean> {
    try {
      await this.transport.connect();
      return true;
    } catch {
      if (!this.autoSpawnBackend) return false;
    }

    const spawned = await this.spawnLocalBackend();
    if (!spawned) return false;
    try {
      await this.transport.connect();
      return true;
    } catch {
      return false;
    }
  }

  private async spawnLocalBackend(): Promise<boolean> {
    if (this.backendProc && !this.backendProc.killed) return true;
    const repoRoot = resolveRepoRoot();
    const venvPy = path.join(repoRoot, ".venv", "Scripts", "python.exe");
    const pyExe = fs.existsSync(venvPy) ? venvPy : "python";
    const parsed = new URL(this.apiBase);

    this.backendProc = spawn(
      pyExe,
      [
        "-m",
        "uvicorn",
        "inference.server:app",
        "--host",
        parsed.hostname,
        "--port",
        parsed.port || "8765",
      ],
      {
        cwd: repoRoot,
        stdio: "ignore",
        windowsHide: true,
      },
    );
    this.backendProc.on("exit", () => {
      this.backendProc = null;
    });

    const deadline = Date.now() + 15000;
    while (Date.now() < deadline) {
      try {
        const res = await fetch(`${this.apiBase}/health`);
        if (res.ok) return true;
      } catch {
        // waiting for server boot
      }
      await new Promise((r) => setTimeout(r, 250));
    }
    return false;
  }
}
