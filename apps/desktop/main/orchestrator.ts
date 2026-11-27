/**
 * End-to-end VoiceAgent desktop runtime orchestrator.
 * Connects AudioCapture, PlaybackQueue, SpeechWsClient, HybridTurnManager,
 * InterruptionController, SpeculativeGenerationController, PiAgentConnector, and Metrics.
 */

import { AudioCapturePipeline, PlaybackQueue } from "@voice-agent/audio";
import {
  type ApprovalRequest,
  type ApprovalResult,
  type LlmStreamProvider,
  PermissionGatekeeper,
  PiAgentConnector,
} from "@voice-agent/agent";
import {
  HybridTurnManager,
  InterruptionController,
  SpeculativeGenerationController,
  type TurnAdapterProbabilities,
} from "@voice-agent/interaction";
import { MetricsCollector, type MetricsSummary } from "@voice-agent/metrics";
import {
  type BinaryAudioFrame,
  type EventEnvelope,
  GenerationTracker,
  StreamKind,
} from "@voice-agent/protocol";
import { SpeechWsClient } from "@voice-agent/speech";

export type InteractionMode = "hands-free" | "push-to-talk" | "text" | "mute";
export type ListeningState = "idle" | "listening" | "muted";
export type AssistantState =
  | "idle"
  | "thinking"
  | "executing"
  | "speaking"
  | "interrupted"
  | "awaiting_approval"
  | "error";

export interface UiSnapshot {
  interactionMode: InteractionMode;
  listeningState: ListeningState;
  assistantState: AssistantState;
  backendConnected: boolean;
  partialTranscript: string;
  lastVadProb: number;
  lastAdapterProb: TurnAdapterProbabilities | null;
  pendingApproval: ApprovalRequest | null;
  metrics: MetricsSummary;
  errorMessage: string | null;
}

export interface OrchestratorConfig {
  sessionId?: string;
  wsUrl?: string;
  allowedRoots: string[];
  llmProvider?: LlmStreamProvider;
  speculativeEnabled?: boolean;
}

export class VoiceAgentOrchestrator {
  readonly capture: AudioCapturePipeline;
  readonly playback: PlaybackQueue;
  readonly speechClient: SpeechWsClient;
  readonly tracker: GenerationTracker;
  readonly turnManager: HybridTurnManager;
  readonly interruption: InterruptionController;
  readonly speculative: SpeculativeGenerationController;
  readonly gatekeeper: PermissionGatekeeper;
  readonly agent: PiAgentConnector;
  readonly metrics: MetricsCollector;

  private interactionMode: InteractionMode = "hands-free";
  private listeningState: ListeningState = "idle";
  private assistantState: AssistantState = "idle";
  private backendConnected = false;
  private partialTranscript = "";
  private lastVadProb = 0;
  private lastAdapterProb: TurnAdapterProbabilities | null = null;
  private lastAudioPtsMs = 0;
  private lastSpeechStartMs = 0;
  private lastSilenceStartMs = 0;
  private vadInSpeech = false;
  private pendingApproval: {
    request: ApprovalRequest;
    resolve: (res: ApprovalResult) => void;
  } | null = null;
  private pendingTtsMeta = new Map<
    number,
    Array<{
      turnId: string;
      generationId: string;
      sentenceIndex: number;
      chunkIndex: number;
      charStart: number;
      charEnd: number;
      sampleRate: number;
    }>
  >();
  private errorMessage: string | null = null;
  private readonly uiListeners = new Set<(snapshot: UiSnapshot) => void>();
  private readonly speculativeEnabled: boolean;

  constructor(config: OrchestratorConfig) {
    const sessionId = config.sessionId ?? `sess_${Date.now().toString(36)}`;
    this.speculativeEnabled = config.speculativeEnabled ?? false;
    this.capture = new AudioCapturePipeline();
    this.playback = new PlaybackQueue();
    this.speechClient = new SpeechWsClient({ url: config.wsUrl });
    this.tracker = new GenerationTracker(sessionId);
    this.turnManager = new HybridTurnManager({ strategy: "hybrid" });
    this.metrics = new MetricsCollector();
    this.speculative = new SpeculativeGenerationController();
    this.gatekeeper = new PermissionGatekeeper({
      mode: "conversation",
      level: "ask",
      allowedRoots: config.allowedRoots,
    });
    this.agent = new PiAgentConnector(this.gatekeeper, config.llmProvider);

    this.interruption = new InterruptionController(
      this.playback,
      this.tracker,
      this.metrics,
      {
        onAbortGeneration: (generationId, generationTag, cursor, reason) => {
          this.pendingTtsMeta.delete(generationTag);
          this.agent.abortGeneration(generationId, reason);
          if (cursor.turnId) {
            this.agent.history.markInterrupted(cursor.turnId, cursor.playedCharIndex);
          }
          this.speechClient.sendEvent(
            "generation.cancel",
            {
              sessionId: this.tracker.ids.sessionId,
              turnId: cursor.turnId || this.tracker.ids.turnId,
              generationId,
            },
            {
              target_generation_id: generationId,
              generation_tag: generationTag,
              reason,
              audible_cursor_chars: cursor.playedCharIndex,
            },
          );
          this.assistantState = "interrupted";
          this.emitUi();
        },
      },
    );

    this.wireSubsystems();
  }

  onUiUpdate(listener: (snapshot: UiSnapshot) => void): () => void {
    this.uiListeners.add(listener);
    return () => this.uiListeners.delete(listener);
  }

  getSnapshot(): UiSnapshot {
    return {
      interactionMode: this.interactionMode,
      listeningState: this.listeningState,
      assistantState: this.assistantState,
      backendConnected: this.backendConnected,
      partialTranscript: this.partialTranscript,
      lastVadProb: this.lastVadProb,
      lastAdapterProb: this.lastAdapterProb,
      pendingApproval: this.pendingApproval?.request ?? null,
      metrics: this.metrics.summarize(),
      errorMessage: this.errorMessage,
    };
  }

  setInteractionMode(mode: InteractionMode): void {
    this.interactionMode = mode;
    if (mode === "mute") {
      this.capture.setMuted(true);
      this.listeningState = "muted";
    } else if (mode === "text") {
      this.capture.setMuted(true);
      this.listeningState = "idle";
    } else {
      this.capture.setMuted(false);
      this.listeningState = mode === "hands-free" ? "listening" : "idle";
    }
    this.emitUi();
  }

  async connectBackend(): Promise<void> {
    await this.speechClient.connect();
    this.speechClient.sendEvent(
      "session.init",
      { sessionId: this.tracker.ids.sessionId },
      { sample_rate: 16000, turn_strategy: this.turnManager.getStrategy() },
    );
  }

  disconnect(): void {
    this.speculative.discard("disconnect");
    this.speechClient.disconnect();
  }

  respondApproval(result: ApprovalResult): void {
    if (!this.pendingApproval) return;
    const { resolve } = this.pendingApproval;
    this.pendingApproval = null;
    this.assistantState = "executing";
    resolve(result);
    this.emitUi();
  }

  /** Submits a text or committed voice turn through Pi Agent -> SentenceBuffer -> Streaming TTS. */
  async submitTurn(transcript: string, speechEndMs = performance.now()): Promise<string> {
    const clean = transcript.trim();
    if (!clean) return "";

    const turnId = this.tracker.startTurn();
    const { generationId, generationTag } = this.tracker.startGeneration();
    this.metrics.startTurn(turnId);
    this.metrics.mark(turnId, { speechEndMs, turnCommitMs: performance.now() });

    this.partialTranscript = "";
    this.assistantState = "thinking";
    this.emitUi();

    let recordedFirstToken = false;
    const { fullReply, aborted } = await this.agent.runTurn({
      turnId,
      generationId,
      userTranscript: clean,
      approvalHandler: (req) =>
        new Promise<ApprovalResult>((resolve) => {
          this.pendingApproval = { request: req, resolve };
          this.assistantState = "awaiting_approval";
          this.emitUi();
        }),
      callbacks: {
        onTextDelta: () => {
          if (!recordedFirstToken) {
            recordedFirstToken = true;
            this.metrics.mark(turnId, { llmFirstTokenMs: performance.now() });
          }
        },
        onToolStart: () => {
          this.metrics.mark(turnId, { usedTool: true });
          this.assistantState = "executing";
          this.emitUi();
        },
        onToolEnd: () => {
          this.assistantState = "thinking";
          this.emitUi();
        },
        onSpeakableSentence: (seg) => {
          if (!this.speechClient.isConnected()) return;
          this.speechClient.sendEvent(
            "tts.synthesize",
            { sessionId: this.tracker.ids.sessionId, turnId, generationId },
            {
              sentence_index: seg.sentenceIndex,
              text: seg.text,
              char_start: seg.charStart,
              char_end: seg.charEnd,
              generation_tag: generationTag,
              speaker_profile: "default",
            },
          );
        },
      },
    });

    if (!aborted && this.playback.getMode() === "idle") {
      this.assistantState = "idle";
      this.emitUi();
    }
    return fullReply;
  }

  private wireSubsystems(): void {
    this.capture.onFrame((frame) => {
      this.lastAudioPtsMs = frame.ptsMs;
      if (this.listeningState !== "muted" && this.interactionMode !== "text") {
        this.listeningState = "listening";
        this.speechClient.sendMicAudio(frame.pcm16, frame.ptsMs);
      }
    });

    // Feed real speaker playback back to capture AEC reference.
    this.playback.onAecReference((ref16k) => {
      this.capture.pushAecReference(ref16k);
    });

    this.playback.onPlaybackAck((ack) => {
      if (ack.playbackState === "playing") {
        this.assistantState = "speaking";
        const turn = ack.turnId ? this.metrics.getTurn(ack.turnId) : undefined;
        if (turn && turn.audiblePlayMs === undefined) {
          this.metrics.mark(ack.turnId, { audiblePlayMs: performance.now() });
        }
      } else if (ack.playbackState === "idle" && this.assistantState === "speaking") {
        this.assistantState = "idle";
      }

      this.speechClient.sendEvent(
        "playback.ack",
        {
          sessionId: this.tracker.ids.sessionId,
          turnId: ack.turnId || null,
          generationId: ack.generationId || null,
        },
        {
          playback_state: ack.playbackState,
          played_char_index: ack.playedCharIndex,
          played_duration_ms: ack.playedDurationMs,
        },
      );
      this.emitUi();
    });

    this.speechClient.onConnectionChange((connected) => {
      this.backendConnected = connected;
      if (!connected) {
        this.errorMessage = "Speech backend disconnected — text mode remains available";
      } else {
        this.errorMessage = null;
      }
      this.emitUi();
    });

    this.speechClient.onEvent((event) => this.handleBackendEvent(event));
    this.speechClient.onBinaryAudio((frame) => this.handleBinaryAudio(frame));
  }

  private handleBackendEvent(event: EventEnvelope): void {
    if (event.type === "error") {
      const payload = event.payload as EventEnvelope<"error">["payload"];
      this.errorMessage = `[${payload.module}] ${payload.message}`;
      if (payload.fallback_action === "text_only") {
        this.setInteractionMode("text");
      }
      this.emitUi();
      return;
    }

    if (event.type === "vad.start") {
      const payload = event.payload as EventEnvelope<"vad.start">["payload"];
      this.vadInSpeech = true;
      this.lastVadProb = payload.vad_prob;
      this.lastSpeechStartMs = payload.speech_start_pts_ms;
      this.lastAudioPtsMs = Math.max(this.lastAudioPtsMs, payload.speech_start_pts_ms);
      if (this.playback.getMode() === "playing") {
        this.interruption.onPotentialUserSpeech();
      }
      this.emitUi();
      return;
    }

    if (event.type === "vad.end") {
      const payload = event.payload as EventEnvelope<"vad.end">["payload"];
      this.vadInSpeech = false;
      this.lastVadProb = payload.vad_prob;
      this.lastSilenceStartMs = payload.silence_start_pts_ms;
      this.lastAudioPtsMs = Math.max(this.lastAudioPtsMs, payload.silence_start_pts_ms);
      this.emitUi();
      return;
    }

    if (event.type === "asr.partial") {
      const payload = event.payload as EventEnvelope<"asr.partial">["payload"];
      this.partialTranscript = payload.text;
      this.lastAudioPtsMs = Math.max(this.lastAudioPtsMs, payload.audio_pts_ms);
      this.emitUi();
      return;
    }

    if (event.type === "turn.prediction") {
      const payload = event.payload as EventEnvelope<"turn.prediction">["payload"];
      this.lastAdapterProb = {
        pCompletion: payload.p_completion,
        pFloorTaking: payload.p_floor_taking,
        pNonFloorFeedback: payload.p_non_floor_feedback,
      };
      const assistantSpeaking =
        this.playback.getMode() === "playing" ||
        this.playback.getMode() === "ducked" ||
        this.interruption.isSuspectedSpeechActive();

      if (assistantSpeaking) {
        const decision = this.turnManager.evaluate({
          vadInSpeech: this.vadInSpeech,
          speechDurationMs: Math.max(0, this.lastAudioPtsMs - this.lastSpeechStartMs),
          silenceDurationMs: this.vadInSpeech ? 0 : Math.max(0, this.lastAudioPtsMs - this.lastSilenceStartMs),
          assistantSpeaking: true,
          transcript: this.partialTranscript,
          adapter: this.lastAdapterProb,
        });
        this.interruption.resolve(decision);
      } else if (
        this.speculativeEnabled &&
        this.speculative.shouldTrigger(payload.p_completion, this.partialTranscript)
      ) {
        const specGenId = `spec_${Date.now().toString(36)}`;
        this.speculative.startCandidate(
          this.tracker.ids.turnId ?? "pending",
          specGenId,
          this.partialTranscript,
        );
      }
      this.emitUi();
      return;
    }

    if (event.type === "asr.final") {
      const payload = event.payload as EventEnvelope<"asr.final">["payload"];
      const decision = this.turnManager.evaluate({
        vadInSpeech: false,
        speechDurationMs: Math.max(0, payload.audio_end_pts_ms - payload.audio_start_pts_ms),
        silenceDurationMs: 1200,
        assistantSpeaking: false,
        transcript: payload.text,
        adapter: this.lastAdapterProb,
      });
      if (decision.event === "EOT" && payload.text.trim()) {
        this.submitTurn(payload.text, performance.now()).catch((err) => {
          this.errorMessage = err instanceof Error ? err.message : String(err);
          this.assistantState = "error";
          this.emitUi();
        });
      }
      return;
    }

    if (event.type === "tts.chunk") {
      const payload = event.payload as EventEnvelope<"tts.chunk">["payload"];
      if (!this.tracker.isGenerationTagActive(payload.generation_tag)) return;
      const turnId = event.turn_id ?? this.tracker.ids.turnId ?? "";
      const generationId = event.generation_id ?? this.tracker.ids.generationId ?? "";
      if (turnId && payload.chunk_index === 0) {
        this.metrics.mark(turnId, { ttsFirstChunkMs: performance.now() });
      }
      const list = this.pendingTtsMeta.get(payload.generation_tag) ?? [];
      list.push({
        turnId,
        generationId,
        sentenceIndex: payload.sentence_index,
        chunkIndex: payload.chunk_index,
        charStart: payload.char_start,
        charEnd: payload.char_end,
        sampleRate: payload.sample_rate,
      });
      this.pendingTtsMeta.set(payload.generation_tag, list);
    }
  }

  private handleBinaryAudio(frame: BinaryAudioFrame): void {
    if (frame.streamKind !== StreamKind.TtsOutput) return;
    if (!this.tracker.isGenerationTagActive(frame.generationTag)) return;
    const list = this.pendingTtsMeta.get(frame.generationTag);
    const meta = list?.shift();
    if (!meta) return;
    if (list && list.length === 0) {
      this.pendingTtsMeta.delete(frame.generationTag);
    }
    this.playback.enqueueChunk({
      turnId: meta.turnId,
      generationId: meta.generationId,
      generationTag: frame.generationTag,
      sentenceIndex: meta.sentenceIndex,
      chunkIndex: meta.chunkIndex,
      charStart: meta.charStart,
      charEnd: meta.charEnd,
      sampleRate: frame.sampleRate,
      pcm16: frame.pcm16,
    });
  }

  private emitUi(): void {
    const snap = this.getSnapshot();
    for (const listener of this.uiListeners) {
      listener(snap);
    }
  }
}
