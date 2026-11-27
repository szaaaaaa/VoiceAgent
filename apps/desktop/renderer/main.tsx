import React, { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  createEvent,
  decodeAudioFrame,
  encodeAudioFrame,
  GenerationTracker,
  parseEventEnvelope,
  StreamKind,
} from "@voice-agent/protocol";
import {
  HybridTurnManager,
  PlaybackAwareHistory,
  type TurnAdapterProbabilities,
  type TurnStrategy,
} from "@voice-agent/interaction";
import { SentenceBuffer } from "@voice-agent/speech";
import { BrowserMicCapture, BrowserStreamPlayer, listAudioDevices, type AudioDeviceOption } from "./audio-io";

interface DesktopPreloadApi {
  isElectron: boolean;
  getBackendConfig: () => Promise<{
    host: string;
    port: number;
    wsUrl: string;
    apiBase: string;
    heavyModelsEnabled: boolean;
  }>;
  restartBackend: (opts: { loadHeavy: boolean }) => Promise<{ ok: boolean; heavyModelsEnabled: boolean }>;
  exportLogs: (jsonlContent: string) => Promise<{ saved: boolean; path?: string }>;
  toggleDevTools: () => Promise<void>;
  onPushToTalkToggle: (cb: () => void) => () => void;
}

declare global {
  interface Window {
    voiceAgentDesktop?: DesktopPreloadApi;
  }
}

interface HealthInfo {
  status: string;
  vram: { device: string; allocated_mb: number; reserved_mb: number; total_mb: number };
  models: {
    vad?: { backend: string };
    asr: { model_id: string; backend?: string; checkpoint: string | null };
    turn_adapter: { version: string; params: number };
    tts: { model_id: string; backend?: string; sample_rate: number };
  };
}

interface PendingApprovalUi {
  approvalId: string;
  toolName: string;
  impactSummary: string;
  targetPath: string | null;
  command: string | null;
}

type InteractionMode = "hands-free" | "push-to-talk" | "text" | "mute";
type TtsEngineChoice = "native_voice" | "backend_pcm";

const DEFAULT_WS_URL = "ws://127.0.0.1:8765/ws/v1/voice";
const DEFAULT_API_BASE = "http://127.0.0.1:8765";

const EXPLICIT_APPROVE = new Set(["yes", "approve", "confirm", "allow", "同意", "确认", "允许", "批准"]);
const EXPLICIT_DENY = new Set(["no", "deny", "cancel", "stop", "不要", "拒绝", "取消"]);

function VoiceAgentDesktop(): React.ReactElement {
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS_URL);
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [heavyGpu, setHeavyGpu] = useState(false);
  const [restartingBackend, setRestartingBackend] = useState(false);

  const [connected, setConnected] = useState(false);
  const [micActive, setMicActive] = useState(false);
  const [pttPressed, setPttPressed] = useState(false);
  const [mode, setMode] = useState<InteractionMode>("hands-free");
  const [ttsEngine, setTtsEngine] = useState<TtsEngineChoice>("native_voice");
  const [agentMode, setAgentMode] = useState<"conversation" | "action">("conversation");
  const [permLevel, setPermLevel] = useState<"readonly" | "ask" | "trusted">("ask");
  const [turnStrategy, setTurnStrategy] = useState<TurnStrategy>("hybrid");
  const [assistantState, setAssistantState] = useState("idle");
  const [partialText, setPartialText] = useState("");
  const [vadProb, setVadProb] = useState(0);
  const [adapterProb, setAdapterProb] = useState<TurnAdapterProbabilities | null>(null);
  const [rmsHistory, setRmsHistory] = useState<number[]>(() => Array(56).fill(-90));
  const [turnsVersion, setTurnsVersion] = useState(0);
  const [textInput, setTextInput] = useState("");
  const [devMode, setDevMode] = useState(true);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [devices, setDevices] = useState<AudioDeviceOption[]>([]);
  const [selectedMic, setSelectedMic] = useState("default");
  const [aecEnabled, setAecEnabled] = useState(true);
  const [lastLatencyMs, setLastLatencyMs] = useState<{ ttfa?: number; e2e?: number; bargeIn?: number }>({});
  const [pendingApproval, setPendingApproval] = useState<PendingApprovalUi | null>(null);
  const [statusNote, setStatusNote] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const micRef = useRef(new BrowserMicCapture());
  const playerRef = useRef(new BrowserStreamPlayer());
  const trackerRef = useRef(new GenerationTracker(`desk_${Date.now().toString(36)}`));
  const historyRef = useRef(new PlaybackAwareHistory());
  const turnManagerRef = useRef(new HybridTurnManager({ strategy: "hybrid" }));
  const pendingMetaRef = useRef(new Map<number, Array<{ charStart: number; charEnd: number; sampleRate: number }>>());
  const audioSeqRef = useRef(0);
  const speechEndRef = useRef(0);
  const vadInSpeechRef = useRef(false);
  const speechStartPtsRef = useRef(0);
  const lastPtsRef = useRef(0);
  const suspectedInterruptMsRef = useRef(0);
  const modeRef = useRef<InteractionMode>(mode);
  const pttPressedRef = useRef(pttPressed);
  const partialTextRef = useRef(partialText);
  const pendingApprovalRef = useRef<PendingApprovalUi | null>(pendingApproval);
  const ttsEngineRef = useRef<TtsEngineChoice>(ttsEngine);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const eventLogRef = useRef<Array<Record<string, unknown>>>([]);

  modeRef.current = mode;
  pttPressedRef.current = pttPressed;
  partialTextRef.current = partialText;
  pendingApprovalRef.current = pendingApproval;
  ttsEngineRef.current = ttsEngine;

  const pushLog = (kind: string, data: Record<string, unknown>): void => {
    eventLogRef.current.push({ ts: Date.now(), kind, ...data });
    if (eventLogRef.current.length > 500) eventLogRef.current.shift();
  };

  // Sync Electron desktop config & global Alt+Space PTT shortcut when running in Electron.
  useEffect(() => {
    const bridge = window.voiceAgentDesktop;
    if (!bridge) return;
    void bridge.getBackendConfig().then((cfg) => {
      setWsUrl(cfg.wsUrl);
      setApiBase(cfg.apiBase);
      setHeavyGpu(cfg.heavyModelsEnabled);
      if (cfg.heavyModelsEnabled) setTtsEngine("backend_pcm");
    });
    return bridge.onPushToTalkToggle(() => {
      setMode("push-to-talk");
      setPttPressed((prev) => !prev);
    });
  }, []);

  // Draw live waveform bars on canvas.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const barW = canvas.width / rmsHistory.length;
    rmsHistory.forEach((db, i) => {
      const norm = Math.max(0.05, Math.min(1, (db + 75) / 65));
      const h = norm * (canvas.height - 8);
      ctx.fillStyle = vadProb >= 0.5 ? "#38bdf8" : mode === "push-to-talk" && pttPressed ? "#a78bfa" : "#475569";
      ctx.fillRect(i * barW + 1.5, (canvas.height - h) / 2, Math.max(2, barW - 3), h);
    });
  }, [rmsHistory, vadProb, mode, pttPressed]);

  // Poll backend /health every 3s.
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${apiBase}/health`);
        if (res.ok) setHealth((await res.json()) as HealthInfo);
      } catch {
        // Offline
      }
    };
    void poll();
    const id = setInterval(() => void poll(), 3000);
    return () => clearInterval(id);
  }, [apiBase]);

  const stopNativeSpeech = (): void => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  };

  const speakNativeVoice = useCallback((text: string, turnId: string): void => {
    if (typeof window === "undefined" || !("speechSynthesis" in window) || !text.trim()) {
      setAssistantState("idle");
      return;
    }
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "zh-CN";
    utter.rate = 1.05;
    const voices = window.speechSynthesis.getVoices();
    const zhVoice =
      voices.find((v) => /xiaoxiao|yunxi|yaoyao|huihui/i.test(v.name) && v.lang.toLowerCase().startsWith("zh")) ??
      voices.find((v) => v.lang.toLowerCase().startsWith("zh-cn")) ??
      voices.find((v) => v.lang.toLowerCase().startsWith("zh"));
    if (zhVoice) utter.voice = zhVoice;
    utter.onstart = () => {
      setAssistantState("speaking");
      if (speechEndRef.current > 0) {
        const e2e = Math.round(performance.now() - speechEndRef.current);
        setLastLatencyMs((prev) => ({ ...prev, ttfa: e2e, e2e }));
        pushLog("e2e_audible", { turnId, e2eMs: e2e, engine: "native_voice" });
        speechEndRef.current = 0;
      }
    };
    utter.onend = () => {
      setAssistantState((cur) => (cur === "speaking" ? "idle" : cur));
    };
    utter.onerror = () => {
      setAssistantState("idle");
    };
    window.speechSynthesis.speak(utter);
  }, []);

  const triggerBargeInCancel = useCallback((reason = "user_interruption"): void => {
    stopNativeSpeech();
    const active = trackerRef.current.cancelCurrentGeneration();
    const cursor = playerRef.current.cancel();
    if (suspectedInterruptMsRef.current > 0) {
      const bargeMs = Math.round(performance.now() - suspectedInterruptMsRef.current);
      setLastLatencyMs((prev) => ({ ...prev, bargeIn: bargeMs }));
      pushLog("barge_in_cancel", { reason, bargeMs, playedChars: cursor.playedCharIndex });
      suspectedInterruptMsRef.current = 0;
    }
    const turnId = trackerRef.current.ids.turnId;
    if (turnId) {
      historyRef.current.markInterrupted(turnId, cursor.playedCharIndex);
      setTurnsVersion((v) => v + 1);
    }
    if (active) {
      pendingMetaRef.current.delete(active.generationTag);
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(
          JSON.stringify(
            createEvent(
              "generation.cancel",
              { sessionId: trackerRef.current.ids.sessionId, turnId, generationId: active.generationId },
              {
                target_generation_id: active.generationId,
                generation_tag: active.generationTag,
                reason,
                audible_cursor_chars: cursor.playedCharIndex,
              },
            ),
          ),
        );
      }
    }
    setAssistantState("interrupted");
  }, []);

  const dispatchSpeakableReply = useCallback(
    (turnId: string, generationId: string, generationTag: number, reply: string): void => {
      const buf = new SentenceBuffer();
      const segs = [...buf.pushDelta(reply), ...buf.flush()];
      const speakable = segs.map((s) => s.text).join(" ");
      historyRef.current.recordAssistantReply(turnId, reply, speakable);
      setTurnsVersion((v) => v + 1);

      if (ttsEngineRef.current === "native_voice") {
        speakNativeVoice(speakable, turnId);
        return;
      }

      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN && trackerRef.current.isGenerationTagActive(generationTag)) {
        for (const seg of segs) {
          ws.send(
            JSON.stringify(
              createEvent(
                "tts.synthesize",
                { sessionId: trackerRef.current.ids.sessionId, turnId, generationId },
                {
                  sentence_index: seg.sentenceIndex,
                  text: seg.text,
                  char_start: seg.charStart,
                  char_end: seg.charEnd,
                  generation_tag: generationTag,
                  speaker_profile: "default",
                },
              ),
            ),
          );
        }
      } else {
        setAssistantState("idle");
      }
    },
    [speakNativeVoice],
  );

  const respondApproval = useCallback(
    async (approved: boolean, source: "ui" | "voice", reason: string): Promise<void> => {
      const current = pendingApprovalRef.current;
      if (!current) return;
      setPendingApproval(null);
      setAssistantState("executing");

      try {
        const res = await fetch(`${apiBase}/api/agent/approval`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            approval_id: current.approvalId,
            approved,
            source,
            reason,
          }),
        });
        const data = (await res.json()) as {
          reply: string;
          tool_calls?: Array<{ id: string; name: string; status: "ok" | "denied" | "error"; output: string }>;
        };
        const turnId = trackerRef.current.ids.turnId ?? trackerRef.current.startTurn();
        const { generationId, generationTag } = trackerRef.current.startGeneration();
        for (const tc of data.tool_calls ?? []) {
          historyRef.current.recordToolExecution(turnId, {
            toolCallId: tc.id,
            toolName: tc.name,
            argsSummary: current.command ?? "{}",
            status: tc.status,
            sideEffectCommitted: tc.status === "ok",
            result: tc.output,
          });
        }
        dispatchSpeakableReply(turnId, generationId, generationTag, data.reply);
      } catch (err) {
        setStatusNote(`Approval dispatch error: ${err instanceof Error ? err.message : String(err)}`);
        setAssistantState("error");
      }
    },
    [apiBase, dispatchSpeakableReply],
  );

  const runTurn = useCallback(
    async (userText: string): Promise<void> => {
      const clean = userText.trim();
      if (!clean) return;

      // If an approval modal is waiting, interpret spoken/typed reply fail-closed.
      if (pendingApprovalRef.current) {
        const norm = clean.toLowerCase().replace(/[.,!?;:。！？]+$/g, "");
        if (EXPLICIT_APPROVE.has(norm)) {
          await respondApproval(true, "voice", `Explicit approval: "${clean}"`);
        } else if (EXPLICIT_DENY.has(norm)) {
          await respondApproval(false, "voice", `Explicit denial: "${clean}"`);
        } else {
          await respondApproval(false, "voice", `Ambiguous response "${clean}" treated as denied`);
        }
        return;
      }

      speechEndRef.current = performance.now();
      const turnId = trackerRef.current.startTurn();
      const { generationId, generationTag } = trackerRef.current.startGeneration();
      historyRef.current.recordUserTurn(turnId, clean);
      setTurnsVersion((v) => v + 1);
      setAssistantState("thinking");
      pushLog("turn_start", { turnId, userText: clean });

      try {
        const res = await fetch(`${apiBase}/api/agent/turn`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: trackerRef.current.ids.sessionId,
            turn_id: turnId,
            user_text: clean,
            history: historyRef.current.toAgentMessages(),
            agent_mode: agentMode,
            permission_level: permLevel,
          }),
        });
        const data = (await res.json()) as {
          reply: string;
          tool_calls?: Array<{ id: string; name: string; status: "ok" | "denied" | "error"; output: string }>;
          pending_approval?: PendingApprovalUi | null;
        };

        if (data.pending_approval) {
          setPendingApproval(data.pending_approval);
          setAssistantState("awaiting_approval");
        }
        for (const tc of data.tool_calls ?? []) {
          historyRef.current.recordToolExecution(turnId, {
            toolCallId: tc.id,
            toolName: tc.name,
            argsSummary: "{}",
            status: tc.status,
            sideEffectCommitted: tc.status === "ok",
            result: tc.output,
          });
        }
        dispatchSpeakableReply(turnId, generationId, generationTag, data.reply);
      } catch (err) {
        setStatusNote(`Agent error: ${err instanceof Error ? err.message : String(err)}`);
        setAssistantState("error");
      }
    },
    [agentMode, apiBase, dispatchSpeakableReply, permLevel, respondApproval],
  );

  const runTurnRef = useRef(runTurn);
  runTurnRef.current = runTurn;

  // Connect local WebSocket & wire audio player state.
  useEffect(() => {
    let closed = false;
    playerRef.current.setListener((st) => {
      if (st.mode === "playing") {
        setAssistantState("speaking");
        if (speechEndRef.current > 0) {
          const e2e = Math.round(performance.now() - speechEndRef.current);
          setLastLatencyMs((prev) => ({ ...prev, e2e }));
          pushLog("e2e_audible", { e2eMs: e2e, engine: "backend_pcm" });
          speechEndRef.current = 0;
        }
      } else if (st.mode === "idle") {
        setAssistantState((cur) => (cur === "speaking" ? "idle" : cur));
      }
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(
          JSON.stringify(
            createEvent("playback.ack", trackerRef.current.ids, {
              playback_state: st.mode,
              played_char_index: st.playedCharIndex,
              played_duration_ms: st.playedDurationMs,
            }),
          ),
        );
      }
    });

    const connect = () => {
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        setStatusNote(null);
        ws.send(
          JSON.stringify(
            createEvent("session.init", { sessionId: trackerRef.current.ids.sessionId }, { sample_rate: 16000 }),
          ),
        );
      };

      ws.onmessage = (ev) => {
        if (ev.data instanceof ArrayBuffer) {
          const frame = decodeAudioFrame(ev.data);
          if (frame.streamKind !== StreamKind.TtsOutput) return;
          if (!trackerRef.current.isGenerationTagActive(frame.generationTag)) return;
          const q = pendingMetaRef.current.get(frame.generationTag);
          const meta = q?.shift();
          if (!meta) return;
          playerRef.current.enqueuePcm16({
            pcm16: frame.pcm16,
            sampleRate: frame.sampleRate,
            generationTag: frame.generationTag,
            charStart: meta.charStart,
            charEnd: meta.charEnd,
          });
          return;
        }

        const msg = parseEventEnvelope(String(ev.data));
        if (msg.type === "vad.start") {
          const p = msg.payload as { vad_prob: number; speech_start_pts_ms: number };
          vadInSpeechRef.current = true;
          speechStartPtsRef.current = p.speech_start_pts_ms;
          setVadProb(p.vad_prob);
          const speaking =
            playerRef.current.getMode() === "playing" ||
            (typeof window !== "undefined" && "speechSynthesis" in window && window.speechSynthesis.speaking);
          if (speaking) {
            suspectedInterruptMsRef.current = performance.now();
            playerRef.current.duck(0.15);
          }
        } else if (msg.type === "vad.end") {
          const p = msg.payload as { vad_prob: number };
          vadInSpeechRef.current = false;
          setVadProb(p.vad_prob);
        } else if (msg.type === "asr.partial") {
          const p = msg.payload as { text: string; audio_pts_ms: number };
          lastPtsRef.current = p.audio_pts_ms;
          setPartialText(p.text);
        } else if (msg.type === "asr.final") {
          const p = msg.payload as { text: string };
          setPartialText("");
          if (p.text.trim()) {
            void runTurnRef.current(p.text.trim());
          }
        } else if (msg.type === "turn.prediction") {
          const p = msg.payload as {
            p_completion: number;
            p_floor_taking: number;
            p_non_floor_feedback: number;
          };
          const probs: TurnAdapterProbabilities = {
            pCompletion: p.p_completion,
            pFloorTaking: p.p_floor_taking,
            pNonFloorFeedback: p.p_non_floor_feedback,
          };
          setAdapterProb(probs);
          const speaking =
            playerRef.current.getMode() === "ducked" ||
            playerRef.current.getMode() === "playing" ||
            (typeof window !== "undefined" && "speechSynthesis" in window && window.speechSynthesis.speaking);
          if (speaking) {
            const decision = turnManagerRef.current.evaluate({
              vadInSpeech: vadInSpeechRef.current,
              speechDurationMs: Math.max(0, lastPtsRef.current - speechStartPtsRef.current),
              silenceDurationMs: vadInSpeechRef.current ? 0 : 250,
              assistantSpeaking: true,
              transcript: partialTextRef.current,
              adapter: probs,
            });
            if (
              decision.event === "BACKCHANNEL" ||
              decision.event === "NOISE" ||
              (decision.event === "SHORT_INTERJECTION" && decision.action === "resume_playback")
            ) {
              playerRef.current.resume();
            } else if (
              decision.event === "INTERRUPTION" ||
              (decision.event === "SHORT_INTERJECTION" && decision.action === "cancel_and_regenerate")
            ) {
              triggerBargeInCancel(decision.reason);
            }
          }
        } else if (msg.type === "tts.chunk") {
          const p = msg.payload as {
            generation_tag: number;
            char_start: number;
            char_end: number;
            sample_rate: number;
            ttfa_ms: number;
          };
          if (!trackerRef.current.isGenerationTagActive(p.generation_tag)) return;
          if (p.ttfa_ms > 0) setLastLatencyMs((prev) => ({ ...prev, ttfa: Math.round(p.ttfa_ms) }));
          const arr = pendingMetaRef.current.get(p.generation_tag) ?? [];
          arr.push({ charStart: p.char_start, charEnd: p.char_end, sampleRate: p.sample_rate });
          pendingMetaRef.current.set(p.generation_tag, arr);
        } else if (msg.type === "error") {
          const p = msg.payload as { module: string; message: string };
          setStatusNote(`[${p.module}] ${p.message}`);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        if (!closed) setTimeout(connect, 1500);
      };
    };

    connect();
    void listAudioDevices().then(setDevices);
    return () => {
      closed = true;
      stopNativeSpeech();
      micRef.current.stop();
      wsRef.current?.close();
    };
  }, [triggerBargeInCancel, wsUrl]);

  const toggleMic = async (): Promise<void> => {
    if (micActive) {
      micRef.current.stop();
      setMicActive(false);
      return;
    }
    try {
      await micRef.current.start(
        (pcm16k, ptsMs, rmsDb) => {
          lastPtsRef.current = ptsMs;
          const curMode = modeRef.current;
          const allowStream =
            curMode === "hands-free" || (curMode === "push-to-talk" && pttPressedRef.current);
          setRmsHistory((prev) => [...prev.slice(1), allowStream ? rmsDb : -90]);

          const ws = wsRef.current;
          if (ws && ws.readyState === WebSocket.OPEN && curMode !== "text" && curMode !== "mute") {
            audioSeqRef.current += 1;
            const payload = allowStream ? pcm16k : new Int16Array(pcm16k.length);
            ws.send(
              encodeAudioFrame({
                streamKind: StreamKind.MicInput,
                seq: audioSeqRef.current,
                ptsMs,
                sampleRate: 16000,
                generationTag: 0,
                pcm16: payload,
              }),
            );
          }
        },
        { deviceId: selectedMic, aecEnabled, noiseSuppression: true },
      );
      setMicActive(true);
      setDevices(await listAudioDevices());
      setStatusNote(null);
    } catch (err) {
      setStatusNote(`Microphone error: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleToggleHeavyGpu = async (): Promise<void> => {
    const bridge = window.voiceAgentDesktop;
    if (!bridge) {
      setStatusNote("Launch via Electron (`npm run desktop`) to hot-restart backend with GPU models.");
      return;
    }
    setRestartingBackend(true);
    const next = !heavyGpu;
    const res = await bridge.restartBackend({ loadHeavy: next });
    setHeavyGpu(res.heavyModelsEnabled);
    setTtsEngine(res.heavyModelsEnabled ? "backend_pcm" : "native_voice");
    setRestartingBackend(false);
    setStatusNote(res.ok ? `Backend restarted (${res.heavyModelsEnabled ? "Full CUDA Models" : "Fast Mode"})` : "Backend restart failed");
  };

  const handleExportLogs = async (): Promise<void> => {
    const jsonl = eventLogRef.current.map((e) => JSON.stringify(e)).join("\n");
    const bridge = window.voiceAgentDesktop;
    if (bridge) {
      const res = await bridge.exportLogs(jsonl);
      if (res.saved && res.path) setStatusNote(`Logs exported to ${res.path}`);
      return;
    }
    const blob = new Blob([jsonl], { type: "application/x-ndjson" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `voiceagent_logs_${Date.now()}.jsonl`;
    a.click();
  };

  const turns = historyRef.current.getTurns();
  void turnsVersion;
  const isElectron = Boolean(window.voiceAgentDesktop?.isElectron);

  return (
    <div style={{ maxWidth: 1160, margin: "0 auto", padding: 18 }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid #1f2937", paddingBottom: 12, gap: 10, flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ width: 10, height: 10, borderRadius: "50%", background: connected ? "#22c55e" : "#ef4444", display: "inline-block" }} />
          <strong style={{ fontSize: 18 }}>VoiceAgent</strong>
          <span style={{ fontSize: 11, padding: "2px 7px", borderRadius: 999, background: "#1e293b", color: "#93c5fd" }}>
            {isElectron ? "Desktop App" : "Web Preview"}
          </span>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", fontSize: 12 }}>
          <select
            aria-label="Microphone"
            value={selectedMic}
            onChange={(e) => setSelectedMic(e.target.value)}
            style={{ background: "#111827", border: "1px solid #374151", borderRadius: 6, padding: "5px 8px" }}
          >
            <option value="default">Default Mic</option>
            {devices.filter((d) => d.kind === "audioinput").map((d) => (
              <option key={d.deviceId} value={d.deviceId}>{d.label}</option>
            ))}
          </select>

          <select
            aria-label="TTS Engine"
            value={ttsEngine}
            onChange={(e) => setTtsEngine(e.target.value as TtsEngineChoice)}
            style={{ background: "#111827", border: "1px solid #374151", borderRadius: 6, padding: "5px 8px" }}
          >
            <option value="native_voice">Voice: Windows Neural (Instant)</option>
            <option value="backend_pcm">Voice: Backend PCM (Qwen3-TTS)</option>
          </select>

          <select
            aria-label="Turn Strategy"
            value={turnStrategy}
            onChange={(e) => {
              const s = e.target.value as TurnStrategy;
              setTurnStrategy(s);
              turnManagerRef.current.setStrategy(s);
            }}
            style={{ background: "#111827", border: "1px solid #374151", borderRadius: 6, padding: "5px 8px" }}
          >
            <option value="hybrid">Turn: Hybrid (GRU+Rules)</option>
            <option value="adapter">Turn: Causal GRU</option>
            <option value="smart_turn">Turn: Smart Turn</option>
            <option value="fixed_silence">Turn: Fixed Silence</option>
          </select>

          <select
            aria-label="Agent Mode"
            value={agentMode}
            onChange={(e) => setAgentMode(e.target.value as "conversation" | "action")}
            style={{ background: "#111827", border: "1px solid #374151", borderRadius: 6, padding: "5px 8px" }}
          >
            <option value="conversation">Mode: Conversation</option>
            <option value="action">Mode: Action</option>
          </select>

          <select
            aria-label="Permission Policy"
            value={permLevel}
            onChange={(e) => setPermLevel(e.target.value as "readonly" | "ask" | "trusted")}
            style={{ background: "#111827", border: "1px solid #374151", borderRadius: 6, padding: "5px 8px" }}
          >
            <option value="readonly">Policy: Readonly</option>
            <option value="ask">Policy: Ask</option>
            <option value="trusted">Policy: Trusted</option>
          </select>

          <button
            type="button"
            disabled={restartingBackend}
            onClick={() => void handleToggleHeavyGpu()}
            style={{ background: heavyGpu ? "#059669" : "#374151", border: "none", borderRadius: 6, padding: "5px 10px", cursor: "pointer" }}
          >
            {restartingBackend ? "Restarting..." : heavyGpu ? "GPU Models: ON" : "GPU Models: Fast"}
          </button>

          <button
            type="button"
            onClick={() => setDevMode((v) => !v)}
            style={{ background: devMode ? "#1d4ed8" : "#1f2937", border: "none", borderRadius: 6, padding: "5px 10px", cursor: "pointer" }}
          >
            Dev HUD
          </button>
        </div>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: devMode ? "1.65fr 1fr" : "1fr", gap: 16, marginTop: 14 }}>
        <section style={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 12, padding: 16, display: "flex", flexDirection: "column", minHeight: 590 }}>
          <div style={{ flex: 1, overflowY: "auto", marginBottom: 14 }}>
            {turns.length === 0 && !partialText && (
              <div style={{ color: "#64748b", padding: "48px 16px", textAlign: "center", lineHeight: 1.7 }}>
                <div style={{ fontSize: 15, color: "#cbd5e1", marginBottom: 6 }}>VoiceAgent Desktop Ready</div>
                Click <strong>Start Mic</strong> for full-duplex conversation (or press <strong>Alt+Space</strong> for Push-to-Talk),
                or type below to test Pi Agent, guarded approvals, and live speech playback.
              </div>
            )}
            {turns.map((t) => (
              <div key={t.turnId} style={{ marginBottom: 16 }}>
                <div style={{ marginBottom: 4 }}>
                  <span style={{ color: "#38bdf8", fontWeight: 600 }}>User: </span>
                  {t.userTranscript}
                </div>
                {t.toolCalls.map((tc) => (
                  <div key={tc.toolCallId} style={{ fontSize: 12, color: "#fbbf24", background: "#1e293b", padding: "6px 10px", borderRadius: 6, margin: "4px 0" }}>
                    Tool <code>{tc.toolName}</code> [{tc.status}] {tc.sideEffectCommitted ? "(committed)" : ""}: {tc.result}
                  </div>
                ))}
                <div>
                  <span style={{ color: "#34d399", fontWeight: 600 }}>Agent: </span>
                  {t.interrupted ? (
                    <>
                      <span>{t.assistantAudibleReply || "(interrupted before first word)"}</span>
                      <span style={{ color: "#f87171", fontSize: 12, marginLeft: 6 }}>[Interrupted — unheard suffix preserved]</span>
                    </>
                  ) : (
                    t.assistantFullReply
                  )}
                </div>
              </div>
            ))}
            {partialText && (
              <div style={{ color: "#93c5fd", fontStyle: "italic" }}>User (live): {partialText}</div>
            )}
          </div>

          {pendingApproval && (
            <div role="dialog" style={{ border: "1px solid #f59e0b", background: "#451a03", borderRadius: 8, padding: 12, marginBottom: 12 }}>
              <strong>Approval Required: {pendingApproval.toolName}</strong>
              <div style={{ fontSize: 13, marginTop: 4 }}>{pendingApproval.impactSummary}</div>
              <div style={{ fontSize: 11, color: "#fde68a", marginTop: 4 }}>
                Click below or say "confirm" / "确认" (silence or ambiguous speech denies by default)
              </div>
              <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                <button
                  type="button"
                  onClick={() => void respondApproval(true, "ui", "Approved via desktop dialog")}
                  style={{ background: "#16a34a", border: "none", borderRadius: 6, padding: "5px 14px", cursor: "pointer", fontWeight: 600 }}
                >
                  Approve
                </button>
                <button
                  type="button"
                  onClick={() => void respondApproval(false, "ui", "Denied via desktop dialog")}
                  style={{ background: "#dc2626", border: "none", borderRadius: 6, padding: "5px 14px", cursor: "pointer", fontWeight: 600 }}
                >
                  Deny
                </button>
              </div>
            </div>
          )}

          <canvas ref={canvasRef} width={640} height={54} style={{ width: "100%", height: 54, background: "#0b0f17", borderRadius: 8, marginBottom: 10 }} />

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
              <button
                type="button"
                onClick={() => void toggleMic()}
                style={{ background: micActive ? "#dc2626" : "#2563eb", border: "none", borderRadius: 8, padding: "7px 14px", fontWeight: 600, cursor: "pointer" }}
              >
                {micActive ? "Stop Mic" : "Start Mic"}
              </button>

              {(["hands-free", "push-to-talk", "text", "mute"] as InteractionMode[]).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => {
                    setMode(m);
                    micRef.current.setMuted(m === "mute");
                  }}
                  style={{
                    background: mode === m ? "#0284c7" : "#1f2937",
                    border: "none",
                    borderRadius: 6,
                    padding: "6px 10px",
                    fontSize: 12,
                    cursor: "pointer",
                  }}
                >
                  {m}
                </button>
              ))}

              {mode === "push-to-talk" && (
                <button
                  type="button"
                  onMouseDown={() => setPttPressed(true)}
                  onMouseUp={() => setPttPressed(false)}
                  onMouseLeave={() => setPttPressed(false)}
                  style={{
                    background: pttPressed ? "#7c3aed" : "#312e81",
                    border: "1px solid #8b5cf6",
                    borderRadius: 8,
                    padding: "7px 12px",
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  {pttPressed ? "Talking... (Release)" : "Hold to Talk"}
                </button>
              )}

              <button
                type="button"
                onClick={() => triggerBargeInCancel("manual_stop")}
                style={{ background: "#374151", border: "none", borderRadius: 8, padding: "7px 12px", cursor: "pointer", fontSize: 12 }}
              >
                Stop / Barge-in
              </button>
            </div>

            <div style={{ display: "flex", gap: 8, fontSize: 12 }}>
              <label style={{ display: "flex", alignItems: "center", gap: 4, color: "#94a3b8" }}>
                <input type="checkbox" checked={aecEnabled} onChange={(e) => setAecEnabled(e.target.checked)} />
                AEC
              </label>
              <span style={{ padding: "4px 8px", borderRadius: 999, background: micActive ? "#0c4a6e" : "#1f2937" }}>
                Mic: {micActive ? (mode === "mute" ? "Muted" : "Listening") : "Idle"}
              </span>
              <span style={{ padding: "4px 8px", borderRadius: 999, background: assistantState === "speaking" ? "#065f46" : "#1f2937" }}>
                Agent: {assistantState}
              </span>
            </div>
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (!textInput.trim()) return;
              const q = textInput.trim();
              setTextInput("");
              void runTurn(q);
            }}
            style={{ display: "flex", gap: 8 }}
          >
            <input
              type="text"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="Speak or type here (try typing 'delete temp.log' to test approval guard)..."
              style={{ flex: 1, background: "#0b0f17", border: "1px solid #374151", borderRadius: 8, padding: "8px 12px" }}
            />
            <button type="submit" style={{ background: "#0ea5e9", border: "none", borderRadius: 8, padding: "8px 16px", fontWeight: 600, cursor: "pointer" }}>
              Send
            </button>
          </form>
          {statusNote && <div style={{ color: "#fbbf24", fontSize: 12, marginTop: 8 }}>{statusNote}</div>}
        </section>

        {devMode && (
          <aside style={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 12, padding: 16, fontSize: 13, display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <strong>Developer Telemetry HUD</strong>
              <button
                type="button"
                onClick={() => void handleExportLogs()}
                style={{ background: "#1f2937", border: "1px solid #374151", borderRadius: 6, padding: "3px 8px", fontSize: 11, cursor: "pointer" }}
              >
                Export JSONL
              </button>
            </div>

            <div>
              <div style={{ color: "#94a3b8", marginBottom: 4 }}>Silero VAD Probability: {(vadProb * 100).toFixed(1)}%</div>
              <div style={{ height: 8, background: "#1f2937", borderRadius: 4, overflow: "hidden" }}>
                <div style={{ width: `${Math.min(100, vadProb * 100)}%`, height: "100%", background: "#38bdf8" }} />
              </div>
            </div>

            <div>
              <div style={{ color: "#94a3b8", marginBottom: 6 }}>Causal GRU Turn Adapter</div>
              {[
                ["Completion (EOT)", adapterProb?.pCompletion ?? 0, "#22c55e"],
                ["Floor Taking (Interrupt)", adapterProb?.pFloorTaking ?? 0, "#ef4444"],
                ["Non-Floor (Backchannel)", adapterProb?.pNonFloorFeedback ?? 0, "#eab308"],
              ].map(([label, val, color]) => (
                <div key={String(label)} style={{ marginBottom: 6 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                    <span>{label}</span>
                    <span>{((val as number) * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{ height: 6, background: "#1f2937", borderRadius: 3 }}>
                    <div style={{ width: `${Math.min(100, (val as number) * 100)}%`, height: "100%", background: String(color) }} />
                  </div>
                </div>
              ))}
            </div>

            <div style={{ borderTop: "1px solid #1f2937", paddingTop: 10, display: "grid", gap: 4 }}>
              <div><strong>Latency</strong></div>
              <div>TTFA: {lastLatencyMs.ttfa ?? "-"} ms</div>
              <div>E2E Audible: {lastLatencyMs.e2e ?? "-"} ms</div>
              <div>Barge-in Stop: {lastLatencyMs.bargeIn ?? "-"} ms</div>
            </div>

            <div style={{ borderTop: "1px solid #1f2937", paddingTop: 10, display: "grid", gap: 4, fontSize: 12 }}>
              <div><strong>RTX 5070 & Active Models</strong></div>
              <div>Device: {health?.vram.device ?? "checking..."}</div>
              <div>VRAM: {health ? `${health.vram.allocated_mb} / ${health.vram.total_mb} MB` : "-"}</div>
              <div>VAD: {health?.models.vad?.backend ?? "-"}</div>
              <div>ASR: {health?.models.asr.backend ?? health?.models.asr.model_id ?? "-"}</div>
              <div>TTS: {health?.models.tts.backend ?? health?.models.tts.model_id ?? "-"}</div>
              <div>Turn Adapter: {health?.models.turn_adapter.version ?? "-"} ({health?.models.turn_adapter.params ?? 0} params)</div>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}

const rootEl = document.getElementById("root");
if (rootEl) {
  createRoot(rootEl).render(<VoiceAgentDesktop />);
}
