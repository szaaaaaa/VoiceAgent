/**
 * VoiceAgent R&D & Experiment Observation Workbench (V2.0).
 * Four focused views wired to the local Python speech core (127.0.0.1:8765):
 *  1. 实时信号示波台 (Observer mode watching live Pi /voice sessions + Sandbox mic/WAV injector)
 *  2. 模型热插拔注册表 (models/registry.json hot-swap & turn strategy switch)
 *  3. A/B 盲测实验台 (ASR multi-checkpoint diff/WER + TTS blind MOS evaluation & CSV export)
 *  4. 实验报告与 Bad Case 回放 (docs/experiments/ viewer + 20ms streaming audio replay)
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  createEvent,
  decodeAudioFrame,
  encodeAudioFrame,
  GenerationTracker,
  parseEventEnvelope,
  StreamKind,
  type ClientRole,
  type EventPayloadMap,
  type RegistrySnapshotResponse,
} from "@voice-agent/protocol";
import { computeRmsDb } from "@voice-agent/audio";
import { BrowserMicCapture, BrowserStreamPlayer } from "./audio-io";
import { ScopeCanvas, type TelemetryPoint } from "./ScopeCanvas";
import {
  activateRegistryModel,
  decodeAudioFileToPcm16k,
  encodePcm16kToWavBase64,
  fetchExperiments,
  fetchHealth,
  fetchRegistry,
  formatCsvRows,
  makeSyntheticProbeWavBase64,
  runAbAsr,
  runAbTts,
  type AbAsrResponse,
  type AbTtsResponse,
  type ExperimentsResponse,
  type HealthSnapshot,
} from "./workbench-api";

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

type WorkbenchTab = "scope" | "registry" | "ab_lab" | "experiments";

interface LiveEventEntry {
  id: number;
  ts: string;
  type: string;
  summary: string;
}

const DEFAULT_WS_URL = "ws://127.0.0.1:8765/ws/v1/voice";
const DEFAULT_API_BASE = "http://127.0.0.1:8765";
const SCOPE_WINDOW = 90;

const TURN_STRATEGY_LABELS: Record<string, string> = {
  hybrid: "混合控制 (CausalGRU + 规则 + 静音兜底)",
  adapter: "纯 CausalTurnGRU 模型",
  smart_turn: "Smart Turn 启发式基线",
  vad_only: "纯 Silero VAD 基线",
  fixed_silence: "固定静音阈值 (700ms)",
};

function makeEmptyScope(): TelemetryPoint[] {
  return Array.from({ length: SCOPE_WINDOW }, () => ({
    rmsDb: -90,
    vadProb: 0,
    pCompletion: 0,
    pFloorTaking: 0,
    pBackchannel: 0,
  }));
}

function formatTimeNow(): string {
  const d = new Date();
  return d.toTimeString().slice(0, 8);
}

function cardStyle(extra?: React.CSSProperties): React.CSSProperties {
  return {
    background: "#111827",
    border: "1px solid #1f2937",
    borderRadius: 10,
    padding: 14,
    ...extra,
  };
}

function btnStyle(active = false, tone: "blue" | "green" | "red" | "gray" = "blue"): React.CSSProperties {
  const palette = {
    blue: active ? "#2563eb" : "#1e293b",
    green: active ? "#059669" : "#1e293b",
    red: active ? "#dc2626" : "#1e293b",
    gray: "#1f2937",
  };
  return {
    background: palette[tone],
    border: "1px solid #334155",
    borderRadius: 6,
    padding: "6px 12px",
    cursor: "pointer",
    fontSize: 12,
    fontWeight: active ? 600 : 400,
  };
}

function VoiceWorkbench(): React.ReactElement {
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS_URL);
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [activeTab, setActiveTab] = useState<WorkbenchTab>("scope");
  const [clientRole, setClientRole] = useState<ClientRole>("observer");
  const [wsConnected, setWsConnected] = useState(false);
  const [heavyGpu, setHeavyGpu] = useState(false);
  const [restartingBackend, setRestartingBackend] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const [health, setHealth] = useState<HealthSnapshot | null>(null);
  const [registry, setRegistry] = useState<RegistrySnapshotResponse | null>(null);
  const [experiments, setExperiments] = useState<ExperimentsResponse | null>(null);
  const [selectedExpId, setSelectedExpId] = useState<string | null>(null);

  // Live oscilloscope & telemetry state
  const [scopePoints, setScopePoints] = useState<TelemetryPoint[]>(makeEmptyScope);
  const [liveEvents, setLiveEvents] = useState<LiveEventEntry[]>([]);
  const [liveTranscript, setLiveTranscript] = useState("");
  const [cursorState, setCursorState] = useState<{ mode: string; charIdx: number; durMs: number }>({
    mode: "idle",
    charIdx: 0,
    durMs: 0,
  });
  const [stepMs, setStepMs] = useState<number | null>(null);
  const [micActive, setMicActive] = useState(false);
  const [sandboxTtsText, setSandboxTtsText] = useState("你好，这里是 VoiceAgent 语音算法观测工作台。");
  const [replayingAudio, setReplayingAudio] = useState(false);

  // A/B Lab state
  const [selectedAsrIds, setSelectedAsrIds] = useState<string[]>([]);
  const [asrRefText, setAsrRefText] = useState("");
  const [asrAudioB64, setAsrAudioB64] = useState<string>("");
  const [asrAudioLabel, setAsrAudioLabel] = useState<string>("未上传（默认将使用内置 1.2s 测试音）");
  const [asrAbLoading, setAsrAbLoading] = useState(false);
  const [asrAbResult, setAsrAbResult] = useState<AbAsrResponse | null>(null);

  const [selectedTtsIds, setSelectedTtsIds] = useState<string[]>([]);
  const [ttsPrompt, setTtsPrompt] = useState("当前四模型同卡常驻显存仅占用两千四百兆，具备充足微调余量。");
  const [ttsBlindMode, setTtsBlindMode] = useState(true);
  const [ttsAbLoading, setTtsAbLoading] = useState(false);
  const [ttsAbResult, setTtsAbResult] = useState<AbTtsResponse | null>(null);
  const [ttsRevealed, setTtsRevealed] = useState(false);
  const [mosScores, setMosScores] = useState<Record<string, number>>({});

  const wsRef = useRef<WebSocket | null>(null);
  const micRef = useRef(new BrowserMicCapture());
  const playerRef = useRef(new BrowserStreamPlayer());
  const trackerRef = useRef(new GenerationTracker(`wb_${Date.now().toString(36)}`));
  const pendingTtsMetaRef = useRef(new Map<number, Array<{ charStart: number; charEnd: number; sampleRate: number }>>());
  const seqRef = useRef(0);
  const logSeqRef = useRef(0);
  const latestPointRef = useRef<TelemetryPoint>({
    rmsDb: -90,
    vadProb: 0,
    pCompletion: 0,
    pFloorTaking: 0,
    pBackchannel: 0,
  });
  const telemetryJsonlRef = useRef<Array<Record<string, unknown>>>([]);
  const pendingReplayRef = useRef<{ pcm16k: Int16Array; label: string } | null>(null);
  const replayFnRef = useRef<((pcm16k: Int16Array, label: string) => Promise<void>) | null>(null);

  const appendEventLog = useCallback((type: string, summary: string, raw?: Record<string, unknown>) => {
    logSeqRef.current += 1;
    const id = logSeqRef.current;
    setLiveEvents((prev) => [{ id, ts: formatTimeNow(), type, summary }, ...prev].slice(0, 80));
    telemetryJsonlRef.current.push({ ts: Date.now(), type, summary, ...(raw ?? {}) });
    if (telemetryJsonlRef.current.length > 600) telemetryJsonlRef.current.shift();
  }, []);

  const pushScopePatch = useCallback((patch: Partial<TelemetryPoint>) => {
    const next: TelemetryPoint = { ...latestPointRef.current, ...patch };
    latestPointRef.current = next;
    setScopePoints((prev) => [...prev.slice(1), next]);
  }, []);

  // Load Electron desktop config if running inside Electron.
  useEffect(() => {
    const bridge = window.voiceAgentDesktop;
    if (!bridge) return;
    void bridge.getBackendConfig().then((cfg) => {
      setWsUrl(cfg.wsUrl);
      setApiBase(cfg.apiBase);
      setHeavyGpu(cfg.heavyModelsEnabled);
    });
  }, []);

  const refreshMetadata = useCallback(async () => {
    try {
      const [h, reg, exps] = await Promise.all([
        fetchHealth(apiBase),
        fetchRegistry(apiBase),
        fetchExperiments(apiBase),
      ]);
      setHealth(h);
      setRegistry(reg);
      setExperiments(exps);
      setSelectedExpId((cur) => cur ?? exps.reports[0]?.id ?? null);
      setSelectedAsrIds((cur) =>
        cur.length > 0
          ? cur
          : reg.checkpoints.filter((c) => c.model_type === "asr").slice(0, 2).map((c) => c.checkpoint_id),
      );
      setSelectedTtsIds((cur) =>
        cur.length > 0
          ? cur
          : reg.checkpoints.filter((c) => c.model_type === "tts").slice(0, 2).map((c) => c.checkpoint_id),
      );
    } catch {
      // Backend still booting or offline.
    }
  }, [apiBase]);

  useEffect(() => {
    void refreshMetadata();
    const id = setInterval(() => {
      void fetchHealth(apiBase).then(setHealth).catch(() => {});
    }, 3000);
    return () => clearInterval(id);
  }, [apiBase, refreshMetadata]);

  // Connect WebSocket in either `observer` (watches live Pi /voice) or `controller` (sandbox) mode.
  useEffect(() => {
    let disposed = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        ws.send(
          JSON.stringify(
            createEvent(
              "session.init",
              { sessionId: trackerRef.current.ids.sessionId },
              {
                sample_rate: 16000,
                role: clientRole,
                audio_mode: "ws_pcm",
              },
            ),
          ),
        );
        appendEventLog("session.init", `已连接内核（角色：${clientRole === "observer" ? "Observer 旁路监听" : "Sandbox 独立实验"}）`);
        if (clientRole === "controller" && pendingReplayRef.current && replayFnRef.current) {
          const job = pendingReplayRef.current;
          pendingReplayRef.current = null;
          void replayFnRef.current(job.pcm16k, job.label);
        }
      };

      ws.onmessage = (ev) => {
        if (ev.data instanceof ArrayBuffer) {
          if (clientRole !== "controller") return;
          const frame = decodeAudioFrame(ev.data);
          if (frame.streamKind !== StreamKind.TtsOutput) return;
          const meta = pendingTtsMetaRef.current.get(frame.generationTag)?.shift();
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
        if (msg.type === "audio.level") {
          const p = msg.payload as EventPayloadMap["audio.level"];
          pushScopePatch({ rmsDb: p.mic_rms_db });
        } else if (msg.type === "vad.state") {
          const p = msg.payload as EventPayloadMap["vad.state"];
          pushScopePatch({ vadProb: p.vad_prob });
        } else if (msg.type === "vad.start") {
          const p = msg.payload as EventPayloadMap["vad.start"];
          pushScopePatch({ vadProb: p.vad_prob });
          appendEventLog("vad.start", `检测到语音起点 (prob=${p.vad_prob.toFixed(2)})`);
        } else if (msg.type === "vad.end") {
          const p = msg.payload as EventPayloadMap["vad.end"];
          pushScopePatch({ vadProb: p.vad_prob });
          appendEventLog("vad.end", `语音段结束 (${p.speech_duration_ms.toFixed(0)} ms)`);
        } else if (msg.type === "turn.prediction") {
          const p = msg.payload as EventPayloadMap["turn.prediction"];
          pushScopePatch({
            pCompletion: p.p_completion,
            pFloorTaking: p.p_floor_taking,
            pBackchannel: p.p_non_floor_feedback,
          });
          if (p.step_ms !== undefined) setStepMs(p.step_ms);
        } else if (msg.type === "asr.partial") {
          const p = msg.payload as EventPayloadMap["asr.partial"];
          setLiveTranscript(p.text);
        } else if (msg.type === "asr.final") {
          const p = msg.payload as EventPayloadMap["asr.final"];
          setLiveTranscript(p.text);
          appendEventLog("asr.final", `识别结果：${p.text} (${p.compute_ms} ms)`);
        } else if (msg.type === "playback.cursor") {
          const p = msg.payload as EventPayloadMap["playback.cursor"];
          setCursorState({ mode: p.mode, charIdx: p.played_char_index, durMs: p.played_duration_ms });
          appendEventLog("playback.cursor", `状态=${p.mode} | 已播字符=${p.played_char_index} | ${p.played_duration_ms} ms`);
        } else if (msg.type === "tts.chunk") {
          const p = msg.payload as EventPayloadMap["tts.chunk"];
          if (clientRole === "controller") {
            const arr = pendingTtsMetaRef.current.get(p.generation_tag) ?? [];
            arr.push({ charStart: p.char_start, charEnd: p.char_end, sampleRate: p.sample_rate });
            pendingTtsMetaRef.current.set(p.generation_tag, arr);
          }
          if (p.chunk_index === 0) {
            appendEventLog("tts.chunk", `句 #${p.sentence_index} 首包 TTFA=${p.ttfa_ms} ms (len=${p.duration_ms} ms)`);
          }
        } else if (msg.type === "telemetry.gpu") {
          void fetchRegistry(apiBase).then(setRegistry).catch(() => {});
        } else if (msg.type === "error") {
          const p = msg.payload as EventPayloadMap["error"];
          setNotice(`[${p.module}] ${p.message}`);
          appendEventLog("error", `[${p.module}] ${p.message}`);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        if (!disposed) {
          reconnectTimer = setTimeout(connect, 1500);
        }
      };
    };

    connect();
    return () => {
      disposed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      micRef.current.stop();
      setMicActive(false);
      wsRef.current?.close();
    };
  }, [wsUrl, clientRole, apiBase, appendEventLog, pushScopePatch]);

  const toggleSandboxMic = async (): Promise<void> => {
    if (clientRole !== "controller") {
      setClientRole("controller");
      setNotice("已切换为 Sandbox 独立实验模式，请再次点击开启麦克风。");
      return;
    }
    if (micActive) {
      micRef.current.stop();
      setMicActive(false);
      return;
    }
    try {
      await micRef.current.start((pcm16k, ptsMs, rmsDb) => {
        pushScopePatch({ rmsDb });
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) {
          seqRef.current += 1;
          ws.send(
            encodeAudioFrame({
              streamKind: StreamKind.MicInput,
              seq: seqRef.current,
              ptsMs,
              sampleRate: 16000,
              generationTag: 0,
              pcm16: pcm16k,
            }),
          );
        }
      });
      setMicActive(true);
      setNotice(null);
    } catch (err) {
      setNotice(`麦克风启动失败：${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const replayPcm16kThroughPipeline = async (pcm16k: Int16Array, label: string): Promise<void> => {
    if (clientRole !== "controller") {
      pendingReplayRef.current = { pcm16k, label };
      setClientRole("controller");
      setActiveTab("scope");
      setNotice(`已自动切换至 Sandbox 示波模式，连接就绪后自动回放：${label}`);
      return;
    }
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setNotice("WebSocket 未连接，无法注入回放音频。");
      return;
    }
    setActiveTab("scope");
    setReplayingAudio(true);
    appendEventLog("replay.start", `开始流式注入音频：${label} (${(pcm16k.length / 16000).toFixed(2)} s)`);
    try {
      const frameSamples = 320; // 20ms @ 16kHz
      let pts = 0;
      for (let off = 0; off < pcm16k.length; off += frameSamples) {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) break;
        const slice = pcm16k.subarray(off, Math.min(pcm16k.length, off + frameSamples));
        pushScopePatch({ rmsDb: computeRmsDb(slice) });
        seqRef.current += 1;
        wsRef.current.send(
          encodeAudioFrame({
            streamKind: StreamKind.MicInput,
            seq: seqRef.current,
            ptsMs: pts,
            sampleRate: 16000,
            generationTag: 0,
            pcm16: slice,
          }),
        );
        pts += 20;
        await new Promise((r) => setTimeout(r, 18));
      }
      // Append 140ms silence tail so VAD & ASR finalize cleanly.
      const silence = new Int16Array(frameSamples);
      for (let k = 0; k < 7; k += 1) {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) break;
        seqRef.current += 1;
        wsRef.current.send(
          encodeAudioFrame({
            streamKind: StreamKind.MicInput,
            seq: seqRef.current,
            ptsMs: pts + k * 20,
            sampleRate: 16000,
            generationTag: 0,
            pcm16: silence,
          }),
        );
        await new Promise((r) => setTimeout(r, 18));
      }
      appendEventLog("replay.end", `流式注入完成：${label}`);
    } finally {
      setReplayingAudio(false);
    }
  };
  replayFnRef.current = replayPcm16kThroughPipeline;

  const handleReplayFile = async (e: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    try {
      const pcm = await decodeAudioFileToPcm16k(file);
      await replayPcm16kThroughPipeline(pcm, file.name);
    } catch (err) {
      setNotice(`音频解码失败：${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleSandboxTtsPreview = (): void => {
    if (clientRole !== "controller") {
      setClientRole("controller");
      setNotice("已切换至 Sandbox 模式，重连后即可试听 TTS 合成。");
      return;
    }
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !sandboxTtsText.trim()) return;
    const gen = trackerRef.current.startGeneration();
    ws.send(
      JSON.stringify(
        createEvent(
          "tts.synthesize",
          { sessionId: trackerRef.current.ids.sessionId, turnId: gen.turnId, generationId: gen.generationId },
          {
            sentence_index: 0,
            text: sandboxTtsText.trim(),
            char_start: 0,
            char_end: sandboxTtsText.trim().length,
            generation_tag: gen.generationTag,
            speaker_profile: "default",
          },
        ),
      ),
    );
  };

  const handleActivateModel = async (task: string, checkpointId: string): Promise<void> => {
    try {
      const updated = await activateRegistryModel(apiBase, { task, checkpoint_id: checkpointId });
      setRegistry(updated);
      setHealth(await fetchHealth(apiBase));
      setNotice(`已热切换 [${task}] → ${checkpointId}`);
      appendEventLog("model.activate", `${task} -> ${checkpointId}`);
    } catch (err) {
      setNotice(`切换失败：${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleAsrFileSelect = async (e: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    try {
      const pcm = await decodeAudioFileToPcm16k(file);
      setAsrAudioB64(encodePcm16kToWavBase64(pcm, 16000));
      setAsrAudioLabel(`${file.name} (${(pcm.length / 16000).toFixed(2)} s)`);
    } catch (err) {
      setNotice(`读取 ASR 测试音频失败：${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleRunAsrAb = async (): Promise<void> => {
    setAsrAbLoading(true);
    setNotice(null);
    try {
      const b64 = asrAudioB64 || makeSyntheticProbeWavBase64(1.2);
      const res = await runAbAsr(apiBase, {
        audio_base64: b64,
        checkpoint_ids: selectedAsrIds,
        reference_text: asrRefText.trim() || null,
      });
      setAsrAbResult(res);
    } catch (err) {
      setNotice(`ASR A/B 对比失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setAsrAbLoading(false);
    }
  };

  const handleRunTtsAb = async (): Promise<void> => {
    if (!ttsPrompt.trim()) return;
    setTtsAbLoading(true);
    setNotice(null);
    setTtsRevealed(!ttsBlindMode);
    setMosScores({});
    try {
      const res = await runAbTts(apiBase, {
        text: ttsPrompt.trim(),
        checkpoint_ids: selectedTtsIds,
        blind: ttsBlindMode,
      });
      setTtsAbResult(res);
    } catch (err) {
      setNotice(`TTS A/B 盲测生成失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setTtsAbLoading(false);
    }
  };

  const handleExportMosCsv = (): void => {
    if (!ttsAbResult) return;
    const csv = formatCsvRows(
      ["slot_id", "checkpoint_id", "label", "mos_score", "ttfa_ms", "rtf", "duration_ms", "text"],
      ttsAbResult.candidates.map((c) => {
        const revealed = ttsAbResult.reveal_map[c.slot_id];
        return [
          c.slot_id,
          revealed?.checkpoint_id ?? c.checkpoint_id ?? "",
          revealed?.label ?? c.display_label,
          mosScores[c.slot_id] ?? "",
          c.ttfa_ms,
          c.rtf,
          c.duration_ms,
          ttsAbResult.text,
        ];
      }),
    );
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `voiceagent_tts_mos_${Date.now()}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const handleExportLogs = async (): Promise<void> => {
    const jsonl = telemetryJsonlRef.current.map((row) => JSON.stringify(row)).join("\n");
    if (window.voiceAgentDesktop) {
      const res = await window.voiceAgentDesktop.exportLogs(jsonl);
      if (res.saved && res.path) setNotice(`观测日志已保存至：${res.path}`);
      return;
    }
    const blob = new Blob([jsonl], { type: "application/x-ndjson" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `voiceagent_telemetry_${Date.now()}.jsonl`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const handleToggleHeavyGpu = async (): Promise<void> => {
    const bridge = window.voiceAgentDesktop;
    if (!bridge) {
      setNotice("请从桌面 Electron 工作台启动以支持一键热重启 GPU 重型权重。");
      return;
    }
    setRestartingBackend(true);
    const res = await bridge.restartBackend({ loadHeavy: !heavyGpu });
    setHeavyGpu(res.heavyModelsEnabled);
    setRestartingBackend(false);
    if (res.ok) {
      await refreshMetadata();
      setNotice(`内核已切换为：${res.heavyModelsEnabled ? "RTX 5070 完整 GPU 模型" : "轻量极速联调模式"}`);
    } else {
      setNotice("推理内核重启失败，请检查端口占用。");
    }
  };

  const latest = scopePoints[scopePoints.length - 1] ?? {
    rmsDb: -90,
    vadProb: 0,
    pCompletion: 0,
    pFloorTaking: 0,
    pBackchannel: 0,
  };
  const selectedExp = experiments?.reports.find((r) => r.id === selectedExpId) ?? experiments?.reports[0] ?? null;
  const asrList = registry?.checkpoints.filter((c) => c.model_type === "asr") ?? [];
  const turnList = registry?.checkpoints.filter((c) => c.model_type === "turn_adapter") ?? [];
  const ttsList = registry?.checkpoints.filter((c) => c.model_type === "tts") ?? [];

  return (
    <div style={{ maxWidth: 1260, margin: "0 auto", padding: 16 }}>
      {/* Header */}
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          borderBottom: "1px solid #1f2937",
          paddingBottom: 12,
          gap: 10,
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: wsConnected ? "#22c55e" : "#ef4444",
              display: "inline-block",
            }}
          />
          <strong style={{ fontSize: 18 }}>VoiceAgent 实验与观测工作台</strong>
          <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 999, background: "#1e293b", color: "#93c5fd" }}>
            V2.0 R&amp;D Workbench
          </span>
          {health?.vram && (
            <span style={{ fontSize: 11, color: "#94a3b8" }}>
              {health.vram.device} · VRAM {health.vram.allocated_mb} / {health.vram.total_mb} MB
            </span>
          )}
        </div>

        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <button
            type="button"
            style={btnStyle(clientRole === "observer", "blue")}
            onClick={() => setClientRole("observer")}
          >
            Observer 旁路观测 Pi
          </button>
          <button
            type="button"
            style={btnStyle(clientRole === "controller", "green")}
            onClick={() => setClientRole("controller")}
          >
            Sandbox 独立实验
          </button>
          <button
            type="button"
            disabled={restartingBackend}
            style={btnStyle(heavyGpu, "green")}
            onClick={() => void handleToggleHeavyGpu()}
          >
            {restartingBackend ? "重启中..." : heavyGpu ? "GPU 完整权重：开" : "GPU 轻量模式"}
          </button>
          <button type="button" style={btnStyle(false, "gray")} onClick={() => void handleExportLogs()}>
            导出日志
          </button>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav style={{ display: "flex", gap: 8, marginTop: 12, marginBottom: 14, flexWrap: "wrap" }}>
        {(
          [
            ["scope", "1. 实时信号示波台"],
            ["registry", "2. 模型热插拔注册表"],
            ["ab_lab", "3. ASR / TTS A/B 盲测实验台"],
            ["experiments", "4. 实验报告与 Bad Case 回放"],
          ] as Array<[WorkbenchTab, string]>
        ).map(([tab, label]) => (
          <button
            key={tab}
            type="button"
            style={btnStyle(activeTab === tab, "blue")}
            onClick={() => setActiveTab(tab)}
          >
            {label}
          </button>
        ))}
      </nav>

      {notice && (
        <div
          style={{
            marginBottom: 12,
            padding: "8px 12px",
            borderRadius: 8,
            background: "#1e293b",
            border: "1px solid #38bdf8",
            fontSize: 12,
            display: "flex",
            justifyContent: "space-between",
          }}
        >
          <span>{notice}</span>
          <button
            type="button"
            onClick={() => setNotice(null)}
            style={{ background: "transparent", border: "none", cursor: "pointer", color: "#94a3b8" }}
          >
            ✕
          </button>
        </div>
      )}

      {/* TAB 1: LIVE OSCILLOSCOPE & OBSERVER */}
      {activeTab === "scope" && (
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <section style={cardStyle()}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <strong>全双工实时信号示波器（VAD + CausalTurnGRU 三头概率）</strong>
                <span style={{ fontSize: 12, color: "#94a3b8" }}>
                  {clientRole === "observer"
                    ? "正在旁路监听本机 Pi /voice 实时信号"
                    : "Sandbox 独立音频注入模式"}
                </span>
              </div>
              <ScopeCanvas points={scopePoints} height={210} />
              <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8, marginTop: 12, fontSize: 12 }}>
                <div style={cardStyle({ padding: 8, background: "#0b1220" })}>
                  <div style={{ color: "#94a3b8" }}>Silero VAD</div>
                  <strong style={{ color: "#38bdf8", fontSize: 16 }}>{(latest.vadProb * 100).toFixed(1)}%</strong>
                </div>
                <div style={cardStyle({ padding: 8, background: "#0b1220" })}>
                  <div style={{ color: "#94a3b8" }}>EOT 完成</div>
                  <strong style={{ color: "#34d399", fontSize: 16 }}>{(latest.pCompletion * 100).toFixed(1)}%</strong>
                </div>
                <div style={cardStyle({ padding: 8, background: "#0b1220" })}>
                  <div style={{ color: "#94a3b8" }}>Floor 打断</div>
                  <strong style={{ color: "#f43f5e", fontSize: 16 }}>{(latest.pFloorTaking * 100).toFixed(1)}%</strong>
                </div>
                <div style={cardStyle({ padding: 8, background: "#0b1220" })}>
                  <div style={{ color: "#94a3b8" }}>Backchannel</div>
                  <strong style={{ color: "#fbbf24", fontSize: 16 }}>{(latest.pBackchannel * 100).toFixed(1)}%</strong>
                </div>
                <div style={cardStyle({ padding: 8, background: "#0b1220" })}>
                  <div style={{ color: "#94a3b8" }}>播放游标 / 步耗时</div>
                  <strong style={{ fontSize: 13 }}>
                    {cursorState.mode} · {cursorState.charIdx}字{stepMs !== null ? ` · ${stepMs}ms` : ""}
                  </strong>
                </div>
              </div>
            </section>

            <section style={cardStyle()}>
              <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 6 }}>最新流式 ASR 识别文本</div>
              <div
                style={{
                  minHeight: 42,
                  padding: 10,
                  borderRadius: 8,
                  background: "#060911",
                  border: "1px solid #1e293b",
                  fontSize: 14,
                }}
              >
                {liveTranscript || "等待 Pi /voice 或 Sandbox 语音输入..."}
              </div>

              <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap", alignItems: "center" }}>
                <button
                  type="button"
                  style={btnStyle(micActive, micActive ? "red" : "green")}
                  onClick={() => void toggleSandboxMic()}
                >
                  {micActive ? "停止 Sandbox 麦克风" : "开启 Sandbox 麦克风"}
                </button>
                <label style={{ ...btnStyle(false, "gray"), display: "inline-block" }}>
                  {replayingAudio ? "正在注入回放..." : "注入音频文件回放 (WAV/MP3)"}
                  <input
                    type="file"
                    accept="audio/*"
                    disabled={replayingAudio}
                    onChange={(e) => void handleReplayFile(e)}
                    style={{ display: "none" }}
                  />
                </label>
                <input
                  type="text"
                  value={sandboxTtsText}
                  onChange={(e) => setSandboxTtsText(e.target.value)}
                  style={{
                    flex: 1,
                    minWidth: 200,
                    background: "#060911",
                    border: "1px solid #334155",
                    borderRadius: 6,
                    padding: "6px 10px",
                    fontSize: 12,
                  }}
                />
                <button type="button" style={btnStyle(true, "blue")} onClick={handleSandboxTtsPreview}>
                  试听 TTS
                </button>
              </div>
            </section>
          </div>

          {/* Right telemetry & event log */}
          <aside style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <section style={cardStyle()}>
              <strong style={{ fontSize: 13 }}>当前激活模型栈</strong>
              <div style={{ fontSize: 12, color: "#cbd5e1", marginTop: 8, lineHeight: 1.8 }}>
                <div>VAD：<code>{health?.models.vad?.backend ?? "-"}</code></div>
                <div>ASR：<code>{health?.models.asr.active_id ?? health?.models.asr.backend ?? "-"}</code></div>
                <div>
                  Turn：<code>{health?.models.turn_adapter.version ?? "-"}</code> ({health?.models.turn_adapter.strategy ?? "hybrid"})
                </div>
                <div>TTS：<code>{health?.models.tts.active_id ?? health?.models.tts.backend ?? "-"}</code></div>
                <div>当前观测端数量：<code>{health?.observers_connected ?? 0}</code></div>
              </div>
            </section>

            <section style={cardStyle({ flex: 1, display: "flex", flexDirection: "column", maxHeight: 360 })}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <strong style={{ fontSize: 13 }}>实时协议事件流</strong>
                <button type="button" style={btnStyle(false, "gray")} onClick={() => setLiveEvents([])}>
                  清空
                </button>
              </div>
              <div style={{ overflowY: "auto", fontSize: 11, lineHeight: 1.65 }}>
                {liveEvents.length === 0 ? (
                  <div style={{ color: "#64748b" }}>暂无事件。请在终端启用 Pi `/voice` 或切换到 Sandbox 注入测试音频。</div>
                ) : (
                  liveEvents.map((ev) => (
                    <div key={ev.id} style={{ borderBottom: "1px solid #1e293b", padding: "4px 0" }}>
                      <span style={{ color: "#64748b" }}>[{ev.ts}]</span>{" "}
                      <code style={{ color: "#38bdf8" }}>{ev.type}</code> {ev.summary}
                    </div>
                  ))
                )}
              </div>
            </section>
          </aside>
        </div>
      )}

      {/* TAB 2: CHECKPOINT REGISTRY HOT-SWAP */}
      {activeTab === "registry" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <section style={cardStyle()}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
              <div>
                <strong>全双工轮次决策策略（Turn Strategy 热切换）</strong>
                <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 2 }}>
                  无需重启进程，实时切换在线轮次控制器策略以对比消融效果
                </div>
              </div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {Object.entries(TURN_STRATEGY_LABELS).map(([key, label]) => (
                  <button
                    key={key}
                    type="button"
                    style={btnStyle(registry?.turn_strategy === key, "blue")}
                    onClick={() => void handleActivateModel("turn_strategy", key)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </section>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
            {(
              [
                ["asr", "ASR 流式识别与 LoRA 权重", asrList],
                ["turn_adapter", "CausalTurnGRU 与消融权重", turnList],
                ["tts", "TTS 引擎与音色微调权重", ttsList],
              ] as const
            ).map(([task, title, list]) => (
              <section key={task} style={cardStyle()}>
                <strong style={{ fontSize: 14 }}>{title}</strong>
                <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 10 }}>
                  {list.map((ckpt) => {
                    const isActive = registry?.active_ids[task] === ckpt.checkpoint_id;
                    return (
                      <div
                        key={ckpt.checkpoint_id}
                        style={{
                          padding: 10,
                          borderRadius: 8,
                          background: isActive ? "#0f2942" : "#0b1220",
                          border: `1px solid ${isActive ? "#38bdf8" : "#1e293b"}`,
                          fontSize: 12,
                        }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6 }}>
                          <strong>{ckpt.label}</strong>
                          <button
                            type="button"
                            disabled={isActive}
                            style={btnStyle(isActive, "green")}
                            onClick={() => void handleActivateModel(task, ckpt.checkpoint_id)}
                          >
                            {isActive ? "已激活" : "热切换"}
                          </button>
                        </div>
                        <div style={{ color: "#94a3b8", marginTop: 4 }}>
                          ID: <code>{ckpt.checkpoint_id}</code> · {ckpt.experiment_id}
                        </div>
                        {ckpt.path && (
                          <div style={{ color: "#64748b", marginTop: 2, wordBreak: "break-all" }}>{ckpt.path}</div>
                        )}
                        {Object.keys(ckpt.metrics).length > 0 && (
                          <div style={{ color: "#34d399", marginTop: 4 }}>
                            指标：
                            {Object.entries(ckpt.metrics)
                              .map(([k, v]) => `${k}=${Number(v).toFixed(3)}`)
                              .join(" | ")}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </section>
            ))}
          </div>
        </div>
      )}

      {/* TAB 3: ASR & TTS A/B EVALUATION LAB */}
      {activeTab === "ab_lab" && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          {/* ASR A/B */}
          <section style={cardStyle()}>
            <strong style={{ fontSize: 15 }}>ASR 同音频多模型对比（WER / CER / RTF）</strong>
            <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 4 }}>
              勾选多个 ASR Checkpoint，对同一段音频执行并发对照识别
            </div>

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
              {asrList.map((c) => {
                const checked = selectedAsrIds.includes(c.checkpoint_id);
                return (
                  <label key={c.checkpoint_id} style={{ fontSize: 12, display: "flex", gap: 4, alignItems: "center" }}>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() =>
                        setSelectedAsrIds((prev) =>
                          checked ? prev.filter((id) => id !== c.checkpoint_id) : [...prev, c.checkpoint_id],
                        )
                      }
                    />
                    {c.label}
                  </label>
                );
              })}
            </div>

            <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 10, flexWrap: "wrap" }}>
              <label style={{ ...btnStyle(false, "gray"), display: "inline-block" }}>
                上传测试音频 (WAV/MP3)
                <input type="file" accept="audio/*" onChange={(e) => void handleAsrFileSelect(e)} style={{ display: "none" }} />
              </label>
              <span style={{ fontSize: 11, color: "#94a3b8" }}>{asrAudioLabel}</span>
            </div>

            <input
              type="text"
              placeholder="可选：输入标准参考文本（Reference），自动计算 WER / CER"
              value={asrRefText}
              onChange={(e) => setAsrRefText(e.target.value)}
              style={{
                width: "100%",
                marginTop: 10,
                background: "#060911",
                border: "1px solid #334155",
                borderRadius: 6,
                padding: "7px 10px",
                fontSize: 12,
              }}
            />

            <button
              type="button"
              disabled={asrAbLoading}
              style={{ ...btnStyle(true, "blue"), marginTop: 10 }}
              onClick={() => void handleRunAsrAb()}
            >
              {asrAbLoading ? "对比识别中..." : "运行 ASR A/B 对比"}
            </button>

            {asrAbResult && (
              <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 8 }}>
                {asrAbResult.candidates.map((cand) => (
                  <div key={cand.checkpoint_id} style={cardStyle({ background: "#060911", padding: 10, fontSize: 12 })}>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <strong>{cand.label}</strong>
                      <span style={{ color: "#94a3b8" }}>
                        {cand.compute_ms} ms · RTF {cand.rtf}
                        {cand.wer !== null ? ` · WER ${(cand.wer * 100).toFixed(1)}% · CER ${(cand.cer! * 100).toFixed(1)}%` : ""}
                      </span>
                    </div>
                    <div style={{ marginTop: 6, color: "#e2e8f0" }}>{cand.transcript || "(空转写)"}</div>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* TTS Blind MOS A/B */}
          <section style={cardStyle()}>
            <strong style={{ fontSize: 15 }}>TTS 双盲音色对比与 MOS 评分台</strong>
            <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 4 }}>
              匿名打乱 Pretrained / Zero-shot / Finetuned 音色顺序，盲听打分后揭晓身份并导出 CSV
            </div>

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
              {ttsList.map((c) => {
                const checked = selectedTtsIds.includes(c.checkpoint_id);
                return (
                  <label key={c.checkpoint_id} style={{ fontSize: 12, display: "flex", gap: 4, alignItems: "center" }}>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() =>
                        setSelectedTtsIds((prev) =>
                          checked ? prev.filter((id) => id !== c.checkpoint_id) : [...prev, c.checkpoint_id],
                        )
                      }
                    />
                    {c.label}
                  </label>
                );
              })}
            </div>

            <textarea
              rows={2}
              value={ttsPrompt}
              onChange={(e) => setTtsPrompt(e.target.value)}
              style={{
                width: "100%",
                marginTop: 10,
                background: "#060911",
                border: "1px solid #334155",
                borderRadius: 6,
                padding: 8,
                fontSize: 12,
              }}
            />

            <div style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 8, flexWrap: "wrap" }}>
              <label style={{ fontSize: 12, display: "flex", gap: 4, alignItems: "center" }}>
                <input type="checkbox" checked={ttsBlindMode} onChange={(e) => setTtsBlindMode(e.target.checked)} />
                开启双盲随机打乱
              </label>
              <button
                type="button"
                disabled={ttsAbLoading}
                style={btnStyle(true, "green")}
                onClick={() => void handleRunTtsAb()}
              >
                {ttsAbLoading ? "合成对比音频中..." : "生成盲测对比音频"}
              </button>
              {ttsAbResult && ttsBlindMode && !ttsRevealed && (
                <button type="button" style={btnStyle(true, "blue")} onClick={() => setTtsRevealed(true)}>
                  揭晓真实模型身份
                </button>
              )}
              {ttsAbResult && (
                <button type="button" style={btnStyle(false, "gray")} onClick={handleExportMosCsv}>
                  导出 MOS CSV
                </button>
              )}
            </div>

            {ttsAbResult && (
              <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 10 }}>
                {ttsAbResult.candidates.map((cand) => {
                  const revealed = ttsRevealed ? ttsAbResult.reveal_map[cand.slot_id] : null;
                  return (
                    <div key={cand.slot_id} style={cardStyle({ background: "#060911", padding: 10, fontSize: 12 })}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                        <strong>
                          {cand.display_label}
                          {revealed ? ` → ${revealed.label} (${revealed.checkpoint_id})` : ""}
                        </strong>
                        <span style={{ color: "#94a3b8" }}>
                          TTFA {cand.ttfa_ms} ms · RTF {cand.rtf} · {cand.duration_ms} ms
                        </span>
                      </div>
                      <audio
                        controls
                        src={`data:audio/wav;base64,${cand.wav_base64}`}
                        style={{ width: "100%", height: 34 }}
                      />
                      <div style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 6 }}>
                        <span style={{ color: "#94a3b8" }}>主观 MOS 评分：</span>
                        {[1, 2, 3, 4, 5].map((score) => (
                          <button
                            key={score}
                            type="button"
                            style={btnStyle(mosScores[cand.slot_id] === score, "blue")}
                            onClick={() => setMosScores((prev) => ({ ...prev, [cand.slot_id]: score }))}
                          >
                            {score} 分
                          </button>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        </div>
      )}

      {/* TAB 4: EXPERIMENT REPORTS & BAD CASE REPLAY */}
      {activeTab === "experiments" && (
        <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 14 }}>
          <aside style={cardStyle({ display: "flex", flexDirection: "column", gap: 8 })}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <strong style={{ fontSize: 13 }}>实验报告列表 (docs/experiments)</strong>
              <button type="button" style={btnStyle(false, "gray")} onClick={() => void refreshMetadata()}>
                刷新
              </button>
            </div>
            {(experiments?.reports ?? []).map((rep) => (
              <button
                key={rep.id}
                type="button"
                onClick={() => setSelectedExpId(rep.id)}
                style={{
                  textAlign: "left",
                  padding: 10,
                  borderRadius: 8,
                  cursor: "pointer",
                  background: selectedExp?.id === rep.id ? "#0f2942" : "#0b1220",
                  border: `1px solid ${selectedExp?.id === rep.id ? "#38bdf8" : "#1e293b"}`,
                  fontSize: 12,
                }}
              >
                <div style={{ fontWeight: 600 }}>{rep.title}</div>
                <div style={{ color: "#64748b", fontSize: 11, marginTop: 2 }}>
                  {rep.markdown_file ?? rep.manifest_file}
                </div>
              </button>
            ))}

            <div style={{ borderTop: "1px solid #1f2937", paddingTop: 10, marginTop: 8 }}>
              <strong style={{ fontSize: 12 }}>Bad Case 录音回放诊断</strong>
              <div style={{ fontSize: 11, color: "#94a3b8", margin: "4px 0 8px" }}>
                导入误截断或误识别样本，自动跳转示波台按 20ms 实时帧回放定位问题帧
              </div>
              <label style={{ ...btnStyle(true, "blue"), display: "inline-block" }}>
                选择 Bad Case 音频回放
                <input type="file" accept="audio/*" onChange={(e) => void handleReplayFile(e)} style={{ display: "none" }} />
              </label>
            </div>
          </aside>

          <section style={cardStyle()}>
            {selectedExp ? (
              <div>
                <h3 style={{ marginTop: 0 }}>{selectedExp.title}</h3>
                {selectedExp.markdown && (
                  <pre
                    style={{
                      whiteSpace: "pre-wrap",
                      background: "#060911",
                      border: "1px solid #1e293b",
                      borderRadius: 8,
                      padding: 14,
                      fontSize: 12,
                      lineHeight: 1.65,
                      overflowX: "auto",
                    }}
                  >
                    {selectedExp.markdown}
                  </pre>
                )}
                {selectedExp.manifest && (
                  <details style={{ marginTop: 10 }}>
                    <summary style={{ cursor: "pointer", fontSize: 12, color: "#38bdf8" }}>
                      展开原始 JSON Manifest ({selectedExp.manifest_file})
                    </summary>
                    <pre
                      style={{
                        whiteSpace: "pre-wrap",
                        background: "#060911",
                        border: "1px solid #1e293b",
                        borderRadius: 8,
                        padding: 12,
                        fontSize: 11,
                        marginTop: 8,
                        maxHeight: 280,
                        overflowY: "auto",
                      }}
                    >
                      {JSON.stringify(selectedExp.manifest, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            ) : (
              <div style={{ color: "#64748b" }}>暂无已归档实验报告。</div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

const rootEl = document.getElementById("root");
if (rootEl) {
  createRoot(rootEl).render(<VoiceWorkbench />);
}
