import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  createEvent,
  decodeAudioFrame,
  encodeAudioFrame,
  GenerationTracker,
  parseEventEnvelope,
  StreamKind,
} from "@voice-agent/protocol";
import { HybridTurnManager, PlaybackAwareHistory, type TurnAdapterProbabilities, type TurnStrategy } from "@voice-agent/interaction";
import { SentenceBuffer } from "@voice-agent/speech";
import { BrowserMicCapture, BrowserStreamPlayer, listAudioDevices, type AudioDeviceOption } from "./audio-io";

const WS_URL = "ws://127.0.0.1:8765/ws/v1/voice";
const API_BASE = "http://127.0.0.1:8765";

interface HealthInfo {
  status: string;
  vram: { device: string; allocated_mb: number; reserved_mb: number; total_mb: number };
  models: {
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

function VoiceAgentDesktop(): React.ReactElement {
  const [connected, setConnected] = useState(false);
  const [micActive, setMicActive] = useState(false);
  const [mode, setMode] = useState<"hands-free" | "push-to-talk" | "text" | "mute">("hands-free");
  const [agentMode, setAgentMode] = useState<"conversation" | "action">("conversation");
  const [permLevel, setPermLevel] = useState<"readonly" | "ask" | "trusted">("ask");
  const [turnStrategy, setTurnStrategy] = useState<TurnStrategy>("hybrid");
  const [assistantState, setAssistantState] = useState("idle");
  const [partialText, setPartialText] = useState("");
  const [vadProb, setVadProb] = useState(0);
  const [adapterProb, setAdapterProb] = useState<TurnAdapterProbabilities | null>(null);
  const [rmsHistory, setRmsHistory] = useState<number[]>(() => Array(48).fill(-90));
  const [turnsVersion, setTurnsVersion] = useState(0);
  const [textInput, setTextInput] = useState("");
  const [devMode, setDevMode] = useState(true);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [devices, setDevices] = useState<AudioDeviceOption[]>([]);
  const [selectedMic, setSelectedMic] = useState("default");
  const [lastLatencyMs, setLastLatencyMs] = useState<{ ttfa?: number; e2e?: number; bargeIn?: number }>({});
  const [pendingApproval, setPendingApproval] = useState<PendingApprovalUi | null>(null);
  const [statusNote, setStatusNote] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const micRef = useRef(new BrowserMicCapture());
  const playerRef = useRef(new BrowserStreamPlayer());
  const trackerRef = useRef(new GenerationTracker(`gui_${Date.now().toString(36)}`));
  const historyRef = useRef(new PlaybackAwareHistory());
  const turnManagerRef = useRef(new HybridTurnManager({ strategy: "hybrid" }));
  const pendingMetaRef = useRef(new Map<number, Array<{ charStart: number; charEnd: number; sampleRate: number }>>());
  const audioSeqRef = useRef(0);
  const speechEndRef = useRef(0);
  const vadInSpeechRef = useRef(false);
  const speechStartPtsRef = useRef(0);
  const lastPtsRef = useRef(0);
  const suspectedInterruptMsRef = useRef(0);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

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
      ctx.fillStyle = vadProb >= 0.5 ? "#38bdf8" : "#475569";
      ctx.fillRect(i * barW + 2, (canvas.height - h) / 2, Math.max(2, barW - 4), h);
    });
  }, [rmsHistory, vadProb]);

  // Poll backend health & VRAM telemetry every 3s.
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) setHealth((await res.json()) as HealthInfo);
      } catch {
        // Offline
      }
    };
    void poll();
    const id = setInterval(() => void poll(), 3000);
    return () => clearInterval(id);
  }, []);

  // Connect local WebSocket + wire player state.
  useEffect(() => {
    let closed = false;
    playerRef.current.setListener((st) => {
      if (st.mode === "playing") {
        setAssistantState("speaking");
        if (speechEndRef.current > 0) {
          setLastLatencyMs((prev) => ({ ...prev, e2e: Math.round(performance.now() - speechEndRef.current) }));
          speechEndRef.current = 0;
        }
      } else if (st.mode === "idle") {
        setAssistantState((cur) => (cur === "speaking" ? "idle" : cur));
      }
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(
          JSON.stringify(
            createEvent(
              "playback.ack",
              trackerRef.current.ids,
              {
                playback_state: st.mode,
                played_char_index: st.playedCharIndex,
                played_duration_ms: st.playedDurationMs,
              },
            ),
          ),
        );
      }
    });

    const connect = () => {
      const ws = new WebSocket(WS_URL);
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
          if (playerRef.current.getMode() === "playing") {
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
            void runTurn(p.text.trim());
          }
        } else if (msg.type === "turn.prediction") {
          const p = msg.payload as {
            p_completion: number;
            p_floor_taking: number;
            p_non_floor_feedback: number;
            assistant_speaking: boolean;
          };
          const probs: TurnAdapterProbabilities = {
            pCompletion: p.p_completion,
            pFloorTaking: p.p_floor_taking,
            pNonFloorFeedback: p.p_non_floor_feedback,
          };
          setAdapterProb(probs);
          if (playerRef.current.getMode() === "ducked" || playerRef.current.getMode() === "playing") {
            const decision = turnManagerRef.current.evaluate({
              vadInSpeech: vadInSpeechRef.current,
              speechDurationMs: Math.max(0, lastPtsRef.current - speechStartPtsRef.current),
              silenceDurationMs: vadInSpeechRef.current ? 0 : 250,
              assistantSpeaking: true,
              transcript: partialText,
              adapter: probs,
            });
            if (decision.event === "BACKCHANNEL" || decision.event === "NOISE") {
              playerRef.current.resume();
            } else if (decision.event === "INTERRUPTION") {
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
          if (p.ttfa_ms > 0) {
            setLastLatencyMs((prev) => ({ ...prev, ttfa: Math.round(p.ttfa_ms) }));
          }
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
      micRef.current.stop();
      wsRef.current?.close();
    };
  }, []);

  const triggerBargeInCancel = (reason = "user_interruption"): void => {
    const active = trackerRef.current.cancelCurrentGeneration();
    const cursor = playerRef.current.cancel();
    if (suspectedInterruptMsRef.current > 0) {
      setLastLatencyMs((prev) => ({
        ...prev,
        bargeIn: Math.round(performance.now() - suspectedInterruptMsRef.current),
      }));
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
  };

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
          setRmsHistory((prev) => [...prev.slice(1), rmsDb]);
          const ws = wsRef.current;
          if (ws && ws.readyState === WebSocket.OPEN && mode !== "text" && mode !== "mute") {
            audioSeqRef.current += 1;
            ws.send(
              encodeAudioFrame({
                streamKind: StreamKind.MicInput,
                seq: audioSeqRef.current,
                ptsMs,
                sampleRate: 16000,
                generationTag: 0,
                pcm16: pcm16k,
              }),
            );
          }
        },
        { deviceId: selectedMic, aecEnabled: true, noiseSuppression: true },
      );
      setMicActive(true);
      setDevices(await listAudioDevices());
      setStatusNote(null);
    } catch (err) {
      setStatusNote(`Microphone error: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const runTurn = async (userText: string): Promise<void> => {
    speechEndRef.current = performance.now();
    const turnId = trackerRef.current.startTurn();
    const { generationId, generationTag } = trackerRef.current.startGeneration();
    historyRef.current.recordUserTurn(turnId, userText);
    setTurnsVersion((v) => v + 1);
    setAssistantState("thinking");

    try {
      const res = await fetch(`${API_BASE}/api/agent/turn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: trackerRef.current.ids.sessionId,
          turn_id: turnId,
          user_text: userText,
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

      const buf = new SentenceBuffer();
      const segs = [...buf.pushDelta(data.reply), ...buf.flush()];
      const speakable = segs.map((s) => s.text).join(" ");
      historyRef.current.recordAssistantReply(turnId, data.reply, speakable);
      setTurnsVersion((v) => v + 1);

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
    } catch (err) {
      setStatusNote(`Agent error: ${err instanceof Error ? err.message : String(err)}`);
      setAssistantState("error");
    }
  };

  const turns = historyRef.current.getTurns();
  void turnsVersion;

  return (
    <div style={{ maxWidth: 1080, margin: "0 auto", padding: 20 }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid #1f2937", paddingBottom: 14, gap: 12, flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ width: 10, height: 10, borderRadius: "50%", background: connected ? "#22c55e" : "#ef4444", display: "inline-block" }} />
          <strong style={{ fontSize: 18 }}>VoiceAgent</strong>
          <span style={{ fontSize: 12, color: "#94a3b8" }}>Full-Duplex Local Voice Assistant</span>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", fontSize: 13 }}>
          <select
            value={selectedMic}
            onChange={(e) => setSelectedMic(e.target.value)}
            style={{ background: "#111827", border: "1px solid #374151", borderRadius: 6, padding: "4px 8px" }}
          >
            <option value="default">Default Mic</option>
            {devices.filter((d) => d.kind === "audioinput").map((d) => (
              <option key={d.deviceId} value={d.deviceId}>{d.label}</option>
            ))}
          </select>

          <select
            value={turnStrategy}
            onChange={(e) => {
              const s = e.target.value as TurnStrategy;
              setTurnStrategy(s);
              turnManagerRef.current.setStrategy(s);
            }}
            style={{ background: "#111827", border: "1px solid #374151", borderRadius: 6, padding: "4px 8px" }}
          >
            <option value="hybrid">Turn: Hybrid (GRU+Rules)</option>
            <option value="adapter">Turn: Causal GRU</option>
            <option value="smart_turn">Turn: Smart Turn</option>
            <option value="fixed_silence">Turn: Fixed Silence</option>
          </select>

          <select
            value={agentMode}
            onChange={(e) => setAgentMode(e.target.value as "conversation" | "action")}
            style={{ background: "#111827", border: "1px solid #374151", borderRadius: 6, padding: "4px 8px" }}
          >
            <option value="conversation">Mode: Conversation</option>
            <option value="action">Mode: Action</option>
          </select>

          <select
            value={permLevel}
            onChange={(e) => setPermLevel(e.target.value as "readonly" | "ask" | "trusted")}
            style={{ background: "#111827", border: "1px solid #374151", borderRadius: 6, padding: "4px 8px" }}
          >
            <option value="readonly">Policy: Readonly</option>
            <option value="ask">Policy: Ask</option>
            <option value="trusted">Policy: Trusted</option>
          </select>

          <button
            type="button"
            onClick={() => setDevMode((v) => !v)}
            style={{ background: devMode ? "#1d4ed8" : "#1f2937", border: "none", borderRadius: 6, padding: "5px 10px", cursor: "pointer" }}
          >
            Dev HUD
          </button>
        </div>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: devMode ? "1.6fr 1fr" : "1fr", gap: 16, marginTop: 16 }}>
        <section style={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 12, padding: 16, display: "flex", flexDirection: "column", minHeight: 520 }}>
          <div style={{ flex: 1, overflowY: "auto", marginBottom: 14 }}>
            {turns.length === 0 && !partialText && (
              <div style={{ color: "#64748b", padding: "32px 0", textAlign: "center" }}>
                Click <strong>Start Mic</strong> for hands-free full-duplex voice, or type below to test instantly.
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
              <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                <button type="button" onClick={() => setPendingApproval(null)} style={{ background: "#16a34a", border: "none", borderRadius: 6, padding: "4px 12px", cursor: "pointer" }}>
                  Approve
                </button>
                <button type="button" onClick={() => setPendingApproval(null)} style={{ background: "#dc2626", border: "none", borderRadius: 6, padding: "4px 12px", cursor: "pointer" }}>
                  Deny
                </button>
              </div>
            </div>
          )}

          <canvas ref={canvasRef} width={600} height={54} style={{ width: "100%", height: 54, background: "#0b0f17", borderRadius: 8, marginBottom: 10 }} />

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                type="button"
                onClick={() => void toggleMic()}
                style={{ background: micActive ? "#dc2626" : "#2563eb", border: "none", borderRadius: 8, padding: "8px 16px", fontWeight: 600, cursor: "pointer" }}
              >
                {micActive ? "Stop Mic" : "Start Mic"}
              </button>
              <button
                type="button"
                onClick={() => triggerBargeInCancel("manual_stop")}
                style={{ background: "#374151", border: "none", borderRadius: 8, padding: "8px 14px", cursor: "pointer" }}
              >
                Barge-in / Stop
              </button>
              <button
                type="button"
                onClick={() => {
                  const next = mode === "mute" ? "hands-free" : "mute";
                  setMode(next);
                  micRef.current.setMuted(next === "mute");
                }}
                style={{ background: mode === "mute" ? "#b45309" : "#1f2937", border: "none", borderRadius: 8, padding: "8px 14px", cursor: "pointer" }}
              >
                {mode === "mute" ? "Unmute" : "Mute"}
              </button>
            </div>

            <div style={{ display: "flex", gap: 10, fontSize: 12 }}>
              <span style={{ padding: "4px 8px", borderRadius: 999, background: micActive ? "#0c4a6e" : "#1f2937" }}>
                Mic: {micActive ? (mode === "mute" ? "Muted" : "Listening") : "Idle"}
              </span>
              <span style={{ padding: "4px 8px", borderRadius: 999, background: assistantState === "speaking" ? "#065f46" : "#1f2937" }}>
                Assistant: {assistantState}
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
              placeholder="Say something or type here to test full pipeline..."
              style={{ flex: 1, background: "#0b0f17", border: "1px solid #374151", borderRadius: 8, padding: "8px 12px" }}
            />
            <button type="submit" style={{ background: "#0ea5e9", border: "none", borderRadius: 8, padding: "8px 16px", fontWeight: 600, cursor: "pointer" }}>
              Send
            </button>
          </form>
          {statusNote && <div style={{ color: "#f87171", fontSize: 12, marginTop: 8 }}>{statusNote}</div>}
        </section>

        {devMode && (
          <aside style={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 12, padding: 16, fontSize: 13, display: "flex", flexDirection: "column", gap: 12 }}>
            <strong>Developer Telemetry HUD</strong>
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
              <div>TTS TTFA: {lastLatencyMs.ttfa ?? "-"} ms</div>
              <div>E2E Audible: {lastLatencyMs.e2e ?? "-"} ms</div>
              <div>Barge-in Stop: {lastLatencyMs.bargeIn ?? "-"} ms</div>
            </div>

            <div style={{ borderTop: "1px solid #1f2937", paddingTop: 10, display: "grid", gap: 4 }}>
              <div><strong>GPU & Models</strong></div>
              <div>Device: {health?.vram.device ?? "checking..."}</div>
              <div>VRAM: {health ? `${health.vram.allocated_mb} / ${health.vram.total_mb} MB` : "-"}</div>
              <div>ASR: {health?.models.asr.model_id ?? "-"}</div>
              <div>TTS: {health?.models.tts.model_id ?? "-"}</div>
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
