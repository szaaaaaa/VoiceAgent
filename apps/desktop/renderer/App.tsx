/**
 * VoiceAgent Desktop Main View & Developer Mode HUD (PRD §5.1-§5.4).
 * Displays dual-track state (Listening + Speaking concurrently), conversation stream,
 * tool approval dialog, and developer latency/probability diagnostics.
 */

import React, { useState } from "react";
import type { ApprovalResult, AgentMode, PermissionLevel } from "@voice-agent/agent";
import type { ConversationTurnRecord } from "@voice-agent/interaction";
import type { InteractionMode, UiSnapshot } from "../main/orchestrator.js";

export interface AppProps {
  snapshot: UiSnapshot;
  turns: readonly ConversationTurnRecord[];
  onSetMode: (mode: InteractionMode) => void;
  onSetPolicy: (mode: AgentMode, level: PermissionLevel) => void;
  onSubmitText: (text: string) => void;
  onStop: () => void;
  onRespondApproval: (result: ApprovalResult) => void;
  onExportLogs?: () => void;
}

export function App({
  snapshot,
  turns,
  onSetMode,
  onSetPolicy,
  onSubmitText,
  onStop,
  onRespondApproval,
  onExportLogs,
}: AppProps): React.ReactElement {
  const [textInput, setTextInput] = useState("");
  const [devMode, setDevMode] = useState(false);
  const [agentMode, setAgentMode] = useState<AgentMode>("conversation");
  const [permLevel, setPermLevel] = useState<PermissionLevel>("ask");

  const handleSend = (e: React.FormEvent): void => {
    e.preventDefault();
    if (!textInput.trim()) return;
    onSubmitText(textInput.trim());
    setTextInput("");
  };

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", maxWidth: 920, margin: "0 auto", padding: 16 }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid #333", paddingBottom: 12 }}>
        <strong>VoiceAgent</strong>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <label>
            Mode:
            <select
              aria-label="Agent Mode"
              value={agentMode}
              onChange={(e) => {
                const next = e.target.value as AgentMode;
                setAgentMode(next);
                onSetPolicy(next, permLevel);
              }}
            >
              <option value="conversation">Conversation</option>
              <option value="action">Action</option>
            </select>
          </label>
          <label>
            Policy:
            <select
              aria-label="Permission Level"
              value={permLevel}
              onChange={(e) => {
                const next = e.target.value as PermissionLevel;
                setPermLevel(next);
                onSetPolicy(agentMode, next);
              }}
            >
              <option value="readonly">Readonly</option>
              <option value="ask">Ask</option>
              <option value="trusted">Trusted</option>
            </select>
          </label>
          <button type="button" onClick={() => setDevMode((v) => !v)}>
            {devMode ? "Hide Dev HUD" : "Developer Mode"}
          </button>
        </div>
      </header>

      <section aria-label="Conversation Transcript" style={{ minHeight: 280, padding: "16px 0" }}>
        {turns.map((turn) => (
          <div key={turn.turnId} style={{ marginBottom: 14 }}>
            <div>
              <strong>User:</strong> {turn.userTranscript}
            </div>
            {turn.toolCalls.map((tool) => (
              <div key={tool.toolCallId} style={{ fontSize: 13, opacity: 0.8 }}>
                Tool [{tool.toolName}] ({tool.status}): {tool.result}
              </div>
            ))}
            <div>
              <strong>Agent:</strong>{" "}
              {turn.interrupted ? `${turn.assistantAudibleReply} [Interrupted]` : turn.assistantFullReply}
            </div>
          </div>
        ))}
        {snapshot.partialTranscript && (
          <div style={{ fontStyle: "italic", opacity: 0.75 }}>
            <strong>User (live):</strong> {snapshot.partialTranscript}
          </div>
        )}
      </section>

      {snapshot.pendingApproval && (
        <aside role="dialog" aria-label="Tool Approval Required" style={{ border: "2px solid #d97706", padding: 12, marginBottom: 12 }}>
          <div><strong>Approval Required: {snapshot.pendingApproval.toolName}</strong></div>
          <div>Impact: {snapshot.pendingApproval.impactSummary}</div>
          {snapshot.pendingApproval.targetPath && <div>Path: {snapshot.pendingApproval.targetPath}</div>}
          {snapshot.pendingApproval.command && <div>Command: {snapshot.pendingApproval.command}</div>}
          <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
            <button
              type="button"
              onClick={() => onRespondApproval({ approved: true, source: "ui", reason: "Approved via UI" })}
            >
              Approve
            </button>
            <button
              type="button"
              onClick={() => onRespondApproval({ approved: false, source: "ui", reason: "Denied via UI" })}
            >
              Deny
            </button>
          </div>
        </aside>
      )}

      <section aria-label="Voice Controls" style={{ borderTop: "1px solid #333", paddingTop: 12 }}>
        <div style={{ display: "flex", gap: 12, marginBottom: 8 }}>
          <span>Mic: <strong>{snapshot.listeningState}</strong></span>
          <span>Assistant: <strong>{snapshot.assistantState}</strong></span>
          <span>Backend: <strong>{snapshot.backendConnected ? "Connected" : "Offline (Text Fallback)"}</strong></span>
        </div>

        <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
          {(["hands-free", "push-to-talk", "text", "mute"] as InteractionMode[]).map((m) => (
            <button
              key={m}
              type="button"
              aria-pressed={snapshot.interactionMode === m}
              onClick={() => onSetMode(m)}
            >
              {m}
            </button>
          ))}
          <button type="button" onClick={onStop}>Stop</button>
        </div>

        <form onSubmit={handleSend} style={{ display: "flex", gap: 8 }}>
          <input
            type="text"
            aria-label="Text input"
            placeholder="Type a message (always available even if voice backend is offline)..."
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            style={{ flex: 1, padding: 6 }}
          />
          <button type="submit">Send</button>
        </form>
      </section>

      {devMode && (
        <section aria-label="Developer Diagnostics" style={{ marginTop: 16, padding: 12, background: "#111", color: "#eee", fontSize: 13 }}>
          <div>VAD Prob: {snapshot.lastVadProb.toFixed(3)}</div>
          <div>
            Turn Adapter:{" "}
            {snapshot.lastAdapterProb
              ? `EOT=${snapshot.lastAdapterProb.pCompletion.toFixed(2)} | Floor=${snapshot.lastAdapterProb.pFloorTaking.toFixed(2)} | BC=${snapshot.lastAdapterProb.pNonFloorFeedback.toFixed(2)}`
              : "N/A"}
          </div>
          <div>
            E2E Latency (Hot P50 / P95):{" "}
            {snapshot.metrics.hotE2eAudibleLatencyMs.p50 ?? "-"} ms /{" "}
            {snapshot.metrics.hotE2eAudibleLatencyMs.p95 ?? "-"} ms
          </div>
          <div>Peak VRAM: {snapshot.metrics.peakVramMb ?? 0} MB</div>
          {onExportLogs && (
            <button type="button" onClick={onExportLogs} style={{ marginTop: 8 }}>
              Export Experiment Logs
            </button>
          )}
        </section>
      )}

      {snapshot.errorMessage && (
        <div role="alert" style={{ marginTop: 10, color: "#ef4444" }}>
          {snapshot.errorMessage}
        </div>
      )}
    </main>
  );
}
