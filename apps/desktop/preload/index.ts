/**
 * Strongly-typed preload bridge contract between Electron Renderer and Main.
 * Never exposes Node.js primitives directly to the renderer window.
 */

import type { ApprovalResult, AgentMode, PermissionLevel } from "@voice-agent/agent";
import type { InteractionMode, UiSnapshot } from "../main/orchestrator.js";

export interface VoiceAgentIpcBridge {
  getSnapshot(): Promise<UiSnapshot>;
  subscribe(listener: (snapshot: UiSnapshot) => void): () => void;
  setInteractionMode(mode: InteractionMode): Promise<void>;
  setPermissionPolicy(mode: AgentMode, level: PermissionLevel): Promise<void>;
  submitText(text: string): Promise<void>;
  stopPlayback(): Promise<void>;
  respondApproval(result: ApprovalResult): Promise<void>;
  exportMetricsJsonl(): Promise<string>;
}

export const IPC_CHANNELS = {
  GET_SNAPSHOT: "voice-agent:get-snapshot",
  SNAPSHOT_UPDATE: "voice-agent:snapshot-update",
  SET_INTERACTION_MODE: "voice-agent:set-interaction-mode",
  SET_PERMISSION_POLICY: "voice-agent:set-permission-policy",
  SUBMIT_TEXT: "voice-agent:submit-text",
  STOP_PLAYBACK: "voice-agent:stop-playback",
  RESPOND_APPROVAL: "voice-agent:respond-approval",
  EXPORT_METRICS: "voice-agent:export-metrics",
} as const;
