/**
 * Structural TypeScript interfaces mirroring @earendil-works/pi-coding-agent v1.4.0 ExtensionAPI.
 * Keeping structural types here avoids runtime host imports (preventing duplicate ESM instances
 * under jiti) and lets the monorepo type-check cleanly without vendoring the global Pi install.
 */

export interface AutocompleteItem {
  value: string;
  label: string;
  description?: string;
}

export interface PiExtensionUIContext {
  notify(message: string, type?: "info" | "warning" | "error"): void;
  setStatus(key: string, text: string | undefined): void;
  confirm(
    title: string,
    message: string,
    opts?: { signal?: AbortSignal; timeout?: number },
  ): Promise<boolean>;
}

export interface PiExtensionContext {
  ui: PiExtensionUIContext;
  hasUI: boolean;
  cwd: string;
  isIdle(): boolean;
  abort(): void;
}

export interface PiSessionStartEvent {
  type: "session_start";
  reason: "startup" | "reload" | "new" | "resume" | "fork";
}

export interface PiSessionShutdownEvent {
  type: "session_shutdown";
  reason: "quit" | "reload" | "new" | "resume" | "fork";
}

export interface PiBeforeAgentStartEvent {
  type: "before_agent_start";
  messageType?: "user" | "custom";
  prompt?: string;
  readonly systemPrompt: string;
}

export interface PiBeforeAgentStartResult {
  message?: {
    customType: string;
    content: string;
    display: boolean;
    details?: unknown;
  };
  systemPrompt?: string;
}

export interface PiAssistantMessageEvent {
  type: string;
  delta?: string;
  content?: string;
}

export interface PiMessageUpdateEvent {
  type: "message_update";
  message: unknown;
  assistantMessageEvent: PiAssistantMessageEvent;
}

export interface PiTurnEndEvent {
  type: "turn_end";
  turnIndex: number;
  message: unknown;
  outcome?: "completed" | "aborted" | "error";
}

export interface PiAgentSettledEvent {
  type: "agent_settled";
  aborted: boolean;
}

export interface PiToolCallEvent {
  type: "tool_call";
  toolCallId: string;
  toolName: string;
  input: Record<string, unknown>;
}

export interface PiToolCallResult {
  block?: boolean;
  reason?: string;
  terminate?: boolean;
}

export interface PiToolExecutionEndEvent {
  type: "tool_execution_end";
  toolCallId: string;
  toolName: string;
  result: unknown;
  isError: boolean;
  durationMs?: number;
}

export interface PiExtensionAPI {
  on(event: "session_start", handler: (e: PiSessionStartEvent, ctx: PiExtensionContext) => Promise<void> | void): () => void;
  on(event: "session_shutdown", handler: (e: PiSessionShutdownEvent, ctx: PiExtensionContext) => Promise<void> | void): () => void;
  on(
    event: "before_agent_start",
    handler: (e: PiBeforeAgentStartEvent, ctx: PiExtensionContext) => Promise<PiBeforeAgentStartResult | void> | PiBeforeAgentStartResult | void,
  ): () => void;
  on(event: "message_update", handler: (e: PiMessageUpdateEvent, ctx: PiExtensionContext) => Promise<void> | void): () => void;
  on(event: "turn_end", handler: (e: PiTurnEndEvent, ctx: PiExtensionContext) => Promise<void> | void): () => void;
  on(event: "agent_settled", handler: (e: PiAgentSettledEvent, ctx: PiExtensionContext) => Promise<void> | void): () => void;
  on(
    event: "tool_call",
    handler: (e: PiToolCallEvent, ctx: PiExtensionContext) => Promise<PiToolCallResult | void> | PiToolCallResult | void,
  ): () => void;
  on(event: "tool_execution_end", handler: (e: PiToolExecutionEndEvent, ctx: PiExtensionContext) => Promise<void> | void): () => void;

  registerCommand(
    name: string,
    options: {
      description?: string;
      getArgumentCompletions?: (argumentPrefix: string) => AutocompleteItem[] | null;
      handler: (args: string, ctx: PiExtensionContext) => Promise<void>;
    },
  ): void;

  registerShortcut(
    shortcut: string,
    options: {
      description?: string;
      handler: (ctx: PiExtensionContext) => Promise<void> | void;
    },
  ): void;

  registerFlag(
    name: string,
    options:
      | { description?: string; type: "boolean"; default?: boolean }
      | { description?: string; type: "string"; default?: string },
  ): void;

  getFlag(name: string): boolean | string | undefined;

  sendUserMessage(
    content: string,
    options?: {
      deliverAs?: "steer" | "followUp";
      expandPromptTemplates?: boolean;
    },
  ): void;
}
