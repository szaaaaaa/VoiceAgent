/**
 * Pi Agent SDK connector & turn execution orchestrator (FR-21~FR-28).
 * Reuses Pi model/session runtime when installed, streams sanitized speakable text
 * separately from raw UI output, and enforces guarded tool execution + AbortController cancel.
 */

import { type AgentHistoryMessage, PlaybackAwareHistory } from "@voice-agent/interaction";
import { SentenceBuffer, type SentenceSegment } from "@voice-agent/speech";
import type { ApprovalPromptHandler, PermissionGatekeeper, ToolRiskLevel } from "./permissions.js";
import { createDefaultTools, type GuardedToolRegistry } from "./tools/index.js";

export interface LlmToolRequest {
  toolCallId: string;
  toolName: string;
  args: Record<string, unknown>;
}

export type LlmStreamEvent =
  | { type: "delta"; text: string }
  | { type: "tool_call"; call: LlmToolRequest };

export interface LlmStreamRequest {
  messages: AgentHistoryMessage[];
  spaceName: string;
  model: string;
  allowExternalTools: boolean;
  signal: AbortSignal;
}

export type LlmStreamProvider = (req: LlmStreamRequest) => AsyncIterable<LlmStreamEvent>;

export interface SpaceConfig {
  name: string;
  allowedRoots: string[];
  model?: string;
}

export interface TurnExecutionCallbacks {
  onTextDelta?: (rawDelta: string) => void;
  onSpeakableSentence?: (segment: SentenceSegment) => void;
  onToolStart?: (call: LlmToolRequest, riskLevel: ToolRiskLevel) => void;
  onToolEnd?: (call: LlmToolRequest, status: "ok" | "error" | "denied", output: string, durationMs: number) => void;
}

/**
 * Default OpenAI-compatible / Pi-compatible streaming provider reading local env config.
 * shortcut: uses fetch SSE when OPENAI_API_KEY/PI_API_KEY is set and falls back to an explicit
 * unconfigured notice; upgrade to direct `@mariozechner/pi-agent-core` session binding once Pi
 * package is installed in the local workspace.
 */
export function createEnvLlmProvider(): LlmStreamProvider {
  return async function* (req: LlmStreamRequest): AsyncIterable<LlmStreamEvent> {
    const apiKey = process.env.PI_API_KEY ?? process.env.OPENAI_API_KEY;
    const baseUrl = (process.env.PI_API_BASE_URL ?? "https://api.openai.com/v1").replace(/\/$/, "");
    if (!apiKey) {
      yield {
        type: "delta",
        text: "Pi Agent model credentials are not configured yet. Please set PI_API_KEY or OPENAI_API_KEY in your local environment.",
      };
      return;
    }

    const res = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: req.model,
        stream: true,
        messages: req.messages.map((m) => ({
          role: m.role === "tool" ? "assistant" : m.role,
          content: m.content,
        })),
      }),
      signal: req.signal,
    });

    if (!res.ok || !res.body) {
      throw new Error(`LLM upstream error: HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") return;
        const parsed = JSON.parse(payload) as {
          choices?: Array<{ delta?: { content?: string } }>;
        };
        const delta = parsed.choices?.[0]?.delta?.content;
        if (delta) yield { type: "delta", text: delta };
      }
    }
  };
}

export class PiAgentConnector {
  readonly history: PlaybackAwareHistory;
  readonly tools: GuardedToolRegistry;
  private readonly activeControllers = new Map<string, AbortController>();
  private currentSpace: SpaceConfig;
  private model: string;

  constructor(
    private readonly gatekeeper: PermissionGatekeeper,
    private readonly provider: LlmStreamProvider = createEnvLlmProvider(),
    options: {
      history?: PlaybackAwareHistory;
      tools?: GuardedToolRegistry;
      space?: SpaceConfig;
      model?: string;
    } = {},
  ) {
    this.history = options.history ?? new PlaybackAwareHistory();
    this.tools = options.tools ?? createDefaultTools();
    this.currentSpace = options.space ?? {
      name: "default",
      allowedRoots: gatekeeper.getPolicy().allowedRoots,
    };
    this.model = options.model ?? process.env.PI_MODEL ?? "gpt-4o-mini";
  }

  switchSpace(space: SpaceConfig): void {
    this.currentSpace = space;
    this.gatekeeper.setAllowedRoots(space.allowedRoots);
    if (space.model) this.model = space.model;
  }

  getSpace(): SpaceConfig {
    return this.currentSpace;
  }

  /** Cancels in-flight LLM generation without rolling back already completed external tools. */
  abortGeneration(generationId: string, reason = "user_interruption"): boolean {
    const controller = this.activeControllers.get(generationId);
    if (!controller) return false;
    controller.abort(reason);
    this.activeControllers.delete(generationId);
    return true;
  }

  async runTurn(params: {
    turnId: string;
    generationId: string;
    userTranscript: string;
    allowExternalTools?: boolean;
    approvalHandler?: ApprovalPromptHandler;
    callbacks?: TurnExecutionCallbacks;
  }): Promise<{ fullReply: string; speakableReply: string; aborted: boolean }> {
    const allowExternalTools = params.allowExternalTools ?? true;
    const controller = new AbortController();
    this.activeControllers.set(params.generationId, controller);

    if (allowExternalTools) {
      this.history.recordUserTurn(params.turnId, params.userTranscript);
    }

    const messages = allowExternalTools
      ? this.history.toAgentMessages()
      : [...this.history.toAgentMessages(), { role: "user" as const, content: params.userTranscript }];

    const sentenceBuffer = new SentenceBuffer();
    let fullReply = "";
    const speakableParts: string[] = [];
    let aborted = false;

    try {
      for await (const event of this.provider({
        messages,
        spaceName: this.currentSpace.name,
        model: this.model,
        allowExternalTools,
        signal: controller.signal,
      })) {
        if (controller.signal.aborted) {
          aborted = true;
          break;
        }

        if (event.type === "delta") {
          fullReply += event.text;
          params.callbacks?.onTextDelta?.(event.text);
          const segments = sentenceBuffer.pushDelta(event.text);
          for (const seg of segments) {
            speakableParts.push(seg.text);
            params.callbacks?.onSpeakableSentence?.(seg);
          }
        } else if (event.type === "tool_call") {
          const toolDef = this.tools.get(event.call.toolName);
          const risk: ToolRiskLevel = toolDef?.riskLevel ?? "high_risk";

          // Speculative runs are strictly forbidden from executing external tools (PRD §4.8).
          if (!allowExternalTools) {
            params.callbacks?.onToolEnd?.(
              event.call,
              "denied",
              "External tool execution forbidden during speculative generation",
              0,
            );
            continue;
          }

          params.callbacks?.onToolStart?.(event.call, risk);
          const startedAt = performance.now();
          const result = await this.tools.invoke(
            event.call.toolName,
            event.call.args,
            this.gatekeeper,
            params.approvalHandler,
            controller.signal,
          );
          const durationMs = Math.round(performance.now() - startedAt);
          this.history.recordToolExecution(params.turnId, {
            toolCallId: event.call.toolCallId,
            toolName: event.call.toolName,
            argsSummary: JSON.stringify(event.call.args).slice(0, 200),
            status: result.status,
            result: result.output,
          });
          params.callbacks?.onToolEnd?.(event.call, result.status, result.output, durationMs);
        }
      }

      if (!aborted && !controller.signal.aborted) {
        const trailing = sentenceBuffer.flush();
        for (const seg of trailing) {
          speakableParts.push(seg.text);
          params.callbacks?.onSpeakableSentence?.(seg);
        }
      }
    } catch (err) {
      if (controller.signal.aborted || (err instanceof Error && err.name === "AbortError")) {
        aborted = true;
      } else {
        throw err;
      }
    } finally {
      this.activeControllers.delete(params.generationId);
    }

    const speakableReply = speakableParts.join(" ").trim();
    if (allowExternalTools) {
      this.history.recordAssistantReply(params.turnId, fullReply, speakableReply);
    }
    return { fullReply, speakableReply, aborted };
  }
}
