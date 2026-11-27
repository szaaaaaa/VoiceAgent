import { describe, expect, it, vi } from "vitest";
import {
  createEvent,
  type EventEnvelope,
  type EventPayloadMap,
  type ProtocolEventType,
} from "@voice-agent/protocol";
import {
  createPiVoiceExtension,
  type PiExtensionAPI,
  type PiExtensionContext,
  type VoiceTransport,
} from "@voice-agent/pi-voice";

class FakeVoiceTransport implements VoiceTransport {
  connected = false;
  readonly sent: EventEnvelope[] = [];
  private readonly listeners = new Set<(ev: EventEnvelope) => void>();
  private readonly connListeners = new Set<(c: boolean) => void>();

  isConnected(): boolean {
    return this.connected;
  }

  async connect(): Promise<void> {
    this.connected = true;
    for (const cb of this.connListeners) cb(true);
  }

  disconnect(): void {
    this.connected = false;
    for (const cb of this.connListeners) cb(false);
  }

  onEvent(listener: (event: EventEnvelope) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  onConnectionChange(listener: (connected: boolean) => void): () => void {
    this.connListeners.add(listener);
    return () => this.connListeners.delete(listener);
  }

  sendEvent<T extends ProtocolEventType>(
    type: T,
    ids: { sessionId: string; turnId?: string | null; generationId?: string | null },
    payload: EventPayloadMap[T],
  ): boolean {
    if (!this.connected) return false;
    this.sent.push(createEvent(type, ids, payload));
    return true;
  }

  emitIncoming<T extends ProtocolEventType>(
    type: T,
    sessionId: string,
    payload: EventPayloadMap[T],
  ): void {
    const ev = createEvent(type, { sessionId }, payload);
    for (const cb of this.listeners) cb(ev);
  }
}

function createMockPiHost() {
  const handlers = new Map<string, (...args: any[]) => any>();
  const commands = new Map<string, { handler: (args: string, ctx: PiExtensionContext) => Promise<void> }>();
  const shortcuts = new Map<string, { handler: (ctx: PiExtensionContext) => Promise<void> | void }>();
  const flags = new Map<string, boolean | string>();
  const sentUserMessages: Array<{ content: string; options?: unknown }> = [];
  const statuses: Array<string | undefined> = [];
  const notifications: Array<{ msg: string; type?: string }> = [];

  let idle = true;
  const abortMock = vi.fn(() => {
    idle = true;
  });
  let nextConfirmResult = false;

  const ctx: PiExtensionContext = {
    cwd: "d:/voice-agent",
    hasUI: true,
    isIdle: () => idle,
    abort: abortMock,
    ui: {
      notify(msg, type) {
        notifications.push({ msg, type });
      },
      setStatus(_key, text) {
        statuses.push(text);
      },
      async confirm() {
        return nextConfirmResult;
      },
    },
  };

  const api: PiExtensionAPI = {
    on(event: string, handler: (...args: any[]) => any) {
      handlers.set(event, handler);
      return () => handlers.delete(event);
    },
    registerCommand(name, opts) {
      commands.set(name, { handler: opts.handler });
    },
    registerShortcut(shortcut, opts) {
      shortcuts.set(shortcut, { handler: opts.handler });
    },
    registerFlag(name, opts) {
      if (opts.default !== undefined) flags.set(name, opts.default);
    },
    getFlag(name) {
      return flags.get(name);
    },
    sendUserMessage(content, options) {
      sentUserMessages.push({ content, options });
    },
  };

  return {
    api,
    ctx,
    handlers,
    commands,
    shortcuts,
    flags,
    sentUserMessages,
    statuses,
    notifications,
    abortMock,
    setIdle: (v: boolean) => {
      idle = v;
    },
    setConfirmResult: (v: boolean) => {
      nextConfirmResult = v;
    },
  };
}

describe("Stage 2: @voice-agent/pi-voice hot-pluggable Pi extension", () => {
  it("has zero side effects at registration time and activates lazily via /voice on", async () => {
    const transport = new FakeVoiceTransport();
    const host = createMockPiHost();
    const ext = createPiVoiceExtension({ transport, sessionId: "pi_test_1", autoSpawnBackend: false });

    const ctrl = ext(host.api);
    expect(transport.isConnected()).toBe(false);
    expect(ctrl.isEnabled()).toBe(false);
    expect(host.commands.has("voice")).toBe(true);
    expect(host.shortcuts.has("alt+v")).toBe(true);

    // /voice on connects, handshakes local_device controller, and starts mic.
    await host.commands.get("voice")!.handler("on", host.ctx);
    expect(transport.isConnected()).toBe(true);
    expect(ctrl.isEnabled()).toBe(true);
    expect(
      transport.sent.some(
        (e) =>
          e.type === "session.init" &&
          (e.payload as EventPayloadMap["session.init"]).audio_mode === "local_device",
      ),
    ).toBe(true);
    expect(
      transport.sent.some(
        (e) =>
          e.type === "audio.control" &&
          (e.payload as EventPayloadMap["audio.control"]).action === "start_mic",
      ),
    ).toBe(true);

    // Shutdown cleanly tears down mic and websocket.
    host.handlers.get("session_shutdown")?.({ type: "session_shutdown", reason: "quit" }, host.ctx);
    expect(transport.isConnected()).toBe(false);
  });

  it("streams sanitized text_delta to TTS while filtering thinking_delta and fenced code blocks", async () => {
    const transport = new FakeVoiceTransport();
    const host = createMockPiHost();
    createPiVoiceExtension({ transport, sessionId: "pi_test_2", autoSpawnBackend: false })(host.api);
    await host.commands.get("voice")!.handler("on", host.ctx);

    // Simulated user utterance committed from Python ASR.
    transport.emitIncoming("asr.final", "pi_test_2", {
      text: "帮我检查显存配置",
      audio_start_pts_ms: 100,
      audio_end_pts_ms: 1200,
      compute_ms: 25,
    });
    expect(host.sentUserMessages).toHaveLength(1);
    expect(host.sentUserMessages[0]!.content).toBe("帮我检查显存配置");

    // Pi starts agent turn and streams thinking + code + prose.
    host.handlers.get("before_agent_start")?.({ type: "before_agent_start", systemPrompt: "" }, host.ctx);
    host.handlers.get("message_update")?.(
      { type: "message_update", message: {}, assistantMessageEvent: { type: "thinking_delta", delta: "internal chain of thought" } },
      host.ctx,
    );
    host.handlers.get("message_update")?.(
      {
        type: "message_update",
        message: {},
        assistantMessageEvent: {
          type: "text_delta",
          delta: "我已经检查完显存配置，一切正常。```bash\nrm -rf /tmp\n``` 接下来可以启动训练。",
        },
      },
      host.ctx,
    );
    host.handlers.get("turn_end")?.({ type: "turn_end", turnIndex: 0, message: {}, outcome: "completed" }, host.ctx);

    const ttsEvents = transport.sent.filter((e) => e.type === "tts.synthesize");
    expect(ttsEvents.length).toBeGreaterThanOrEqual(1);
    const spokenAll = ttsEvents.map((e) => (e.payload as EventPayloadMap["tts.synthesize"]).text).join(" ");
    expect(spokenAll).toContain("我已经检查完显存配置");
    expect(spokenAll).not.toContain("internal chain of thought");
    expect(spokenAll).not.toContain("rm -rf");
  });

  it("distinguishes backchannel resume from full barge-in abort + audible-prefix context injection", async () => {
    const transport = new FakeVoiceTransport();
    const host = createMockPiHost();
    createPiVoiceExtension({ transport, sessionId: "pi_test_3", autoSpawnBackend: false })(host.api);
    await host.commands.get("voice")!.handler("on", host.ctx);

    // Start assistant reply and mark speaker playing up to char 8.
    transport.emitIncoming("asr.final", "pi_test_3", {
      text: "详细解释一下显存分配方案",
      audio_start_pts_ms: 0,
      audio_end_pts_ms: 1000,
      compute_ms: 20,
    });
    host.setIdle(false);
    host.handlers.get("before_agent_start")?.({ type: "before_agent_start", systemPrompt: "" }, host.ctx);
    host.handlers.get("message_update")?.(
      {
        type: "message_update",
        message: {},
        assistantMessageEvent: {
          type: "text_delta",
          delta: "当前四模型同卡常驻显存仅占用两千四百兆，剩余超过五千兆可用于训练缓存。",
        },
      },
      host.ctx,
    );
    host.handlers.get("turn_end")?.({ type: "turn_end", turnIndex: 0, message: {}, outcome: "completed" }, host.ctx);
    transport.emitIncoming("playback.cursor", "pi_test_3", {
      mode: "playing",
      played_char_index: 12,
      played_duration_ms: 900,
      generation_id: "gen_1",
      turn_id: "turn_0001",
    });

    // Case A: User says brief backchannel "嗯" -> ducks then resumes without aborting Pi.
    transport.emitIncoming("vad.start", "pi_test_3", {
      vad_prob: 0.9,
      speech_start_pts_ms: 2000,
      preroll_included_ms: 300,
    });
    expect(
      transport.sent.some(
        (e) =>
          e.type === "audio.control" &&
          (e.payload as EventPayloadMap["audio.control"]).action === "duck",
      ),
    ).toBe(true);

    transport.emitIncoming("asr.final", "pi_test_3", {
      text: "嗯",
      audio_start_pts_ms: 2000,
      audio_end_pts_ms: 2250,
      compute_ms: 10,
    });
    expect(host.abortMock).not.toHaveBeenCalled();
    expect(
      transport.sent.some(
        (e) =>
          e.type === "audio.control" &&
          (e.payload as EventPayloadMap["audio.control"]).action === "resume",
      ),
    ).toBe(true);

    // Case B: User actively interrupts -> cancels TTS, aborts Pi, steers new message, injects audible prefix!
    transport.emitIncoming("playback.cursor", "pi_test_3", {
      mode: "playing",
      played_char_index: 15,
      played_duration_ms: 1200,
      generation_id: "gen_1",
      turn_id: "turn_0001",
    });
    transport.emitIncoming("asr.final", "pi_test_3", {
      text: "停一下，直接告诉我结论就行",
      audio_start_pts_ms: 3000,
      audio_end_pts_ms: 4100,
      compute_ms: 18,
    });

    expect(host.abortMock).toHaveBeenCalledTimes(1);
    expect(transport.sent.some((e) => e.type === "generation.cancel")).toBe(true);
    expect(host.sentUserMessages.at(-1)?.content).toBe("停一下，直接告诉我结论就行");

    const injected = host.handlers.get("before_agent_start")?.(
      { type: "before_agent_start", systemPrompt: "" },
      host.ctx,
    );
    expect(injected?.message?.customType).toBe("voice-interruption-context");
    expect(injected?.message?.content).toContain("当前四模型同卡常驻显存仅占用");
    expect(injected?.message?.content).toContain("用户未听到的剩余内容");
  });

  it("blocks destructive shell commands fail-closed when voice tool confirmation is enabled", async () => {
    const transport = new FakeVoiceTransport();
    const host = createMockPiHost();
    createPiVoiceExtension({
      transport,
      sessionId: "pi_test_4",
      autoSpawnBackend: false,
      voiceApproveTools: true,
    })(host.api);
    await host.commands.get("voice")!.handler("on", host.ctx);

    host.setConfirmResult(false);
    const blocked = await host.handlers.get("tool_call")?.(
      { type: "tool_call", toolCallId: "c1", toolName: "bash", input: { command: "rm -rf ./models" } },
      host.ctx,
    );
    expect(blocked?.block).toBe(true);

    host.setConfirmResult(true);
    const allowed = await host.handlers.get("tool_call")?.(
      { type: "tool_call", toolCallId: "c2", toolName: "bash", input: { command: "rm -rf ./tmp_file" } },
      host.ctx,
    );
    expect(allowed).toBeUndefined();
  });
});
