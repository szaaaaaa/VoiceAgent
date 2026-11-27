/**
 * Hot-pluggable Pi Coding Agent extension entry point (`@voice-agent/pi-voice`).
 * Zero side effects during registration: sockets and audio streams open lazily on
 * `session_start` (with `--voice`) or `/voice on`, and close cleanly on `session_shutdown`.
 */

import type { AutocompleteItem, PiExtensionAPI } from "./pi-host-types.js";
import { PiVoiceSessionController, type PiVoiceControllerOptions } from "./voice-session.js";

export { PiVoiceSessionController } from "./voice-session.js";
export type { PiVoiceControllerOptions, VoiceInputMode, VoiceTtsEngine, VoiceTransport } from "./voice-session.js";
export type { PiExtensionAPI, PiExtensionContext } from "./pi-host-types.js";

const VOICE_SUBCOMMANDS: AutocompleteItem[] = [
  { value: "on", label: "on", description: "开启免提全双工语音模式" },
  { value: "off", label: "off", description: "关闭语音模式" },
  { value: "handsfree", label: "handsfree", description: "免提持续监听模式" },
  { value: "ptt", label: "ptt", description: "按键说话模式 (Alt+V)" },
  { value: "engine qwen3", label: "engine qwen3", description: "切换至 Qwen3-TTS 音色引擎" },
  { value: "engine fast", label: "engine fast", description: "切换至极速低延迟 TTS 引擎" },
  { value: "speaker default", label: "speaker <id>", description: "切换目标说话人音色" },
  { value: "approve on", label: "approve on|off", description: "开启/关闭高危工具语音确认辅助" },
  { value: "status", label: "status", description: "查看语音内核连接与模型状态" },
];

export function createPiVoiceExtension(options: PiVoiceControllerOptions = {}) {
  return function piVoiceExtension(pi: PiExtensionAPI): PiVoiceSessionController {
    const controller = new PiVoiceSessionController(pi, options);

    pi.registerFlag("voice", {
      description: "启动 Pi 时自动启用本地全双工语音模式",
      type: "boolean",
      default: false,
    });

    pi.registerCommand("voice", {
      description: "控制 VoiceAgent 全双工语音模块 (on|off|ptt|handsfree|engine|speaker|approve|status)",
      getArgumentCompletions(prefix: string): AutocompleteItem[] | null {
        const lower = prefix.trim().toLowerCase();
        const filtered = VOICE_SUBCOMMANDS.filter(
          (item) => item.value.toLowerCase().startsWith(lower) || item.label.toLowerCase().startsWith(lower),
        );
        return filtered.length > 0 ? filtered : null;
      },
      async handler(args, ctx) {
        await controller.handleCommand(args, ctx);
      },
    });

    pi.registerShortcut("alt+v", {
      description: "VoiceAgent 语音开关 / PTT 按键说话切换",
      async handler(ctx) {
        await controller.toggleShortcut(ctx);
      },
    });

    pi.on("session_start", async (_event, ctx) => {
      controller.setContext(ctx);
      if (pi.getFlag("voice") === true) {
        await controller.enable(ctx, "handsfree");
      }
    });

    pi.on("before_agent_start", (event, ctx) => controller.onBeforeAgentStart(event, ctx));
    pi.on("message_update", (event, ctx) => controller.onMessageUpdate(event, ctx));
    pi.on("tool_call", (event, ctx) => controller.onToolCall(event, ctx));
    pi.on("tool_execution_end", (event) => controller.onToolExecutionEnd(event));
    pi.on("turn_end", (event, ctx) => controller.onTurnEnd(event, ctx));
    pi.on("agent_settled", (event, ctx) => controller.onAgentSettled(event, ctx));
    pi.on("session_shutdown", () => controller.shutdown());

    return controller;
  };
}

export default createPiVoiceExtension();
