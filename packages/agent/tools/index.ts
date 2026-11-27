/**
 * Guarded built-in local tools for Pi Agent (PRD §4.6, §4.9).
 * Every tool passes through PermissionGatekeeper before touching the filesystem or OS.
 */

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { ApprovalPromptHandler, PermissionGatekeeper, ToolRiskLevel } from "../permissions.js";

const execFileAsync = promisify(execFile);

export interface AgentToolDefinition {
  name: string;
  description: string;
  riskLevel: ToolRiskLevel;
  extractAuditMeta: (args: Record<string, unknown>) => {
    targetPath?: string | null;
    command?: string | null;
    impactSummary: string;
  };
  execute: (args: Record<string, unknown>, signal?: AbortSignal) => Promise<string>;
}

export class GuardedToolRegistry {
  private readonly tools = new Map<string, AgentToolDefinition>();

  register(tool: AgentToolDefinition): void {
    this.tools.set(tool.name, tool);
  }

  get(name: string): AgentToolDefinition | undefined {
    return this.tools.get(name);
  }

  list(): AgentToolDefinition[] {
    return Array.from(this.tools.values());
  }

  async invoke(
    name: string,
    args: Record<string, unknown>,
    gatekeeper: PermissionGatekeeper,
    promptHandler?: ApprovalPromptHandler,
    signal?: AbortSignal,
  ): Promise<{ status: "ok" | "error" | "denied"; output: string; riskLevel: ToolRiskLevel }> {
    const tool = this.tools.get(name);
    if (!tool) {
      return { status: "error", output: `Unknown tool: ${name}`, riskLevel: "high_risk" };
    }
    const meta = tool.extractAuditMeta(args);
    const decision = await gatekeeper.authorize(
      {
        toolName: tool.name,
        riskLevel: tool.riskLevel,
        targetPath: meta.targetPath,
        command: meta.command,
        impactSummary: meta.impactSummary,
      },
      promptHandler,
    );
    if (!decision.approved) {
      return { status: "denied", output: decision.reason, riskLevel: tool.riskLevel };
    }
    try {
      const output = await tool.execute(args, signal);
      return { status: "ok", output, riskLevel: tool.riskLevel };
    } catch (err) {
      return {
        status: "error",
        output: err instanceof Error ? err.message : String(err),
        riskLevel: tool.riskLevel,
      };
    }
  }
}

export function createDefaultTools(): GuardedToolRegistry {
  const registry = new GuardedToolRegistry();

  registry.register({
    name: "system.info",
    description: "Read local OS, CPU count, and free/total RAM summary",
    riskLevel: "readonly",
    extractAuditMeta: () => ({ impactSummary: "Read system hardware summary" }),
    execute: async () => {
      const freeGb = (os.freemem() / 1024 ** 3).toFixed(2);
      const totalGb = (os.totalmem() / 1024 ** 3).toFixed(2);
      return JSON.stringify({
        platform: os.platform(),
        release: os.release(),
        cpus: os.cpus().length,
        memoryGb: `${freeGb} free / ${totalGb} total`,
      });
    },
  });

  registry.register({
    name: "fs.read",
    description: "Read a UTF-8 text file inside an allowed workspace root",
    riskLevel: "readonly",
    extractAuditMeta: (args) => ({
      targetPath: String(args.path ?? ""),
      impactSummary: `Read file ${String(args.path ?? "")}`,
    }),
    execute: async (args) => {
      const filePath = path.resolve(String(args.path ?? ""));
      const content = await fs.readFile(filePath, "utf8");
      return content.slice(0, 16000);
    },
  });

  registry.register({
    name: "fs.list",
    description: "List directory entries inside an allowed workspace root",
    riskLevel: "readonly",
    extractAuditMeta: (args) => ({
      targetPath: String(args.path ?? ""),
      impactSummary: `List directory ${String(args.path ?? "")}`,
    }),
    execute: async (args) => {
      const dirPath = path.resolve(String(args.path ?? ""));
      const entries = await fs.readdir(dirPath, { withFileTypes: true });
      return entries.map((e) => (e.isDirectory() ? `${e.name}/` : e.name)).join("\n");
    },
  });

  registry.register({
    name: "fs.write",
    description: "Write UTF-8 text to a file inside an allowed workspace root",
    riskLevel: "write",
    extractAuditMeta: (args) => ({
      targetPath: String(args.path ?? ""),
      impactSummary: `Write ${String(args.content ?? "").length} chars to ${String(args.path ?? "")}`,
    }),
    execute: async (args) => {
      const filePath = path.resolve(String(args.path ?? ""));
      await fs.mkdir(path.dirname(filePath), { recursive: true });
      await fs.writeFile(filePath, String(args.content ?? ""), "utf8");
      return `Wrote file ${filePath}`;
    },
  });

  registry.register({
    name: "shell.exec",
    description: "Run an approved single command executable with arguments (no shell injection)",
    riskLevel: "high_risk",
    extractAuditMeta: (args) => {
      const cmd = String(args.executable ?? "");
      const cmdArgs = Array.isArray(args.args) ? args.args.map(String) : [];
      return {
        targetPath: args.cwd ? String(args.cwd) : null,
        command: `${cmd} ${cmdArgs.join(" ")}`.trim(),
        impactSummary: `Execute command: ${cmd} ${cmdArgs.join(" ")}`,
      };
    },
    execute: async (args, signal) => {
      const executable = String(args.executable ?? "");
      const cmdArgs = Array.isArray(args.args) ? args.args.map(String) : [];
      const cwd = args.cwd ? path.resolve(String(args.cwd)) : undefined;
      const { stdout, stderr } = await execFileAsync(executable, cmdArgs, {
        cwd,
        timeout: 15000,
        signal,
        windowsHide: true,
      });
      return `${stdout}${stderr}`.trim().slice(0, 8000);
    },
  });

  return registry;
}
