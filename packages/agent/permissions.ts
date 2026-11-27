/**
 * Tool permission & spoken/UI approval gatekeeper (PRD §4.9, §8.3).
 * Enforces Conversation vs Action mode, Readonly/Ask/Trusted levels, independent
 * path-boundary validation (cwd is not a sandbox), dangerous command blocking,
 * and fail-closed approval semantics (silence / timeout / ambiguity => denied).
 */

import fs from "node:fs";
import path from "node:path";

export type AgentMode = "conversation" | "action";
export type PermissionLevel = "readonly" | "ask" | "trusted";
export type ToolRiskLevel = "readonly" | "write" | "high_risk";

export interface ApprovalRequest {
  approvalId: string;
  toolName: string;
  riskLevel: ToolRiskLevel;
  targetPath: string | null;
  command: string | null;
  impactSummary: string;
  expiresInMs: number;
}

export interface ApprovalResult {
  approved: boolean;
  source: "ui" | "voice" | "timeout" | "policy";
  reason: string;
}

export type ApprovalPromptHandler = (request: ApprovalRequest) => Promise<ApprovalResult>;

export interface PermissionPolicyConfig {
  mode?: AgentMode;
  level?: PermissionLevel;
  allowedRoots: string[];
  trustedTools?: string[];
  approvalTimeoutMs?: number;
}

const EXPLICIT_APPROVE_PHRASES = new Set([
  "yes",
  "approve",
  "approved",
  "confirm",
  "confirmed",
  "allow",
  "go ahead",
  "do it",
  "同意",
  "确认",
  "允许",
  "批准",
  "可以执行",
]);

const EXPLICIT_DENY_PHRASES = new Set([
  "no",
  "deny",
  "cancel",
  "stop",
  "reject",
  "don't",
  "不要",
  "拒绝",
  "取消",
  "不允许",
]);

// Commands that must never be silently executed or allowed through shell tooling.
const DANGEROUS_COMMAND_PATTERNS: RegExp[] = [
  /\brm\s+-[a-z]*r[a-z]*f\b/i,
  /\brm\s+-[a-z]*f[a-z]*r\b/i,
  /\bremove-item\b[^\n]*-recurse[^\n]*-force/i,
  /\bremove-item\b[^\n]*-force[^\n]*-recurse/i,
  /\b(?:ri|del|rm|erase)\b[^\n]*-(?:r|rec|recurse)\b[^\n]*-(?:fo|force)\b/i,
  /\bdel\b[^\n]*\/[fsq]/i,
  /\b(?:rmdir|rd)\b[^\n]*\/s/i,
  /\bformat\s+[a-z]:/i,
  /\bmkfs\b/i,
  /\bshutdown\b/i,
  /\bgit\s+reset\s+--hard\b/i,
  /\bgit\s+clean\s+-[a-z]*f/i,
  /\bgit\s+push\s+[^\n]*--force\b/i,
];

/** Interprets spoken approval strictly: only explicit affirmative passes; everything else denies. */
export function parseSpokenApproval(utterance: string): ApprovalResult {
  const norm = utterance
    .trim()
    .toLowerCase()
    .replace(/[.,!?;:。！？，]+$/g, "")
    .replace(/\s+/g, " ");

  if (!norm) {
    return { approved: false, source: "voice", reason: "Empty or silent response treated as denied" };
  }
  if (EXPLICIT_DENY_PHRASES.has(norm)) {
    return { approved: false, source: "voice", reason: `Explicit spoken denial: "${norm}"` };
  }
  if (EXPLICIT_APPROVE_PHRASES.has(norm)) {
    return { approved: true, source: "voice", reason: `Explicit spoken approval: "${norm}"` };
  }
  return {
    approved: false,
    source: "voice",
    reason: `Ambiguous spoken response "${norm}" treated as denied (fail-closed)`,
  };
}

/** Resolves symlinks where possible and checks targetPath stays strictly inside allowedRoots. */
export function isPathWithinAllowedRoots(targetPath: string, allowedRoots: string[]): boolean {
  if (!targetPath || allowedRoots.length === 0) return false;
  const resolvedTarget = resolveRealOrParent(targetPath);

  return allowedRoots.some((root) => {
    const resolvedRoot = resolveRealOrParent(root);
    const rel = path.relative(resolvedRoot, resolvedTarget);
    return rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel));
  });
}

function resolveRealOrParent(p: string): string {
  const abs = path.resolve(p);
  try {
    return fs.realpathSync(abs);
  } catch {
    // Target file may not exist yet (e.g., fs.write); canonicalize deepest existing ancestor.
    const parent = path.dirname(abs);
    if (parent === abs) return abs;
    return path.join(resolveRealOrParent(parent), path.basename(abs));
  }
}

export function isDangerousCommand(command: string): boolean {
  return DANGEROUS_COMMAND_PATTERNS.some((pattern) => pattern.test(command));
}

export class PermissionGatekeeper {
  private mode: AgentMode;
  private level: PermissionLevel;
  private allowedRoots: string[];
  private readonly trustedTools: Set<string>;
  private readonly approvalTimeoutMs: number;
  private approvalSeq = 0;

  constructor(config: PermissionPolicyConfig) {
    this.mode = config.mode ?? "conversation";
    this.level = config.level ?? "ask";
    this.allowedRoots = config.allowedRoots.map((r) => path.resolve(r));
    this.trustedTools = new Set(config.trustedTools ?? []);
    this.approvalTimeoutMs = config.approvalTimeoutMs ?? 15000;
  }

  setMode(mode: AgentMode): void {
    this.mode = mode;
  }

  setLevel(level: PermissionLevel): void {
    this.level = level;
  }

  setAllowedRoots(roots: string[]): void {
    this.allowedRoots = roots.map((r) => path.resolve(r));
  }

  getPolicy(): { mode: AgentMode; level: PermissionLevel; allowedRoots: string[] } {
    return { mode: this.mode, level: this.level, allowedRoots: [...this.allowedRoots] };
  }

  async authorize(
    params: {
      toolName: string;
      riskLevel: ToolRiskLevel;
      targetPath?: string | null;
      command?: string | null;
      impactSummary: string;
    },
    promptHandler?: ApprovalPromptHandler,
  ): Promise<ApprovalResult> {
    // 1. Independent path boundary verification (cwd is never trusted as a sandbox).
    if (params.targetPath && !isPathWithinAllowedRoots(params.targetPath, this.allowedRoots)) {
      return {
        approved: false,
        source: "policy",
        reason: `Path "${params.targetPath}" escapes allowed workspace roots`,
      };
    }

    // 2. Hard block on destructive / force-delete commands regardless of mode.
    if (params.command && isDangerousCommand(params.command)) {
      return {
        approved: false,
        source: "policy",
        reason: `Blocked dangerous command by security policy: "${params.command}"`,
      };
    }

    // 3. Readonly tools inside allowed boundaries are permitted in all modes.
    if (params.riskLevel === "readonly") {
      return { approved: true, source: "policy", reason: "Readonly tool permitted" };
    }

    // 4. Conversation mode forbids any state-mutating or high-risk tools.
    if (this.mode === "conversation") {
      return {
        approved: false,
        source: "policy",
        reason: `Tool "${params.toolName}" requires Action mode (currently in Conversation mode)`,
      };
    }

    // 5. Readonly level forbids all side-effecting operations.
    if (this.level === "readonly") {
      return {
        approved: false,
        source: "policy",
        reason: `Tool "${params.toolName}" blocked by Readonly permission level`,
      };
    }

    // 6. Trusted level only auto-approves explicitly configured non-high-risk tools.
    if (
      this.level === "trusted" &&
      params.riskLevel === "write" &&
      this.trustedTools.has(params.toolName)
    ) {
      return {
        approved: true,
        source: "policy",
        reason: `Tool "${params.toolName}" pre-approved in Trusted list`,
      };
    }

    // 7. All remaining write & high-risk operations require explicit confirmation with fail-closed timeout.
    if (!promptHandler) {
      return {
        approved: false,
        source: "policy",
        reason: "No interactive approval handler available (fail-closed)",
      };
    }

    this.approvalSeq += 1;
    const req: ApprovalRequest = {
      approvalId: `appr_${this.approvalSeq}`,
      toolName: params.toolName,
      riskLevel: params.riskLevel,
      targetPath: params.targetPath ?? null,
      command: params.command ?? null,
      impactSummary: params.impactSummary,
      expiresInMs: this.approvalTimeoutMs,
    };

    return this.awaitWithTimeout(promptHandler(req), this.approvalTimeoutMs);
  }

  private async awaitWithTimeout(
    promise: Promise<ApprovalResult>,
    timeoutMs: number,
  ): Promise<ApprovalResult> {
    let timer: ReturnType<typeof setTimeout> | null = null;
    try {
      const timeoutPromise = new Promise<ApprovalResult>((resolve) => {
        timer = setTimeout(() => {
          resolve({
            approved: false,
            source: "timeout",
            reason: `Approval timed out after ${timeoutMs}ms (fail-closed)`,
          });
        }, timeoutMs);
      });
      return await Promise.race([promise, timeoutPromise]);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }
}
