/**
 * Typed REST client & audio/CSV helpers for the Desktop R&D Experiment Workbench.
 */

import { toMono16kPcm16 } from "@voice-agent/audio";
import type {
  AbAsrCandidateResult,
  AbTtsCandidateResult,
  ExperimentReportItem,
  RegistrySnapshotResponse,
} from "@voice-agent/protocol";

export interface HealthSnapshot {
  status: string;
  version?: string;
  observers_connected?: number;
  audio_capture_active?: boolean;
  vram: { device: string; allocated_mb: number; reserved_mb: number; total_mb: number };
  models: {
    vad?: { backend: string };
    asr: { model_id: string; backend?: string; checkpoint: string | null; active_id?: string };
    turn_adapter: { version: string; params: number; strategy?: string; active_id?: string };
    tts: { model_id: string; backend?: string; engine?: string; sample_rate: number; active_id?: string };
  };
}

export interface AbAsrResponse {
  audio_duration_ms: number;
  reference_text: string | null;
  candidates: AbAsrCandidateResult[];
}

export interface AbTtsResponse {
  text: string;
  blind: boolean;
  candidates: AbTtsCandidateResult[];
  reveal_map: Record<string, { checkpoint_id: string; label: string }>;
}

export interface ExperimentsResponse {
  experiments_dir: string;
  reports: ExperimentReportItem[];
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const errBody = (await res.json()) as { detail?: string };
      if (errBody.detail) detail = errBody.detail;
    } catch {
      // ignore non-json error body
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export function fetchHealth(apiBase: string): Promise<HealthSnapshot> {
  return fetchJson<HealthSnapshot>(`${apiBase}/health`);
}

export function fetchRegistry(apiBase: string): Promise<RegistrySnapshotResponse> {
  return fetchJson<RegistrySnapshotResponse>(`${apiBase}/v1/models/registry`);
}

export function activateRegistryModel(
  apiBase: string,
  payload: { task: string; checkpoint_id: string; checkpoint_path?: string | null },
): Promise<RegistrySnapshotResponse> {
  return fetchJson<RegistrySnapshotResponse>(`${apiBase}/v1/models/activate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function runAbAsr(
  apiBase: string,
  payload: { audio_base64: string; checkpoint_ids: string[]; reference_text?: string | null },
): Promise<AbAsrResponse> {
  return fetchJson<AbAsrResponse>(`${apiBase}/v1/ab-test/asr`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function runAbTts(
  apiBase: string,
  payload: { text: string; checkpoint_ids: string[]; blind: boolean; speaker_profile?: string },
): Promise<AbTtsResponse> {
  return fetchJson<AbTtsResponse>(`${apiBase}/v1/ab-test/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function fetchExperiments(apiBase: string): Promise<ExperimentsResponse> {
  return fetchJson<ExperimentsResponse>(`${apiBase}/v1/experiments`);
}

export function encodePcm16kToWavBase64(pcm16: Int16Array, sampleRate = 16000): string {
  const dataBytes = pcm16.length * 2;
  const buf = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buf);
  const writeAscii = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i += 1) view.setUint8(offset + i, s.charCodeAt(i));
  };
  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(36, "data");
  view.setUint32(40, dataBytes, true);
  for (let i = 0; i < pcm16.length; i += 1) {
    view.setInt16(44 + i * 2, pcm16[i]!, true);
  }
  const bytes = new Uint8Array(buf);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

export function makeSyntheticProbeWavBase64(durationSec = 1.2): string {
  const sr = 16000;
  const total = Math.floor(sr * durationSec);
  const pcm = new Int16Array(total);
  for (let i = 0; i < total; i += 1) {
    const t = i / sr;
    const env = Math.sin(Math.PI * Math.min(1, t / durationSec));
    const sig = 0.32 * Math.sin(2 * Math.PI * 220 * t) + 0.14 * Math.sin(2 * Math.PI * 440 * t);
    pcm[i] = Math.round(sig * env * 32767);
  }
  return encodePcm16kToWavBase64(pcm, sr);
}

export async function decodeAudioFileToPcm16k(file: File): Promise<Int16Array> {
  const arrayBuf = await file.arrayBuffer();
  const ctx = new AudioContext();
  try {
    const decoded = await ctx.decodeAudioData(arrayBuf.slice(0));
    const ch0 = decoded.getChannelData(0);
    return toMono16kPcm16(ch0, decoded.sampleRate, 1);
  } finally {
    await ctx.close();
  }
}

export function formatCsvRows(headers: string[], rows: Array<Array<string | number | null | undefined>>): string {
  const escapeCell = (val: string | number | null | undefined): string => {
    const str = val === null || val === undefined ? "" : String(val);
    if (/[",\n\r]/.test(str)) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };
  return [headers.map(escapeCell).join(","), ...rows.map((r) => r.map(escapeCell).join(","))].join("\n");
}
