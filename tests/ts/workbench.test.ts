import { describe, expect, it } from "vitest";
import {
  encodePcm16kToWavBase64,
  formatCsvRows,
  makeSyntheticProbeWavBase64,
} from "../../apps/desktop/renderer/workbench-api.js";

describe("Stage 3: Desktop R&D Workbench helpers", () => {
  it("encodes 16kHz PCM16 into valid RIFF WAVE base64 payload", () => {
    const b64 = makeSyntheticProbeWavBase64(0.25);
    const bytes = Buffer.from(b64, "base64");
    expect(bytes.subarray(0, 4).toString("ascii")).toBe("RIFF");
    expect(bytes.subarray(8, 12).toString("ascii")).toBe("WAVE");
    expect(bytes.readUInt32LE(24)).toBe(16000);
    // 0.25s * 16000 * 2 bytes + 44 header bytes = 8044
    expect(bytes.length).toBe(44 + 4000 * 2);

    const custom = encodePcm16kToWavBase64(new Int16Array([0, 100, -100]), 16000);
    expect(Buffer.from(custom, "base64").length).toBe(44 + 6);
  });

  it("escapes commas, quotes, and newlines safely when exporting MOS CSV", () => {
    const csv = formatCsvRows(
      ["slot", "label", "mos", "text"],
      [["A", 'Qwen3 "Custom", v1', 5, "第一行\n第二行"]],
    );
    expect(csv).toBe('slot,label,mos,text\nA,"Qwen3 ""Custom"", v1",5,"第一行\n第二行"');
  });
});
