/**
 * Real-time 60fps multi-trace oscilloscope rendering mic RMS bars alongside
 * Silero VAD and CausalTurnGRU 3-head probabilities (EOT / Interrupt / Backchannel).
 */

import React, { useEffect, useRef } from "react";

export interface TelemetryPoint {
  rmsDb: number;
  vadProb: number;
  pCompletion: number;
  pFloorTaking: number;
  pBackchannel: number;
}

interface ScopeCanvasProps {
  points: TelemetryPoint[];
  height?: number;
}

export function ScopeCanvas({ points, height = 190 }: ScopeCanvasProps): React.ReactElement {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Background grid (0.25, 0.50, 0.75 probability lines).
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1;
    for (const frac of [0.25, 0.5, 0.75]) {
      const y = h - frac * (h - 16) - 8;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    if (points.length === 0) return;
    const stepX = w / Math.max(1, points.length - 1);

    // 1. Subtle RMS waveform bars at the bottom layer.
    const barW = Math.max(2, w / points.length - 1.5);
    points.forEach((pt, idx) => {
      const norm = Math.max(0.03, Math.min(1, (pt.rmsDb + 75) / 65));
      const barH = norm * (h * 0.48);
      ctx.fillStyle = "rgba(100, 116, 139, 0.28)";
      ctx.fillRect(idx * (w / points.length), h - barH - 4, barW, barH);
    });

    // 2. Probability curves helper.
    const drawCurve = (getter: (p: TelemetryPoint) => number, color: string, lineWidth = 2) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      points.forEach((pt, idx) => {
        const val = Math.max(0, Math.min(1, getter(pt)));
        const x = idx * stepX;
        const y = h - val * (h - 16) - 8;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    drawCurve((p) => p.vadProb, "#38bdf8", 1.8);
    drawCurve((p) => p.pCompletion, "#34d399", 2.2);
    drawCurve((p) => p.pFloorTaking, "#f43f5e", 2.0);
    drawCurve((p) => p.pBackchannel, "#fbbf24", 1.8);
  }, [points]);

  return (
    <div>
      <canvas
        ref={canvasRef}
        width={960}
        height={height}
        style={{
          width: "100%",
          height,
          background: "#060911",
          border: "1px solid #1e293b",
          borderRadius: 8,
          display: "block",
        }}
      />
      <div style={{ display: "flex", gap: 16, fontSize: 11, color: "#94a3b8", marginTop: 6, flexWrap: "wrap" }}>
        <span><span style={{ color: "#38bdf8" }}>━━</span> Silero VAD 概率</span>
        <span><span style={{ color: "#34d399" }}>━━</span> EOT 发言完成 (p_completion)</span>
        <span><span style={{ color: "#f43f5e" }}>━━</span> 主动打断 (p_floor_taking)</span>
        <span><span style={{ color: "#fbbf24" }}>━━</span> 短暂附和 (p_backchannel)</span>
        <span><span style={{ color: "#64748b" }}>▮▮</span> 麦克风 RMS 能量</span>
      </div>
    </div>
  );
}
