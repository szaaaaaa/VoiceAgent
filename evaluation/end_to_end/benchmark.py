"""
System-level 10-scenario benchmark catalog & standardized Experiment Manifest (PRD §9.2-§9.3).
Explicitly supports archiving negative experimental outcomes without suppression.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

SYSTEM_SCENARIOS: tuple[dict[str, str], ...] = (
    {"id": "S01", "name": "normal_continuous_dialogue", "goal": "Multi-turn baseline dialogue"},
    {
        "id": "S02",
        "name": "long_mid_utterance_pause",
        "goal": "Avoid false EOT cutoff during hesitation",
    },
    {"id": "S03", "name": "active_user_interruption", "goal": "Cancel reply within P95 <= 500ms"},
    {"id": "S04", "name": "short_backchannel", "goal": "Resume reply without cancelling"},
    {"id": "S05", "name": "cough_and_background_noise", "goal": "Avoid false triggers"},
    {"id": "S06", "name": "loudspeaker_echo_playback", "goal": "Prevent self-echo ASR triggers"},
    {
        "id": "S07",
        "name": "interrupt_during_tool_execution",
        "goal": "Preserve executed tool state",
    },
    {"id": "S08", "name": "unauthorized_tool_call", "goal": "Intercept & fail-closed"},
    {"id": "S09", "name": "low_vram_oom_recovery", "goal": "Graceful fallback to text mode"},
    {"id": "S10", "name": "long_soak_stability", "goal": "Zero latency drift or memory leak"},
)


@dataclass
class PerformanceProfileSnapshot:
    """Core 6 hardware/system performance metrics for RTX 5070 Laptop 8GB optimization tracking."""

    label: str  # e.g., "baseline_fp32" or "optimized_bf16_stream"
    asr_rtf: float
    tts_ttfa_ms: float
    e2e_latency_p50_ms: float
    e2e_latency_p95_ms: float
    barge_in_latency_p95_ms: float
    peak_vram_mb: float
    false_interruption_rate: float


def compare_before_after_optimization(
    before: PerformanceProfileSnapshot,
    after: PerformanceProfileSnapshot,
) -> dict[str, Any]:
    """Computes before/after deltas across all 6 core performance metrics (negative = improved)."""
    return {
        "before_label": before.label,
        "after_label": after.label,
        "before": asdict(before),
        "after": asdict(after),
        "delta": {
            "asr_rtf": round(after.asr_rtf - before.asr_rtf, 5),
            "tts_ttfa_ms": round(after.tts_ttfa_ms - before.tts_ttfa_ms, 2),
            "e2e_latency_p50_ms": round(after.e2e_latency_p50_ms - before.e2e_latency_p50_ms, 2),
            "e2e_latency_p95_ms": round(after.e2e_latency_p95_ms - before.e2e_latency_p95_ms, 2),
            "barge_in_latency_p95_ms": round(
                after.barge_in_latency_p95_ms - before.barge_in_latency_p95_ms, 2
            ),
            "peak_vram_mb": round(after.peak_vram_mb - before.peak_vram_mb, 1),
            "false_interruption_rate": round(
                after.false_interruption_rate - before.false_interruption_rate, 5
            ),
        },
    }


@dataclass
class ExperimentManifest:
    experiment_id: str
    dataset_version: str
    split_hash: str
    model_checkpoints: dict[str, str | None]
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any]
    bad_cases: list[dict[str, Any]] = field(default_factory=list)
    conclusion: str = ""
    beats_baseline: bool = False
    git_commit: str = field(default_factory=lambda: _current_git_commit())
    hardware_info: dict[str, Any] = field(default_factory=lambda: _collect_hardware_info())

    def save(self, output_path: str | Path) -> Path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return target


def _current_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, timeout=3)
        return out.strip()
    except Exception:
        return "unknown"


def _collect_hardware_info() -> dict[str, Any]:
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
