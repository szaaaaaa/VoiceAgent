"""
TTS speaker adaptation manifest validation, audio quality filtering, and
zero-shot baseline promotion gate (PRD §6.3, TTS-A0..A3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TtsSampleAudit:
    accepted: bool
    duration_s: float
    rms_db: float
    reason: str


def audit_tts_recording(
    pcm_f32: np.ndarray,
    sample_rate: int,
    *,
    min_duration_s: float = 1.0,
    max_duration_s: float = 15.0,
    min_rms_db: float = -40.0,
    max_clip_ratio: float = 0.01,
) -> TtsSampleAudit:
    """Filters low-energy, clipped, or out-of-range utterances prior to Qwen3-TTS fine-tuning."""
    duration_s = pcm_f32.size / float(sample_rate) if sample_rate > 0 else 0.0
    if duration_s < min_duration_s or duration_s > max_duration_s:
        return TtsSampleAudit(False, duration_s, -120.0, "Duration out of bounds")
    rms = float(np.sqrt(np.mean(pcm_f32 * pcm_f32) + 1e-12))
    rms_db = float(20.0 * np.log10(max(rms, 1e-6)))
    if rms_db < min_rms_db:
        return TtsSampleAudit(False, duration_s, rms_db, "Signal too quiet")
    clip_ratio = float(np.mean(np.abs(pcm_f32) >= 0.995))
    if clip_ratio > max_clip_ratio:
        return TtsSampleAudit(False, duration_s, rms_db, f"Excessive clipping ({clip_ratio:.3f})")
    return TtsSampleAudit(True, duration_s, rms_db, "ok")


def verify_finetune_beats_zeroshot(
    zeroshot_similarity: float,
    finetuned_similarity: float,
    zeroshot_cer: float,
    finetuned_cer: float,
    *,
    min_similarity_gain: float = 0.01,
    max_cer_regression: float = 0.01,
) -> dict[str, Any]:
    """Enforces PRD §6.3: TTS fine-tuning must prove real gain over Zero-shot (TTS-A1)."""
    sim_gain = finetuned_similarity - zeroshot_similarity
    cer_delta = finetuned_cer - zeroshot_cer
    passed = sim_gain >= min_similarity_gain and cer_delta <= max_cer_regression
    return {
        "passed": passed,
        "similarity_gain": round(sim_gain, 4),
        "cer_delta": round(cer_delta, 4),
    }
