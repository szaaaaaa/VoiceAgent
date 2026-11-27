"""
ASR fine-tuning data preparation, augmentation, and regression gate (PRD §6.1, ASR-A0..A3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AsrManifestEntry:
    audio_filepath: str
    text: str
    duration: float
    session_id: str
    split: str = "train"


def load_and_validate_manifest(manifest_path: str | Path) -> list[AsrManifestEntry]:
    entries: list[AsrManifestEntry] = []
    lines = Path(manifest_path).read_text(encoding="utf-8").splitlines()
    for line_no, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        item = json.loads(raw)
        if not item.get("audio_filepath") or not item.get("text"):
            raise ValueError(
                f"Invalid manifest entry at line {line_no}: missing audio_filepath or text"
            )
        duration = float(item.get("duration", 0.0))
        if duration <= 0:
            raise ValueError(f"Invalid non-positive duration at line {line_no}: {duration}")
        entries.append(
            AsrManifestEntry(
                audio_filepath=str(item["audio_filepath"]),
                text=str(item["text"]),
                duration=duration,
                session_id=str(item.get("session_id", f"utt_{line_no}")),
                split=str(item.get("split", "train")),
            )
        )
    return entries


def augment_waveform(
    pcm_f32: np.ndarray,
    *,
    snr_db: float = 15.0,
    gain_db: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Applies reproducible gain jitter and additive white noise for ASR-A3 robustness."""
    rng = np.random.default_rng(seed)
    scaled = pcm_f32 * (10.0 ** (gain_db / 20.0))
    sig_power = float(np.mean(scaled * scaled) + 1e-12)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=scaled.shape).astype(np.float32)
    return np.clip(scaled + noise, -1.0, 1.0)


def check_general_wer_regression(
    baseline_general_wer: float,
    finetuned_general_wer: float,
    *,
    max_allowed_degradation: float = 0.015,
) -> dict[str, Any]:
    """Blocks promotion if domain fine-tuning degrades general speech WER beyond tolerance."""
    delta = finetuned_general_wer - baseline_general_wer
    passed = delta <= max_allowed_degradation
    return {
        "passed": passed,
        "baseline_general_wer": baseline_general_wer,
        "finetuned_general_wer": finetuned_general_wer,
        "degradation": round(delta, 5),
        "max_allowed_degradation": max_allowed_degradation,
    }
