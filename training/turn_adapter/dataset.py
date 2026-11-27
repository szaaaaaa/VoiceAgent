"""
Session-level dataset splitting and causal frame-sequence dataset for Turn Adapter training.
Enforces strict session isolation across train / val / test splits (PRD §6.2).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class TurnSessionSample:
    session_id: str
    encoder_features: np.ndarray  # (T, feature_dim) float32
    playback_state: np.ndarray    # (T, 1) float32
    labels: np.ndarray            # (T, 3) float32 -> [completion, floor_taking, non_floor_feedback]


def split_by_session(
    samples: list[TurnSessionSample],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[list[TurnSessionSample], list[TurnSessionSample], list[TurnSessionSample]]:
    """
    Deterministically partitions samples by unique `session_id` so fragments from
    the same conversation NEVER leak across train, val, and test splits.
    """
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio and val_ratio must be positive and sum to < 1.0")

    unique_sessions = sorted({s.session_id for s in samples})

    def _bucket(sid: str) -> float:
        digest = hashlib.sha256(sid.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    ordered = sorted(unique_sessions, key=_bucket)
    n = len(ordered)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n and n >= 3:
        n_train = max(1, n - 2)
        n_val = 1

    train_ids = set(ordered[:n_train])
    val_ids = set(ordered[n_train : n_train + n_val])
    test_ids = set(ordered[n_train + n_val :])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    train = [s for s in samples if s.session_id in train_ids]
    val = [s for s in samples if s.session_id in val_ids]
    test = [s for s in samples if s.session_id in test_ids]
    return train, val, test


def generate_synthetic_turn_dataset(
    num_sessions: int = 30,
    seq_len: int = 40,
    feature_dim: int = 128,
    seed: int = 42,
) -> list[TurnSessionSample]:
    """
    Generates realistic causal turn-taking sequences with 4 distinct regimes:
      0) True EOT: user speaks (high acoustic + completion semantic cue), then drops to true silence.
      1) Mid-utterance HOLD pause: user pauses briefly (true low-energy silence, 4-8 frames) with
         unfinished semantic cue, then resumes speaking (tests false-cutoff resistance!).
      2) Active floor-taking interruption during assistant playback (sustained speech > 600ms).
      3) Short non-floor backchannel during assistant playback ("uh-huh", 3-5 frames).
    """
    rng = np.random.default_rng(seed)
    samples: list[TurnSessionSample] = []

    for i in range(num_sessions):
        # Ambient silence floor has low magnitude (~0.08) so energy-based baselines work honestly.
        feats = rng.normal(0.0, 0.08, size=(seq_len, feature_dim)).astype(np.float32)
        playback = np.zeros((seq_len, 1), dtype=np.float32)
        labels = np.zeros((seq_len, 3), dtype=np.float32)

        mode = i % 4
        if mode == 0:
            # User speaks then cleanly finishes -> EOT during trailing silence.
            end_frame = seq_len // 2 + int(rng.integers(-3, 5))
            feats[:end_frame, :16] += rng.normal(0.85, 0.15, size=(end_frame, 16)).astype(np.float32)
            # Semantic completion cues reside in upper ASR encoder dimensions (80:112).
            if feature_dim >= 112:
                feats[max(0, end_frame - 4) : end_frame, 80:112] += 1.1
            labels[end_frame : min(seq_len, end_frame + 10), 0] = 1.0
        elif mode == 1:
            # Mid-utterance hesitation HOLD pause -> silence gap should NOT trigger EOT!
            pause_start = seq_len // 3
            pause_len = int(rng.integers(5, 10))
            pause_end = min(seq_len - 4, pause_start + pause_len)
            feats[:pause_start, :16] += rng.normal(0.95, 0.18, size=(pause_start, 16)).astype(np.float32)
            if feature_dim >= 112:
                # Semantic "unfinished/hesitation" cue in upper encoder dims.
                feats[max(0, pause_start - 4) : pause_start, 80:112] -= 1.1
            feats[pause_end : seq_len - 5, :16] += rng.normal(0.85, 0.15, size=(seq_len - 5 - pause_end, 16)).astype(np.float32)
            if feature_dim >= 112:
                feats[seq_len - 8 : seq_len - 5, 80:112] += 1.1
            labels[seq_len - 5 :, 0] = 1.0
        elif mode == 2:
            # Assistant speaking; user barges in (sustained Floor Taking).
            playback[:, 0] = 1.0
            barge_start = seq_len // 3
            barge_end = min(seq_len, barge_start + 14)
            feats[barge_start:barge_end, :16] += rng.normal(0.95, 0.18, size=(barge_end - barge_start, 16)).astype(np.float32)
            if feature_dim >= 128:
                feats[barge_start:barge_end, 96:128] += 1.2
            labels[barge_start + 2 : barge_end, 1] = 1.0
        else:
            # Assistant speaking; user gives brief backchannel ("uh-huh", 3-5 frames).
            playback[:, 0] = 1.0
            bc_start = seq_len // 2
            bc_len = int(rng.integers(3, 6))
            bc_end = min(seq_len, bc_start + bc_len)
            feats[bc_start:bc_end, :16] += rng.normal(0.75, 0.15, size=(bc_end - bc_start, 16)).astype(np.float32)
            if feature_dim >= 128:
                feats[bc_start:bc_end, 96:128] -= 1.0
            labels[bc_start + 1 : bc_end, 2] = 1.0

        samples.append(
            TurnSessionSample(
                session_id=f"synth_sess_{i:04d}",
                encoder_features=feats,
                playback_state=playback,
                labels=labels,
            )
        )
    return samples


class TurnSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[TurnSessionSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        return (
            torch.from_numpy(s.encoder_features),
            torch.from_numpy(s.playback_state),
            torch.from_numpy(s.labels),
        )


def load_real_dialogue_jsonl(manifest_path: str | Path) -> list[TurnSessionSample]:
    """
    Loads real conversational turn-taking datasets (e.g., Mundo TurnBench, Switchboard,
    or Voice-Light pre-extracted feature manifests) into `TurnSessionSample` items.
    Each JSONL row expects:
      {"session_id": str, "features_npy": str, "playback_npy": str, "labels_npy": str}
    """
    base_dir = Path(manifest_path).parent
    samples: list[TurnSessionSample] = []
    for raw in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        feats = np.load(base_dir / row["features_npy"]).astype(np.float32)
        playback = np.load(base_dir / row["playback_npy"]).astype(np.float32)
        labels = np.load(base_dir / row["labels_npy"]).astype(np.float32)
        if playback.ndim == 1:
            playback = playback[:, None]
        samples.append(
            TurnSessionSample(
                session_id=str(row["session_id"]),
                encoder_features=feats,
                playback_state=playback,
                labels=labels,
            )
        )
    return samples
