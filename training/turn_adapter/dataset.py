"""
Session-level dataset splitting and causal frame-sequence dataset for Turn Adapter training.
Enforces strict session isolation across train / val / test splits (PRD §6.2).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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
    Generates causal synthetic turn-taking sequences mimicking Voice-Light interaction semantics:
      - EOT completion at end of user utterance after brief trailing drop in speech energy
      - Active floor-taking interruption during assistant playback (sustained high energy)
      - Non-floor backchannel during assistant playback (brief localized energy bump)
    """
    rng = np.random.default_rng(seed)
    samples: list[TurnSessionSample] = []

    for i in range(num_sessions):
        feats = rng.normal(0.0, 0.25, size=(seq_len, feature_dim)).astype(np.float32)
        playback = np.zeros((seq_len, 1), dtype=np.float32)
        labels = np.zeros((seq_len, 3), dtype=np.float32)

        mode = i % 3
        if mode == 0:
            # User speaks then finishes (EOT).
            end_frame = seq_len // 2 + int(rng.integers(2, 8))
            feats[:end_frame, :16] += 1.4
            feats[end_frame:, :16] -= 0.8
            labels[end_frame : min(seq_len, end_frame + 5), 0] = 1.0
        elif mode == 1:
            # Assistant speaking; user barges in (Floor Taking).
            playback[:, 0] = 1.0
            barge_start = seq_len // 3
            feats[barge_start : barge_start + 14, 16:32] += 1.8
            labels[barge_start + 2 : barge_start + 14, 1] = 1.0
        else:
            # Assistant speaking; user gives brief backchannel ("uh-huh").
            playback[:, 0] = 1.0
            bc_start = seq_len // 2
            feats[bc_start : bc_start + 4, 32:48] += 1.5
            labels[bc_start + 1 : bc_start + 4, 2] = 1.0

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
