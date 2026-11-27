"""Abstract streaming TTS backend interface."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class TTSChunk:
    chunk_index: int
    pcm16: np.ndarray
    sample_rate: int
    char_start: int
    char_end: int
    duration_ms: float
    ttfa_ms: float


class StreamingTTSBackend(Protocol):
    model_id: str
    sample_rate: int

    def synthesize_stream(
        self,
        text: str,
        *,
        base_char_start: int,
        speaker_profile: str = "default",
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[TTSChunk]: ...

    def load_checkpoint(self, checkpoint_path: str | None) -> None: ...
