"""Abstract interfaces for streaming VAD and ASR backends with encoder feature taps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch


@dataclass(slots=True)
class ASRStepOutput:
    partial_text: str
    is_final: bool
    encoder_taps: torch.Tensor | None  # Shape: (1, frames, feature_dim), strictly causal
    compute_time_ms: float


class StreamingASRBackend(Protocol):
    model_id: str
    feature_dim: int

    def reset_state(self, session_id: str) -> None: ...

    def process_chunk(
        self,
        session_id: str,
        pcm_16k: np.ndarray,
        *,
        is_speech: bool,
        extract_features: bool = True,
    ) -> ASRStepOutput: ...

    def finalize_utterance(self, session_id: str) -> ASRStepOutput: ...

    def load_checkpoint(self, checkpoint_path: str | None) -> None: ...
