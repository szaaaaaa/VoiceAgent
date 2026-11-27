"""Stateful streaming inference wrapper for CausalTurnGRU."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from inference.turn_adapter.model import CausalTurnGRU, TurnAdapterConfig


@dataclass(slots=True)
class TurnAdapterPrediction:
    p_completion: float
    p_floor_taking: float
    p_non_floor_feedback: float
    inference_time_ms: float
    adapter_version: str


class TurnAdapterPredictor:
    def __init__(
        self,
        config: TurnAdapterConfig | None = None,
        *,
        checkpoint_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.adapter_version = "turn-gru-untrained"
        self.checkpoint_path = checkpoint_path
        self._states: dict[str, torch.Tensor] = {}

        if checkpoint_path and Path(checkpoint_path).exists():
            self.model, meta = CausalTurnGRU.from_checkpoint(checkpoint_path, device=self.device)
            self.adapter_version = str(meta.get("version", Path(checkpoint_path).stem))
        else:
            self.model = CausalTurnGRU(config).to(self.device).eval()

    def reset_state(self, session_id: str) -> None:
        self._states.pop(session_id, None)

    def load_checkpoint(self, checkpoint_path: str | None) -> None:
        self.checkpoint_path = checkpoint_path
        self._states.clear()
        if checkpoint_path and Path(checkpoint_path).exists():
            self.model, meta = CausalTurnGRU.from_checkpoint(checkpoint_path, device=self.device)
            self.adapter_version = str(meta.get("version", Path(checkpoint_path).stem))
        else:
            self.model = CausalTurnGRU(self.model.config).to(self.device).eval()
            self.adapter_version = "turn-gru-untrained"

    def step(
        self,
        session_id: str,
        encoder_taps: torch.Tensor,
        assistant_speaking: bool,
    ) -> TurnAdapterPrediction:
        t0 = time.perf_counter()
        feats = encoder_taps.to(device=self.device, dtype=torch.float32)
        if feats.ndim == 2:
            feats = feats.unsqueeze(0)
        batch, seq_len, _ = feats.shape
        playback_tensor = torch.full(
            (batch, seq_len, 1),
            1.0 if assistant_speaking else 0.0,
            device=self.device,
            dtype=torch.float32,
        )
        prev_hidden = self._states.get(session_id)

        with torch.inference_mode():
            logits, next_hidden = self.model(feats, playback_tensor, prev_hidden)
            probs = torch.sigmoid(logits[:, -1, :]).squeeze(0).cpu().tolist()

        self._states[session_id] = next_hidden.detach()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return TurnAdapterPrediction(
            p_completion=float(probs[0]),
            p_floor_taking=float(probs[1]),
            p_non_floor_feedback=float(probs[2]),
            inference_time_ms=elapsed_ms,
            adapter_version=self.adapter_version,
        )
