"""
Causal GRU Turn-Taking Adapter (PRD §6.2, referencing Voice-Light).
Consumes frozen ASR encoder features + assistant playback state strictly causally
(unidirectional GRU carrying hidden state across chunks) and predicts:
  - p_completion (EOT)
  - p_floor_taking (active interruption)
  - p_non_floor_feedback (backchannel)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class TurnAdapterConfig:
    encoder_feature_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    num_encoder_taps: int = 1
    feature_source: str = "asr_encoder"  # "asr_encoder" | "acoustic_logmel"
    dropout: float = 0.1
    eot_threshold: float = 0.72
    interruption_threshold: float = 0.68
    backchannel_threshold: float = 0.60


class CausalTurnGRU(nn.Module):
    """Lightweight (~183K params at default config) causal turn-taking adapter."""

    def __init__(self, config: TurnAdapterConfig | None = None) -> None:
        super().__init__()
        self.config = config or TurnAdapterConfig()
        # Learnable scalar weights to fuse multi-layer frozen ASR encoder taps when K > 1.
        self.tap_weights = nn.Parameter(torch.zeros(max(1, self.config.num_encoder_taps)))
        in_dim = self.config.encoder_feature_dim + 1  # +1 for assistant_playback_state scalar
        self.input_norm = nn.LayerNorm(in_dim)
        self.input_proj = nn.Linear(in_dim, self.config.hidden_dim)
        self.gru = nn.GRU(
            input_size=self.config.hidden_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=False,  # Strict causality: no future frame leakage.
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        self.head_norm = nn.LayerNorm(self.config.hidden_dim)
        # 3 independent binary heads: [completion, floor_taking, non_floor_feedback]
        self.classifier = nn.Linear(self.config.hidden_dim, 3)

    def forward(
        self,
        encoder_features: torch.Tensor,
        assistant_playback_state: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_features: (batch, seq_len, encoder_feature_dim)
            assistant_playback_state: (batch, seq_len, 1) float in [0, 1]
            hidden_state: optional (num_layers, batch, hidden_dim)
        Returns:
            logits: (batch, seq_len, 3)
            next_hidden: (num_layers, batch, hidden_dim)
        """
        if encoder_features.ndim == 4:
            # Multi-layer encoder taps: (batch, taps, seq_len, dim) -> weighted causal sum.
            weights = torch.softmax(self.tap_weights[: encoder_features.shape[1]], dim=0)
            encoder_features = torch.einsum("k,bktd->btd", weights, encoder_features)
        if assistant_playback_state.ndim == 2:
            assistant_playback_state = assistant_playback_state.unsqueeze(-1)
        x = torch.cat([encoder_features, assistant_playback_state], dim=-1)
        x = torch.nn.functional.gelu(self.input_proj(self.input_norm(x)))
        out, next_hidden = self.gru(x, hidden_state)
        logits = self.classifier(self.head_norm(out))
        return logits, next_hidden

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(self, path: str | Path, extra_meta: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
                "meta": extra_meta or {},
            },
            target,
        )

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> tuple[CausalTurnGRU, dict[str, Any]]:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        cfg = TurnAdapterConfig(**payload["config"])
        model = cls(cfg).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, dict(payload.get("meta", {}))
