"""Local speech runtime manager: config, model registry, VRAM telemetry, and cancel state."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from inference.asr.nemotron_streaming import NemotronStreamingASR, StreamingVAD
from inference.tts.qwen3_tts import Qwen3StreamingTTS
from inference.turn_adapter.predictor import TurnAdapterPredictor


@dataclass
class SessionRuntimeState:
    session_id: str
    in_speech: bool = False
    speech_start_pts_ms: float = 0.0
    silence_frames: int = 0
    assistant_speaking: bool = False
    active_cancel_events: dict[int, asyncio.Event] = field(default_factory=dict)
    cancelled_tags: deque[int] = field(default_factory=lambda: deque(maxlen=256))


class InferenceRuntimeManager:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config = self._load_config(config_path)
        asr_cfg = self.config.get("asr", {})
        tts_cfg = self.config.get("tts", {})
        turn_cfg = self.config.get("turn_manager", {})
        vad_cfg = self.config.get("vad", {})

        self.vad_start_threshold = float(vad_cfg.get("threshold_start", 0.5))
        self.vad_end_threshold = float(vad_cfg.get("threshold_end", 0.35))
        self.vad = StreamingVAD(sample_rate=16000)
        self.asr = NemotronStreamingASR(
            model_id=str(asr_cfg.get("model_id", "nvidia/nemotron-speech-streaming-en-0.6b")),
            chunk_size_ms=int(asr_cfg.get("chunk_size_ms", 80)),
        )
        self.turn_adapter = TurnAdapterPredictor(
            checkpoint_path=turn_cfg.get("adapter_checkpoint"),
        )
        self.tts = Qwen3StreamingTTS(
            model_id=str(tts_cfg.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")),
            sample_rate=int(self.config.get("audio", {}).get("output_sample_rate", 24000)),
        )
        self.sessions: dict[str, SessionRuntimeState] = {}

    @staticmethod
    def _load_config(config_path: str | Path | None) -> dict[str, Any]:
        default_path = Path(config_path) if config_path else Path("configs/default.yaml")
        if default_path.exists():
            return dict(yaml.safe_load(default_path.read_text(encoding="utf-8")) or {})
        return {}

    def get_session(self, session_id: str) -> SessionRuntimeState:
        return self.sessions.setdefault(session_id, SessionRuntimeState(session_id=session_id))

    def reset_session(self, session_id: str) -> None:
        state = self.sessions.pop(session_id, None)
        if state:
            for ev in state.active_cancel_events.values():
                ev.set()
        self.asr.reset_state(session_id)
        self.turn_adapter.reset_state(session_id)

    def register_tts_generation(self, session_id: str, generation_tag: int) -> asyncio.Event:
        state = self.get_session(session_id)
        ev = state.active_cancel_events.get(generation_tag)
        if ev is None:
            ev = asyncio.Event()
            state.active_cancel_events[generation_tag] = ev
        if generation_tag in state.cancelled_tags:
            ev.set()
        return ev

    def cancel_generation(self, session_id: str, generation_tag: int) -> bool:
        state = self.get_session(session_id)
        if generation_tag > 0 and generation_tag not in state.cancelled_tags:
            state.cancelled_tags.append(generation_tag)
        ev = state.active_cancel_events.get(generation_tag)
        if ev is not None:
            ev.set()
            return True
        return generation_tag > 0

    def is_generation_cancelled(self, session_id: str, generation_tag: int) -> bool:
        state = self.sessions.get(session_id)
        return bool(state and generation_tag in state.cancelled_tags)

    def finish_tts_generation(self, session_id: str, generation_tag: int) -> None:
        state = self.sessions.get(session_id)
        if state:
            state.active_cancel_events.pop(generation_tag, None)

    def switch_model(self, target: str, checkpoint_path: str | None) -> None:
        if target == "asr":
            self.asr.load_checkpoint(checkpoint_path)
        elif target == "tts":
            self.tts.load_checkpoint(checkpoint_path)
        elif target == "turn_adapter":
            self.turn_adapter.load_checkpoint(checkpoint_path)
        else:
            raise ValueError(f"Unknown model switch target: {target}")

    def vram_telemetry(self) -> dict[str, Any]:
        if torch.cuda.is_available():
            allocated_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
            reserved_mb = round(torch.cuda.memory_reserved() / (1024 * 1024), 1)
            total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
            return {
                "device": torch.cuda.get_device_name(0),
                "allocated_mb": allocated_mb,
                "reserved_mb": reserved_mb,
                "total_mb": total_mb,
            }
        return {"device": "cpu", "allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0}

    def models_info(self) -> dict[str, Any]:
        return {
            "asr": {"model_id": self.asr.model_id, "checkpoint": self.asr.checkpoint_path},
            "turn_adapter": {
                "version": self.turn_adapter.adapter_version,
                "checkpoint": self.turn_adapter.checkpoint_path,
                "params": self.turn_adapter.model.parameter_count(),
            },
            "tts": {
                "model_id": self.tts.model_id,
                "sample_rate": self.tts.sample_rate,
                "checkpoint": self.tts.checkpoint_path,
            },
        }
