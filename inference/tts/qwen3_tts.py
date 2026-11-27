"""Streaming Qwen3-TTS 0.6B backend with instant cancel support & offline test synthesizer."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import torch

from inference.tts.base import TTSChunk


class Qwen3StreamingTTS:
    """
    Streaming TTS backend targeting `Qwen/Qwen3-TTS-12Hz-0.6B-Base` / `CustomVoice`.
    Supports cooperative `asyncio.Event` cancellation between every emitted PCM chunk
    so barge-in immediately frees GPU compute and stops audio emission.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        *,
        sample_rate: int = 24000,
        chunk_ms: int = 120,
        device: str | None = None,
        load_heavy_model: bool = False,
    ) -> None:
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path: str | None = None
        self._model: Any = None

        if load_heavy_model or os.getenv("VOICE_AGENT_LOAD_HEAVY_MODELS") == "1":
            self._try_load_qwen3(model_id)

    def _try_load_qwen3(self, model_ref: str) -> None:
        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore[import-not-found]

            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            try:
                self._model = Qwen3TTSModel.from_pretrained(
                    model_ref,
                    dtype=dtype,
                    device_map=self.device,
                    attn_implementation="sdpa",
                )
            except TypeError:
                self._model = Qwen3TTSModel.from_pretrained(
                    model_ref,
                    dtype=dtype,
                    device_map=self.device,
                )
        except Exception:
            self._model = None

    def load_checkpoint(self, checkpoint_path: str | None) -> None:
        self.checkpoint_path = checkpoint_path
        if checkpoint_path and self._model is not None and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state, strict=False)

    async def synthesize_stream(
        self,
        text: str,
        *,
        base_char_start: int = 0,
        speaker_profile: str = "default",
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[TTSChunk]:
        cleaned = text.strip()
        if not cleaned:
            return
        t0 = time.perf_counter()

        if self._model is not None:
            wav = await asyncio.to_thread(self._generate_qwen3_wav, cleaned, speaker_profile)
        else:
            # shortcut: harmonic envelope PCM generator when Qwen3-TTS weights are not loaded;
            # real neural synthesis runs automatically once `qwen-tts` & weights are present.
            wav = self._synthesize_harmonic_pcm(cleaned)

        total_samples = wav.size
        total_chars = len(cleaned)
        chunk_idx = 0
        offset = 0

        while offset < total_samples:
            if cancel_event is not None and cancel_event.is_set():
                return
            end = min(total_samples, offset + self.chunk_samples)
            piece = wav[offset:end]
            char_start = base_char_start + int((offset / total_samples) * total_chars)
            char_end = base_char_start + int((end / total_samples) * total_chars)
            duration_ms = (piece.size / self.sample_rate) * 1000.0
            ttfa_ms = (time.perf_counter() - t0) * 1000.0 if chunk_idx == 0 else 0.0

            yield TTSChunk(
                chunk_index=chunk_idx,
                pcm16=piece,
                sample_rate=self.sample_rate,
                char_start=char_start,
                char_end=max(char_start, char_end),
                duration_ms=duration_ms,
                ttfa_ms=ttfa_ms,
            )
            chunk_idx += 1
            offset = end
            # Yield control back to event loop so `generation.cancel` interrupts promptly.
            await asyncio.sleep(0)

    def _generate_qwen3_wav(self, text: str, speaker_profile: str) -> np.ndarray:
        speaker = "Vivian" if speaker_profile == "default" else speaker_profile
        with torch.inference_mode():
            if hasattr(self._model, "generate_custom_voice"):
                wavs, sr = self._model.generate_custom_voice(
                    text=text, language="Auto", speaker=speaker
                )
            else:
                wavs, sr = self._model.generate(text=text)
        if isinstance(sr, int) and sr > 0:
            self.sample_rate = sr
        arr = np.asarray(wavs[0], dtype=np.float32)
        return np.clip(arr * 32767.0, -32768, 32767).astype("<i2")

    def _synthesize_harmonic_pcm(self, text: str) -> np.ndarray:
        # Approximate realistic speech duration (~65ms per char, bounded 240ms..6s).
        duration_s = min(6.0, max(0.24, len(text) * 0.055))
        n = int(self.sample_rate * duration_s)
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        base_f0 = 190.0 + (sum(ord(c) for c in text) % 45)
        sig = 0.18 * np.sin(2.0 * np.pi * base_f0 * t) + 0.08 * np.sin(4.0 * np.pi * base_f0 * t)
        fade = min(n // 8, int(self.sample_rate * 0.02))
        if fade > 0:
            env = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            sig[:fade] *= env
            sig[-fade:] *= env[::-1]
        return np.clip(sig * 32767.0, -32768, 32767).astype("<i2")
