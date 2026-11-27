"""Streaming Nemotron ASR backend + Silero VAD wrapper with causal encoder feature extraction."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from inference.asr.base import ASRStepOutput, FeatureSource, StreamingEncoderCache


class StreamingVAD:
    """Silero VAD v5/v6 wrapper with 512-sample sliding window and offline fallback."""

    def __init__(self, sample_rate: int = 16000, use_silero: bool = True) -> None:
        self.sample_rate = sample_rate
        self.backend_name = "heuristic"
        self._silero_model: Any = None
        self._window = np.zeros(512, dtype=np.float32)
        if use_silero:
            try:
                from silero_vad import load_silero_vad  # type: ignore[import-not-found]

                self._silero_model = load_silero_vad(onnx=True)
                self.backend_name = "silero-vad-onnx"
            except Exception:
                try:
                    from silero_vad import load_silero_vad  # type: ignore[import-not-found]

                    self._silero_model = load_silero_vad(onnx=False)
                    self.backend_name = "silero-vad-torch"
                except Exception:
                    self._silero_model = None

    def reset(self) -> None:
        self._window.fill(0.0)
        if self._silero_model is not None and hasattr(self._silero_model, "reset_states"):
            self._silero_model.reset_states()

    def score_frame(self, pcm_16k: np.ndarray) -> float:
        if pcm_16k.size == 0:
            return 0.0
        audio_f32 = pcm_16k.astype(np.float32) / 32768.0
        if self._silero_model is not None:
            if audio_f32.size >= 512:
                self._window = audio_f32[-512:].copy()
            else:
                shift = audio_f32.size
                self._window[:-shift] = self._window[shift:]
                self._window[-shift:] = audio_f32
            with torch.inference_mode():
                tensor = torch.from_numpy(self._window)
                return float(self._silero_model(tensor, self.sample_rate).item())

        # shortcut: calibrated RMS + zero-crossing heuristic when silero_vad package is absent;
        # real neural Silero VAD activates automatically once `silero-vad` is installed.
        rms = float(np.sqrt(np.mean(audio_f32 * audio_f32) + 1e-12))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(audio_f32)))))
        energy_score = np.clip((rms - 0.008) / 0.045, 0.0, 1.0)
        zcr_penalty = 0.55 if zcr > 0.45 else 1.0
        return float(np.clip(energy_score * zcr_penalty, 0.0, 1.0))


@dataclass
class _SessionASRState:
    buffer: list[np.ndarray] = field(default_factory=list)
    accumulated_samples: int = 0
    speech_chunks: int = 0
    last_partial: str = ""
    step_count: int = 0
    previous_hypotheses: Any = None
    previous_pred_out: Any = None
    encoder_cache: StreamingEncoderCache = field(default_factory=StreamingEncoderCache)


class NemotronStreamingASR:
    """
    Cache-aware streaming ASR interface targeting `nvidia/nemotron-speech-streaming-en-0.6b`.
    When NeMo weights are loaded (`load_heavy_model=True`), hooks intermediate FastConformer
    encoder layers for causal Turn Adapter consumption. Otherwise runs a deterministic causal
    spectral feature extractor so Turn Adapter training/inference and WS tests work offline.
    """

    def __init__(
        self,
        model_id: str = "nvidia/nemotron-speech-streaming-en-0.6b",
        *,
        chunk_size_ms: int = 80,
        feature_dim: int = 128,
        feature_source: FeatureSource = "asr_encoder",
        device: str | None = None,
        load_heavy_model: bool = False,
    ) -> None:
        self.model_id = model_id
        self.chunk_size_ms = chunk_size_ms
        self.chunk_samples = int(16000 * chunk_size_ms / 1000)
        self.feature_dim = feature_dim
        self.feature_source: FeatureSource = feature_source
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path: str | None = None
        self._sessions: dict[str, _SessionASRState] = {}
        self._nemo_model: Any = None
        self._hf_asr_model: Any = None
        self._hf_asr_processor: Any = None
        self.active_backend = "offline-spectral"
        self._encoder_hook_outputs: list[torch.Tensor] = []

        if load_heavy_model or os.getenv("VOICE_AGENT_LOAD_HEAVY_MODELS") == "1":
            self._try_load_nemo(model_id)
            if self._nemo_model is None:
                self._try_load_hf_whisper()

    def _try_load_hf_whisper(self, model_ref: str | None = None) -> None:
        """Loads lightweight CUDA BF16 Whisper via already-installed transformers when NeMo is absent."""
        repo: str = model_ref or os.getenv("VOICE_AGENT_HF_ASR_MODEL") or "openai/whisper-base.en"
        try:
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
            )

            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self._hf_asr_processor = AutoProcessor.from_pretrained(repo)
            self._hf_asr_model = (
                AutoModelForSpeechSeq2Seq.from_pretrained(repo, dtype=dtype)
                .to(self.device)
                .eval()
            )
            self.active_backend = f"transformers:{repo}"
        except Exception:
            self._hf_asr_model = None
            self._hf_asr_processor = None

    def _try_load_nemo(self, model_ref: str) -> None:
        try:
            import nemo.collections.asr as nemo_asr  # type: ignore[import-not-found]

            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_ref)
            model = model.to(device=self.device, dtype=dtype).eval()
            if hasattr(model, "encoder") and hasattr(model.encoder, "set_default_att_context_size"):
                lookahead = 0 if self.chunk_size_ms <= 80 else 1 if self.chunk_size_ms <= 160 else 3
                model.encoder.set_default_att_context_size([70, lookahead])
                if hasattr(model.encoder, "setup_streaming_params"):
                    model.encoder.setup_streaming_params()
            self._nemo_model = model
            self.active_backend = f"nemo:{model_ref}"
            self._register_encoder_hooks()
        except Exception:
            self._nemo_model = None

    def _register_encoder_hooks(self) -> None:
        if self._nemo_model is None or not hasattr(self._nemo_model, "encoder"):
            return
        layers = getattr(self._nemo_model.encoder, "layers", None)
        if not layers:
            return
        tap_indices = sorted({max(0, len(layers) // 2 - 1), len(layers) - 1})

        def _capture_hook(_module: Any, _inp: Any, out: Any) -> None:
            tensor = out[0] if isinstance(out, tuple) else out
            if isinstance(tensor, torch.Tensor):
                self._encoder_hook_outputs.append(tensor.detach())

        for idx in tap_indices:
            layers[idx].register_forward_hook(_capture_hook)

    def reset_state(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def load_checkpoint(self, checkpoint_path: str | None) -> None:
        self.checkpoint_path = checkpoint_path
        if checkpoint_path and self._nemo_model is not None and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self._nemo_model.load_state_dict(state, strict=False)

    def process_chunk(
        self,
        session_id: str,
        pcm_16k: np.ndarray,
        *,
        is_speech: bool,
        extract_features: bool = True,
    ) -> ASRStepOutput:
        t0 = time.perf_counter()
        state = self._sessions.setdefault(session_id, _SessionASRState())
        samples = np.asarray(pcm_16k, dtype=np.int16).reshape(-1)

        if samples.size > 0:
            state.buffer.append(samples)
            state.accumulated_samples += int(samples.size)
            # Cap rolling audio buffer per session to 30 seconds to prevent memory bloat.
            max_samples = 16000 * 30
            while state.accumulated_samples > max_samples and len(state.buffer) > 1:
                dropped = state.buffer.pop(0)
                state.accumulated_samples -= int(dropped.size)

        if is_speech:
            state.speech_chunks += 1

        partial = state.last_partial
        if self._nemo_model is not None and state.accumulated_samples >= self.chunk_samples:
            partial = self._run_nemo_streaming_step(state)
        elif (
            self._hf_asr_model is not None
            and is_speech
            and state.accumulated_samples >= 16000 // 2
            and state.speech_chunks % 4 == 0
        ):
            partial = self._run_hf_whisper_step(state, extract_features=extract_features)

        taps: torch.Tensor | None = None
        active_source: FeatureSource = self.feature_source
        if extract_features and samples.size > 0:
            if self.feature_source == "asr_encoder" and self._encoder_hook_outputs:
                # Stack captured intermediate encoder layers: (1, num_taps, frames, dim)
                stacked = torch.stack(self._encoder_hook_outputs, dim=1)
                taps = stacked[..., : self.feature_dim]
            else:
                active_source = "acoustic_logmel"
                taps = self._extract_causal_features(samples)

        if (
            self._nemo_model is None
            and self._hf_asr_model is None
            and is_speech
            and state.speech_chunks >= 3
        ):
            # shortcut: deterministic offline placeholder when NeMo/Whisper weights are not loaded;
            # real GPU transcription activates when VOICE_AGENT_LOAD_HEAVY_MODELS=1.
            partial = f"[speech_stream_{state.speech_chunks}]"

        state.last_partial = partial
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ASRStepOutput(
            partial_text=partial,
            is_final=False,
            encoder_taps=taps,
            compute_time_ms=elapsed_ms,
            feature_source=active_source,
        )

    def finalize_utterance(self, session_id: str) -> ASRStepOutput:
        t0 = time.perf_counter()
        state = self._sessions.get(session_id)
        if state is not None and self._hf_asr_model is not None and state.accumulated_samples > 0:
            state.last_partial = self._run_hf_whisper_step(state, extract_features=False)
        text = state.last_partial if state else ""
        if state is not None:
            state.buffer.clear()
            state.accumulated_samples = 0
            state.speech_chunks = 0
            state.last_partial = ""
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ASRStepOutput(
            partial_text=text,
            is_final=True,
            encoder_taps=None,
            compute_time_ms=elapsed_ms,
        )

    def _run_hf_whisper_step(
        self, state: _SessionASRState, *, extract_features: bool = True
    ) -> str:
        if self._hf_asr_model is None or self._hf_asr_processor is None:
            return state.last_partial
        audio = np.concatenate(state.buffer).astype(np.float32) / 32768.0
        dtype = next(self._hf_asr_model.parameters()).dtype
        inputs = self._hf_asr_processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device=self.device, dtype=dtype)
        with torch.inference_mode():
            if extract_features:
                enc_out = self._hf_asr_model.model.encoder(
                    input_features, output_hidden_states=True
                )
                hidden_states = enc_out.hidden_states
                if hidden_states and len(hidden_states) >= 2:
                    mid = hidden_states[len(hidden_states) // 2][:, -1:, :].float()
                    top = hidden_states[-1][:, -1:, :].float()
                    self._encoder_hook_outputs = [mid, top]
            pred_ids = self._hf_asr_model.generate(input_features, max_new_tokens=64)
        decoded = self._hf_asr_processor.batch_decode(pred_ids, skip_special_tokens=True)
        return str(decoded[0]).strip() if decoded else state.last_partial

    def _extract_causal_features(self, pcm_16k: np.ndarray) -> torch.Tensor:
        """
        Extracts strictly causal log-magnitude spectral features of shape (1, 1, feature_dim)
        using only samples present in the current chunk (zero future lookahead).
        """
        audio = pcm_16k.astype(np.float32) / 32768.0
        n_fft = max(256, self.feature_dim * 2)
        if audio.size < n_fft:
            padded = np.zeros(n_fft, dtype=np.float32)
            padded[-audio.size :] = audio
        else:
            padded = audio[-n_fft:]
        window = np.hanning(n_fft).astype(np.float32)
        spec = np.abs(np.fft.rfft(padded * window))[: self.feature_dim]
        log_spec = np.log1p(spec).astype(np.float32)
        return torch.from_numpy(log_spec).view(1, 1, self.feature_dim)

    def _run_nemo_streaming_step(self, state: _SessionASRState) -> str:
        if self._nemo_model is None:
            return state.last_partial
        self._encoder_hook_outputs.clear()
        chunk_audio = np.concatenate(state.buffer).astype(np.float32) / 32768.0
        with torch.inference_mode():
            # Prefer true O(1) cache-aware conformer_stream_step when supported by Nemotron/NeMo.
            if hasattr(self._nemo_model, "conformer_stream_step") and hasattr(
                self._nemo_model.encoder, "get_initial_cache_state"
            ):
                if state.encoder_cache.cache_last_channel is None:
                    c_ch, c_t, c_len = self._nemo_model.encoder.get_initial_cache_state(
                        batch_size=1
                    )
                    state.encoder_cache = StreamingEncoderCache(
                        cache_last_channel=c_ch.to(self.device),
                        cache_last_time=c_t.to(self.device),
                        cache_last_channel_len=c_len.to(self.device),
                    )
                sig = torch.from_numpy(chunk_audio).unsqueeze(0).to(self.device)
                sig_len = torch.tensor([sig.shape[1]], dtype=torch.long, device=self.device)
                proc_sig, proc_len = self._nemo_model.preprocessor(
                    input_signal=sig, length=sig_len
                )
                streaming_cfg = getattr(self._nemo_model.encoder, "streaming_cfg", None)
                drop_extra = (
                    getattr(streaming_cfg, "drop_extra_pre_encoded", 0)
                    if state.step_count > 0
                    else 0
                )
                (
                    state.previous_pred_out,
                    texts,
                    c_ch,
                    c_t,
                    c_len,
                    state.previous_hypotheses,
                ) = self._nemo_model.conformer_stream_step(
                    processed_signal=proc_sig,
                    processed_signal_length=proc_len,
                    cache_last_channel=state.encoder_cache.cache_last_channel,
                    cache_last_time=state.encoder_cache.cache_last_time,
                    cache_last_channel_len=state.encoder_cache.cache_last_channel_len,
                    keep_all_outputs=False,
                    previous_hypotheses=state.previous_hypotheses,
                    previous_pred_out=state.previous_pred_out,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )
                state.encoder_cache = StreamingEncoderCache(
                    cache_last_channel=c_ch,
                    cache_last_time=c_t,
                    cache_last_channel_len=c_len,
                )
                state.step_count += 1
                state.buffer.clear()
                state.accumulated_samples = 0
                if texts:
                    first = texts[0]
                    return first.text if hasattr(first, "text") else str(first)
                return state.last_partial

            results = self._nemo_model.transcribe([chunk_audio], verbose=False)
        if not results:
            return state.last_partial
        first = results[0]
        return first.text if hasattr(first, "text") else str(first)
