"""Local speech runtime manager: config, model registry, hot-swap, A/B eval, VRAM telemetry, and cancel state."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import random
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from evaluation.asr.metrics import evaluate_asr_predictions
from inference.asr.nemotron_streaming import NemotronStreamingASR, StreamingVAD
from inference.tts.qwen3_tts import Qwen3StreamingTTS
from inference.turn_adapter.predictor import TurnAdapterPredictor
from training.registry import CheckpointRegistry

VALID_TURN_STRATEGIES = ("fixed_silence", "vad_only", "smart_turn", "adapter", "hybrid")
VALID_TTS_ENGINES = ("qwen3", "fast")
VALID_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".nemo", ".bin")


def pcm16_to_wav_bytes(pcm16: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(np.asarray(pcm16, dtype="<i2").tobytes())
    return buf.getvalue()


def wav_bytes_to_pcm16_16k(raw: bytes) -> np.ndarray:
    """Decodes WAV bytes (or raw PCM16 fallback) into mono 16 kHz int16 samples."""
    if raw[:4] == b"RIFF":
        with wave.open(io.BytesIO(raw), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if sampwidth != 2:
            raise ValueError(f"Only 16-bit PCM WAV is supported, got {sampwidth * 8}-bit")
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if sr != 16000 and samples.size > 0:
            target_len = max(1, int(round(samples.size * 16000.0 / sr)))
            xp = np.linspace(0.0, 1.0, samples.size, dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
            samples = np.interp(x_new, xp, samples).astype(np.float32)
        return np.clip(samples, -32768, 32767).astype("<i2")

    if len(raw) % 2 != 0:
        raise ValueError("Raw PCM16 payload must have even byte count")
    return np.frombuffer(raw, dtype="<i2").copy()


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
    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        registry_path: str | Path = "models/registry.json",
        experiments_dir: str | Path = "docs/experiments",
    ) -> None:
        self.config = self._load_config(config_path)
        self.registry = CheckpointRegistry(registry_path)
        self.experiments_dir = Path(experiments_dir)

        asr_cfg = self.config.get("asr", {})
        tts_cfg = self.config.get("tts", {})
        turn_cfg = self.config.get("turn_manager", {})
        vad_cfg = self.config.get("vad", {})

        self.vad_start_threshold = float(vad_cfg.get("threshold_start", 0.5))
        self.vad_end_threshold = float(vad_cfg.get("threshold_end", 0.35))
        self.turn_strategy = str(turn_cfg.get("strategy", "hybrid"))
        self.tts_engine = "qwen3"

        self.vad = StreamingVAD(sample_rate=16000)
        self.asr = NemotronStreamingASR(
            model_id=str(asr_cfg.get("model_id", "nvidia/nemotron-speech-streaming-en-0.6b")),
            chunk_size_ms=int(asr_cfg.get("chunk_size_ms", 80)),
        )
        default_adapter = turn_cfg.get("adapter_checkpoint")
        self.turn_adapter = TurnAdapterPredictor(checkpoint_path=default_adapter)
        self.tts = Qwen3StreamingTTS(
            model_id=str(tts_cfg.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")),
            sample_rate=int(self.config.get("audio", {}).get("output_sample_rate", 24000)),
        )
        self.active_ids: dict[str, str] = {
            "asr": "asr-base-default",
            "turn_adapter": (
                "turn_gru_2x128_v1"
                if default_adapter and Path(str(default_adapter)).exists()
                else "turn-gru-untrained"
            ),
            "tts": "tts-qwen3-default",
        }
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

    def set_turn_strategy(self, strategy: str) -> str:
        if strategy not in VALID_TURN_STRATEGIES:
            raise ValueError(
                f"Invalid turn strategy '{strategy}'. Expected one of {VALID_TURN_STRATEGIES}"
            )
        self.turn_strategy = strategy
        return self.turn_strategy

    def set_tts_engine(self, engine: str) -> str:
        if engine not in VALID_TTS_ENGINES:
            raise ValueError(f"Invalid tts engine '{engine}'. Expected one of {VALID_TTS_ENGINES}")
        self.tts_engine = engine
        self.tts.force_fast_mode = engine == "fast"
        return self.tts_engine

    def switch_model(self, target: str, checkpoint_path: str | None) -> None:
        if checkpoint_path is not None:
            p = Path(checkpoint_path)
            if p.suffix.lower() not in VALID_CHECKPOINT_SUFFIXES:
                raise ValueError(
                    f"Unsupported checkpoint file suffix '{p.suffix}'. Allowed: {VALID_CHECKPOINT_SUFFIXES}"
                )
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
        if target == "asr":
            self.asr.load_checkpoint(checkpoint_path)
        elif target == "tts":
            self.tts.load_checkpoint(checkpoint_path)
        elif target == "turn_adapter":
            self.turn_adapter.load_checkpoint(checkpoint_path)
        else:
            raise ValueError(f"Unknown model switch target: {target}")

    def list_registry(self) -> dict[str, Any]:
        """Returns deduplicated builtin + registered checkpoints for Workbench hot-swap."""
        entries: dict[tuple[str, str], dict[str, Any]] = {}

        builtins: list[dict[str, Any]] = [
            {
                "checkpoint_id": "asr-base-default",
                "model_type": "asr",
                "path": None,
                "label": f"ASR Pretrained ({self.asr.active_backend})",
                "experiment_id": "ASR-A0",
                "metrics": {},
                "promoted": True,
                "builtin": True,
                "notes": "Base pretrained streaming ASR backend",
            },
            {
                "checkpoint_id": "turn-gru-untrained",
                "model_type": "turn_adapter",
                "path": None,
                "label": "CausalTurnGRU (Untrained Init)",
                "experiment_id": "TURN-INIT",
                "metrics": {},
                "promoted": False,
                "builtin": True,
                "notes": "Fresh randomly initialized GRU baseline",
            },
            {
                "checkpoint_id": "tts-qwen3-default",
                "model_type": "tts",
                "path": None,
                "label": "Qwen3-TTS 0.6B (Default)",
                "experiment_id": "TTS-A0",
                "metrics": {},
                "promoted": True,
                "builtin": True,
                "notes": "Pretrained Qwen3-TTS default speaker",
            },
            {
                "checkpoint_id": "tts-fast-harmonic",
                "model_type": "tts",
                "path": None,
                "label": "Fast Low-Latency TTS",
                "experiment_id": "OPT-FAST",
                "metrics": {"ttfa_ms": 2.0},
                "promoted": False,
                "builtin": True,
                "notes": "Instant local fast TTS engine for low-latency coding mode",
            },
        ]
        for item in builtins:
            entries[(item["model_type"], item["checkpoint_id"])] = item

        # Auto-discover local smoke/ablation checkpoints if present on disk.
        discovered = [
            ("asr", "asr-lora-v1", "models/asr/lora_v1/lora_smoke.pt", "ASR-A2", "ASR LoRA v1"),
            ("tts", "tts-speaker-v1", "models/tts/speaker_v1/tts_speaker_smoke.pt", "TTS-A2", "TTS Speaker v1"),
        ]
        for mtype, cid, rel_path, exp_id, label in discovered:
            if Path(rel_path).exists():
                entries[(mtype, cid)] = {
                    "checkpoint_id": cid,
                    "model_type": mtype,
                    "path": rel_path,
                    "label": label,
                    "experiment_id": exp_id,
                    "metrics": {},
                    "promoted": False,
                    "builtin": False,
                    "notes": f"Discovered local checkpoint at {rel_path}",
                }

        ablation_dir = Path("models/turn_adapter/ablations")
        if ablation_dir.is_dir():
            for pt in sorted(ablation_dir.glob("*.pt")):
                cid = f"ablation_{pt.stem}"
                entries[("turn_adapter", cid)] = {
                    "checkpoint_id": cid,
                    "model_type": "turn_adapter",
                    "path": pt.as_posix(),
                    "label": f"CausalTurnGRU ({pt.stem})",
                    "experiment_id": "TURN-ABLATION",
                    "metrics": {},
                    "promoted": False,
                    "builtin": False,
                    "notes": f"Ablation checkpoint {pt.name}",
                }

        # Overlay explicit registry records (newest wins for duplicate IDs).
        for rec in self.registry.list_records():
            mtype = str(rec.get("model_type", ""))
            cid = str(rec.get("checkpoint_id", ""))
            if not mtype or not cid:
                continue
            entries[(mtype, cid)] = {
                "checkpoint_id": cid,
                "model_type": mtype,
                "path": rec.get("path"),
                "label": f"{cid} ({rec.get('experiment_id', 'exp')})",
                "experiment_id": str(rec.get("experiment_id", "")),
                "metrics": dict(rec.get("metrics", {})),
                "promoted": bool(rec.get("promoted", False)),
                "builtin": False,
                "notes": str(rec.get("notes", "")),
            }

        return {
            "active": {
                "asr": self.asr.checkpoint_path,
                "turn_adapter": self.turn_adapter.checkpoint_path,
                "tts": self.tts.checkpoint_path,
            },
            "active_ids": dict(self.active_ids),
            "turn_strategy": self.turn_strategy,
            "tts_engine": self.tts_engine,
            "checkpoints": list(entries.values()),
            "models": self.models_info(),
        }

    def activate_checkpoint(
        self,
        task: str,
        checkpoint_id: str,
        *,
        checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        """Hot-swaps a model checkpoint, turn strategy, or TTS engine with rollback on failure."""
        if task == "turn_strategy":
            self.set_turn_strategy(checkpoint_id)
            return self.list_registry()
        if task == "tts_engine":
            self.set_tts_engine(checkpoint_id)
            return self.list_registry()

        if task not in ("asr", "turn_adapter", "tts"):
            raise ValueError(f"Unsupported activation task: {task}")

        reg_items = self.list_registry()["checkpoints"]
        matched = next(
            (c for c in reg_items if c["model_type"] == task and c["checkpoint_id"] == checkpoint_id),
            None,
        )
        if checkpoint_path is not None:
            target_path: str | None = checkpoint_path
        elif matched is not None:
            target_path = matched["path"]
        else:
            raise ValueError(f"Unknown checkpoint_id '{checkpoint_id}' for task '{task}'")
        prev_path = (
            self.asr.checkpoint_path
            if task == "asr"
            else self.turn_adapter.checkpoint_path
            if task == "turn_adapter"
            else self.tts.checkpoint_path
        )
        prev_engine = self.tts_engine

        try:
            if task == "tts":
                self.set_tts_engine("fast" if checkpoint_id == "tts-fast-harmonic" else "qwen3")
            self.switch_model(task, target_path)
            self.active_ids[task] = checkpoint_id
        except Exception:
            # Roll back cleanly so live sessions stay healthy.
            self.set_tts_engine(prev_engine)
            if prev_path is None or Path(prev_path).exists():
                self.switch_model(task, prev_path)
            raise

        return self.list_registry()

    def run_ab_asr(
        self,
        pcm_16k: np.ndarray,
        checkpoint_ids: list[str],
        *,
        reference_text: str | None = None,
    ) -> dict[str, Any]:
        """Runs the exact same 16 kHz PCM clip across multiple ASR checkpoints for A/B diff."""
        if pcm_16k.size == 0:
            raise ValueError("Audio clip for ASR A/B test is empty")
        reg_map = {
            c["checkpoint_id"]: c
            for c in self.list_registry()["checkpoints"]
            if c["model_type"] == "asr"
        }
        chosen = checkpoint_ids or list(reg_map.keys())[:2]
        audio_dur_s = max(1e-3, pcm_16k.size / 16000.0)

        prev_path = self.asr.checkpoint_path
        prev_id = self.active_ids.get("asr", "asr-base-default")
        results: list[dict[str, Any]] = []
        try:
            for idx, cid in enumerate(chosen):
                meta = reg_map.get(cid)
                if meta is None:
                    raise ValueError(f"Unknown ASR checkpoint_id: {cid}")
                self.asr.load_checkpoint(meta["path"])
                sess_id = f"__ab_asr_{idx}_{time.time_ns()}"
                t0 = time.perf_counter()
                frame_len = 320
                for start in range(0, pcm_16k.size, frame_len):
                    chunk = pcm_16k[start : start + frame_len]
                    self.asr.process_chunk(sess_id, chunk, is_speech=True, extract_features=False)
                out = self.asr.finalize_utterance(sess_id)
                self.asr.reset_state(sess_id)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                transcript = out.partial_text
                wer_val: float | None = None
                cer_val: float | None = None
                if reference_text and reference_text.strip():
                    asr_eval = evaluate_asr_predictions([reference_text], [transcript or ""])
                    wer_val = round(asr_eval.wer, 4)
                    cer_val = round(asr_eval.cer, 4)
                results.append(
                    {
                        "checkpoint_id": cid,
                        "label": meta["label"],
                        "transcript": transcript,
                        "compute_ms": round(elapsed_ms, 2),
                        "rtf": round((elapsed_ms / 1000.0) / audio_dur_s, 4),
                        "wer": wer_val,
                        "cer": cer_val,
                    }
                )
        finally:
            self.asr.load_checkpoint(prev_path)
            self.active_ids["asr"] = prev_id

        return {
            "audio_duration_ms": round(audio_dur_s * 1000.0, 1),
            "reference_text": reference_text,
            "candidates": results,
        }

    async def run_ab_tts(
        self,
        text: str,
        checkpoint_ids: list[str],
        *,
        blind: bool = True,
        speaker_profile: str = "default",
    ) -> dict[str, Any]:
        """Synthesizes identical text across multiple TTS candidates with optional blind shuffling."""
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text for TTS A/B test cannot be empty")
        reg_map = {
            c["checkpoint_id"]: c
            for c in self.list_registry()["checkpoints"]
            if c["model_type"] == "tts"
        }
        chosen = checkpoint_ids or list(reg_map.keys())[:2]

        prev_path = self.tts.checkpoint_path
        prev_engine = self.tts_engine
        prev_id = self.active_ids.get("tts", "tts-qwen3-default")
        raw_items: list[dict[str, Any]] = []
        try:
            for cid in chosen:
                meta = reg_map.get(cid)
                if meta is None:
                    raise ValueError(f"Unknown TTS checkpoint_id: {cid}")
                self.set_tts_engine("fast" if cid == "tts-fast-harmonic" else "qwen3")
                self.tts.load_checkpoint(meta["path"])

                t0 = time.perf_counter()
                pcm_parts: list[np.ndarray] = []
                ttfa_ms = 0.0
                sample_rate = self.tts.sample_rate
                async for chunk in self.tts.synthesize_stream(
                    cleaned, speaker_profile=speaker_profile
                ):
                    if chunk.chunk_index == 0:
                        ttfa_ms = chunk.ttfa_ms
                    sample_rate = chunk.sample_rate
                    pcm_parts.append(chunk.pcm16)
                total_ms = (time.perf_counter() - t0) * 1000.0
                full_pcm = (
                    np.concatenate(pcm_parts) if pcm_parts else np.zeros(0, dtype="<i2")
                )
                dur_ms = (full_pcm.size / max(1, sample_rate)) * 1000.0
                wav_b64 = base64.b64encode(pcm16_to_wav_bytes(full_pcm, sample_rate)).decode(
                    "ascii"
                )
                raw_items.append(
                    {
                        "checkpoint_id": cid,
                        "label": meta["label"],
                        "wav_base64": wav_b64,
                        "sample_rate": sample_rate,
                        "duration_ms": round(dur_ms, 1),
                        "ttfa_ms": round(ttfa_ms, 2),
                        "rtf": round(total_ms / max(1.0, dur_ms), 4),
                    }
                )
        finally:
            self.set_tts_engine(prev_engine)
            self.tts.load_checkpoint(prev_path)
            self.active_ids["tts"] = prev_id

        order = list(range(len(raw_items)))
        if blind and len(order) > 1:
            random.shuffle(order)

        candidates: list[dict[str, Any]] = []
        reveal_map: dict[str, dict[str, str]] = {}
        for pos, orig_idx in enumerate(order):
            item = raw_items[orig_idx]
            slot_id = chr(ord("A") + pos)
            reveal_map[slot_id] = {
                "checkpoint_id": item["checkpoint_id"],
                "label": item["label"],
            }
            candidates.append(
                {
                    "slot_id": slot_id,
                    "display_label": f"盲测模型 {slot_id}" if blind else item["label"],
                    "checkpoint_id": None if blind else item["checkpoint_id"],
                    "wav_base64": item["wav_base64"],
                    "sample_rate": item["sample_rate"],
                    "duration_ms": item["duration_ms"],
                    "ttfa_ms": item["ttfa_ms"],
                    "rtf": item["rtf"],
                }
            )

        return {
            "text": cleaned,
            "blind": blind,
            "candidates": candidates,
            "reveal_map": reveal_map,
        }

    def list_experiments(self) -> dict[str, Any]:
        """Reads markdown reports and JSON manifests from docs/experiments/."""
        if not self.experiments_dir.is_dir():
            return {"experiments_dir": str(self.experiments_dir), "reports": []}

        md_files = sorted(self.experiments_dir.glob("*.md"))
        json_files = sorted(self.experiments_dir.glob("*.json"))
        manifests_by_key: dict[str, tuple[str, dict[str, Any]]] = {}
        for jf in json_files:
            try:
                parsed = json.loads(jf.read_text(encoding="utf-8"))
                manifests_by_key[jf.stem] = (jf.name, parsed)
            except Exception:
                continue

        reports: list[dict[str, Any]] = []
        used_manifests: set[str] = set()
        for mf in md_files:
            text = mf.read_text(encoding="utf-8")
            first_line = next(
                (ln.lstrip("# ").strip() for ln in text.splitlines() if ln.strip()), mf.stem
            )
            # Match closest manifest by shared tokens (e.g., turn_adapter, perf, 4model/real).
            matched_name: str | None = None
            matched_data: dict[str, Any] | None = None
            stem_tokens = set(mf.stem.lower().split("_"))
            best_overlap = 0
            for jstem, (jname, jdata) in manifests_by_key.items():
                overlap = len(stem_tokens & set(jstem.lower().split("_")))
                if overlap > best_overlap:
                    best_overlap = overlap
                    matched_name = jname
                    matched_data = jdata
            if matched_name is not None:
                used_manifests.add(matched_name)
            reports.append(
                {
                    "id": mf.stem,
                    "title": first_line,
                    "markdown_file": mf.name,
                    "manifest_file": matched_name,
                    "markdown": text,
                    "manifest": matched_data,
                }
            )

        for jstem, (jname, jdata) in manifests_by_key.items():
            if jname not in used_manifests:
                reports.append(
                    {
                        "id": jstem,
                        "title": str(jdata.get("experiment_id", jstem)),
                        "markdown_file": None,
                        "manifest_file": jname,
                        "markdown": "",
                        "manifest": jdata,
                    }
                )

        return {"experiments_dir": self.experiments_dir.as_posix(), "reports": reports}

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
        tts_backend = (
            "fast-harmonic"
            if self.tts.force_fast_mode
            else "qwen3-tts-cuda"
            if self.tts._model is not None
            else "offline-harmonic"
        )
        return {
            "vad": {"backend": self.vad.backend_name},
            "asr": {
                "model_id": self.asr.model_id,
                "backend": self.asr.active_backend,
                "checkpoint": self.asr.checkpoint_path,
                "active_id": self.active_ids.get("asr"),
            },
            "turn_adapter": {
                "version": self.turn_adapter.adapter_version,
                "checkpoint": self.turn_adapter.checkpoint_path,
                "active_id": self.active_ids.get("turn_adapter"),
                "strategy": self.turn_strategy,
                "params": self.turn_adapter.model.parameter_count(),
            },
            "tts": {
                "model_id": self.tts.model_id,
                "backend": tts_backend,
                "engine": self.tts_engine,
                "sample_rate": self.tts.sample_rate,
                "checkpoint": self.tts.checkpoint_path,
                "active_id": self.active_ids.get("tts"),
            },
        }
