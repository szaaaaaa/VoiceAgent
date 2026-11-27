"""
Local WASAPI / PortAudio audio daemon for CLI Pi `/voice` host-managed audio mode.
Handles 16 kHz mono capture, 24 kHz mono playback queue, Duck/Pause/Resume/Cancel,
sample-accurate hardware PlaybackCursor tracking, and lightweight playback-reference AEC.
Falls back cleanly to a deterministic headless software queue when audio hardware is absent.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class HardwarePlaybackCursor:
    turn_id: str
    generation_id: str
    played_samples: int
    played_duration_ms: float
    played_char_index: int
    mode: str  # "idle" | "playing" | "ducked" | "paused"


@dataclass(slots=True)
class _PlaybackSegment:
    pcm_f32: np.ndarray
    turn_id: str
    generation_id: str
    generation_tag: int
    char_start: int
    char_end: int
    consumed_samples: int = 0


def list_audio_devices() -> dict[str, Any]:
    """Lists host input/output audio devices safely even on headless machines."""
    try:
        import sounddevice as sd  # type: ignore[import-untyped]

        raw_devs = sd.query_devices()
        default_in, default_out = sd.default.device
        inputs: list[dict[str, Any]] = []
        outputs: list[dict[str, Any]] = []
        for idx, dev in enumerate(raw_devs):
            name = str(dev.get("name", f"Device {idx}"))
            if int(dev.get("max_input_channels", 0)) > 0:
                inputs.append(
                    {
                        "index": idx,
                        "name": name,
                        "default": idx == default_in,
                        "sample_rate": int(dev.get("default_samplerate", 16000)),
                    }
                )
            if int(dev.get("max_output_channels", 0)) > 0:
                outputs.append(
                    {
                        "index": idx,
                        "name": name,
                        "default": idx == default_out,
                        "sample_rate": int(dev.get("default_samplerate", 24000)),
                    }
                )
        return {"available": True, "inputs": inputs, "outputs": outputs}
    except Exception as exc:
        return {"available": False, "error": str(exc), "inputs": [], "outputs": []}


def compute_rms_db(pcm_f32: np.ndarray) -> float:
    if pcm_f32.size == 0:
        return -90.0
    rms = float(np.sqrt(np.mean(pcm_f32 * pcm_f32) + 1e-12))
    return round(max(-90.0, 20.0 * math.log10(rms)), 2)


class LocalAudioDaemon:
    """Thread-safe local audio capture & playback controller with sample-accurate cursor."""

    def __init__(
        self,
        *,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        frame_ms: int = 20,
        aec_enabled: bool = True,
        headless: bool = False,
    ) -> None:
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.frame_samples = int(input_sample_rate * frame_ms / 1000)
        self.aec_enabled = aec_enabled
        self.headless = headless

        self._lock = threading.Lock()
        self._queue: deque[_PlaybackSegment] = deque()
        self._mode: str = "idle"
        self._gain: float = 1.0
        self._turn_id: str = ""
        self._generation_id: str = ""
        self._played_samples: int = 0
        self._played_char_index: int = 0
        self._last_spk_rms: float = 0.0
        self._last_spk_rms_db: float = -90.0
        self._last_mic_rms_db: float = -90.0
        self._capture_t0: float = time.perf_counter()

        self._in_stream: Any = None
        self._out_stream: Any = None
        self._on_frame: Callable[[np.ndarray, float, float], None] | None = None
        self._capture_active = False

    @property
    def is_capturing(self) -> bool:
        return self._capture_active

    @property
    def last_levels(self) -> tuple[float, float]:
        with self._lock:
            return self._last_mic_rms_db, self._last_spk_rms_db

    def start_capture(
        self,
        on_frame: Callable[[np.ndarray, float, float], None],
        *,
        input_device: int | str | None = None,
    ) -> bool:
        self.stop_capture()
        self._on_frame = on_frame
        self._capture_t0 = time.perf_counter()
        self._capture_active = True

        if self.headless:
            return True

        try:
            import sounddevice as sd  # type: ignore[import-untyped]

            def _make_cb(capture_sr: int) -> Callable[..., None]:
                def _in_cb(indata: np.ndarray, _frames: int, _time_info: Any, _status: Any) -> None:
                    if not self._capture_active or self._on_frame is None:
                        return
                    mono = np.asarray(indata[:, 0], dtype=np.float32)
                    if capture_sr != self.input_sample_rate and mono.size > 0:
                        target_len = max(
                            1, int(round(mono.size * self.input_sample_rate / capture_sr))
                        )
                        xp = np.linspace(0.0, 1.0, mono.size, dtype=np.float32)
                        x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
                        mono = np.interp(x_new, xp, mono).astype(np.float32)
                    processed = self._apply_aec_attenuation(mono)
                    rms_db = compute_rms_db(processed)
                    with self._lock:
                        self._last_mic_rms_db = rms_db
                    pcm16 = np.clip(processed * 32767.0, -32768, 32767).astype("<i2")
                    pts_ms = (time.perf_counter() - self._capture_t0) * 1000.0
                    self._on_frame(pcm16, pts_ms, rms_db)

                return _in_cb

            try:
                self._in_stream = sd.InputStream(
                    samplerate=self.input_sample_rate,
                    blocksize=self.frame_samples,
                    channels=1,
                    dtype="float32",
                    device=input_device,
                    callback=_make_cb(self.input_sample_rate),
                )
                self._in_stream.start()
            except Exception:
                dev_info = sd.query_devices(input_device, "input")
                native_sr = int(dev_info.get("default_samplerate", 48000))
                native_block = max(128, int(round(native_sr * self.frame_samples / self.input_sample_rate)))
                self._in_stream = sd.InputStream(
                    samplerate=native_sr,
                    blocksize=native_block,
                    channels=1,
                    dtype="float32",
                    device=input_device,
                    callback=_make_cb(native_sr),
                )
                self._in_stream.start()
            return True
        except Exception:
            self._in_stream = None
            return False

    def stop_capture(self) -> None:
        self._capture_active = False
        self._on_frame = None
        stream = self._in_stream
        self._in_stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def ensure_output_stream(self, *, output_device: int | str | None = None) -> bool:
        if self.headless:
            return True
        if self._out_stream is not None:
            return True
        try:
            import sounddevice as sd  # type: ignore[import-untyped]

            def _out_cb(outdata: np.ndarray, frames: int, _time_info: Any, _status: Any) -> None:
                buf = self._pull_output_samples(frames)
                outdata[:, 0] = buf

            self._out_stream = sd.OutputStream(
                samplerate=self.output_sample_rate,
                blocksize=int(self.output_sample_rate * 0.02),
                channels=1,
                dtype="float32",
                device=output_device,
                callback=_out_cb,
            )
            self._out_stream.start()
            return True
        except Exception:
            self._out_stream = None
            return False

    def enqueue_tts_pcm24k(
        self,
        pcm16: np.ndarray,
        *,
        turn_id: str,
        generation_id: str,
        generation_tag: int,
        char_start: int,
        char_end: int,
        output_device: int | str | None = None,
    ) -> HardwarePlaybackCursor:
        self.ensure_output_stream(output_device=output_device)
        arr = np.asarray(pcm16, dtype=np.int16).reshape(-1).astype(np.float32) / 32768.0
        if arr.size == 0:
            return self.current_cursor()
        with self._lock:
            if generation_id != self._generation_id:
                self._turn_id = turn_id
                self._generation_id = generation_id
                self._played_samples = 0
                self._played_char_index = char_start
            self._queue.append(
                _PlaybackSegment(
                    pcm_f32=arr,
                    turn_id=turn_id,
                    generation_id=generation_id,
                    generation_tag=generation_tag,
                    char_start=char_start,
                    char_end=max(char_start, char_end),
                )
            )
            if self._mode == "idle":
                self._mode = "playing"
                self._gain = 1.0
            return self._cursor_locked()

    def duck(self, gain: float = 0.15) -> HardwarePlaybackCursor:
        with self._lock:
            if self._mode in ("playing", "ducked"):
                self._mode = "ducked"
                self._gain = float(np.clip(gain, 0.0, 1.0))
            return self._cursor_locked()

    def pause(self) -> HardwarePlaybackCursor:
        with self._lock:
            if self._mode in ("playing", "ducked"):
                self._mode = "paused"
            return self._cursor_locked()

    def resume(self) -> HardwarePlaybackCursor:
        with self._lock:
            if self._queue:
                self._mode = "playing"
                self._gain = 1.0
            else:
                self._mode = "idle"
            return self._cursor_locked()

    def cancel(self, generation_tag: int | None = None) -> HardwarePlaybackCursor:
        with self._lock:
            if generation_tag is None or generation_tag <= 0:
                self._queue.clear()
            else:
                self._queue = deque(
                    seg for seg in self._queue if seg.generation_tag != generation_tag
                )
            cursor = self._cursor_locked()
            if not self._queue:
                self._mode = "idle"
                self._gain = 1.0
                self._last_spk_rms = 0.0
                self._last_spk_rms_db = -90.0
                cursor.mode = "idle"
            return cursor

    def current_cursor(self) -> HardwarePlaybackCursor:
        with self._lock:
            return self._cursor_locked()

    def drain_software_output(self, num_samples: int) -> np.ndarray:
        """Advances playback queue deterministically in headless tests without hardware."""
        return self._pull_output_samples(num_samples)

    def close(self) -> None:
        self.stop_capture()
        self.cancel()
        out = self._out_stream
        self._out_stream = None
        if out is not None:
            try:
                out.stop()
                out.close()
            except Exception:
                pass

    def _apply_aec_attenuation(self, mic_f32: np.ndarray) -> np.ndarray:
        # shortcut: energy-domain ducking gate using speaker reference RMS; upgrade to WebRTC AEC3
        # when open-air loud speaker playback causes residual echo bleed.
        if not self.aec_enabled:
            return mic_f32
        with self._lock:
            spk_rms = self._last_spk_rms
            active = self._mode in ("playing", "ducked")
        if not active or spk_rms < 1e-4:
            return mic_f32
        mic_rms = float(np.sqrt(np.mean(mic_f32 * mic_f32) + 1e-12))
        if mic_rms < spk_rms * 0.45:
            return mic_f32 * 0.15
        return mic_f32

    def _pull_output_samples(self, frames: int) -> np.ndarray:
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            if self._mode not in ("playing", "ducked") or not self._queue:
                self._last_spk_rms = 0.0
                self._last_spk_rms_db = -90.0
                if not self._queue and self._mode != "paused":
                    self._mode = "idle"
                return out

            written = 0
            while written < frames and self._queue:
                seg = self._queue[0]
                remaining = seg.pcm_f32.size - seg.consumed_samples
                take = min(frames - written, remaining)
                chunk = seg.pcm_f32[seg.consumed_samples : seg.consumed_samples + take]
                out[written : written + take] = chunk * self._gain
                seg.consumed_samples += take
                written += take
                self._played_samples += take

                ratio = seg.consumed_samples / max(1, seg.pcm_f32.size)
                seg_char = seg.char_start + int(ratio * (seg.char_end - seg.char_start))
                self._played_char_index = max(self._played_char_index, seg_char)

                if seg.consumed_samples >= seg.pcm_f32.size:
                    self._played_char_index = max(self._played_char_index, seg.char_end)
                    self._queue.popleft()

            if not self._queue and self._mode in ("playing", "ducked"):
                self._mode = "idle"
                self._gain = 1.0

            self._last_spk_rms = float(np.sqrt(np.mean(out * out) + 1e-12))
            self._last_spk_rms_db = compute_rms_db(out)
            return out

    def _cursor_locked(self) -> HardwarePlaybackCursor:
        dur_ms = (self._played_samples / max(1, self.output_sample_rate)) * 1000.0
        return HardwarePlaybackCursor(
            turn_id=self._turn_id,
            generation_id=self._generation_id,
            played_samples=self._played_samples,
            played_duration_ms=round(dur_ms, 2),
            played_char_index=self._played_char_index,
            mode=self._mode,
        )
