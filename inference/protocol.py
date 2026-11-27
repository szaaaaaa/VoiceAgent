"""Binary PCM frame codec and JSON EventEnvelope mirroring packages/protocol/index.ts."""

from __future__ import annotations

import struct
import time
from enum import IntEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

BINARY_MAGIC = 0x5641  # "VA"
PROTOCOL_VERSION = 1
SUPPORTED_PROTOCOL_VERSIONS = (1, 2)
BINARY_HEADER_BYTES = 24
HEADER_STRUCT = struct.Struct("<HBBIdII")


class StreamKind(IntEnum):
    MIC_INPUT = 1
    TTS_OUTPUT = 2
    AEC_REFERENCE = 3


class BinaryAudioFrame(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    stream_kind: StreamKind
    seq: int
    pts_ms: float
    sample_rate: int
    generation_tag: int
    pcm16: np.ndarray


class EventEnvelope(BaseModel):
    v: int = Field(default=PROTOCOL_VERSION)
    type: str
    event_id: str
    timestamp_ms: int
    session_id: str
    turn_id: str | None = None
    generation_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


_event_seq = 0


def make_event(
    event_type: str,
    session_id: str,
    payload: dict[str, Any],
    *,
    turn_id: str | None = None,
    generation_id: str | None = None,
) -> EventEnvelope:
    global _event_seq
    _event_seq += 1
    now_ms = int(time.time() * 1000)
    return EventEnvelope(
        v=PROTOCOL_VERSION,
        type=event_type,
        event_id=f"py_{now_ms}_{_event_seq}",
        timestamp_ms=now_ms,
        session_id=session_id,
        turn_id=turn_id,
        generation_id=generation_id,
        payload=payload,
    )


def encode_audio_frame(
    *,
    stream_kind: StreamKind,
    seq: int,
    pts_ms: float,
    sample_rate: int,
    generation_tag: int,
    pcm16: np.ndarray,
) -> bytes:
    samples = np.asarray(pcm16, dtype="<i2").reshape(-1)
    header = HEADER_STRUCT.pack(
        BINARY_MAGIC,
        PROTOCOL_VERSION,
        int(stream_kind),
        seq & 0xFFFFFFFF,
        float(pts_ms),
        sample_rate & 0xFFFFFFFF,
        generation_tag & 0xFFFFFFFF,
    )
    return header + samples.tobytes()


def decode_audio_frame(raw: bytes) -> BinaryAudioFrame:
    if len(raw) < BINARY_HEADER_BYTES:
        raise ValueError(f"Binary frame too short: {len(raw)} bytes")
    payload_bytes = len(raw) - BINARY_HEADER_BYTES
    if payload_bytes % 2 != 0:
        raise ValueError(f"PCM16 payload must have even byte count, got {payload_bytes}")

    magic, version, kind_val, seq, pts_ms, sample_rate, generation_tag = HEADER_STRUCT.unpack_from(
        raw, 0
    )
    if magic != BINARY_MAGIC:
        raise ValueError(f"Invalid binary frame magic: 0x{magic:04x}")
    if version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ValueError(f"Unsupported protocol version: {version}")
    try:
        stream_kind = StreamKind(kind_val)
    except ValueError as exc:
        raise ValueError(f"Unknown stream_kind: {kind_val}") from exc

    pcm16 = np.frombuffer(raw, dtype="<i2", offset=BINARY_HEADER_BYTES).copy()
    return BinaryAudioFrame(
        stream_kind=stream_kind,
        seq=seq,
        pts_ms=pts_ms,
        sample_rate=sample_rate,
        generation_tag=generation_tag,
        pcm16=pcm16,
    )
