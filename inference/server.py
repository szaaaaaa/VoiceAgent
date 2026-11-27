"""
Local FastAPI + WebSocket speech inference & experiment control server (V2.0).
Binds strictly to 127.0.0.1 by default; supports:
- Controller + Observer multi-client WebSocket broadcast (Pi /voice + Desktop Workbench)
- Dual audio transport modes: `local_device` (WASAPI daemon) & `ws_pcm` (binary PCM stream)
- Streaming VAD / ASR / CausalTurnGRU / Qwen3-TTS inference with instant barge-in cancel
- REST v1 endpoints for device discovery, Checkpoint Registry hot-swap, A/B eval & experiments
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inference.audio_io import (
    HardwarePlaybackCursor,
    LocalAudioDaemon,
    compute_rms_db,
    list_audio_devices,
)
from inference.protocol import (
    EventEnvelope,
    StreamKind,
    decode_audio_frame,
    encode_audio_frame,
    make_event,
)
from inference.runtime_manager import InferenceRuntimeManager, wav_bytes_to_pcm16_16k


class AgentTurnRequest(BaseModel):
    session_id: str
    turn_id: str
    user_text: str
    history: list[dict[str, Any]] = []
    agent_mode: str = "conversation"
    permission_level: str = "ask"


class ApprovalDecisionRequest(BaseModel):
    approval_id: str
    approved: bool
    source: str = "ui"
    reason: str = ""


class ActivateModelRequest(BaseModel):
    task: str
    checkpoint_id: str
    checkpoint_path: str | None = None


class AbTestAsrRequest(BaseModel):
    audio_base64: str
    checkpoint_ids: list[str] = Field(default_factory=list)
    reference_text: str | None = None


class AbTestTtsRequest(BaseModel):
    text: str
    checkpoint_ids: list[str] = Field(default_factory=list)
    blind: bool = True
    speaker_profile: str = "default"


_HIGH_RISK_KEYWORDS = (
    "delete ",
    "remove ",
    "rm ",
    "format ",
    "shutdown",
    "删除",
    "清空",
    "执行命令",
    "写文件",
    "修改文件",
)


def _needs_explicit_approval(req: AgentTurnRequest) -> dict[str, Any] | None:
    lowered = req.user_text.lower()
    if any(k in lowered for k in _HIGH_RISK_KEYWORDS):
        if req.agent_mode == "conversation" or req.permission_level != "trusted":
            return {
                "approvalId": f"appr_{req.turn_id}",
                "toolName": "workspace.action",
                "impactSummary": f"Requested potentially side-effecting action: {req.user_text[:120]}",
                "targetPath": None,
                "command": req.user_text[:120],
            }
    return None


async def _run_pi_turn(req: AgentTurnRequest) -> dict[str, Any]:
    """Bridges standalone sandbox turns to the locally installed `pi` CLI."""
    pending = _needs_explicit_approval(req)
    if pending is not None:
        return {
            "reply": "该操作可能会修改本地文件或系统状态，请在弹窗中点击确认，或者直接说“确认”。",
            "tool_calls": [],
            "pending_approval": pending,
        }

    launcher = os.path.join(os.path.expanduser("~"), ".pi", "agent", "bin", "pi-launcher.js")
    node_exe = shutil.which("node")
    pi_exe = shutil.which("pi")
    if os.getenv("VOICE_AGENT_DISABLE_PI_CLI") == "1" or (
        not (node_exe and os.path.exists(launcher)) and not pi_exe
    ):
        return {
            "reply": f"已收到你的消息：{req.user_text}",
            "tool_calls": [],
            "pending_approval": None,
        }

    tools_flag = "read,grep,find,ls"
    if req.agent_mode == "action" and req.permission_level == "trusted":
        tools_flag = "read,grep,find,ls,write,edit"

    context_lines = [
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in req.history[-8:]
    ]
    prompt = "\n".join(context_lines) if context_lines else req.user_text
    model_override = os.getenv("VOICE_AGENT_PI_MODEL")
    cmd = (
        [node_exe, launcher]
        if (node_exe and os.path.exists(launcher))
        else [pi_exe or "pi"]
    )
    pi_args = [
        "-p",
        "--no-session",
        "--no-mcp",
        "--no-extensions",
        "--no-approve",
        "--thinking",
        "off",
        "--tools",
        tools_flag,
        *(["--model", model_override] if model_override else []),
        "--append-system-prompt",
        (
            "你是 VoiceAgent，一个简洁自然的中文语音助手。"
            "请始终使用清晰、口语化的中文简短回复，避免输出大段代码或 Markdown 符号，适合直接语音朗读。"
        ),
        prompt,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        *pi_args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=12.0)
    except TimeoutError:
        proc.kill()
        return {
            "reply": f"Pi Agent 响应超时，本地语音已收到：{req.user_text}",
            "tool_calls": [],
            "pending_approval": None,
        }

    reply = stdout_b.decode("utf-8", errors="replace").strip()
    if not reply:
        err = stderr_b.decode("utf-8", errors="replace").strip()
        reply = f"Pi Agent 未返回内容（{err[:120]}）。" if err else "暂时没有收到回复。"
    return {"reply": reply, "tool_calls": [], "pending_approval": None}


def _cursor_event(session_id: str, cursor: HardwarePlaybackCursor) -> EventEnvelope:
    return make_event(
        "playback.cursor",
        session_id,
        {
            "mode": cursor.mode,
            "played_char_index": cursor.played_char_index,
            "played_duration_ms": cursor.played_duration_ms,
            "generation_id": cursor.generation_id or None,
            "turn_id": cursor.turn_id or None,
        },
        turn_id=cursor.turn_id or None,
        generation_id=cursor.generation_id or None,
    )


def create_app(
    manager: InferenceRuntimeManager | None = None,
    *,
    audio_daemon: LocalAudioDaemon | None = None,
) -> FastAPI:
    runtime = manager or InferenceRuntimeManager()
    headless_audio = os.getenv("VOICE_AGENT_HEADLESS_AUDIO") == "1"
    daemon = audio_daemon or LocalAudioDaemon(
        input_sample_rate=16000,
        output_sample_rate=runtime.tts.sample_rate,
        headless=headless_audio,
    )

    app = FastAPI(title="VoiceAgent Local Speech Core & Workbench Server", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "null"],
        allow_origin_regex=r"^(file://.*|http://127\.0\.0\.1:\d+|http://localhost:\d+)$",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.runtime = runtime
    app.state.audio_daemon = daemon

    pending_approvals: dict[str, AgentTurnRequest] = {}
    observers: dict[int, tuple[WebSocket, asyncio.Lock]] = {}
    model_op_lock = asyncio.Lock()

    async def broadcast_to_observers(event: EventEnvelope) -> None:
        if not observers:
            return
        payload_text = event.model_dump_json()
        stale_ids: list[int] = []
        for obs_id, (obs_ws, obs_lock) in list(observers.items()):
            try:
                async with obs_lock:
                    await obs_ws.send_text(payload_text)
            except Exception:
                stale_ids.append(obs_id)
        for obs_id in stale_ids:
            observers.pop(obs_id, None)

    def _gpu_snapshot_event(session_id: str) -> EventEnvelope:
        vram = runtime.vram_telemetry()
        return make_event(
            "telemetry.gpu",
            session_id,
            {
                "device": vram["device"],
                "allocated_mb": vram["allocated_mb"],
                "reserved_mb": vram["reserved_mb"],
                "total_mb": vram["total_mb"],
                "active_models": runtime.models_info(),
            },
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "2.0.0",
            "vram": runtime.vram_telemetry(),
            "models": runtime.models_info(),
            "observers_connected": len(observers),
            "audio_capture_active": daemon.is_capturing,
        }

    @app.get("/models")
    async def models() -> dict[str, Any]:
        return runtime.models_info()

    @app.get("/v1/devices")
    async def v1_devices() -> dict[str, Any]:
        return list_audio_devices()

    @app.get("/v1/models/registry")
    async def v1_models_registry() -> dict[str, Any]:
        return runtime.list_registry()

    @app.post("/v1/models/activate")
    async def v1_models_activate(req: ActivateModelRequest) -> dict[str, Any]:
        async with model_op_lock:
            try:
                updated = runtime.activate_checkpoint(
                    req.task, req.checkpoint_id, checkpoint_path=req.checkpoint_path
                )
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        await broadcast_to_observers(_gpu_snapshot_event("system"))
        return updated

    @app.post("/v1/ab-test/asr")
    async def v1_ab_test_asr(req: AbTestAsrRequest) -> dict[str, Any]:
        async with model_op_lock:
            try:
                raw_b64 = req.audio_base64.split(",", 1)[-1].strip()
                raw = base64.b64decode(raw_b64)
                pcm_16k = wav_bytes_to_pcm16_16k(raw)
                return await asyncio.to_thread(
                    runtime.run_ab_asr,
                    pcm_16k,
                    req.checkpoint_ids,
                    reference_text=req.reference_text,
                )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/ab-test/tts")
    async def v1_ab_test_tts(req: AbTestTtsRequest) -> dict[str, Any]:
        async with model_op_lock:
            try:
                return await runtime.run_ab_tts(
                    req.text,
                    req.checkpoint_ids,
                    blind=req.blind,
                    speaker_profile=req.speaker_profile,
                )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/experiments")
    async def v1_experiments() -> dict[str, Any]:
        return runtime.list_experiments()

    @app.post("/api/agent/turn")
    async def api_agent_turn(req: AgentTurnRequest) -> dict[str, Any]:
        res = await _run_pi_turn(req)
        if res.get("pending_approval"):
            appr_id = str(res["pending_approval"]["approvalId"])
            pending_approvals[appr_id] = req
        return res

    @app.post("/api/agent/approval")
    async def api_agent_approval(decision: ApprovalDecisionRequest) -> dict[str, Any]:
        original = pending_approvals.pop(decision.approval_id, None)
        if not decision.approved:
            return {
                "reply": f"Action cancelled ({decision.reason or 'denied by user'}).",
                "tool_calls": [
                    {
                        "id": decision.approval_id,
                        "name": "workspace.action",
                        "status": "denied",
                        "output": decision.reason or "Denied by user",
                    }
                ],
            }
        if original is None:
            return {"reply": "Approval request expired or not found.", "tool_calls": []}
        approved_req = original.model_copy(
            update={"agent_mode": "action", "permission_level": "trusted"}
        )
        return await _run_pi_turn(approved_req)

    @app.websocket("/ws/v1/voice")
    async def voice_ws(ws: WebSocket) -> None:
        await ws.accept()
        ws_id = id(ws)
        session_id = "default"
        role = "controller"
        audio_mode = "ws_pcm"
        output_device: int | str | None = None
        tts_queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=64)
        local_mic_queue: asyncio.Queue[tuple[np.ndarray, float, float]] = asyncio.Queue(maxsize=64)
        tts_seq = 0
        frame_counter = 0
        send_lock = asyncio.Lock()
        loop = asyncio.get_running_loop()

        async def send_json_event(event: EventEnvelope, *, mirror_observers: bool = True) -> None:
            async with send_lock:
                await ws.send_text(event.model_dump_json())
            if mirror_observers and role == "controller":
                await broadcast_to_observers(event)

        async def send_tts_pair(event: EventEnvelope, frame_bytes: bytes) -> None:
            async with send_lock:
                await ws.send_text(event.model_dump_json())
                await ws.send_bytes(frame_bytes)
            if role == "controller":
                await broadcast_to_observers(event)

        async def ingest_mic_pcm(
            pcm16: np.ndarray, pts_ms: float, rms_db: float | None = None
        ) -> None:
            nonlocal frame_counter
            frame_counter += 1
            sess = runtime.get_session(session_id)
            if audio_mode == "local_device":
                cur = daemon.current_cursor()
                sess.assistant_speaking = cur.mode in ("playing", "ducked")

            vad_prob = runtime.vad.score_frame(pcm16)
            vad_event_kind = "ongoing"
            if not sess.in_speech and vad_prob >= runtime.vad_start_threshold:
                sess.in_speech = True
                sess.speech_start_pts_ms = pts_ms
                sess.silence_frames = 0
                vad_event_kind = "start"
                await send_json_event(
                    make_event(
                        "vad.start",
                        session_id,
                        {
                            "vad_prob": round(vad_prob, 4),
                            "speech_start_pts_ms": pts_ms,
                            "preroll_included_ms": 300,
                        },
                    )
                )
            elif sess.in_speech and vad_prob <= runtime.vad_end_threshold:
                sess.silence_frames += 1
                if sess.silence_frames >= 5:
                    sess.in_speech = False
                    vad_event_kind = "end"
                    speech_dur = max(0.0, pts_ms - sess.speech_start_pts_ms)
                    await send_json_event(
                        make_event(
                            "vad.end",
                            session_id,
                            {
                                "vad_prob": round(vad_prob, 4),
                                "silence_start_pts_ms": pts_ms,
                                "speech_duration_ms": round(speech_dur, 1),
                            },
                        )
                    )
                    final_out = runtime.asr.finalize_utterance(session_id)
                    if final_out.partial_text:
                        await send_json_event(
                            make_event(
                                "asr.final",
                                session_id,
                                {
                                    "text": final_out.partial_text,
                                    "audio_start_pts_ms": sess.speech_start_pts_ms,
                                    "audio_end_pts_ms": pts_ms,
                                    "compute_ms": round(final_out.compute_time_ms, 2),
                                    "model_id": runtime.active_ids.get("asr", "asr-base-default"),
                                },
                            )
                        )
            elif sess.in_speech:
                sess.silence_frames = 0

            asr_out = runtime.asr.process_chunk(
                session_id,
                pcm16,
                is_speech=sess.in_speech,
                extract_features=True,
            )
            if asr_out.partial_text:
                await send_json_event(
                    make_event(
                        "asr.partial",
                        session_id,
                        {
                            "text": asr_out.partial_text,
                            "stability": 0.85,
                            "audio_pts_ms": pts_ms,
                            "compute_ms": round(asr_out.compute_time_ms, 2),
                        },
                    )
                )
            if asr_out.encoder_taps is not None:
                pred = runtime.turn_adapter.step(
                    session_id,
                    asr_out.encoder_taps,
                    assistant_speaking=sess.assistant_speaking,
                )
                await send_json_event(
                    make_event(
                        "turn.prediction",
                        session_id,
                        {
                            "p_completion": round(pred.p_completion, 4),
                            "p_floor_taking": round(pred.p_floor_taking, 4),
                            "p_non_floor_feedback": round(pred.p_non_floor_feedback, 4),
                            "assistant_speaking": sess.assistant_speaking,
                            "adapter_version": pred.adapter_version,
                            "step_ms": round(pred.inference_time_ms, 3),
                        },
                    )
                )

            # Broadcast high-frequency oscilloscope telemetry to observers (and local_device controller).
            if observers or audio_mode == "local_device":
                mic_db = (
                    rms_db
                    if rms_db is not None
                    else compute_rms_db(pcm16.astype(np.float32) / 32768.0)
                )
                _, spk_db = daemon.last_levels
                vad_state_evt = make_event(
                    "vad.state",
                    session_id,
                    {
                        "is_speech": sess.in_speech,
                        "vad_prob": round(vad_prob, 4),
                        "pts_ms": pts_ms,
                        "event": vad_event_kind,
                    },
                )
                await broadcast_to_observers(vad_state_evt)
                if frame_counter % 3 == 0:
                    lvl_evt = make_event(
                        "audio.level",
                        session_id,
                        {
                            "mic_rms_db": mic_db,
                            "spk_rms_db": spk_db,
                            "aec_active": daemon.aec_enabled,
                        },
                    )
                    await broadcast_to_observers(lvl_evt)
                    if audio_mode == "local_device":
                        await send_json_event(lvl_evt, mirror_observers=False)

        async def run_tts_request(envelope: EventEnvelope) -> None:
            nonlocal tts_seq
            payload = envelope.payload
            text = str(payload.get("text", ""))
            sentence_index = int(payload.get("sentence_index", 0))
            char_start = int(payload.get("char_start", 0))
            generation_tag = int(payload.get("generation_tag", 0))
            speaker = str(payload.get("speaker_profile", "default"))
            if runtime.is_generation_cancelled(envelope.session_id, generation_tag):
                return
            cancel_event = runtime.register_tts_generation(envelope.session_id, generation_tag)

            try:
                async for chunk in runtime.tts.synthesize_stream(
                    text,
                    base_char_start=char_start,
                    speaker_profile=speaker,
                    cancel_event=cancel_event,
                ):
                    if cancel_event.is_set():
                        return
                    tts_seq += 1
                    meta_evt = make_event(
                        "tts.chunk",
                        envelope.session_id,
                        {
                            "sentence_index": sentence_index,
                            "chunk_index": chunk.chunk_index,
                            "char_start": chunk.char_start,
                            "char_end": chunk.char_end,
                            "duration_ms": round(chunk.duration_ms, 2),
                            "ttfa_ms": round(chunk.ttfa_ms, 2),
                            "generation_tag": generation_tag,
                            "sample_rate": chunk.sample_rate,
                        },
                        turn_id=envelope.turn_id,
                        generation_id=envelope.generation_id,
                    )
                    if audio_mode == "local_device":
                        cursor = daemon.enqueue_tts_pcm24k(
                            chunk.pcm16,
                            turn_id=envelope.turn_id or "",
                            generation_id=envelope.generation_id or "",
                            generation_tag=generation_tag,
                            char_start=chunk.char_start,
                            char_end=chunk.char_end,
                            output_device=output_device,
                        )
                        sess = runtime.get_session(envelope.session_id)
                        sess.assistant_speaking = cursor.mode in ("playing", "ducked")
                        await send_json_event(meta_evt)
                        await send_json_event(_cursor_event(envelope.session_id, cursor))
                    else:
                        frame_bytes = encode_audio_frame(
                            stream_kind=StreamKind.TTS_OUTPUT,
                            seq=tts_seq,
                            pts_ms=0.0,
                            sample_rate=chunk.sample_rate,
                            generation_tag=generation_tag,
                            pcm16=chunk.pcm16,
                        )
                        await send_tts_pair(meta_evt, frame_bytes)
            except Exception as exc:
                err_evt = make_event(
                    "error",
                    envelope.session_id,
                    {
                        "module": "tts",
                        "code": "TTS_SYNTHESIS_ERROR",
                        "message": str(exc),
                        "recoverable": True,
                        "fallback_action": "text_only",
                    },
                    turn_id=envelope.turn_id,
                    generation_id=envelope.generation_id,
                )
                await send_json_event(err_evt)
            finally:
                if runtime.is_generation_cancelled(envelope.session_id, generation_tag):
                    runtime.finish_tts_generation(envelope.session_id, generation_tag)

        async def tts_worker() -> None:
            while True:
                envelope = await tts_queue.get()
                try:
                    await run_tts_request(envelope)
                finally:
                    tts_queue.task_done()

        async def local_mic_worker() -> None:
            while True:
                pcm16, pts_ms, rms_db = await local_mic_queue.get()
                try:
                    await ingest_mic_pcm(pcm16, pts_ms, rms_db)
                finally:
                    local_mic_queue.task_done()

        async def cursor_sync_worker() -> None:
            last_key: tuple[str, int, str] | None = None
            while True:
                await asyncio.sleep(0.08)
                if role != "controller" or audio_mode != "local_device":
                    continue
                cur = daemon.current_cursor()
                key = (cur.mode, cur.played_char_index, cur.generation_id)
                if key == last_key:
                    continue
                last_key = key
                sess = runtime.get_session(session_id)
                sess.assistant_speaking = cur.mode in ("playing", "ducked")
                await send_json_event(_cursor_event(session_id, cur))

        def _on_hardware_mic_frame(pcm16: np.ndarray, pts_ms: float, rms_db: float) -> None:
            def _push() -> None:
                if local_mic_queue.full():
                    try:
                        local_mic_queue.get_nowait()
                        local_mic_queue.task_done()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    local_mic_queue.put_nowait((pcm16, pts_ms, rms_db))
                except asyncio.QueueFull:
                    pass

            loop.call_soon_threadsafe(_push)

        tts_worker_task = asyncio.create_task(tts_worker())
        mic_worker_task = asyncio.create_task(local_mic_worker())
        cursor_worker_task = asyncio.create_task(cursor_sync_worker())

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break

                if msg.get("bytes") is not None:
                    if role == "observer":
                        continue
                    try:
                        frame = decode_audio_frame(msg["bytes"])
                    except ValueError as exc:
                        await send_json_event(
                            make_event(
                                "error",
                                session_id,
                                {
                                    "module": "protocol",
                                    "code": "INVALID_BINARY_FRAME",
                                    "message": str(exc),
                                    "recoverable": True,
                                    "fallback_action": "none",
                                },
                            )
                        )
                        continue

                    if frame.stream_kind != StreamKind.MIC_INPUT:
                        continue
                    await ingest_mic_pcm(frame.pcm16, frame.pts_ms)

                elif msg.get("text") is not None:
                    try:
                        envelope = EventEnvelope.model_validate(json.loads(msg["text"]))
                    except Exception as exc:
                        await send_json_event(
                            make_event(
                                "error",
                                session_id,
                                {
                                    "module": "protocol",
                                    "code": "INVALID_EVENT_JSON",
                                    "message": str(exc),
                                    "recoverable": True,
                                    "fallback_action": "none",
                                },
                            )
                        )
                        continue

                    session_id = envelope.session_id
                    if envelope.type == "session.init":
                        new_role = str(envelope.payload.get("role", "controller"))
                        role = "observer" if new_role == "observer" else "controller"
                        if role == "observer":
                            observers[ws_id] = (ws, send_lock)
                            await send_json_event(
                                _gpu_snapshot_event(session_id), mirror_observers=False
                            )
                            continue

                        observers.pop(ws_id, None)
                        audio_mode = (
                            "local_device"
                            if envelope.payload.get("audio_mode") == "local_device"
                            else "ws_pcm"
                        )
                        output_device = envelope.payload.get("output_device")
                        if envelope.payload.get("turn_strategy"):
                            try:
                                runtime.set_turn_strategy(str(envelope.payload["turn_strategy"]))
                            except ValueError:
                                pass
                        if envelope.payload.get("tts_engine"):
                            try:
                                runtime.set_tts_engine(str(envelope.payload["tts_engine"]))
                            except ValueError:
                                pass
                        runtime.get_session(session_id)
                        await broadcast_to_observers(_gpu_snapshot_event(session_id))

                    elif role == "observer":
                        # Observers are read-only subscribers.
                        continue

                    elif envelope.type == "session.reset":
                        if audio_mode == "local_device":
                            daemon.cancel()
                        runtime.reset_session(session_id)

                    elif envelope.type == "audio.control":
                        action = str(envelope.payload.get("action", ""))
                        if action in ("start_mic", "ptt_down"):
                            daemon.start_capture(
                                _on_hardware_mic_frame,
                                input_device=envelope.payload.get("input_device"),
                            )
                        elif action in ("stop_mic", "ptt_up"):
                            daemon.stop_capture()
                            sess = runtime.get_session(session_id)
                            if sess.in_speech:
                                sess.in_speech = False
                                final_out = runtime.asr.finalize_utterance(session_id)
                                if final_out.partial_text:
                                    await send_json_event(
                                        make_event(
                                            "asr.final",
                                            session_id,
                                            {
                                                "text": final_out.partial_text,
                                                "audio_start_pts_ms": sess.speech_start_pts_ms,
                                                "audio_end_pts_ms": sess.speech_start_pts_ms,
                                                "compute_ms": round(final_out.compute_time_ms, 2),
                                                "model_id": runtime.active_ids.get("asr", "asr-base-default"),
                                            },
                                        )
                                    )
                        elif action == "duck":
                            gain = float(envelope.payload.get("duck_gain", 0.15))
                            cur = daemon.duck(gain=gain)
                            runtime.get_session(session_id).assistant_speaking = cur.mode in (
                                "playing",
                                "ducked",
                            )
                            await send_json_event(_cursor_event(session_id, cur))
                        elif action == "pause":
                            cur = daemon.pause()
                            runtime.get_session(session_id).assistant_speaking = False
                            await send_json_event(_cursor_event(session_id, cur))
                        elif action == "resume":
                            cur = daemon.resume()
                            runtime.get_session(session_id).assistant_speaking = cur.mode in (
                                "playing",
                                "ducked",
                            )
                            await send_json_event(_cursor_event(session_id, cur))

                    elif envelope.type == "playback.ack":
                        sess = runtime.get_session(session_id)
                        pb_state = str(envelope.payload.get("playback_state", "idle"))
                        sess.assistant_speaking = pb_state in ("playing", "ducked")
                        await broadcast_to_observers(envelope)

                    elif envelope.type == "generation.cancel":
                        tag = int(envelope.payload.get("generation_tag", 0))
                        runtime.cancel_generation(session_id, tag)
                        if audio_mode == "local_device":
                            cur = daemon.cancel(tag if tag > 0 else None)
                            runtime.get_session(session_id).assistant_speaking = cur.mode in (
                                "playing",
                                "ducked",
                            )
                            await send_json_event(_cursor_event(session_id, cur))
                        else:
                            await broadcast_to_observers(envelope)

                    elif envelope.type == "model.switch":
                        try:
                            ckpt_id = envelope.payload.get("checkpoint_id")
                            target = str(envelope.payload.get("target", ""))
                            if ckpt_id:
                                runtime.activate_checkpoint(
                                    target,
                                    str(ckpt_id),
                                    checkpoint_path=envelope.payload.get("checkpoint_path"),
                                )
                            else:
                                runtime.switch_model(target, envelope.payload.get("checkpoint_path"))
                            await broadcast_to_observers(_gpu_snapshot_event(session_id))
                        except (ValueError, FileNotFoundError) as exc:
                            await send_json_event(
                                make_event(
                                    "error",
                                    session_id,
                                    {
                                        "module": "runtime",
                                        "code": "MODEL_SWITCH_ERROR",
                                        "message": str(exc),
                                        "recoverable": True,
                                        "fallback_action": "none",
                                    },
                                )
                            )

                    elif envelope.type == "tts.synthesize":
                        if not tts_queue.full():
                            tts_queue.put_nowait(envelope)

        except WebSocketDisconnect:
            pass
        finally:
            observers.pop(ws_id, None)
            tts_worker_task.cancel()
            mic_worker_task.cancel()
            cursor_worker_task.cancel()
            if role == "controller":
                if audio_mode == "local_device":
                    daemon.stop_capture()
                    daemon.cancel()
                runtime.reset_session(session_id)

    # Serve built desktop renderer directly on 127.0.0.1 so Electron loads via secure same-origin HTTP.
    renderer_dist = Path(__file__).resolve().parent.parent / "apps" / "desktop" / "dist" / "renderer"
    if renderer_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(renderer_dist), html=True), name="desktop_ui")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("VOICE_AGENT_WS_HOST", "127.0.0.1")
    port = int(os.getenv("VOICE_AGENT_WS_PORT", "8765"))
    uvicorn.run("inference.server:app", host=host, port=port, reload=False)
