"""
Local FastAPI + WebSocket speech inference server (PRD §7.3, §8.3).
Binds strictly to 127.0.0.1 by default; supports full-duplex audio ingestion,
streaming VAD/ASR/Turn Adapter inference, and cancellable streaming TTS synthesis.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference.protocol import (
    EventEnvelope,
    StreamKind,
    decode_audio_frame,
    encode_audio_frame,
    make_event,
)
from inference.runtime_manager import InferenceRuntimeManager


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
    """Bridges live GUI turns to the locally installed `pi` CLI with low-latency --thinking off."""
    pending = _needs_explicit_approval(req)
    if pending is not None:
        return {
            "reply": "This action may modify local files or system state. Please confirm in the approval dialog or say confirm.",
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
            "reply": f"VoiceAgent received: {req.user_text}",
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
            "You are VoiceAgent, a concise spoken voice assistant. "
            "Respond in clear natural sentences suitable for speech."
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
            "reply": (
                f"Pi Agent timed out. Local voice pipeline received: {req.user_text}"
            ),
            "tool_calls": [],
            "pending_approval": None,
        }

    reply = stdout_b.decode("utf-8", errors="replace").strip()
    if not reply:
        err = stderr_b.decode("utf-8", errors="replace").strip()
        reply = f"Pi Agent returned no text ({err[:160]})." if err else "I did not receive a reply."
    return {"reply": reply, "tool_calls": [], "pending_approval": None}


def create_app(manager: InferenceRuntimeManager | None = None) -> FastAPI:
    runtime = manager or InferenceRuntimeManager()
    app = FastAPI(title="VoiceAgent Local Inference Server", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "null"],
        allow_origin_regex=r"^(file://.*|http://127\.0\.0\.1:\d+|http://localhost:\d+)$",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.runtime = runtime
    pending_approvals: dict[str, AgentTurnRequest] = {}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "vram": runtime.vram_telemetry(),
            "models": runtime.models_info(),
        }

    @app.get("/models")
    async def models() -> dict[str, Any]:
        return runtime.models_info()

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
        # Re-dispatch approved request in trusted action scope.
        approved_req = original.model_copy(
            update={"agent_mode": "action", "permission_level": "trusted"}
        )
        return await _run_pi_turn(approved_req)

    @app.websocket("/ws/v1/voice")
    async def voice_ws(ws: WebSocket) -> None:
        await ws.accept()
        session_id = "default"
        tts_queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=64)
        tts_seq = 0
        send_lock = asyncio.Lock()

        async def send_json_event(event: EventEnvelope) -> None:
            async with send_lock:
                await ws.send_text(event.model_dump_json())

        async def send_tts_pair(event: EventEnvelope, frame_bytes: bytes) -> None:
            # Send metadata JSON and binary PCM atomically so chunks never interleave.
            async with send_lock:
                await ws.send_text(event.model_dump_json())
                await ws.send_bytes(frame_bytes)

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

        tts_worker_task = asyncio.create_task(tts_worker())

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break

                if msg.get("bytes") is not None:
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

                    sess = runtime.get_session(session_id)
                    vad_prob = runtime.vad.score_frame(frame.pcm16)
                    if not sess.in_speech and vad_prob >= runtime.vad_start_threshold:
                        sess.in_speech = True
                        sess.speech_start_pts_ms = frame.pts_ms
                        sess.silence_frames = 0
                        await send_json_event(
                            make_event(
                                "vad.start",
                                session_id,
                                {
                                    "vad_prob": round(vad_prob, 4),
                                    "speech_start_pts_ms": frame.pts_ms,
                                    "preroll_included_ms": 300,
                                },
                            )
                        )
                    elif sess.in_speech and vad_prob <= runtime.vad_end_threshold:
                        sess.silence_frames += 1
                        if sess.silence_frames >= 5:
                            sess.in_speech = False
                            speech_dur = max(0.0, frame.pts_ms - sess.speech_start_pts_ms)
                            await send_json_event(
                                make_event(
                                    "vad.end",
                                    session_id,
                                    {
                                        "vad_prob": round(vad_prob, 4),
                                        "silence_start_pts_ms": frame.pts_ms,
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
                                            "audio_end_pts_ms": frame.pts_ms,
                                            "compute_ms": round(final_out.compute_time_ms, 2),
                                        },
                                    )
                                )
                    elif sess.in_speech:
                        sess.silence_frames = 0

                    asr_out = runtime.asr.process_chunk(
                        session_id,
                        frame.pcm16,
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
                                    "audio_pts_ms": frame.pts_ms,
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
                                },
                            )
                        )

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
                        runtime.get_session(session_id)
                    elif envelope.type == "session.reset":
                        runtime.reset_session(session_id)
                    elif envelope.type == "playback.ack":
                        sess = runtime.get_session(session_id)
                        pb_state = str(envelope.payload.get("playback_state", "idle"))
                        sess.assistant_speaking = pb_state in ("playing", "ducked")
                    elif envelope.type == "generation.cancel":
                        tag = int(envelope.payload.get("generation_tag", 0))
                        runtime.cancel_generation(session_id, tag)
                    elif envelope.type == "model.switch":
                        try:
                            runtime.switch_model(
                                str(envelope.payload.get("target", "")),
                                envelope.payload.get("checkpoint_path"),
                            )
                        except ValueError as exc:
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
            tts_worker_task.cancel()
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
