"""Stage 1 tests: LocalAudioDaemon, Controller+Observer WS broadcast, REST registry & A/B APIs."""

from __future__ import annotations

import base64
import json

import numpy as np
from starlette.testclient import TestClient

from inference.audio_io import LocalAudioDaemon
from inference.protocol import StreamKind, encode_audio_frame, make_event
from inference.runtime_manager import pcm16_to_wav_bytes
from inference.server import create_app


def test_local_audio_daemon_cursor_duck_pause_resume_cancel_and_aec() -> None:
    daemon = LocalAudioDaemon(headless=True, aec_enabled=True)
    pcm = (np.sin(np.linspace(0, 80, 24000, dtype=np.float32)) * 12000).astype("<i2")

    c0 = daemon.enqueue_tts_pcm24k(
        pcm,
        turn_id="turn_01",
        generation_id="gen_01",
        generation_tag=1,
        char_start=0,
        char_end=20,
    )
    assert c0.mode == "playing"
    assert c0.played_char_index == 0

    # Play half of the 1.0s clip -> cursor should be at ~10 chars & 500ms.
    out_half = daemon.drain_software_output(12000)
    assert out_half.size == 12000
    c_mid = daemon.current_cursor()
    assert c_mid.played_char_index == 10
    assert 490.0 <= c_mid.played_duration_ms <= 510.0

    # Low-energy mic frame during active speaker playback gets attenuated by AEC gate.
    weak_echo = np.full(320, 0.02, dtype=np.float32)
    attenuated = daemon._apply_aec_attenuation(weak_echo)
    assert float(np.max(np.abs(attenuated))) < float(np.max(np.abs(weak_echo)))

    # Duck -> gain reduced; Pause -> cursor stops advancing; Resume -> plays again.
    c_duck = daemon.duck(gain=0.15)
    assert c_duck.mode == "ducked"
    c_pause = daemon.pause()
    assert c_pause.mode == "paused"
    paused_buf = daemon.drain_software_output(2400)
    assert float(np.max(np.abs(paused_buf))) == 0.0
    assert daemon.current_cursor().played_char_index == c_pause.played_char_index

    assert daemon.resume().mode == "playing"
    c_cancel = daemon.cancel(generation_tag=1)
    assert c_cancel.mode == "idle"
    assert c_cancel.played_char_index == c_pause.played_char_index
    daemon.close()


def test_controller_observer_broadcast_and_local_device_mode() -> None:
    daemon = LocalAudioDaemon(headless=True)
    app = create_app(audio_daemon=daemon)
    client = TestClient(app)

    with client.websocket_connect("/ws/v1/voice") as obs_ws:
        obs_ws.send_text(
            make_event(
                "session.init", "obs_1", {"sample_rate": 16000, "role": "observer"}
            ).model_dump_json()
        )
        # Observer receives initial telemetry.gpu snapshot immediately.
        init_gpu = json.loads(obs_ws.receive_text())
        assert init_gpu["type"] == "telemetry.gpu"
        assert "active_models" in init_gpu["payload"]

        with client.websocket_connect("/ws/v1/voice") as ctrl_ws:
            ctrl_ws.send_text(
                make_event(
                    "session.init",
                    "pi_sess_1",
                    {
                        "sample_rate": 16000,
                        "role": "controller",
                        "audio_mode": "local_device",
                        "tts_engine": "fast",
                    },
                ).model_dump_json()
            )
            # Observer receives updated GPU snapshot on controller init.
            obs_gpu2 = json.loads(obs_ws.receive_text())
            assert obs_gpu2["type"] == "telemetry.gpu"

            # Synthesize TTS in local_device mode -> daemon plays & sends tts.chunk + playback.cursor.
            ctrl_ws.send_text(
                make_event(
                    "tts.synthesize",
                    "pi_sess_1",
                    {
                        "sentence_index": 0,
                        "text": "你好，这是本地语音内核测试。",
                        "char_start": 0,
                        "char_end": 14,
                        "generation_tag": 42,
                        "speaker_profile": "default",
                    },
                    turn_id="turn_0001",
                    generation_id="gen_42",
                ).model_dump_json()
            )

            ctrl_types: list[str] = []
            for _ in range(4):
                ev = json.loads(ctrl_ws.receive_text())
                ctrl_types.append(ev["type"])
                if "playback.cursor" in ctrl_types and "tts.chunk" in ctrl_types:
                    break
            assert "tts.chunk" in ctrl_types
            assert "playback.cursor" in ctrl_types

            # Observer also received the mirrored tts.chunk & playback.cursor events.
            obs_types: list[str] = []
            for _ in range(6):
                ev = json.loads(obs_ws.receive_text())
                obs_types.append(ev["type"])
                if "playback.cursor" in obs_types and "tts.chunk" in obs_types:
                    break
            assert "tts.chunk" in obs_types
            assert "playback.cursor" in obs_types

            # Advance playback partially and issue barge-in generation.cancel.
            daemon.drain_software_output(4800)
            ctrl_ws.send_text(
                make_event(
                    "generation.cancel",
                    "pi_sess_1",
                    {
                        "target_generation_id": "gen_42",
                        "generation_tag": 42,
                        "reason": "user_interruption",
                        "audible_cursor_chars": 0,
                    },
                ).model_dump_json()
            )
            cancel_cursor = None
            for _ in range(20):
                ev = json.loads(ctrl_ws.receive_text())
                if ev["type"] == "playback.cursor" and ev["payload"]["mode"] == "idle":
                    cancel_cursor = ev["payload"]
                    break
            assert cancel_cursor is not None
            assert cancel_cursor["played_duration_ms"] >= 190.0

            # Binary PCM input also mirrors turn.prediction + vad.state to observer.
            voiced = (np.sin(np.linspace(0, 20, 320, dtype=np.float32)) * 14000).astype("<i2")
            ctrl_ws.send_bytes(
                encode_audio_frame(
                    stream_kind=StreamKind.MIC_INPUT,
                    seq=1,
                    pts_ms=20.0,
                    sample_rate=16000,
                    generation_tag=0,
                    pcm16=voiced,
                )
            )
            obs_seen: set[str] = set()
            for _ in range(25):
                ev = json.loads(obs_ws.receive_text())
                obs_seen.add(ev["type"])
                if "turn.prediction" in obs_seen and "vad.state" in obs_seen:
                    break
            assert "turn.prediction" in obs_seen
            assert "vad.state" in obs_seen


def test_rest_registry_hot_swap_ab_tests_and_experiments() -> None:
    app = create_app(audio_daemon=LocalAudioDaemon(headless=True))
    client = TestClient(app)

    # 1. Devices & Registry listing.
    dev_res = client.get("/v1/devices")
    assert dev_res.status_code == 200
    assert "inputs" in dev_res.json()

    reg_res = client.get("/v1/models/registry")
    assert reg_res.status_code == 200
    reg = reg_res.json()
    assert len(reg["checkpoints"]) >= 4
    assert "active_ids" in reg

    # 2. Hot-swap turn strategy & TTS engine + reject invalid task cleanly.
    act1 = client.post(
        "/v1/models/activate",
        json={"task": "turn_strategy", "checkpoint_id": "smart_turn"},
    )
    assert act1.status_code == 200
    assert act1.json()["turn_strategy"] == "smart_turn"

    act_bad = client.post(
        "/v1/models/activate",
        json={"task": "asr", "checkpoint_id": "does-not-exist"},
    )
    assert act_bad.status_code == 400

    # Reject non-checkpoint file suffix & missing file without corrupting active state.
    act_bad_path = client.post(
        "/v1/models/activate",
        json={"task": "asr", "checkpoint_id": "custom", "checkpoint_path": "README.md"},
    )
    assert act_bad_path.status_code == 400
    assert client.get("/v1/models/registry").json()["active_ids"]["asr"] == "asr-base-default"

    # 3. ASR A/B evaluation endpoint with data-URI WAV base64 + WER/CER computation.
    pcm = (np.sin(np.linspace(0, 60, 16000, dtype=np.float32)) * 12000).astype("<i2")
    wav_b64 = "data:audio/wav;base64," + base64.b64encode(pcm16_to_wav_bytes(pcm, 16000)).decode(
        "ascii"
    )
    asr_ab = client.post(
        "/v1/ab-test/asr",
        json={
            "audio_base64": wav_b64,
            "checkpoint_ids": ["asr-base-default"],
            "reference_text": "hello world",
        },
    )
    assert asr_ab.status_code == 200
    asr_body = asr_ab.json()
    assert len(asr_body["candidates"]) == 1
    assert asr_body["candidates"][0]["wer"] is not None

    # 4. TTS Blind A/B evaluation endpoint -> anonymized slots + reveal_map + valid WAV.
    tts_ab = client.post(
        "/v1/ab-test/tts",
        json={
            "text": "测试双盲音色对比接口。",
            "checkpoint_ids": ["tts-qwen3-default", "tts-fast-harmonic"],
            "blind": True,
        },
    )
    assert tts_ab.status_code == 200
    tts_body = tts_ab.json()
    assert len(tts_body["candidates"]) == 2
    assert all(c["checkpoint_id"] is None for c in tts_body["candidates"])
    assert set(tts_body["reveal_map"].keys()) == {"A", "B"}
    wav_decoded = base64.b64decode(tts_body["candidates"][0]["wav_base64"])
    assert wav_decoded[:4] == b"RIFF"

    # 5. Experiments listing reads docs/experiments/.
    exp_res = client.get("/v1/experiments")
    assert exp_res.status_code == 200
    exp_body = exp_res.json()
    assert len(exp_body["reports"]) >= 3
