"""Python unit & integration tests for inference, causal training, registry, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from evaluation.asr.metrics import evaluate_asr_predictions
from evaluation.end_to_end.benchmark import SYSTEM_SCENARIOS, ExperimentManifest
from evaluation.interruption.metrics import compute_erle_db, evaluate_interruption
from evaluation.tts.metrics import evaluate_tts_samples
from evaluation.turn_taking.metrics import evaluate_turn_taking
from inference.protocol import StreamKind, decode_audio_frame, encode_audio_frame, make_event
from inference.server import create_app
from inference.turn_adapter.model import CausalTurnGRU, TurnAdapterConfig
from inference.turn_adapter.predictor import TurnAdapterPredictor
from training.asr.finetune import augment_waveform, check_general_wer_regression
from training.registry import CheckpointRecord, CheckpointRegistry
from training.tts.finetune import audit_tts_recording, verify_finetune_beats_zeroshot
from training.turn_adapter.dataset import generate_synthetic_turn_dataset, split_by_session
from training.turn_adapter.train import train_turn_adapter, verify_streaming_causality


def test_binary_protocol_roundtrip_and_validation() -> None:
    pcm = np.array([0, 1000, -1000, 32767, -32768], dtype="<i2")
    raw = encode_audio_frame(
        stream_kind=StreamKind.MIC_INPUT,
        seq=5,
        pts_ms=42.25,
        sample_rate=16000,
        generation_tag=0,
        pcm16=pcm,
    )
    decoded = decode_audio_frame(raw)
    assert decoded.stream_kind == StreamKind.MIC_INPUT
    assert decoded.seq == 5
    assert decoded.sample_rate == 16000
    assert np.array_equal(decoded.pcm16, pcm)

    with pytest.raises(ValueError, match="too short"):
        decode_audio_frame(b"short")
    with pytest.raises(ValueError, match="magic"):
        decode_audio_frame(b"\x00\x00" + raw[2:])


def test_causal_turn_gru_strict_streaming_equivalence_and_training(tmp_path: Path) -> None:
    model = CausalTurnGRU(TurnAdapterConfig(encoder_feature_dim=64, hidden_dim=128, num_layers=2))
    # Voice-Light scale lightweight adapter (~150k-250k params).
    assert 100_000 < model.parameter_count() < 300_000
    assert verify_streaming_causality(model, feature_dim=64, seq_len=20) < 1e-5

    samples = generate_synthetic_turn_dataset(num_sessions=24, seq_len=30, feature_dim=64, seed=7)
    train_s, val_s, test_s = split_by_session(samples)
    train_ids = {s.session_id for s in train_s}
    val_ids = {s.session_id for s in val_s}
    test_ids = {s.session_id for s in test_s}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    ckpt_path = tmp_path / "causal_gru_test.pt"
    result = train_turn_adapter(
        samples,
        output_path=ckpt_path,
        epochs=6,
        batch_size=8,
        device="cpu",
    )
    assert ckpt_path.exists()
    assert result.causality_max_diff < 1e-5
    assert result.test_f1_by_head["completion"] > 0.5

    # Reload into live inference predictor & register into CheckpointRegistry.
    predictor = TurnAdapterPredictor(checkpoint_path=str(ckpt_path), device="cpu")
    reg = CheckpointRegistry(tmp_path / "registry.json")
    reg.register(
        CheckpointRecord(
            checkpoint_id="ckpt_turn_1",
            model_type="turn_adapter",
            path=str(ckpt_path),
            experiment_id="TURN-A4",
            git_commit="test",
            metrics=result.test_f1_by_head,
        ),
        promote=True,
    )
    assert reg.get_active("turn_adapter") == str(ckpt_path)
    assert predictor.adapter_version == "causal_gru_test"


def test_fastapi_server_health_and_websocket_full_duplex() -> None:
    app = create_app()
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert "asr" in body["models"]

    with client.websocket_connect("/ws/v1/voice") as ws:
        ws.send_text(
            make_event("session.init", "py_sess_1", {"sample_rate": 16000}).model_dump_json()
        )
        # Send voiced 20ms frame -> expect vad.start + turn.prediction.
        t = np.arange(320, dtype=np.float32) / 16000.0
        voiced = (np.sin(2 * np.pi * 240 * t) * 14000).astype("<i2")
        ws.send_bytes(
            encode_audio_frame(
                stream_kind=StreamKind.MIC_INPUT,
                seq=1,
                pts_ms=20.0,
                sample_rate=16000,
                generation_tag=0,
                pcm16=voiced,
            )
        )
        ev1 = json.loads(ws.receive_text())
        ev2 = json.loads(ws.receive_text())
        assert {ev1["type"], ev2["type"]} == {"vad.start", "turn.prediction"}

        # Trigger streaming TTS synthesis -> receive JSON tts.chunk + binary 24kHz PCM frame.
        ws.send_text(
            make_event(
                "tts.synthesize",
                "py_sess_1",
                {
                    "sentence_index": 0,
                    "text": "Hello from local TTS.",
                    "char_start": 0,
                    "char_end": 21,
                    "generation_tag": 9,
                    "speaker_profile": "default",
                },
                turn_id="turn_0001",
                generation_id="gen_9",
            ).model_dump_json()
        )
        tts_meta = json.loads(ws.receive_text())
        tts_bin = ws.receive_bytes()
        assert tts_meta["type"] == "tts.chunk"
        assert tts_meta["payload"]["generation_tag"] == 9
        decoded_tts = decode_audio_frame(tts_bin)
        assert decoded_tts.stream_kind == StreamKind.TTS_OUTPUT
        assert decoded_tts.sample_rate == 24000
        assert decoded_tts.generation_tag == 9

        # Cancel generation_tag=10 BEFORE tts.synthesize arrives -> must be skipped cleanly.
        ws.send_text(
            make_event(
                "generation.cancel",
                "py_sess_1",
                {
                    "target_generation_id": "gen_10",
                    "generation_tag": 10,
                    "reason": "user_interruption",
                    "audible_cursor_chars": 0,
                },
            ).model_dump_json()
        )
        ws.send_text(
            make_event(
                "tts.synthesize",
                "py_sess_1",
                {
                    "sentence_index": 0,
                    "text": "This stale sentence must never synthesize.",
                    "char_start": 0,
                    "char_end": 42,
                    "generation_tag": 10,
                    "speaker_profile": "default",
                },
                turn_id="turn_0001",
                generation_id="gen_10",
            ).model_dump_json()
        )
        # Follow immediately with active generation_tag=11 -> next TTS chunk must be tag 11.
        ws.send_text(
            make_event(
                "tts.synthesize",
                "py_sess_1",
                {
                    "sentence_index": 0,
                    "text": "Active sentence.",
                    "char_start": 0,
                    "char_end": 16,
                    "generation_tag": 11,
                    "speaker_profile": "default",
                },
                turn_id="turn_0002",
                generation_id="gen_11",
            ).model_dump_json()
        )
        # Drain remaining chunks from tag 9 until tag 11 arrives; tag 10 must never appear.
        seen_tags: list[int] = []
        for _ in range(30):
            meta = json.loads(ws.receive_text())
            _raw_audio = ws.receive_bytes()
            if meta["type"] == "tts.chunk":
                tag = meta["payload"]["generation_tag"]
                seen_tags.append(tag)
                if tag == 11:
                    break
        assert 10 not in seen_tags
        assert 11 in seen_tags


def test_training_gates_and_evaluation_metrics_suite(tmp_path: Path) -> None:
    # ASR evaluation + general-set WER regression gate.
    asr_report = evaluate_asr_predictions(
        ["open the browser", "check gpu memory"],
        ["open the browser", "check cpu memory"],
        command_flags=[True, True],
        audio_durations_s=[1.5, 1.5],
        compute_times_s=[0.15, 0.15],
    )
    assert asr_report.wer > 0.0
    assert asr_report.mean_rtf == pytest.approx(0.1, abs=1e-4)
    assert check_general_wer_regression(0.05, 0.055)["passed"] is True
    assert check_general_wer_regression(0.05, 0.09)["passed"] is False

    # Waveform augmentation stays clipped inside [-1, 1].
    aug = augment_waveform(np.ones(1600, dtype=np.float32) * 0.5, snr_db=15.0)
    assert float(np.max(np.abs(aug))) <= 1.0

    # Turn-taking & interruption metrics + ERLE.
    tt = evaluate_turn_taking(
        [True, False, True, False],
        [True, False, True, True],
        is_mid_pause=[False, True, False, True],
        decision_latencies_ms=[180.0, 210.0],
    )
    assert tt.eot_recall == 1.0
    assert tt.false_cutoff_rate == 0.5

    inter = evaluate_interruption(
        ["interruption", "backchannel", "noise", "interruption"],
        [True, False, False, True],
        stop_latencies_ms=[220.0, 310.0],
    )
    assert inter.interruption_recall == 1.0
    assert inter.backchannel_false_cancel_rate == 0.0
    assert compute_erle_db(np.ones(1000) * 0.5, np.ones(1000) * 0.05) > 19.0

    # TTS audit + zero-shot comparison gate.
    clean_tone = np.sin(np.linspace(0, 100, 24000 * 2, dtype=np.float32)) * 0.25
    assert audit_tts_recording(clean_tone, 24000).accepted is True
    assert verify_finetune_beats_zeroshot(0.80, 0.85, 0.03, 0.03)["passed"] is True
    assert verify_finetune_beats_zeroshot(0.80, 0.79, 0.03, 0.03)["passed"] is False

    tts_eval = evaluate_tts_samples(
        ["hello world"],
        ["hello world"],
        ref_embeddings=[np.array([1.0, 0.0])],
        gen_embeddings=[np.array([0.9, 0.1])],
        ttfa_ms_list=[240.0],
        rtf_list=[0.25],
    )
    assert tts_eval.cer == 0.0
    assert tts_eval.mean_speaker_similarity > 0.9

    # Experiment manifest preserves negative results cleanly.
    assert len(SYSTEM_SCENARIOS) == 10
    manifest = ExperimentManifest(
        experiment_id="TURN-A4-neg",
        dataset_version="synth-v1",
        split_hash="abc123",
        model_checkpoints={"turn_adapter": None},
        hyperparameters={"lr": 3e-3},
        metrics={"eot_f1": 0.61},
        conclusion="Did not outperform Smart Turn baseline; retained hybrid fallback.",
        beats_baseline=False,
    )
    out = manifest.save(tmp_path / "exp_manifest.json")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["beats_baseline"] is False
    assert "hybrid fallback" in loaded["conclusion"]
