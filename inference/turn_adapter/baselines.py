"""
Unified Python turn-taking baselines (Fixed Silence & Smart Turn) sharing the exact frame-level
evaluation contract with CausalTurnGRU for apples-to-apples benchmarking.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class FrameTurnPredictor(Protocol):
    name: str

    def predict_sequence(
        self,
        encoder_features: np.ndarray,
        playback_state: np.ndarray,
        *,
        frame_ms: float = 80.0,
    ) -> np.ndarray:
        """Returns frame-level probabilities (T, 3): [completion, floor_taking, backchannel]."""
        ...


class FixedSilenceBaseline:
    """Predicts EOT purely once low-energy silence exceeds `silence_ms` after active speech."""

    name = "fixed_silence"

    def __init__(self, silence_ms: float = 700.0, energy_threshold: float = 0.35) -> None:
        self.silence_ms = silence_ms
        self.energy_threshold = energy_threshold

    def predict_sequence(
        self,
        encoder_features: np.ndarray,
        playback_state: np.ndarray,
        *,
        frame_ms: float = 80.0,
    ) -> np.ndarray:
        seq_len = encoder_features.shape[0]
        probs = np.zeros((seq_len, 3), dtype=np.float32)
        energy = np.mean(np.abs(encoder_features[:, :16]), axis=-1)
        seen_speech = False
        silence_acc_ms = 0.0

        for t in range(seq_len):
            in_speech = bool(energy[t] >= self.energy_threshold)
            assistant_on = bool(playback_state[t, 0] >= 0.5)
            if in_speech:
                seen_speech = True
                silence_acc_ms = 0.0
                if assistant_on:
                    # Naive baseline treats any speech during playback as an interruption.
                    probs[t, 1] = 0.85
            elif seen_speech and not assistant_on:
                silence_acc_ms += frame_ms
                if silence_acc_ms >= self.silence_ms:
                    probs[t, 0] = 0.90
        return probs


class SmartTurnBaseline:
    """
    Prosodic/duration heuristic baseline: separates brief backchannels (< backchannel_max_ms)
    from sustained interruptions, and waits longer after high-energy abrupt endings.
    """

    name = "smart_turn"

    def __init__(
        self,
        base_silence_ms: float = 450.0,
        hold_silence_ms: float = 950.0,
        backchannel_max_ms: float = 560.0,
        energy_threshold: float = 0.35,
    ) -> None:
        self.base_silence_ms = base_silence_ms
        self.hold_silence_ms = hold_silence_ms
        self.backchannel_max_ms = backchannel_max_ms
        self.energy_threshold = energy_threshold

    def predict_sequence(
        self,
        encoder_features: np.ndarray,
        playback_state: np.ndarray,
        *,
        frame_ms: float = 80.0,
    ) -> np.ndarray:
        seq_len = encoder_features.shape[0]
        probs = np.zeros((seq_len, 3), dtype=np.float32)
        energy = np.mean(np.abs(encoder_features[:, :32]), axis=-1)
        seen_speech = False
        speech_acc_ms = 0.0
        silence_acc_ms = 0.0
        last_voiced_energy = 0.0

        for t in range(seq_len):
            in_speech = bool(energy[t] >= self.energy_threshold)
            assistant_on = bool(playback_state[t, 0] >= 0.5)

            if in_speech:
                seen_speech = True
                speech_acc_ms += frame_ms
                silence_acc_ms = 0.0
                last_voiced_energy = float(energy[t])
                if assistant_on:
                    if speech_acc_ms > self.backchannel_max_ms:
                        probs[t, 1] = 0.88
                    else:
                        probs[t, 2] = 0.72
            else:
                if assistant_on and 0 < speech_acc_ms <= self.backchannel_max_ms:
                    probs[t, 2] = 0.80
                if seen_speech and not assistant_on:
                    silence_acc_ms += frame_ms
                    # Abrupt high-energy cutoff suggests hesitation/hold -> wait longer.
                    req = self.hold_silence_ms if last_voiced_energy > 1.1 else self.base_silence_ms
                    if silence_acc_ms >= req:
                        probs[t, 0] = 0.86
                speech_acc_ms = 0.0
        return probs
