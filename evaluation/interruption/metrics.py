"""Full-duplex interruption & AEC evaluation metrics (PRD §6.2, §9.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InterruptionEvalReport:
    interruption_recall: float
    false_interrupt_rate: float
    backchannel_false_cancel_rate: float
    p50_stop_latency_ms: float
    p95_stop_latency_ms: float


def evaluate_interruption(
    true_events: list[str],  # "interruption" | "backchannel" | "noise"
    cancelled_reply: list[bool],
    *,
    stop_latencies_ms: list[float] | None = None,
) -> InterruptionEvalReport:
    if len(true_events) != len(cancelled_reply) or len(true_events) == 0:
        raise ValueError("true_events and cancelled_reply must be non-empty and equal length")

    int_total = sum(1 for e in true_events if e == "interruption")
    int_hits = sum(
        1 for e, c in zip(true_events, cancelled_reply, strict=True) if e == "interruption" and c
    )

    non_int_total = sum(1 for e in true_events if e != "interruption")
    false_ints = sum(
        1 for e, c in zip(true_events, cancelled_reply, strict=True) if e != "interruption" and c
    )

    bc_total = sum(1 for e in true_events if e == "backchannel")
    bc_cancels = sum(
        1 for e, c in zip(true_events, cancelled_reply, strict=True) if e == "backchannel" and c
    )

    lats = sorted(stop_latencies_ms or [])
    p50 = float(np.percentile(lats, 50)) if lats else 0.0
    p95 = float(np.percentile(lats, 95)) if lats else 0.0

    return InterruptionEvalReport(
        interruption_recall=round(int_hits / int_total if int_total else 1.0, 5),
        false_interrupt_rate=round(false_ints / non_int_total if non_int_total else 0.0, 5),
        backchannel_false_cancel_rate=round(bc_cancels / bc_total if bc_total else 0.0, 5),
        p50_stop_latency_ms=round(p50, 2),
        p95_stop_latency_ms=round(p95, 2),
    )


def compute_erle_db(mic_echo_signal: np.ndarray, aec_residual_signal: np.ndarray) -> float:
    """Computes Echo Return Loss Enhancement (ERLE) in dB: 10 * log10(E[mic^2] / E[residual^2])."""
    mic_f = mic_echo_signal.astype(np.float64)
    res_f = aec_residual_signal.astype(np.float64)
    mic_power = float(np.mean(mic_f * mic_f))
    res_power = float(np.mean(res_f * res_f) + 1e-12)
    if mic_power <= 1e-12:
        return 0.0
    return float(10.0 * np.log10(mic_power / res_power))
