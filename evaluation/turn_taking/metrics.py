"""Turn-taking evaluation: EOT Recall, Precision, F1, False Cutoff Rate, Decision Latency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TurnTakingEvalReport:
    eot_recall: float
    eot_precision: float
    eot_f1: float
    false_cutoff_rate: float
    mean_decision_latency_ms: float


def evaluate_turn_taking(
    true_eot: list[bool],
    pred_eot: list[bool],
    *,
    is_mid_pause: list[bool] | None = None,
    decision_latencies_ms: list[float] | None = None,
) -> TurnTakingEvalReport:
    if len(true_eot) != len(pred_eot) or len(true_eot) == 0:
        raise ValueError("true_eot and pred_eot must be non-empty and equal length")

    t = np.asarray(true_eot, dtype=bool)
    p = np.asarray(pred_eot, dtype=bool)
    tp = int(np.sum(t & p))
    fp = int(np.sum((~t) & p))
    fn = int(np.sum(t & (~p)))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if is_mid_pause is not None:
        pauses = np.asarray(is_mid_pause, dtype=bool)
        pause_count = int(np.sum(pauses))
        false_cutoff = int(np.sum(pauses & p)) / pause_count if pause_count > 0 else 0.0
    else:
        neg_count = int(np.sum(~t))
        false_cutoff = fp / neg_count if neg_count > 0 else 0.0

    mean_lat = float(np.mean(decision_latencies_ms)) if decision_latencies_ms else 0.0
    return TurnTakingEvalReport(
        eot_recall=round(recall, 5),
        eot_precision=round(precision, 5),
        eot_f1=round(f1, 5),
        false_cutoff_rate=round(false_cutoff, 5),
        mean_decision_latency_ms=round(mean_lat, 2),
    )
