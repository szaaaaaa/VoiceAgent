"""ASR module evaluation metrics: WER, CER, command accuracy, and RTF (PRD §6.1, §9.1)."""

from __future__ import annotations

from dataclasses import dataclass

from jiwer import cer, wer


@dataclass
class AsrEvalReport:
    wer: float
    cer: float
    command_accuracy: float
    mean_rtf: float
    sample_count: int


def evaluate_asr_predictions(
    references: list[str],
    hypotheses: list[str],
    *,
    command_flags: list[bool] | None = None,
    audio_durations_s: list[float] | None = None,
    compute_times_s: list[float] | None = None,
) -> AsrEvalReport:
    if len(references) != len(hypotheses) or len(references) == 0:
        raise ValueError("references and hypotheses must be non-empty and of equal length")

    norm_refs = [r.strip().lower() for r in references]
    norm_hyps = [h.strip().lower() for h in hypotheses]

    w = float(wer(norm_refs, norm_hyps))
    c = float(cer(norm_refs, norm_hyps))

    if command_flags is not None:
        cmd_pairs = [
            (r, h)
            for r, h, is_cmd in zip(norm_refs, norm_hyps, command_flags, strict=True)
            if is_cmd
        ]
        cmd_acc = sum(1 for r, h in cmd_pairs if r == h) / len(cmd_pairs) if cmd_pairs else 1.0
    else:
        matches = sum(1 for r, h in zip(norm_refs, norm_hyps, strict=True) if r == h)
        cmd_acc = matches / len(norm_refs)

    if audio_durations_s and compute_times_s:
        rtfs = [
            ct / dur
            for dur, ct in zip(audio_durations_s, compute_times_s, strict=True)
            if dur > 0
        ]
        mean_rtf = sum(rtfs) / len(rtfs) if rtfs else 0.0
    else:
        mean_rtf = 0.0

    return AsrEvalReport(
        wer=round(w, 5),
        cer=round(c, 5),
        command_accuracy=round(cmd_acc, 5),
        mean_rtf=round(mean_rtf, 5),
        sample_count=len(references),
    )
