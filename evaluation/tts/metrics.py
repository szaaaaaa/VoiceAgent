"""TTS evaluation metrics: roundtrip CER/WER, speaker embedding cosine similarity, TTFA, and RTF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jiwer import cer, wer


@dataclass
class TtsEvalReport:
    cer: float
    wer: float
    mean_speaker_similarity: float
    mean_ttfa_ms: float
    mean_rtf: float


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def evaluate_tts_samples(
    target_texts: list[str],
    asr_transcriptions: list[str],
    *,
    ref_embeddings: list[np.ndarray] | None = None,
    gen_embeddings: list[np.ndarray] | None = None,
    ttfa_ms_list: list[float] | None = None,
    rtf_list: list[float] | None = None,
) -> TtsEvalReport:
    if len(target_texts) != len(asr_transcriptions) or len(target_texts) == 0:
        raise ValueError("target_texts and asr_transcriptions must be non-empty and equal length")

    refs = [t.strip().lower() for t in target_texts]
    hyps = [t.strip().lower() for t in asr_transcriptions]

    sims: list[float] = []
    if ref_embeddings and gen_embeddings:
        sims = [
            cosine_similarity(r, g)
            for r, g in zip(ref_embeddings, gen_embeddings, strict=True)
        ]

    return TtsEvalReport(
        cer=round(float(cer(refs, hyps)), 5),
        wer=round(float(wer(refs, hyps)), 5),
        mean_speaker_similarity=round(float(np.mean(sims)) if sims else 0.0, 5),
        mean_ttfa_ms=round(float(np.mean(ttfa_ms_list)) if ttfa_ms_list else 0.0, 2),
        mean_rtf=round(float(np.mean(rtf_list)) if rtf_list else 0.0, 5),
    )
