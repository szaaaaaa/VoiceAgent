"""
Reproducible Causal Turn Adapter training & validation threshold calibration pipeline (PRD §6.2).
Strictly calibrates probability thresholds on the validation split — never touches test for tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from inference.turn_adapter.model import CausalTurnGRU, TurnAdapterConfig
from training.turn_adapter.dataset import (
    TurnSequenceDataset,
    TurnSessionSample,
    generate_synthetic_turn_dataset,
    split_by_session,
)


@dataclass
class TurnTrainResult:
    checkpoint_path: str
    final_train_loss: float
    val_loss: float
    calibrated_thresholds: dict[str, float]
    test_f1_by_head: dict[str, float]
    causality_max_diff: float
    parameter_count: int = 0


@dataclass
class TurnAblationSpec:
    """Configuration unit for GRU architecture & feature-source ablation sweeps."""

    name: str
    feature_source: str = "asr_encoder"  # "asr_encoder" | "acoustic_logmel"
    hidden_dim: int = 128
    num_layers: int = 2
    num_encoder_taps: int = 1


def _binary_f1(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    pred = (y_prob >= threshold).astype(np.int32)
    true = (y_true >= 0.5).astype(np.int32)
    tp = int(np.sum((pred == 1) & (true == 1)))
    fp = int(np.sum((pred == 1) & (true == 0)))
    fn = int(np.sum((pred == 0) & (true == 1)))
    if tp == 0:
        return 0.0
    return float((2 * tp) / (2 * tp + fp + fn))


def calibrate_thresholds_on_val(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Grid-searches per-head decision thresholds strictly on the validation set."""
    names = ("eot_threshold", "interruption_threshold", "backchannel_threshold")
    calibrated: dict[str, float] = {}
    grid = np.linspace(0.25, 0.85, 13)
    for head_idx, name in enumerate(names):
        best_th = 0.5
        best_f1 = -1.0
        for th in grid:
            f1 = _binary_f1(y_true[:, head_idx], y_prob[:, head_idx], float(th))
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(round(th, 3))
        calibrated[name] = best_th
    return calibrated


def verify_streaming_causality(model: CausalTurnGRU, feature_dim: int, seq_len: int = 16) -> float:
    """
    Verifies strict frame-by-frame causality: running chunk-by-chunk with carried GRU state
    must match full-sequence forward pass within float32 numerical tolerance (zero future leakage).
    """
    model.eval()
    device = next(model.parameters()).device
    torch.manual_seed(0)
    feats = torch.randn(1, seq_len, feature_dim, device=device)
    pb = torch.randint(0, 2, (1, seq_len, 1), device=device, dtype=torch.float32)

    with torch.inference_mode():
        full_logits, _ = model(feats, pb, None)
        hidden = None
        step_outputs = []
        for t in range(seq_len):
            step_logit, hidden = model(feats[:, t : t + 1, :], pb[:, t : t + 1, :], hidden)
            step_outputs.append(step_logit)
        stream_logits = torch.cat(step_outputs, dim=1)
        return float((full_logits - stream_logits).abs().max().item())


def _collect_predictions(
    model: CausalTurnGRU, loader: DataLoader[Any], device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    trues: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    with torch.inference_mode():
        for feats, pb, labels in loader:
            feats, pb, labels = feats.to(device), pb.to(device), labels.to(device)
            logits, _ = model(feats, pb)
            losses.append(float(criterion(logits, labels).item()))
            trues.append(labels.reshape(-1, 3).cpu().numpy())
            probs.append(torch.sigmoid(logits).reshape(-1, 3).cpu().numpy())
    return float(np.mean(losses)), np.concatenate(trues, axis=0), np.concatenate(probs, axis=0)


def train_turn_adapter(
    samples: list[TurnSessionSample] | None = None,
    *,
    output_path: str | Path = "models/turn_adapter/causal_gru_v1.pt",
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_encoder_taps: int = 1,
    feature_source: str = "asr_encoder",
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 3e-3,
    seed: int = 42,
    device: str | None = None,
) -> TurnTrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    all_samples = samples if samples is not None else generate_synthetic_turn_dataset(seed=seed)
    train_s, val_s, test_s = split_by_session(all_samples)
    feature_dim = int(train_s[0].encoder_features.shape[-1])

    train_loader = DataLoader(TurnSequenceDataset(train_s), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TurnSequenceDataset(val_s), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TurnSequenceDataset(test_s), batch_size=batch_size, shuffle=False)

    cfg = TurnAdapterConfig(
        encoder_feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_encoder_taps=num_encoder_taps,
        feature_source=feature_source,
    )
    model = CausalTurnGRU(cfg).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    final_train_loss = 0.0
    for _epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for feats, pb, labels in train_loader:
            feats, pb, labels = feats.to(dev), pb.to(dev), labels.to(dev)
            optimizer.zero_grad()
            logits, _ = model(feats, pb)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        final_train_loss = float(np.mean(epoch_losses))

    val_loss, val_true, val_prob = _collect_predictions(model, val_loader, dev)
    thresholds = calibrate_thresholds_on_val(val_true, val_prob)
    model.config.eot_threshold = thresholds["eot_threshold"]
    model.config.interruption_threshold = thresholds["interruption_threshold"]
    model.config.backchannel_threshold = thresholds["backchannel_threshold"]

    _, test_true, test_prob = _collect_predictions(model, test_loader, dev)
    test_f1 = {
        "completion": _binary_f1(test_true[:, 0], test_prob[:, 0], thresholds["eot_threshold"]),
        "floor_taking": _binary_f1(
            test_true[:, 1], test_prob[:, 1], thresholds["interruption_threshold"]
        ),
        "non_floor_feedback": _binary_f1(
            test_true[:, 2], test_prob[:, 2], thresholds["backchannel_threshold"]
        ),
    }

    causality_diff = verify_streaming_causality(model, feature_dim)
    if causality_diff > 1e-4:
        raise RuntimeError(f"Causality audit failed: streaming max diff {causality_diff:.6f}")

    model.save_checkpoint(
        output_path,
        extra_meta={
            "version": Path(output_path).stem,
            "val_loss": val_loss,
            "thresholds": thresholds,
            "test_f1": test_f1,
            "causality_max_diff": causality_diff,
        },
    )
    return TurnTrainResult(
        checkpoint_path=str(output_path),
        final_train_loss=final_train_loss,
        val_loss=val_loss,
        calibrated_thresholds=thresholds,
        test_f1_by_head=test_f1,
        causality_max_diff=causality_diff,
        parameter_count=model.parameter_count(),
    )


def run_turn_ablation_grid(
    specs: list[TurnAblationSpec],
    samples: list[TurnSessionSample],
    output_dir: str | Path,
    *,
    epochs: int = 8,
    device: str | None = None,
) -> dict[str, TurnTrainResult]:
    """Runs controlled GRU depth/width/feature-tap ablations on identical session splits."""
    out_root = Path(output_dir)
    results: dict[str, TurnTrainResult] = {}
    for spec in specs:
        ckpt = out_root / f"{spec.name}.pt"
        results[spec.name] = train_turn_adapter(
            samples,
            output_path=ckpt,
            hidden_dim=spec.hidden_dim,
            num_layers=spec.num_layers,
            num_encoder_taps=spec.num_encoder_taps,
            feature_source=spec.feature_source,
            epochs=epochs,
            device=device,
        )
    return results
