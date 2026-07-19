"""CUDA training and evaluation loops for the star heatmap detector."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from starmap_fusion.inference.heatmap import decode_heatmap
from starmap_fusion.training.losses import center_focal_loss


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Serializable hyperparameters for one training run."""

    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    confidence_threshold: float = 0.3
    match_radius: float = 2.0
    gradient_clip_norm: float = 5.0
    seed: int = 42


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Loss and point-matching metrics accumulated over a dataset."""

    loss: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _match_points(predictions: Tensor, targets: Tensor, radius: float) -> tuple[int, int, int]:
    """Greedily match predicted centers to targets within a pixel radius."""

    if predictions.shape[0] == 0:
        return 0, 0, int(targets.shape[0])
    if targets.shape[0] == 0:
        return 0, int(predictions.shape[0]), 0

    prediction_coordinates = predictions[:, :2].to(dtype=torch.float32)
    target_coordinates = targets.to(dtype=torch.float32)
    distances = torch.cdist(prediction_coordinates, target_coordinates)
    matched_predictions: set[int] = set()
    matched_targets: set[int] = set()
    flat_indices = distances.flatten().argsort()
    target_count = targets.shape[0]
    for flat_index in flat_indices.tolist():
        prediction_index = flat_index // target_count
        target_index = flat_index % target_count
        if distances[prediction_index, target_index].item() > radius:
            break
        if prediction_index in matched_predictions or target_index in matched_targets:
            continue
        matched_predictions.add(prediction_index)
        matched_targets.add(target_index)

    true_positives = len(matched_predictions)
    false_positives = predictions.shape[0] - true_positives
    false_negatives = targets.shape[0] - true_positives
    return true_positives, false_positives, false_negatives


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader[dict[str, Any]],
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    """Train for one epoch with CUDA automatic mixed precision."""

    model.train()
    total_loss = 0.0
    sample_count = 0
    progress = tqdm(data_loader, desc="train", leave=False)
    for batch in progress:
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["heatmaps"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = center_focal_loss(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(1, sample_count)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader[dict[str, Any]],
    device: torch.device,
    confidence_threshold: float,
    match_radius: float,
) -> EvaluationMetrics:
    """Evaluate heatmap loss and center precision/recall on one split."""

    model.eval()
    total_loss = 0.0
    sample_count = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for batch in tqdm(data_loader, desc="validation", leave=False):
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["heatmaps"].to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = center_focal_loss(logits, targets)
        decoded_batch = decode_heatmap(logits, confidence_threshold=confidence_threshold)
        for predictions, centers in zip(decoded_batch, batch["centers"], strict=True):
            matched = _match_points(predictions.cpu(), centers, match_radius)
            true_positives += matched[0]
            false_positives += matched[1]
            false_negatives += matched[2]

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return EvaluationMetrics(
        loss=total_loss / max(1, sample_count),
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    config: TrainingConfig,
    metrics: EvaluationMetrics,
) -> None:
    """Save a resumable model checkpoint."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "metrics": asdict(metrics),
        },
        output_path,
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader[dict[str, Any]],
    validation_loader: DataLoader[dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    config: TrainingConfig,
) -> list[dict[str, float]]:
    """Train, validate, checkpoint, and return the epoch history."""

    if device.type != "cuda":
        raise RuntimeError("Training requires CUDA. Select a GPU runtime in Colab.")
    seed_everything(config.seed)
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler("cuda", enabled=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_f1 = -1.0

    for epoch_index in range(config.epochs):
        epoch = epoch_index + 1
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        metrics = evaluate(
            model=model,
            data_loader=validation_loader,
            device=device,
            confidence_threshold=config.confidence_threshold,
            match_radius=config.match_radius,
        )
        scheduler.step()
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": metrics.loss,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        save_checkpoint(
            output_dir / "last.pt", model, optimizer, epoch, config, metrics
        )
        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, epoch, config, metrics
            )
        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} "
            f"train_loss={train_loss:.4f} val_loss={metrics.loss:.4f} "
            f"precision={metrics.precision:.4f} recall={metrics.recall:.4f} "
            f"f1={metrics.f1:.4f}"
        )
    return history
