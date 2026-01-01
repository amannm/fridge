from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augment import Augmenter
from .beats_frontend import beats_feature_dim, load_beats
from .config import Config, config_to_dict
from .dataset import WindowedFridgeDataset
from .eval import collect_predictions, compute_metrics, find_best_threshold
from .model import FridgeClassifier


class TrainError(RuntimeError):
    pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise TrainError("CUDA requested but not available")
    if device_str == "mps" and not torch.backends.mps.is_available():
        raise TrainError("MPS requested but not available")
    return torch.device(device_str)


def build_model(cfg: Config, device: torch.device) -> FridgeClassifier:
    beats = load_beats(cfg.model.beats_checkpoint, device)
    feature_dim = beats_feature_dim(beats)
    model = FridgeClassifier(
        backbone=beats,
        feature_dim=feature_dim,
        head_hidden=cfg.model.head_hidden,
        dropout=cfg.model.dropout,
        pooling=cfg.model.pooling,
        fbank_mean=cfg.fbank.mean,
        fbank_std=cfg.fbank.std,
    )
    model.to(device)
    return model


def create_dataloader(
    dataset: WindowedFridgeDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
) -> float:
    model.train()
    if hasattr(model, "backbone"):
        if all(not p.requires_grad for p in model.backbone.parameters()):
            model.backbone.eval()
    total_loss = 0.0
    total_samples = 0
    for waveforms, labels in dataloader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float | None = 0.5,
) -> Dict[str, float | list]:
    y_true, y_prob = collect_predictions(model, dataloader, device)
    if threshold is None:
        threshold = find_best_threshold(y_true, y_prob)
    metrics = compute_metrics(y_true, y_prob, threshold)
    return {
        "auc": metrics.auc,
        "f1": metrics.f1,
        "accuracy": metrics.accuracy,
        "confusion": metrics.confusion,
    }


def _score_metrics(metrics: Dict[str, float | list]) -> float:
    return float(metrics["auc"]) + float(metrics["f1"])


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_score = -float("inf")
        self.counter = 0

    def update(self, score: float) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        return False

    def should_stop(self) -> bool:
        return self.counter >= self.patience


def save_checkpoint(path: Path, model: FridgeClassifier, cfg: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config_to_dict(cfg),
        },
        path,
    )


def load_checkpoint(path: Path, cfg: Config, device: torch.device) -> FridgeClassifier:
    checkpoint = torch.load(path, map_location=device)
    model = build_model(cfg, device)
    if "model_state" not in checkpoint:
        raise TrainError("Checkpoint missing model_state")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_ratio: float,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_fold(
    cfg: Config,
    train_df,
    val_df,
    decoded_dir: str | Path,
    run_dir: Path,
    fold_name: str,
) -> Tuple[Dict[str, float | list], Path]:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    model = build_model(cfg, device)
    model.freeze_backbone()

    augmenter = Augmenter(
        sample_rate=cfg.audio.sample_rate,
        gain_db_min=cfg.augment.gain_db_min,
        gain_db_max=cfg.augment.gain_db_max,
        time_shift=cfg.augment.time_shift,
        eq_tilt=cfg.augment.eq_tilt,
        noise_dir=cfg.augment.noise_dir,
        noise_snr_min=cfg.augment.noise_snr_min,
        noise_snr_max=cfg.augment.noise_snr_max,
    )

    train_dataset = WindowedFridgeDataset(
        train_df,
        decoded_dir=decoded_dir,
        sample_rate=cfg.audio.sample_rate,
        window_sec=cfg.window.train_window_sec,
        value_range=(cfg.audio.value_range_min, cfg.audio.value_range_max),
        augment=augmenter,
    )
    val_dataset = WindowedFridgeDataset(
        val_df,
        decoded_dir=decoded_dir,
        sample_rate=cfg.audio.sample_rate,
        window_sec=cfg.window.train_window_sec,
        value_range=(cfg.audio.value_range_min, cfg.audio.value_range_max),
        augment=None,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.head_trainable_params(),
        lr=cfg.train.lr_head,
        weight_decay=cfg.train.weight_decay,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    early = EarlyStopping(cfg.train.early_stop_patience)
    best_metrics: Dict[str, float | list] = {}
    best_path = run_dir / "checkpoints" / f"{fold_name}_best.pt"

    for epoch in range(1, cfg.train.epochs_max + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, device, threshold=None)
        score = _score_metrics(val_metrics)
        improved = early.update(score)
        print(
            f"[{fold_name}] epoch {epoch} loss={train_loss:.4f} "
            f"auc={val_metrics['auc']:.4f} f1={val_metrics['f1']:.4f} acc={val_metrics['accuracy']:.4f}"
        )
        if improved:
            best_metrics = val_metrics
            save_checkpoint(best_path, model, cfg)
        if early.should_stop():
            break

    if cfg.finetune.enabled:
        model = load_checkpoint(best_path, cfg, device)
        model.unfreeze_last_blocks(cfg.finetune.unfreeze_last_blocks)
        backbone_params = model.backbone_trainable_params()
        head_params = model.head_trainable_params()
        if not backbone_params:
            raise TrainError("No backbone parameters available for fine-tuning")
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.finetune.lr_backbone},
                {"params": head_params, "lr": cfg.finetune.lr_head},
            ],
            weight_decay=cfg.train.weight_decay,
        )
        total_steps = cfg.finetune.epochs * len(train_loader)
        scheduler = _build_scheduler(optimizer, cfg.finetune.warmup_ratio, total_steps)
        early = EarlyStopping(cfg.train.early_stop_patience)

        for epoch in range(1, cfg.finetune.epochs + 1):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scheduler=scheduler,
            )
            val_metrics = evaluate_model(model, val_loader, device, threshold=None)
            score = _score_metrics(val_metrics)
            improved = early.update(score)
            print(
                f"[{fold_name}][ft] epoch {epoch} loss={train_loss:.4f} "
                f"auc={val_metrics['auc']:.4f} f1={val_metrics['f1']:.4f} acc={val_metrics['accuracy']:.4f}"
            )
            if improved:
                best_metrics = val_metrics
                save_checkpoint(best_path, model, cfg)
            if early.should_stop():
                break

    return best_metrics, best_path


def train_final(
    cfg: Config,
    windows_df,
    decoded_dir: str | Path,
    run_dir: Path,
) -> Path:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    model = build_model(cfg, device)
    model.freeze_backbone()

    augmenter = Augmenter(
        sample_rate=cfg.audio.sample_rate,
        gain_db_min=cfg.augment.gain_db_min,
        gain_db_max=cfg.augment.gain_db_max,
        time_shift=cfg.augment.time_shift,
        eq_tilt=cfg.augment.eq_tilt,
        noise_dir=cfg.augment.noise_dir,
        noise_snr_min=cfg.augment.noise_snr_min,
        noise_snr_max=cfg.augment.noise_snr_max,
    )

    dataset = WindowedFridgeDataset(
        windows_df,
        decoded_dir=decoded_dir,
        sample_rate=cfg.audio.sample_rate,
        window_sec=cfg.window.train_window_sec,
        value_range=(cfg.audio.value_range_min, cfg.audio.value_range_max),
        augment=augmenter,
    )
    loader = create_dataloader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.head_trainable_params(),
        lr=cfg.train.lr_head,
        weight_decay=cfg.train.weight_decay,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    best_path = run_dir / "checkpoints" / "final_best.pt"
    early = EarlyStopping(cfg.train.early_stop_patience)

    for epoch in range(1, cfg.train.epochs_max + 1):
        train_loss = train_epoch(model, loader, optimizer, criterion, device)
        score = -train_loss
        improved = early.update(score)
        print(f"[final] epoch {epoch} loss={train_loss:.4f}")
        if improved:
            save_checkpoint(best_path, model, cfg)
        if early.should_stop():
            break

    if cfg.finetune.enabled:
        model = load_checkpoint(best_path, cfg, device)
        model.unfreeze_last_blocks(cfg.finetune.unfreeze_last_blocks)
        backbone_params = model.backbone_trainable_params()
        head_params = model.head_trainable_params()
        if not backbone_params:
            raise TrainError("No backbone parameters available for fine-tuning")
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.finetune.lr_backbone},
                {"params": head_params, "lr": cfg.finetune.lr_head},
            ],
            weight_decay=cfg.train.weight_decay,
        )
        total_steps = cfg.finetune.epochs * len(loader)
        scheduler = _build_scheduler(optimizer, cfg.finetune.warmup_ratio, total_steps)
        early = EarlyStopping(cfg.train.early_stop_patience)

        for epoch in range(1, cfg.finetune.epochs + 1):
            train_loss = train_epoch(
                model,
                loader,
                optimizer,
                criterion,
                device,
                scheduler=scheduler,
            )
            score = -train_loss
            improved = early.update(score)
            print(f"[final][ft] epoch {epoch} loss={train_loss:.4f}")
            if improved:
                save_checkpoint(best_path, model, cfg)
            if early.should_stop():
                break

    return best_path
