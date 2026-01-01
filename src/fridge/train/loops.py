from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..eval.metrics import window_metrics
from ..eval.stream_eval import smoothed_f1


class EarlyStopper:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_score: Optional[float] = None
        self.bad_epochs = 0

    def update(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    device: torch.device,
    grad_accum_steps: int,
    grad_clip: float,
    aux_weight: float,
) -> dict:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        waveforms = batch.waveforms.to(device)
        padding_mask = batch.padding_mask.to(device)
        fridge_labels = batch.fridge_labels.to(device)
        ac_labels = batch.ac_labels.to(device)

        fridge_logit, ac_logit = model(waveforms, padding_mask=padding_mask)
        loss = criterion(fridge_logit, fridge_labels)
        if ac_logit is not None:
            loss = loss + aux_weight * criterion(ac_logit, ac_labels)

        loss = loss / grad_accum_steps
        loss.backward()

        if step % grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps
        total_steps += 1

    if total_steps % grad_accum_steps != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return {"train_loss": total_loss / max(1, total_steps)}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    recordings: dict[str, dict],
    ema_alpha: float,
    hysteresis_on: float,
    hysteresis_off: float,
) -> dict:
    model.eval()
    all_scores: list[float] = []
    all_labels: list[int] = []
    all_recording_ids: list[str] = []
    all_start_times: list[float] = []

    with torch.no_grad():
        for batch in loader:
            waveforms = batch.waveforms.to(device)
            padding_mask = batch.padding_mask.to(device)
            fridge_labels = batch.fridge_labels.to(device)

            fridge_logit, _ = model(waveforms, padding_mask=padding_mask)
            scores = torch.sigmoid(fridge_logit).cpu().tolist()
            labels = fridge_labels.cpu().int().tolist()
            all_scores.extend(scores)
            all_labels.extend(labels)
            all_recording_ids.extend(batch.recording_ids)
            all_start_times.extend(batch.start_times)

    metrics = window_metrics(all_labels, all_scores)
    metrics["smoothed_f1"] = smoothed_f1(
        recordings=recordings,
        y_scores=all_scores,
        recording_ids=all_recording_ids,
        start_times=all_start_times,
        ema_alpha=ema_alpha,
        hysteresis_on=hysteresis_on,
        hysteresis_off=hysteresis_off,
    )
    return metrics
