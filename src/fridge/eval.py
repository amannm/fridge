from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


class EvalError(RuntimeError):
    pass


@dataclass
class EvalResults:
    auc: float
    f1: float
    accuracy: float
    confusion: list


def collect_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            logits = model(waveforms)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    return y_true, y_prob


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> EvalResults:
    if len(np.unique(y_true)) < 2:
        raise EvalError("AUC requires both classes in evaluation data")
    y_pred = (y_prob >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, y_prob))
    f1 = float(f1_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    conf = confusion_matrix(y_true, y_pred).tolist()
    return EvalResults(auc=auc, f1=f1, accuracy=acc, confusion=conf)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.0, 1.0, num=101):
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold
