from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def window_metrics(y_true: Iterable[int], y_score: Iterable[float], threshold: float = 0.5) -> dict:
    y_true = np.asarray(list(y_true))
    y_score = np.asarray(list(y_score))
    if y_true.ndim != 1:
        raise RuntimeError("y_true must be 1D")
    if y_true.shape[0] != y_score.shape[0]:
        raise RuntimeError("y_true and y_score length mismatch")
    if len(np.unique(y_true)) < 2:
        raise RuntimeError("Cannot compute AUROC with single-class labels")
    y_pred = (y_score >= threshold).astype(int)
    return {
        "window_auroc": float(roc_auc_score(y_true, y_score)),
        "window_f1": float(f1_score(y_true, y_pred)),
        "window_accuracy": float(accuracy_score(y_true, y_pred)),
    }
