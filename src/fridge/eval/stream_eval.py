from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score

from ..streaming.hysteresis import HysteresisState


def smoothed_f1(
    recordings: dict[str, dict],
    y_scores: Iterable[float],
    recording_ids: Iterable[str],
    start_times: Iterable[float],
    ema_alpha: float,
    hysteresis_on: float,
    hysteresis_off: float,
) -> float:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for score, rec_id, start in zip(y_scores, recording_ids, start_times):
        grouped[rec_id].append((start, float(score)))

    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    for rec_id, items in grouped.items():
        items.sort(key=lambda x: x[0])
        state = HysteresisState(
            ema_alpha=ema_alpha,
            hysteresis_on=hysteresis_on,
            hysteresis_off=hysteresis_off,
        )
        true_label = int(recordings[rec_id]["fridge_on"])
        for _, score in items:
            pred = state.update(score)
            y_true_all.append(true_label)
            y_pred_all.append(pred)

    if not y_true_all:
        raise RuntimeError("No samples provided for smoothed F1")
    return float(f1_score(np.asarray(y_true_all), np.asarray(y_pred_all)))
