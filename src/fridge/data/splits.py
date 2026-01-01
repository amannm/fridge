from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Split:
    train_ids: list[str]
    val_ids: list[str]


def make_split(recordings: Iterable[dict], strategy: str, fold: int) -> Split:
    records = list(recordings)
    if strategy == "ac_on_night_holdout":
        return _split_ac_on_night(records)
    if strategy == "environment_holdout":
        return _split_environment_holdout(records, fold)
    raise ValueError(f"Unknown split strategy: {strategy}")


def _split_ac_on_night(records: list[dict]) -> Split:
    train_ids: list[str] = []
    val_ids: list[str] = []
    for rec in records:
        is_val = bool(rec["ac_on"]) and rec["environment"] == "night"
        (val_ids if is_val else train_ids).append(rec["id"])
    if not val_ids:
        raise ValueError("Validation split empty for ac_on_night_holdout")
    if set(train_ids) & set(val_ids):
        raise ValueError("Train/val overlap detected")
    return Split(train_ids=train_ids, val_ids=val_ids)


def _split_environment_holdout(records: list[dict], fold: int) -> Split:
    folds = [
        ("ac_on", "night"),
        ("ac_on", "day"),
        ("ac_off", "night"),
        ("ac_off", "day"),
    ]
    idx = fold % len(folds)
    target_ac, target_env = folds[idx]
    train_ids: list[str] = []
    val_ids: list[str] = []
    for rec in records:
        ac_label = "ac_on" if rec["ac_on"] else "ac_off"
        is_val = ac_label == target_ac and rec["environment"] == target_env
        (val_ids if is_val else train_ids).append(rec["id"])
    if not val_ids:
        raise ValueError("Validation split empty for environment_holdout")
    if set(train_ids) & set(val_ids):
        raise ValueError("Train/val overlap detected")
    return Split(train_ids=train_ids, val_ids=val_ids)
