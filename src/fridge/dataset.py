from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .augment import Augmenter
from .io_audio import AudioError, load_audio
from .windowing import window_indices


class DatasetError(RuntimeError):
    pass


RECORDINGS_COLUMNS = ["recording_id", "path", "label", "group", "duration_sec"]
WINDOWS_COLUMNS = [
    "window_id",
    "recording_id",
    "start_sample",
    "end_sample",
    "label",
    "group",
]


@dataclass
class RecordingEntry:
    recording_id: str
    path: str
    label: int
    group: str
    duration_sec: Optional[float]


def load_recordings_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise DatasetError(f"Recordings manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    missing = [col for col in RECORDINGS_COLUMNS if col not in df.columns]
    if missing:
        raise DatasetError(f"Recordings manifest missing columns: {missing}")
    return df


def save_recordings_manifest(df: pd.DataFrame, path: str | Path) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)


def generate_windows_manifest(
    recordings_df: pd.DataFrame,
    decoded_dir: str | Path,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    output_path: str | Path,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> pd.DataFrame:
    decoded_dir = Path(decoded_dir)
    window_size = int(window_sec * sample_rate)
    hop_size = int(hop_sec * sample_rate)
    rows = []
    for _, rec in recordings_df.iterrows():
        recording_id = rec["recording_id"]
        label = int(rec["label"])
        group = rec["group"]
        decoded_path = decoded_dir / f"{recording_id}.wav"
        waveform = load_audio(
            decoded_path,
            expected_sample_rate=sample_rate,
            expected_mono=True,
            expected_dtype="float32",
            value_range=value_range,
        )
        indices = window_indices(len(waveform), window_size, hop_size)
        for idx, (start, end) in enumerate(indices):
            rows.append(
                {
                    "window_id": f"{recording_id}_{idx:05d}",
                    "recording_id": recording_id,
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "label": label,
                    "group": group,
                }
            )
    if not rows:
        raise DatasetError("No windows generated")
    windows_df = pd.DataFrame(rows, columns=WINDOWS_COLUMNS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows_df.to_csv(output_path, index=False)
    return windows_df


def load_windows_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise DatasetError(f"Windows manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    missing = [col for col in WINDOWS_COLUMNS if col not in df.columns]
    if missing:
        raise DatasetError(f"Windows manifest missing columns: {missing}")
    return df


class AudioCache:
    def __init__(self) -> None:
        self._cache: Dict[str, np.ndarray] = {}

    def get(self, key: str) -> Optional[np.ndarray]:
        return self._cache.get(key)

    def set(self, key: str, value: np.ndarray) -> None:
        self._cache[key] = value


class WindowedFridgeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        windows_df: pd.DataFrame,
        decoded_dir: str | Path,
        sample_rate: int,
        window_sec: float,
        value_range: tuple[float, float] = (-1.0, 1.0),
        augment: Optional[Augmenter] = None,
    ) -> None:
        self.windows_df = windows_df.reset_index(drop=True)
        self.decoded_dir = Path(decoded_dir)
        self.sample_rate = sample_rate
        self.window_samples = int(window_sec * sample_rate)
        self.value_range = value_range
        self.augment = augment
        self._cache = AudioCache()

    def __len__(self) -> int:
        return len(self.windows_df)

    def _load_recording(self, recording_id: str) -> np.ndarray:
        cached = self._cache.get(recording_id)
        if cached is not None:
            return cached
        path = self.decoded_dir / f"{recording_id}.wav"
        waveform = load_audio(
            path,
            expected_sample_rate=self.sample_rate,
            expected_mono=True,
            expected_dtype="float32",
            value_range=self.value_range,
        )
        self._cache.set(recording_id, waveform)
        return waveform

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.windows_df.iloc[idx]
        recording_id = row["recording_id"]
        start = int(row["start_sample"])
        end = int(row["end_sample"])
        label = float(row["label"])
        waveform = self._load_recording(recording_id)
        window = waveform[start:end]
        if window.shape[0] != self.window_samples:
            raise DatasetError(
                f"Window length mismatch for {recording_id}: {window.shape[0]} != {self.window_samples}"
            )
        tensor = torch.from_numpy(window.astype(np.float32))
        if self.augment is not None:
            tensor = self.augment(tensor)
        return tensor, torch.tensor(label, dtype=torch.float32)


def split_by_group(df: pd.DataFrame, train_groups: List[str], eval_groups: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["group"].isin(train_groups)].copy()
    eval_df = df[df["group"].isin(eval_groups)].copy()
    if train_df.empty or eval_df.empty:
        raise DatasetError("Train/eval split produced empty dataset")
    overlap = set(train_df["recording_id"]).intersection(set(eval_df["recording_id"]))
    if overlap:
        raise DatasetError(f"Leakage detected in split: {sorted(overlap)}")
    return train_df, eval_df
