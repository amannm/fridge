from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import Dataset

from ..io.audio import load_audio, trim_audio
from .augment import AudioAugmenter
from .windowing import WindowSpec


@dataclass(frozen=True)
class Batch:
    waveforms: torch.Tensor
    padding_mask: torch.Tensor
    fridge_labels: torch.Tensor
    ac_labels: torch.Tensor
    recording_ids: list[str]
    start_times: list[float]


def collate_batch(batch: list[tuple[torch.Tensor, int, int, str, float]]) -> Batch:
    waveforms, fridge_labels, ac_labels, recording_ids, start_times = zip(*batch)
    lengths = [w.shape[-1] for w in waveforms]
    max_len = max(lengths)
    padded = torch.zeros(len(waveforms), max_len, dtype=waveforms[0].dtype)
    padding_mask = torch.ones(len(waveforms), max_len, dtype=torch.bool)
    for idx, waveform in enumerate(waveforms):
        length = waveform.shape[-1]
        padded[idx, :length] = waveform
        padding_mask[idx, :length] = False
    return Batch(
        waveforms=padded,
        padding_mask=padding_mask,
        fridge_labels=torch.tensor(fridge_labels, dtype=torch.float32),
        ac_labels=torch.tensor(ac_labels, dtype=torch.float32),
        recording_ids=list(recording_ids),
        start_times=list(start_times),
    )


class WindowDataset(Dataset):
    def __init__(
        self,
        windows: Iterable[WindowSpec],
        recordings: dict[str, dict],
        sample_rate: int,
        mono: bool,
        trim_start_s: float,
        trim_end_s: float,
        train: bool,
        random_crop_s: tuple[float, float],
        time_shift_s: float,
        augmenter: Optional[AudioAugmenter] = None,
    ) -> None:
        self.windows = list(windows)
        self.recordings = recordings
        self.sample_rate = sample_rate
        self.mono = mono
        self.trim_start_s = trim_start_s
        self.trim_end_s = trim_end_s
        self.train = train
        self.random_crop_s = random_crop_s
        self.time_shift_s = time_shift_s
        self.augmenter = augmenter
        self._cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        spec = self.windows[idx]
        waveform = self._load_recording(spec.recording_id, spec.path)
        if self.train:
            start_s, crop_s = _sample_train_window(
                anchor_s=spec.start_s,
                crop_bounds=self.random_crop_s,
                trim_start_s=self.trim_start_s,
                trim_end_s=self._trimmed_end(spec.recording_id),
                time_shift_s=self.time_shift_s,
            )
        else:
            crop_s = spec.duration_s
            start_s = _sample_eval_start(
                anchor_s=spec.start_s,
                crop_s=crop_s,
                trim_start_s=self.trim_start_s,
                trim_end_s=self._trimmed_end(spec.recording_id),
            )
        start_idx = int((start_s - self.trim_start_s) * self.sample_rate)
        num_samples = int(crop_s * self.sample_rate)
        segment = waveform[..., start_idx : start_idx + num_samples]
        if segment.shape[-1] != num_samples:
            raise RuntimeError("Segment length mismatch")
        segment = segment.squeeze(0)
        if self.train and self.augmenter is not None:
            segment = self.augmenter(segment, spec.fridge_on)
        return segment, spec.fridge_on, spec.ac_on, spec.recording_id, spec.start_s

    def _load_recording(self, recording_id: str, path: str) -> torch.Tensor:
        if recording_id in self._cache:
            return self._cache[recording_id]
        record = self.recordings[recording_id]
        waveform = load_audio(Path(record["path"]), self.sample_rate, self.mono)
        waveform = trim_audio(waveform, self.sample_rate, self.trim_start_s, self.trim_end_s)
        self._cache[recording_id] = waveform
        return waveform

    def _trimmed_end(self, recording_id: str) -> float:
        record = self.recordings[recording_id]
        return record["duration_s"] - self.trim_end_s


def _sample_uniform(bounds: tuple[float, float]) -> float:
    low, high = bounds
    return torch.empty(1).uniform_(low, high).item()


def _sample_train_window(
    anchor_s: float,
    crop_bounds: tuple[float, float],
    trim_start_s: float,
    trim_end_s: float,
    time_shift_s: float,
) -> tuple[float, float]:
    for _ in range(20):
        crop_s = _sample_uniform(crop_bounds)
        shift = torch.empty(1).uniform_(-time_shift_s, time_shift_s).item()
        start = anchor_s + shift
        if start < trim_start_s:
            continue
        if start + crop_s > trim_end_s:
            continue
        return start, crop_s
    raise RuntimeError("Failed to sample a valid training window within bounds")


def _sample_eval_start(
    anchor_s: float,
    crop_s: float,
    trim_start_s: float,
    trim_end_s: float,
) -> float:
    start = anchor_s
    if start < trim_start_s or start + crop_s > trim_end_s:
        raise RuntimeError("Evaluation window exceeds trimmed bounds")
    return start
