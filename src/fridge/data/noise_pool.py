from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from ..io.audio import load_audio, trim_audio
from .windowing import compute_window_starts


@dataclass
class NoisePool:
    fridge_off: list[torch.Tensor]
    fridge_on: list[torch.Tensor]

    def sample_off(self) -> torch.Tensor:
        if not self.fridge_off:
            raise RuntimeError("Noise pool (fridge_off) is empty")
        idx = torch.randint(0, len(self.fridge_off), (1,)).item()
        return self.fridge_off[idx]

    def sample_on(self) -> torch.Tensor:
        if not self.fridge_on:
            raise RuntimeError("Noise pool (fridge_on) is empty")
        idx = torch.randint(0, len(self.fridge_on), (1,)).item()
        return self.fridge_on[idx]


def build_noise_pool(
    recordings: Iterable[dict],
    train_ids: set[str],
    sample_rate: int,
    mono: bool,
    trim_start_s: float,
    trim_end_s: float,
    window_s: float,
    hop_s: float,
) -> NoisePool:
    fridge_off: list[torch.Tensor] = []
    fridge_on: list[torch.Tensor] = []
    for record in recordings:
        if record["id"] not in train_ids:
            continue
        waveform = load_audio(Path(record["path"]), sample_rate, mono)
        waveform = trim_audio(waveform, sample_rate, trim_start_s, trim_end_s)
        total_s = waveform.size(-1) / sample_rate
        starts = compute_window_starts(total_s, window_s, hop_s, 0.0, 0.0)
        for start_s in starts:
            start = int(start_s * sample_rate)
            end = start + int(window_s * sample_rate)
            segment = waveform[..., start:end]
            if segment.size(-1) != int(window_s * sample_rate):
                continue
            segment = segment.squeeze(0)
            if record["fridge_on"]:
                fridge_on.append(segment)
            else:
                fridge_off.append(segment)
    if not fridge_off:
        raise RuntimeError("No fridge_off windows available for noise pool")
    if not fridge_on:
        raise RuntimeError("No fridge_on windows available for hard negatives")
    return NoisePool(fridge_off=fridge_off, fridge_on=fridge_on)
