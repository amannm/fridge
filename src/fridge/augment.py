from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch


class AugmentError(RuntimeError):
    pass


def _db_to_amp(db: float) -> float:
    return float(10 ** (db / 20.0))


def apply_random_gain(waveform: torch.Tensor, min_db: float, max_db: float) -> torch.Tensor:
    gain_db = random.uniform(min_db, max_db)
    gain = _db_to_amp(gain_db)
    return waveform * gain


def apply_time_shift(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim != 1:
        raise AugmentError("Time shift expects mono waveform")
    shift = random.randint(0, waveform.shape[0] - 1)
    return torch.roll(waveform, shifts=shift, dims=0)


def apply_eq_tilt(waveform: torch.Tensor, min_db: float = -2.0, max_db: float = 2.0) -> torch.Tensor:
    if waveform.ndim != 1:
        raise AugmentError("EQ tilt expects mono waveform")
    tilt_db = random.uniform(min_db, max_db)
    spectrum = torch.fft.rfft(waveform)
    freqs = torch.linspace(0.0, 1.0, spectrum.shape[0], device=waveform.device)
    tilt = torch.linspace(-tilt_db, tilt_db, spectrum.shape[0], device=waveform.device)
    gain = torch.pow(10.0, tilt / 20.0)
    spectrum = spectrum * gain
    return torch.fft.irfft(spectrum, n=waveform.shape[0])


class NoiseLibrary:
    def __init__(self, noise_dir: str | Path, sample_rate: int) -> None:
        self.noise_dir = Path(noise_dir)
        self.sample_rate = sample_rate
        if not self.noise_dir.exists():
            raise AugmentError(f"Noise directory not found: {self.noise_dir}")
        self.files = sorted(
            [p for p in self.noise_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
        )
        if not self.files:
            raise AugmentError(f"No .wav noise files found in {self.noise_dir}")

    def load_random(self, length: int) -> torch.Tensor:
        path = random.choice(self.files)
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != self.sample_rate:
            raise AugmentError(f"Noise sample rate mismatch for {path}: {sr}")
        if data.ndim == 2:
            data = data[:, 0]
        noise = torch.from_numpy(data.astype(np.float32))
        if noise.shape[0] < length:
            repeats = int(np.ceil(length / noise.shape[0]))
            noise = noise.repeat(repeats)
        return noise[:length]


def add_noise_snr(waveform: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    if waveform.shape != noise.shape:
        raise AugmentError("Noise length mismatch")
    signal_rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)
    noise_rms = torch.sqrt(torch.mean(noise ** 2) + 1e-8)
    desired_noise_rms = signal_rms / _db_to_amp(snr_db)
    scaled_noise = noise * (desired_noise_rms / (noise_rms + 1e-8))
    return waveform + scaled_noise


class Augmenter:
    def __init__(
        self,
        sample_rate: int,
        gain_db_min: float,
        gain_db_max: float,
        time_shift: bool,
        eq_tilt: bool,
        noise_dir: Optional[str],
        noise_snr_min: float,
        noise_snr_max: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.gain_db_min = gain_db_min
        self.gain_db_max = gain_db_max
        self.time_shift = time_shift
        self.eq_tilt = eq_tilt
        self.noise_snr_min = noise_snr_min
        self.noise_snr_max = noise_snr_max
        self.noise_library = NoiseLibrary(noise_dir, sample_rate) if noise_dir else None

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 1:
            raise AugmentError("Augmenter expects mono waveform")
        augmented = waveform
        augmented = apply_random_gain(augmented, self.gain_db_min, self.gain_db_max)
        if self.time_shift:
            augmented = apply_time_shift(augmented)
        if self.eq_tilt:
            augmented = apply_eq_tilt(augmented)
        if self.noise_library is not None:
            snr_db = random.uniform(self.noise_snr_min, self.noise_snr_max)
            noise = self.noise_library.load_random(augmented.shape[0]).to(augmented.device)
            augmented = add_noise_snr(augmented, noise, snr_db)
        return augmented
