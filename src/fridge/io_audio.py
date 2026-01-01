from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


class AudioError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise AudioError("ffmpeg is required for decoding .m4a files")


def decode_to_wav(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int,
    force: bool = False,
) -> Path:
    require_ffmpeg()
    input_path = Path(input_path)
    if not input_path.exists():
        raise AudioError(f"Input audio not found: {input_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path

    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    args.append("-y" if force else "-n")
    args += [
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_f32le",
        str(output_path),
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioError(
            f"ffmpeg decode failed for {input_path}: {result.stderr.strip()}"
        )
    if not output_path.exists():
        raise AudioError(f"ffmpeg did not produce output: {output_path}")
    return output_path


def load_audio(
    path: str | Path,
    expected_sample_rate: int,
    expected_mono: bool = True,
    expected_dtype: str = "float32",
    value_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise AudioError(f"Audio file not found: {path}")
    try:
        data, sample_rate = sf.read(path, dtype=expected_dtype, always_2d=True)
    except Exception as exc:  # pragma: no cover - backend errors vary by platform
        raise AudioError(f"Failed to read audio: {path}") from exc
    if sample_rate != expected_sample_rate:
        raise AudioError(
            f"Sample rate mismatch for {path}: {sample_rate} != {expected_sample_rate}"
        )
    if expected_mono and data.shape[1] != 1:
        raise AudioError(f"Expected mono audio for {path}, got {data.shape[1]} channels")
    if expected_mono:
        data = data[:, 0]
    if data.dtype.name != expected_dtype:
        raise AudioError(f"Expected dtype {expected_dtype} for {path}, got {data.dtype}")
    min_val, max_val = float(np.min(data)), float(np.max(data))
    if min_val < value_range[0] - 1e-4 or max_val > value_range[1] + 1e-4:
        raise AudioError(
            f"Audio out of range for {path}: min {min_val}, max {max_val}"
        )
    return data.astype(np.float32, copy=False)


def get_duration_seconds(samples: np.ndarray, sample_rate: int) -> float:
    if samples.ndim != 1:
        raise AudioError("Duration requires mono waveform")
    return float(samples.shape[0]) / float(sample_rate)
