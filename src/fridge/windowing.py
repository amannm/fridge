from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


class WindowingError(RuntimeError):
    pass


def window_indices(num_samples: int, window_size: int, hop_size: int) -> List[Tuple[int, int]]:
    if num_samples < window_size:
        raise WindowingError(
            f"Audio shorter than window size: {num_samples} < {window_size}"
        )
    if window_size <= 0 or hop_size <= 0:
        raise WindowingError("Window size and hop size must be positive")
    indices = []
    start = 0
    while start + window_size <= num_samples:
        end = start + window_size
        indices.append((start, end))
        start += hop_size
    if not indices:
        raise WindowingError("No windows generated")
    return indices


@dataclass
class RingBuffer:
    size: int

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise WindowingError("RingBuffer size must be positive")
        self._buffer = np.zeros(self.size, dtype=np.float32)
        self._write_pos = 0
        self._filled = 0

    def append(self, samples: np.ndarray) -> None:
        if samples.ndim != 1:
            raise WindowingError("RingBuffer expects mono samples")
        num = samples.shape[0]
        if num > self.size:
            raise WindowingError("Chunk larger than ring buffer")
        end = self._write_pos + num
        if end <= self.size:
            self._buffer[self._write_pos:end] = samples
        else:
            first = self.size - self._write_pos
            self._buffer[self._write_pos:] = samples[:first]
            self._buffer[: end - self.size] = samples[first:]
        self._write_pos = (self._write_pos + num) % self.size
        self._filled = min(self.size, self._filled + num)

    @property
    def filled(self) -> bool:
        return self._filled == self.size

    def get(self) -> np.ndarray:
        if not self.filled:
            raise WindowingError("RingBuffer is not yet full")
        if self._write_pos == 0:
            return self._buffer.copy()
        return np.concatenate((self._buffer[self._write_pos :], self._buffer[: self._write_pos])).astype(
            np.float32
        )
