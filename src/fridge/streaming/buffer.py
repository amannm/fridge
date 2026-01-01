from __future__ import annotations

import numpy as np


class RollingBuffer:
    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be > 0")
        self.capacity = capacity_samples
        self.buffer = np.zeros(capacity_samples, dtype=np.float32)
        self.offset = 0
        self.filled = 0

    def append(self, samples: np.ndarray) -> None:
        if samples.ndim != 1:
            raise ValueError("samples must be 1D")
        if samples.size == 0:
            return
        if samples.size >= self.capacity:
            self.buffer[:] = samples[-self.capacity :]
            self.offset = 0
            self.filled = self.capacity
            return
        end = self.offset + samples.size
        if end <= self.capacity:
            self.buffer[self.offset : end] = samples
        else:
            first = self.capacity - self.offset
            self.buffer[self.offset :] = samples[:first]
            self.buffer[: end - self.capacity] = samples[first:]
        self.offset = end % self.capacity
        self.filled = min(self.capacity, self.filled + samples.size)

    def is_full(self) -> bool:
        return self.filled == self.capacity

    def get(self) -> np.ndarray:
        if not self.is_full():
            raise RuntimeError("Buffer not yet full")
        if self.offset == 0:
            return self.buffer.copy()
        return np.concatenate((self.buffer[self.offset :], self.buffer[: self.offset])).copy()
