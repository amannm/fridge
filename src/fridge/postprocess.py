from __future__ import annotations

from dataclasses import dataclass


class PostProcessError(RuntimeError):
    pass


@dataclass
class PostProcessor:
    ema_alpha: float
    on_threshold: float
    on_frames: int
    off_threshold: float
    off_frames: int

    def __post_init__(self) -> None:
        if not 0.0 < self.ema_alpha < 1.0:
            raise PostProcessError("EMA alpha must be between 0 and 1")
        if self.on_frames <= 0 or self.off_frames <= 0:
            raise PostProcessError("Frame counts must be positive")
        self._p_hat = None
        self._state = False
        self._on_count = 0
        self._off_count = 0

    def update(self, prob: float) -> tuple[float, bool]:
        if self._p_hat is None:
            self._p_hat = prob
        else:
            self._p_hat = self.ema_alpha * self._p_hat + (1.0 - self.ema_alpha) * prob

        if self._p_hat > self.on_threshold:
            self._on_count += 1
            self._off_count = 0
        elif self._p_hat < self.off_threshold:
            self._off_count += 1
            self._on_count = 0
        else:
            self._on_count = 0
            self._off_count = 0

        if not self._state and self._on_count >= self.on_frames:
            self._state = True
            self._on_count = 0
        elif self._state and self._off_count >= self.off_frames:
            self._state = False
            self._off_count = 0

        return float(self._p_hat), self._state
