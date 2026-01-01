from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HysteresisState:
    ema_alpha: float
    hysteresis_on: float
    hysteresis_off: float
    state: int = 0
    p_ema: float = 0.0
    _initialized: bool = False

    def update(self, p: float) -> int:
        if not self._initialized:
            self.p_ema = p
            self._initialized = True
        else:
            self.p_ema = self.ema_alpha * p + (1.0 - self.ema_alpha) * self.p_ema
        if self.p_ema >= self.hysteresis_on:
            self.state = 1
        elif self.p_ema <= self.hysteresis_off:
            self.state = 0
        return self.state
