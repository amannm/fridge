from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class WindowSpec:
    recording_id: str
    path: str
    start_s: float
    duration_s: float
    fridge_on: int
    ac_on: int
    environment: str


def _trim_bounds(duration_s: float, trim_start_s: float, trim_end_s: float) -> tuple[float, float]:
    start = trim_start_s
    end = duration_s - trim_end_s
    if end <= start:
        raise ValueError("Trim bounds exceed duration")
    return start, end


def compute_window_starts(
    duration_s: float,
    window_s: float,
    hop_s: float,
    trim_start_s: float,
    trim_end_s: float,
) -> list[float]:
    start, end = _trim_bounds(duration_s, trim_start_s, trim_end_s)
    if end - start < window_s:
        raise ValueError("Window larger than trimmed audio")
    starts: list[float] = []
    t = start
    while t + window_s <= end + 1e-8:
        starts.append(t)
        t += hop_s
    return starts


def build_windows(
    recordings: Iterable[dict],
    window_s: float,
    hop_s: float,
    trim_start_s: float,
    trim_end_s: float,
) -> list[WindowSpec]:
    windows: list[WindowSpec] = []
    for record in recordings:
        starts = compute_window_starts(
            duration_s=record["duration_s"],
            window_s=window_s,
            hop_s=hop_s,
            trim_start_s=trim_start_s,
            trim_end_s=trim_end_s,
        )
        for start_s in starts:
            windows.append(
                WindowSpec(
                    recording_id=record["id"],
                    path=record["path"],
                    start_s=start_s,
                    duration_s=window_s,
                    fridge_on=int(record["fridge_on"]),
                    ac_on=int(record["ac_on"]),
                    environment=record["environment"],
                )
            )
    return windows
