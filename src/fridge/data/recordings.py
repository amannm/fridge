from __future__ import annotations

import re
from pathlib import Path

FILENAME_RE = re.compile(r"^ac-(on|off)-fridge-(on|off)-(day|night)$")


class RecordingNameError(ValueError):
    pass


def parse_recording_filename(path: Path) -> dict:
    stem = path.stem
    match = FILENAME_RE.match(stem)
    if not match:
        raise RecordingNameError(
            f"Invalid filename '{path.name}'. Expected format: ac-(on|off)-fridge-(on|off)-(day|night)."
        )
    ac_state, fridge_state, environment = match.groups()
    return {
        "id": stem,
        "ac_on": 1 if ac_state == "on" else 0,
        "fridge_on": 1 if fridge_state == "on" else 0,
        "environment": environment,
    }
