from __future__ import annotations

import shutil
import sys


def main() -> None:
    missing = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")

    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import torch/torchaudio: {exc}")

    if missing:
        raise RuntimeError(f"Missing required system tools: {', '.join(missing)}")

    print("Environment OK")


if __name__ == "__main__":
    main()
