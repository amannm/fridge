from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fridge.config import load_config
from fridge.data.recordings import parse_recording_filename
from fridge.io.manifest import write_jsonl
from fridge.utils.logging import setup_logging
from fridge.utils.paths import ensure_dir, find_project_root


def _convert_to_wav(src: Path, dst: Path, sample_rate: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}: {result.stderr.strip()}")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    root = find_project_root(Path(args.config))

    ensure_dir(config.paths.processed_wav_dir)
    ensure_dir(config.paths.manifests_dir)

    samples_dir = config.paths.samples_dir
    if not samples_dir.exists():
        raise RuntimeError(f"Samples directory not found: {samples_dir}")

    records: list[dict] = []
    for src in sorted(samples_dir.glob("*.m4a")) + sorted(samples_dir.glob("*.wav")):
        labels = parse_recording_filename(src)
        wav_path = config.paths.processed_wav_dir / f"{src.stem}.wav"
        if src.suffix.lower() == ".m4a":
            _convert_to_wav(src, wav_path, config.audio.sample_rate)
        else:
            if src.resolve() != wav_path.resolve():
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                wav_path.write_bytes(src.read_bytes())
        waveform, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        if sample_rate != config.audio.sample_rate:
            raise RuntimeError(f"Expected {config.audio.sample_rate} Hz, got {sample_rate} for {wav_path}")
        if waveform.ndim != 2 or waveform.shape[1] != 1:
            raise RuntimeError(f"Expected mono audio for {wav_path}")
        duration_s = waveform.shape[0] / sample_rate
        record = {
            **labels,
            "path": str(wav_path.relative_to(root)),
            "duration_s": duration_s,
            "sample_rate": sample_rate,
        }
        records.append(record)

    if not records:
        raise RuntimeError("No samples found in samples/ directory")

    manifest_path = config.paths.manifests_dir / "recordings.jsonl"
    write_jsonl(manifest_path, records)


if __name__ == "__main__":
    main()
