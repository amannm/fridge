from __future__ import annotations

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    if start is None:
        start_path = Path(__file__).resolve()
    else:
        start_path = Path(start).resolve()
    if start_path.is_file():
        start_path = start_path.parent
    for parent in (start_path, *start_path.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not locate pyproject.toml in parent directories")


def resolve_path(root: Path, path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
