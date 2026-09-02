from __future__ import annotations

import json
import shutil
from pathlib import Path


ARTIFACT_DIRS = ("models", "submissions", "reports")


def _export(root: Path) -> Path:
    export = root / ".crhp" / "artifact_export"
    if export.exists():
        shutil.rmtree(export)
    export.mkdir(parents=True, exist_ok=True)

    state = root / ".crhp" / "state.json"
    if state.exists():
        shutil.copy2(state, export / "state.json")

    for name in ARTIFACT_DIRS:
        source = root / name
        if source.exists():
            shutil.copytree(
                source,
                export / name,
                ignore=shutil.ignore_patterns(".gitkeep", "__pycache__", "*.pyc"),
                dirs_exist_ok=True,
            )
    return export


def snapshot(root: Path, handle: str, notes: str = "competition checkpoint") -> None:
    import kagglehub

    export = _export(root)
    kagglehub.dataset_upload(handle, str(export), version_notes=notes)
    print(f"Checkpoint uploaded to Kaggle Dataset: {handle}")


def restore(root: Path, handle: str) -> None:
    import kagglehub

    cache = root / ".crhp" / "artifact_restore"
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)

    downloaded = Path(
        kagglehub.dataset_download(handle, output_dir=str(cache), force_download=True)
    )
    source_root = downloaded if downloaded.is_dir() else cache

    state_candidates = list(source_root.rglob("state.json"))
    if state_candidates:
        target = root / ".crhp" / "state.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(state_candidates[0], target)

    for name in ARTIFACT_DIRS:
        candidates = [p for p in source_root.rglob(name) if p.is_dir()]
        if not candidates:
            continue
        shutil.copytree(candidates[0], root / name, dirs_exist_ok=True)

    print(f"Checkpoint restored from Kaggle Dataset: {handle}")

