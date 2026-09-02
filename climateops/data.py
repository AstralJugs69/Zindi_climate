from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .config import REQUIRED_DATA_FILES


def _find_unique(input_root: Path, filename: str) -> Path:
    matches = [p for p in input_root.rglob(filename) if p.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} below {input_root}. "
            "Attach the Zindi competition files to the Kaggle notebook as a Dataset."
        )
    if len(matches) > 1:
        print(f"Warning: multiple {filename} files found; using {matches[0]}")
    return matches[0]


def hydrate(root: Path, input_root: Path = Path("/kaggle/input")) -> dict[str, str]:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    # Outside Kaggle, existing canonical files are accepted but nothing is copied.
    if not input_root.exists():
        missing = [name for name in REQUIRED_DATA_FILES if not (raw / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing data files: {', '.join(missing)}")
        return {name: str((raw / name).resolve()) for name in REQUIRED_DATA_FILES}

    for filename in REQUIRED_DATA_FILES:
        source = _find_unique(input_root, filename)
        target = raw / filename

        if target.is_symlink() or target.exists():
            try:
                if target.resolve() == source.resolve():
                    manifest[filename] = str(source)
                    continue
            except Exception:
                pass
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

        try:
            os.symlink(source, target)
        except OSError:
            # Symlinks are preferable because Kaggle Inputs are already mounted,
            # but a copy is a safe fallback if the runtime disallows symlinks.
            shutil.copy2(source, target)
        manifest[filename] = str(source)

    runtime = root / ".crhp"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest

