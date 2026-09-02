from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable


def run(
    args: Iterable[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    cmd = [str(x) for x in args]
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged,
        check=check,
        text=True,
    )


def capture(args: Iterable[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        [str(x) for x in args],
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()

