from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def state_path(root: Path) -> Path:
    path = root / ".crhp" / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load(root: Path) -> dict[str, Any]:
    path = state_path(root)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save(root: Path, **values: Any) -> dict[str, Any]:
    state = load(root)
    state.update(values)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    state_path(root).write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return state

