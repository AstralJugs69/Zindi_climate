from __future__ import annotations

import sys


# Named commands are intentionally boring and auditable. They point at ordinary
# Python modules/scripts in the repo; no notebook state is required.
STAGES: dict[str, list[str]] = {
    "ablate": [sys.executable, "src/ablate_features.py"],
    "baseline": [sys.executable, "src/evaluate_baselines.py"],
    "diagnose-shift": [sys.executable, "src/diagnose_shift.py"],
    "robust-candidates": [sys.executable, "src/robust_candidates.py"],
    "robust-validate": [sys.executable, "src/robust_validate.py"],
    "structured-validate": [sys.executable, "src/structured_validate.py"],
    "suite": [sys.executable, "src/evaluate_model_suite.py"],
    "candidates": [sys.executable, "src/train_candidates.py"],
    "tune-catboost": [sys.executable, "src/tune_catboost.py"],
}


def get(name: str) -> list[str]:
    try:
        return list(STAGES[name])
    except KeyError as exc:
        valid = ", ".join(sorted(STAGES))
        raise KeyError(f"Unknown stage {name!r}. Valid stages: {valid}") from exc
