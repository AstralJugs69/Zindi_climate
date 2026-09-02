from __future__ import annotations

import sys


# Named commands are intentionally boring and auditable. They point at ordinary
# Python modules/scripts in the repo; no notebook state is required.
STAGES: dict[str, list[str]] = {
    "ablate": [sys.executable, "src/ablate_features.py"],
    "age-expert-validate": [sys.executable, "src/age_expert_validate.py"],
    "baseline": [sys.executable, "src/evaluate_baselines.py"],
    "chirps-validate": [sys.executable, "src/chirps_validate.py"],
    "cohort-candidates": [sys.executable, "src/cohort_candidates.py"],
    "demographic-validate": [sys.executable, "src/demographic_validate.py"],
    "diagnose-shift": [sys.executable, "src/diagnose_shift.py"],
    "fine-demo-validate": [sys.executable, "src/fine_demo_validate.py"],
    "interaction-validate": [sys.executable, "src/interaction_validate.py"],
    "interaction-select": [sys.executable, "src/interaction_select.py"],
    "interaction-candidates": [sys.executable, "src/interaction_candidates.py"],
    "low-shift-select": [sys.executable, "src/low_shift_select.py"],
    "lagged-climate-validate": [sys.executable, "src/lagged_climate_validate.py"],
    "power-validate": [sys.executable, "src/power_climate_validate.py"],
    "profile-validate": [sys.executable, "src/profile_validate.py"],
    "robust-candidates": [sys.executable, "src/robust_candidates.py"],
    "robust-validate": [sys.executable, "src/robust_validate.py"],
    "structured-validate": [sys.executable, "src/structured_validate.py"],
    "temporal-density-validate": [sys.executable, "src/temporal_density_validate.py"],
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
