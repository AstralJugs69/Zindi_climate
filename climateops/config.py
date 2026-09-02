from __future__ import annotations

import os
from pathlib import Path


PROJECT_SLUG = "climate-risk-health-prediction"
DEFAULT_KAGGLE_ROOT = Path("/kaggle/working") / PROJECT_SLUG
REQUIRED_DATA_FILES = (
    "Train.csv",
    "Test.csv",
    "SampleSubmission.csv",
    "data_dictionary.csv",
    "downloaded_climate_features_data_dictionary.csv",
    "climate_features.csv",
)


def repo_root() -> Path:
    override = os.getenv("CRHP_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    return here.parents[1]


def is_kaggle() -> bool:
    return Path("/kaggle").exists() and Path("/kaggle/working").exists()


def branch() -> str:
    return os.getenv("CRHP_BRANCH", "main")


def artifact_handle() -> str | None:
    value = os.getenv("CRHP_ARTIFACT_DATASET", "").strip()
    return value or None

