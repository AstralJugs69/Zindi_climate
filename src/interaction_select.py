from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import official_metrics


CONFIGS = ("base_ctr1", "categorical_ctr1", "numeric_ctr1", "all_ctr1")
WEIGHTS = (0.25, 0.50, 0.75)


def _load_oof(report_dir: Path, config: str) -> tuple[np.ndarray, np.ndarray]:
    path = report_dir / f"{config}_oof.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `crhp run interaction-validate` first."
        )
    frame = pd.read_csv(path)
    return frame["target"].to_numpy(dtype=int), frame["oof_probability"].to_numpy(dtype=float)


def main():
    report_dir = Path("reports/interaction_validation")
    probabilities: dict[str, np.ndarray] = {}
    y_reference = None

    for config in CONFIGS:
        y, p = _load_oof(report_dir, config)
        if y_reference is None:
            y_reference = y
        elif not np.array_equal(y_reference, y):
            raise RuntimeError("OOF target ordering differs across interaction configs")
        probabilities[config] = p

    assert y_reference is not None
    rows = []

    for name, p in probabilities.items():
        rows.append({"candidate": name, "kind": "single", **official_metrics(y_reference, p)})

    for left, right in combinations(CONFIGS, 2):
        for w_left in WEIGHTS:
            p = w_left * probabilities[left] + (1.0 - w_left) * probabilities[right]
            rows.append(
                {
                    "candidate": f"{w_left:.2f}_{left}+{1.0-w_left:.2f}_{right}",
                    "kind": "blend",
                    **official_metrics(y_reference, p),
                }
            )

    result = pd.DataFrame(rows).sort_values(["score", "auc"], ascending=False)
    result.to_csv(report_dir / "blend_search.csv", index=False)
    print("\nInteraction selection:\n", result.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
