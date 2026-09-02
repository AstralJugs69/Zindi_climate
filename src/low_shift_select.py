from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import official_metrics


SOURCES = {
    "all_ctr1": Path("reports/interaction_validation/all_ctr1_oof.csv"),
    "categorical_ctr1": Path("reports/interaction_validation/categorical_ctr1_oof.csv"),
    "numeric_ctr1": Path("reports/interaction_validation/numeric_ctr1_oof.csv"),
    "fine_age": Path("reports/fine_demo_validation/fine_age_oof.csv"),
    "cohort_calendar": Path("reports/fine_demo_validation/cohort_calendar_oof.csv"),
    "rich": Path("reports/fine_demo_validation/rich_oof.csv"),
}


def load_oof():
    probs = {}
    y = None
    ids = None
    for name, path in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run interaction-validate and fine-demo-validate first."
            )
        frame = pd.read_csv(path)
        if y is None:
            y = frame["target"].astype(int).to_numpy()
            ids = frame["ID"].astype(str).to_numpy()
        else:
            if not np.array_equal(y, frame["target"].astype(int).to_numpy()):
                raise ValueError(f"Target order mismatch in {path}")
            if not np.array_equal(ids, frame["ID"].astype(str).to_numpy()):
                raise ValueError(f"ID order mismatch in {path}")
        probs[name] = frame["oof_probability"].astype(float).to_numpy()
    return y, probs


def main():
    out_dir = Path("reports/low_shift_selection")
    out_dir.mkdir(parents=True, exist_ok=True)
    y, probs = load_oof()

    rows = []
    for name, p in probs.items():
        rows.append({"candidate": name, "kind": "single", **official_metrics(y, p)})

    keys = sorted(probs)
    for a, b in combinations(keys, 2):
        for wa in (0.25, 0.50, 0.75):
            p = wa * probs[a] + (1.0 - wa) * probs[b]
            rows.append(
                {
                    "candidate": f"{wa:.2f}_{a}+{1-wa:.2f}_{b}",
                    "kind": "pair",
                    **official_metrics(y, p),
                }
            )

    # Pre-declared three-way blends centered on the most transfer-safe all_ctr1 model.
    triples = [
        (0.50, "all_ctr1", 0.25, "fine_age", 0.25, "cohort_calendar"),
        (0.50, "all_ctr1", 0.25, "categorical_ctr1", 0.25, "cohort_calendar"),
        (0.50, "all_ctr1", 0.25, "categorical_ctr1", 0.25, "fine_age"),
        (0.50, "categorical_ctr1", 0.25, "fine_age", 0.25, "cohort_calendar"),
        (0.40, "all_ctr1", 0.30, "categorical_ctr1", 0.30, "cohort_calendar"),
        (0.40, "all_ctr1", 0.30, "categorical_ctr1", 0.30, "fine_age"),
    ]
    for w1, a, w2, b, w3, c in triples:
        p = w1 * probs[a] + w2 * probs[b] + w3 * probs[c]
        rows.append(
            {
                "candidate": f"{w1:.2f}_{a}+{w2:.2f}_{b}+{w3:.2f}_{c}",
                "kind": "triple",
                **official_metrics(y, p),
            }
        )

    result = pd.DataFrame(rows).sort_values(["score", "auc"], ascending=False)
    result.to_csv(out_dir / "blend_search.csv", index=False)
    print("\nTop low-shift blends:\n", result.head(25).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
