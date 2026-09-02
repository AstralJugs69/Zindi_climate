from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import (
    ID_COL,
    TARGET,
    categorical_columns,
    engineer_features,
    feature_columns,
    load_competition_data,
)


def _feature_view(frame: pd.DataFrame, view: str) -> pd.DataFrame:
    x = engineer_features(frame, include_location=(view == "full"))
    x = x[feature_columns(x)]

    if view == "full":
        return x
    if view == "no_location":
        return x
    if view == "no_spatial":
        spatial = {
            "location",
            "latitude",
            "longitude",
            "elevation",
            "slope",
            "lat_bin_025",
            "lon_bin_025",
            "spatial_cell_025",
        }
        return x.drop(columns=[c for c in spatial if c in x.columns])
    if view == "demographics_time":
        keep = {
            "zone",
            "gender",
            "age",
            "year",
            "month",
            "day_of_year",
            "week_of_year",
            "month_sin",
            "month_cos",
            "doy_sin",
            "doy_cos",
            "age_band",
            "age_log1p",
            "age_sq",
            "is_infant",
            "is_under5",
            "is_child",
            "is_elderly",
        }
        return x[[c for c in x.columns if c in keep]]
    raise ValueError(view)


def evaluate_view(raw: pd.DataFrame, domain: np.ndarray, view: str, out_dir: Path):
    x = _feature_view(raw, view)
    cats = categorical_columns(x)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros(len(x), dtype=float)
    importances = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, domain)):
        model = CatBoostClassifier(
            iterations=350,
            depth=6,
            learning_rate=0.04,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=2026 + fold,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(
            x.iloc[tr_idx],
            domain[tr_idx],
            cat_features=cats,
            eval_set=(x.iloc[va_idx], domain[va_idx]),
            early_stopping_rounds=60,
            verbose=False,
        )
        oof[va_idx] = model.predict_proba(x.iloc[va_idx])[:, 1]
        importances.append(model.get_feature_importance())

    auc = float(roc_auc_score(domain, oof))
    imp = pd.DataFrame(
        {
            "feature": x.columns,
            "importance": np.mean(importances, axis=0),
        }
    ).sort_values("importance", ascending=False)
    imp.to_csv(out_dir / f"{view}_feature_importance.csv", index=False)
    pd.DataFrame({"domain_test": domain, "oof_probability": oof}).to_csv(
        out_dir / f"{view}_oof.csv", index=False
    )
    return {"view": view, "auc": auc, "n_features": x.shape[1]}


def main():
    train, test, _ = load_competition_data()
    raw_train = train.drop(columns=[TARGET])
    raw = pd.concat([raw_train, test], ignore_index=True)
    domain = np.concatenate(
        [np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)]
    )

    out_dir = Path("reports/shift")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for view in ["full", "no_location", "no_spatial", "demographics_time"]:
        print(f"Running adversarial validation: {view} ...", flush=True)
        result = evaluate_view(raw, domain, view, out_dir)
        print(result, flush=True)
        rows.append(result)

    summary = pd.DataFrame(rows).sort_values("auc", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("\nShift summary:\n", summary.to_string(index=False), flush=True)

    for view in ["full", "no_location", "no_spatial"]:
        path = out_dir / f"{view}_feature_importance.csv"
        print(f"\nTop shift drivers ({view}):\n")
        print(pd.read_csv(path).head(15).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
