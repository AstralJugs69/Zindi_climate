from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold

from features import (
    ID_COL,
    TARGET,
    categorical_columns,
    engineer_features,
    feature_columns,
    load_competition_data,
)
from metrics import official_metrics


BASE_CLIMATE = {
    "avg_temperature",
    "max_temperature",
    "min_temperature",
    "precipitation",
    "temperature_range",
    "is_rainy_day_current",
    "precip_log1p",
    "elevation",
    "slope",
    "hot_days_30d",
    "max_daily_rain_30d",
    "ndvi_30d",
    "ndvi_90d",
    "rain_days_30d",
    "rain_sum_30d",
    "rain_sum_7d",
    "rain_sum_90d",
    "tavg_30d",
    "tavg_7d",
    "tavg_90d",
    "temp_range_mean_30d",
    "tmax_30d",
    "tmin_30d",
    "tavg_7_minus_30",
    "tavg_30_minus_90",
    "rain_7_share_30",
    "rain_30_share_90",
    "rain_intensity_30",
    "ndvi_30_minus_90",
}

SPATIAL = {
    "location",
    "latitude",
    "longitude",
    "elevation",
    "slope",
    "lat_bin_025",
    "lon_bin_025",
    "spatial_cell_025",
}


def make_view(train: pd.DataFrame, name: str) -> pd.DataFrame:
    x = engineer_features(train.drop(columns=[TARGET]), include_location=True)
    x = x[feature_columns(x)]

    interaction_cols = [
        c
        for c in x.columns
        if c.startswith("age_x_")
        or c.startswith("under5_x_")
        or c.startswith("elderly_x_")
    ]

    if name == "full":
        return x
    if name == "no_location":
        return x.drop(columns=["location"], errors="ignore")
    if name == "no_spatial":
        return x.drop(columns=[c for c in SPATIAL if c in x.columns])
    if name == "no_interactions":
        return x.drop(columns=interaction_cols, errors="ignore")
    if name == "no_climate":
        drop = set(BASE_CLIMATE) | set(interaction_cols)
        return x.drop(columns=[c for c in drop if c in x.columns])
    if name == "demographics_time":
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
    raise ValueError(name)


def evaluate(train: pd.DataFrame, name: str, out_dir: Path):
    x = make_view(train, name)
    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()
    cats = categorical_columns(x)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros(len(train), dtype=float)
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups)):
        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.035,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=2026 + fold,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(
            x.iloc[tr_idx],
            y[tr_idx],
            cat_features=cats,
            eval_set=(x.iloc[va_idx], y[va_idx]),
            early_stopping_rounds=60,
            verbose=False,
        )
        pred = model.predict_proba(x.iloc[va_idx])[:, 1]
        oof[va_idx] = pred
        fold_rows.append({"fold": fold, **official_metrics(y[va_idx], pred)})

    metrics = official_metrics(y, oof)
    pd.DataFrame(fold_rows).to_csv(out_dir / f"{name}_folds.csv", index=False)
    pd.DataFrame(
        {ID_COL: train[ID_COL], "target": y, "oof_probability": oof}
    ).to_csv(out_dir / f"{name}_oof.csv", index=False)
    return {"view": name, "n_features": x.shape[1], **metrics}


def main():
    train, _, _ = load_competition_data()
    out_dir = Path("reports/ablations")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in [
        "full",
        "no_location",
        "no_spatial",
        "no_interactions",
        "no_climate",
        "demographics_time",
    ]:
        print(f"Running grouped-CV ablation: {name} ...", flush=True)
        result = evaluate(train, name, out_dir)
        print(result, flush=True)
        rows.append(result)

    summary = pd.DataFrame(rows).sort_values("score", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("\nAblation summary:\n", summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
