from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold

from ablate_features import make_view
from features import ID_COL, TARGET, categorical_columns, load_competition_data
from metrics import official_metrics
from robust_validate import SPLIT_SEEDS


TARGET_ENCODINGS = (
    ("te_year", ("year",), 30.0),
    ("te_age_band", ("age_band",), 40.0),
    ("te_year_age", ("year", "age_band"), 55.0),
    ("te_year_under5", ("year", "is_under5"), 45.0),
    ("te_month_age", ("month", "age_band"), 65.0),
    ("te_gender_age", ("gender", "age_band"), 70.0),
    ("te_zone_age", ("zone", "age_band"), 70.0),
)


def _meta(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["deathdate"], errors="raise")
    age_bins = [-1, 0, 1, 2, 4, 9, 14, 24, 44, 64, 200]
    age_labels = [
        "age0", "age1", "age2", "age3_4", "age5_9", "age10_14",
        "age15_24", "age25_44", "age45_64", "age65plus",
    ]
    out = pd.DataFrame(index=df.index)
    out["year"] = dt.dt.year.astype(int)
    out["month"] = dt.dt.month.astype(int)
    out["age_band"] = pd.cut(df["age"], bins=age_bins, labels=age_labels).astype(str)
    out["is_under5"] = (df["age"] < 5).astype(int)
    out["gender"] = df["gender"].astype(str)
    out["zone"] = df["zone"].astype(str)
    out["location"] = df["location"].astype(str)
    out["latitude"] = df["latitude"].astype(float)
    out["longitude"] = df["longitude"].astype(float)
    return out


def _key(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    if len(columns) == 1:
        return frame[columns[0]].astype(str)
    return frame[list(columns)].astype(str).agg("||".join, axis=1)


def _target_encode_pair(meta_train, meta_apply, y_train, columns, alpha):
    """Leave-one-out encoding for fit rows; train-only encoding for apply rows."""
    y = np.asarray(y_train, dtype=float)
    global_mean = float(y.mean())
    key_train = _key(meta_train, columns)
    key_apply = _key(meta_apply, columns)
    stats = (
        pd.DataFrame({"key": key_train.to_numpy(), "target": y})
        .groupby("key", sort=False)["target"]
        .agg(["sum", "count"])
    )
    sums_train = key_train.map(stats["sum"]).to_numpy(dtype=float)
    counts_train = key_train.map(stats["count"]).to_numpy(dtype=float)
    train_encoded = (sums_train - y + alpha * global_mean) / np.maximum(
        counts_train - 1.0 + alpha, 1.0
    )
    sums_apply = key_apply.map(stats["sum"]).to_numpy(dtype=float)
    counts_apply = key_apply.map(stats["count"]).to_numpy(dtype=float)
    apply_encoded = (sums_apply + alpha * global_mean) / (counts_apply + alpha)
    apply_encoded[np.isnan(apply_encoded)] = global_mean
    return train_encoded, apply_encoded


def _haversine_matrix(lat_a, lon_a, lat_b, lon_b):
    lat1 = np.radians(lat_a)[:, None]
    lon1 = np.radians(lon_a)[:, None]
    lat2 = np.radians(lat_b)[None, :]
    lon2 = np.radians(lon_b)[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _location_table(meta_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
    temp = meta_train[["location", "latitude", "longitude"]].copy()
    temp["target"] = np.asarray(y_train, dtype=float)
    grouped = temp.groupby("location", sort=False).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        target_sum=("target", "sum"),
        target_count=("target", "size"),
    )
    global_mean = float(np.mean(y_train))
    alpha = 20.0
    grouped["target_rate"] = (
        grouped["target_sum"] + alpha * global_mean
    ) / (grouped["target_count"] + alpha)
    return grouped.reset_index()


def _spatial_prior_features(meta_train, meta_apply, y_train, exclude_same_location):
    locations = _location_table(meta_train, y_train)
    global_mean = float(np.mean(y_train))
    d = _haversine_matrix(
        meta_apply["latitude"].to_numpy(dtype=float),
        meta_apply["longitude"].to_numpy(dtype=float),
        locations["latitude"].to_numpy(dtype=float),
        locations["longitude"].to_numpy(dtype=float),
    )
    if exclude_same_location:
        loc_to_col = {loc: i for i, loc in enumerate(locations["location"])}
        for row_i, loc in enumerate(meta_apply["location"].astype(str)):
            col_i = loc_to_col.get(loc)
            if col_i is not None:
                d[row_i, col_i] = np.inf

    rates = locations["target_rate"].to_numpy(dtype=float)
    order = np.argsort(d, axis=1)
    sorted_d = np.take_along_axis(d, order, axis=1)
    sorted_rates = rates[order]
    result = pd.DataFrame(index=meta_apply.index)
    result["geo_nearest_km"] = sorted_d[:, 0]
    for k in (1, 3, 5):
        k_eff = min(k, sorted_rates.shape[1])
        dk = sorted_d[:, :k_eff]
        rk = sorted_rates[:, :k_eff]
        weights = 1.0 / np.maximum(dk, 5.0)
        values = np.sum(weights * rk, axis=1) / np.sum(weights, axis=1)
        values[~np.isfinite(values)] = global_mean
        result[f"geo_knn{k_eff}_target"] = values
    for bandwidth in (40.0, 100.0):
        weights = np.exp(-0.5 * (d / bandwidth) ** 2)
        weights[~np.isfinite(weights)] = 0.0
        denominator = weights.sum(axis=1)
        values = (weights * rates[None, :]).sum(axis=1) / np.maximum(denominator, 1e-12)
        values[denominator < 1e-8] = global_mean
        result[f"geo_kernel_{int(bandwidth)}km_target"] = values
    return result


def _augment(x_train, x_apply, raw_train, raw_apply, y_train, add_te, add_geo):
    out_train = x_train.copy()
    out_apply = x_apply.copy()
    meta_train = _meta(raw_train)
    meta_apply = _meta(raw_apply)
    if add_te:
        for name, columns, alpha in TARGET_ENCODINGS:
            tr_enc, ap_enc = _target_encode_pair(meta_train, meta_apply, y_train, columns, alpha)
            out_train[name] = tr_enc
            out_apply[name] = ap_enc
    if add_geo:
        train_geo = _spatial_prior_features(meta_train, meta_train, y_train, True)
        apply_geo = _spatial_prior_features(meta_train, meta_apply, y_train, False)
        for column in train_geo.columns:
            out_train[column] = train_geo[column].to_numpy()
            out_apply[column] = apply_geo[column].to_numpy()
    return out_train, out_apply


def _fit_predict(x_train, y_train, x_valid, y_valid, seed):
    model = CatBoostClassifier(
        iterations=420,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    model.fit(
        x_train,
        y_train,
        cat_features=categorical_columns(x_train),
        eval_set=(x_valid, y_valid),
        early_stopping_rounds=70,
        verbose=False,
    )
    return model.predict_proba(x_valid)[:, 1]


CONFIGS = {
    "demo_te": ("demographics_time", True, False),
    "demo_te_geo": ("demographics_time", True, True),
    "no_spatial_te": ("no_spatial", True, False),
    "no_spatial_te_geo": ("no_spatial", True, True),
}


def evaluate(train: pd.DataFrame, config_name: str, out_dir: Path):
    view_name, add_te, add_geo = CONFIGS[config_name]
    x_full = make_view(train, view_name)
    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()
    repeated_oof = []
    repeat_rows = []
    for repeat, split_seed in enumerate(SPLIT_SEEDS):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
        oof = np.zeros(len(train), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x_full, y, groups)):
            xtr, xva = _augment(
                x_full.iloc[tr_idx].copy(),
                x_full.iloc[va_idx].copy(),
                train.iloc[tr_idx].copy(),
                train.iloc[va_idx].copy(),
                y[tr_idx],
                add_te,
                add_geo,
            )
            oof[va_idx] = _fit_predict(
                xtr, y[tr_idx], xva, y[va_idx], split_seed + fold
            )
        metrics = official_metrics(y, oof)
        repeat_rows.append(
            {"config": config_name, "repeat": repeat, "split_seed": split_seed, **metrics}
        )
        repeated_oof.append(oof)

    mean_oof = np.vstack(repeated_oof).mean(axis=0)
    metrics = official_metrics(y, mean_oof)
    repeats = pd.DataFrame(repeat_rows)
    summary = {
        "config": config_name,
        "base_view": view_name,
        **metrics,
        "repeat_score_mean": float(repeats["score"].mean()),
        "repeat_score_std": float(repeats["score"].std(ddof=0)),
        "repeat_auc_std": float(repeats["auc"].std(ddof=0)),
        "repeat_f1_std": float(repeats["f1"].std(ddof=0)),
    }
    pd.DataFrame(
        {ID_COL: train[ID_COL], "target": y, "oof_probability": mean_oof}
    ).to_csv(out_dir / f"{config_name}_oof.csv", index=False)
    return summary, repeat_rows


def main():
    train, _, _ = load_competition_data()
    out_dir = Path("reports/structured_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries, all_repeats = [], []
    for config_name in CONFIGS:
        print(f"Structured repeated CV: {config_name} ...", flush=True)
        summary, repeats = evaluate(train, config_name, out_dir)
        print(summary, flush=True)
        summaries.append(summary)
        all_repeats.extend(repeats)
    summary_df = pd.DataFrame(summaries).sort_values(
        ["score", "repeat_score_std"], ascending=[False, True]
    )
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(all_repeats).to_csv(out_dir / "repeat_scores.csv", index=False)
    print("\nStructured validation summary:\n", summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
