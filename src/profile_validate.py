from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ablate_features import make_view
from features import ID_COL, TARGET, categorical_columns, load_competition_data
from interaction_validate import add_interactions
from metrics import official_metrics
from robust_validate import SPLIT_SEEDS


CLIMATE_PROFILE_COLS = [
    "avg_temperature",
    "max_temperature",
    "min_temperature",
    "precipitation",
    "rain_sum_7d",
    "rain_sum_30d",
    "rain_sum_90d",
    "rain_days_30d",
    "max_daily_rain_30d",
    "tavg_7d",
    "tavg_30d",
    "tavg_90d",
    "tmax_30d",
    "tmin_30d",
    "hot_days_30d",
    "temp_range_mean_30d",
    "ndvi_30d",
    "ndvi_90d",
    "elevation",
    "slope",
]


def _profile_key(raw: pd.DataFrame) -> pd.Series:
    # Keep train and test location summaries independent even for the one location
    # name that occurs in both datasets. This mirrors inference: test profiles are
    # computed only from unlabeled test rows, never by mixing in training labels.
    return raw["__domain"].astype(str) + "||" + raw["location"].astype(str)


def build_profile_features(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frame = raw.copy()
    frame["__profile_key"] = _profile_key(frame)
    dt = pd.to_datetime(frame["deathdate"], errors="raise")
    frame["__year"] = dt.dt.year.astype(float)
    frame["__month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12.0)
    frame["__month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12.0)
    frame["__doy_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
    frame["__doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
    frame["__under5"] = (frame["age"] < 5).astype(float)
    frame["__child"] = (frame["age"] < 15).astype(float)
    frame["__elderly"] = (frame["age"] >= 65).astype(float)

    age = frame["age"].astype(float)
    age_bands = {
        "age0": age < 1,
        "age1_4": (age >= 1) & (age < 5),
        "age5_14": (age >= 5) & (age < 15),
        "age15_44": (age >= 15) & (age < 45),
        "age45_64": (age >= 45) & (age < 65),
        "age65plus": age >= 65,
    }
    for name, mask in age_bands.items():
        frame[f"__{name}"] = mask.astype(float)

    grouped = frame.groupby("__profile_key", sort=False)
    demo_table = grouped.agg(
        profile_n_rows=(ID_COL, "size"),
        profile_age_mean=("age", "mean"),
        profile_age_std=("age", "std"),
        profile_age_median=("age", "median"),
        profile_under5_rate=("__under5", "mean"),
        profile_child_rate=("__child", "mean"),
        profile_elderly_rate=("__elderly", "mean"),
        profile_age0_rate=("__age0", "mean"),
        profile_age1_4_rate=("__age1_4", "mean"),
        profile_age5_14_rate=("__age5_14", "mean"),
        profile_age15_44_rate=("__age15_44", "mean"),
        profile_age45_64_rate=("__age45_64", "mean"),
        profile_age65plus_rate=("__age65plus", "mean"),
        profile_month_sin_mean=("__month_sin", "mean"),
        profile_month_cos_mean=("__month_cos", "mean"),
        profile_doy_sin_mean=("__doy_sin", "mean"),
        profile_doy_cos_mean=("__doy_cos", "mean"),
    )

    # Gender composition without assuming organizer-specific category names.
    gender = pd.crosstab(
        frame["__profile_key"], frame["gender"].astype(str), normalize="index"
    )
    gender.columns = [f"profile_gender_rate__{str(c)}" for c in gender.columns]
    demo_table = demo_table.join(gender, how="left")

    time_table = grouped.agg(
        profile_year_mean=("__year", "mean"),
        profile_year_std=("__year", "std"),
        profile_year_min=("__year", "min"),
        profile_year_max=("__year", "max"),
    )
    time_table["profile_year_span"] = (
        time_table["profile_year_max"] - time_table["profile_year_min"]
    )

    available_climate = [c for c in CLIMATE_PROFILE_COLS if c in frame.columns]
    climate_parts = []
    for column in available_climate:
        stats = grouped[column].agg(["mean", "std", "median"])
        stats.columns = [
            f"profile_{column}_mean",
            f"profile_{column}_std",
            f"profile_{column}_median",
        ]
        climate_parts.append(stats)
    climate_table = pd.concat(climate_parts, axis=1) if climate_parts else pd.DataFrame()

    keys = frame["__profile_key"]

    def expand(table: pd.DataFrame) -> pd.DataFrame:
        out = table.reindex(keys.to_numpy()).reset_index(drop=True)
        out.index = raw.index
        return out

    demo = expand(demo_table)
    time = expand(time_table)
    climate_abs = expand(climate_table) if not climate_table.empty else pd.DataFrame(index=raw.index)

    relative = pd.DataFrame(index=raw.index)
    for column in available_climate:
        mean_col = f"profile_{column}_mean"
        std_col = f"profile_{column}_std"
        mean = climate_abs[mean_col]
        std = climate_abs[std_col].fillna(0.0)
        relative[f"profile_rel_{column}_diff"] = frame[column].to_numpy(dtype=float) - mean.to_numpy(dtype=float)
        relative[f"profile_rel_{column}_z"] = (
            frame[column].to_numpy(dtype=float) - mean.to_numpy(dtype=float)
        ) / (std.to_numpy(dtype=float) + 1e-3)

    # Individual-vs-location demographic deviations are often more transferable
    # than absolute site identity.
    demo["profile_age_diff"] = frame["age"].to_numpy(dtype=float) - demo["profile_age_mean"].to_numpy(dtype=float)
    time["profile_year_diff"] = frame["__year"].to_numpy(dtype=float) - time["profile_year_mean"].to_numpy(dtype=float)

    return {
        "case_mix": demo,
        "time": time,
        "climate_relative": relative,
        "climate_absolute": climate_abs,
    }


def make_config_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw = train.drop(columns=[TARGET]).copy()
    test_raw = test.copy()
    train_raw["__domain"] = "train"
    test_raw["__domain"] = "test"
    raw_all = pd.concat([train_raw, test_raw], ignore_index=True)
    profiles = build_profile_features(raw_all)

    base_train = add_interactions(make_view(train, "demographics_time"), "all").reset_index(drop=True)
    test_dummy = test.copy()
    test_dummy[TARGET] = 0
    base_test = add_interactions(make_view(test_dummy, "demographics_time"), "all").reset_index(drop=True)

    n_train = len(train)
    additions: list[pd.DataFrame] = []
    if config in {"case_mix", "case_mix_time", "case_mix_climate_relative", "full_profile"}:
        additions.append(profiles["case_mix"])
    if config in {"case_mix_time", "case_mix_climate_relative", "full_profile"}:
        additions.append(profiles["time"])
    if config in {"case_mix_climate_relative", "full_profile"}:
        additions.append(profiles["climate_relative"])
    if config == "full_profile":
        additions.append(profiles["climate_absolute"])

    if not additions:
        return base_train, base_test

    extra = pd.concat(additions, axis=1).reset_index(drop=True)
    train_extra = extra.iloc[:n_train].reset_index(drop=True)
    test_extra = extra.iloc[n_train:].reset_index(drop=True)
    return (
        pd.concat([base_train, train_extra], axis=1),
        pd.concat([base_test, test_extra], axis=1),
    )


def _fit_target_cv(train: pd.DataFrame, x: pd.DataFrame):
    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()
    repeated_oof = []
    repeat_rows = []
    for repeat, split_seed in enumerate(SPLIT_SEEDS):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
        oof = np.zeros(len(train), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups)):
            model = CatBoostClassifier(
                iterations=420,
                depth=6,
                learning_rate=0.03,
                l2_leaf_reg=6.0,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=split_seed + fold,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
                max_ctr_complexity=1,
            )
            model.fit(
                x.iloc[tr_idx],
                y[tr_idx],
                cat_features=categorical_columns(x),
                eval_set=(x.iloc[va_idx], y[va_idx]),
                early_stopping_rounds=70,
                verbose=False,
            )
            oof[va_idx] = model.predict_proba(x.iloc[va_idx])[:, 1]
        metrics = official_metrics(y, oof)
        repeat_rows.append({"repeat": repeat, "split_seed": split_seed, **metrics})
        repeated_oof.append(oof)

    mean_oof = np.vstack(repeated_oof).mean(axis=0)
    aggregate = official_metrics(y, mean_oof)
    repeat_df = pd.DataFrame(repeat_rows)
    return {
        **aggregate,
        "repeat_score_mean": float(repeat_df["score"].mean()),
        "repeat_score_std": float(repeat_df["score"].std(ddof=0)),
        "repeat_auc_std": float(repeat_df["auc"].std(ddof=0)),
        "repeat_f1_std": float(repeat_df["f1"].std(ddof=0)),
    }, repeat_rows, mean_oof


def _shift_auc(train_x: pd.DataFrame, test_x: pd.DataFrame) -> float:
    x = pd.concat([train_x, test_x], ignore_index=True)
    domain = np.concatenate(
        [np.zeros(len(train_x), dtype=int), np.ones(len(test_x), dtype=int)]
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros(len(x), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, domain)):
        model = CatBoostClassifier(
            iterations=240,
            depth=5,
            learning_rate=0.05,
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
            cat_features=categorical_columns(x),
            eval_set=(x.iloc[va_idx], domain[va_idx]),
            early_stopping_rounds=50,
            verbose=False,
        )
        oof[va_idx] = model.predict_proba(x.iloc[va_idx])[:, 1]
    return float(roc_auc_score(domain, oof))


def main():
    train, test, _ = load_competition_data()
    out_dir = Path("reports/profile_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        "reference",
        "case_mix",
        "case_mix_time",
        "case_mix_climate_relative",
        "full_profile",
    ]
    summaries = []
    all_repeats = []
    for config in configs:
        print(f"Profile validation: {config} ...", flush=True)
        x_train, x_test = make_config_features(train, test, config)
        metrics, repeats, oof = _fit_target_cv(train, x_train)
        shift = _shift_auc(x_train, x_test)
        summary = {
            "config": config,
            "n_features": x_train.shape[1],
            "shift_auc": shift,
            **metrics,
        }
        print(summary, flush=True)
        summaries.append(summary)
        for row in repeats:
            all_repeats.append({"config": config, **row})
        pd.DataFrame(
            {ID_COL: train[ID_COL], "target": train[TARGET], "oof_probability": oof}
        ).to_csv(out_dir / f"{config}_oof.csv", index=False)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["score", "shift_auc"], ascending=[False, True]
    )
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(all_repeats).to_csv(out_dir / "repeat_scores.csv", index=False)
    print("\nProfile validation summary:\n", summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
