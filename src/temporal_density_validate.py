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


ROLLING_WINDOWS = (7, 14, 28, 56, 84, 168)
CONTEXT_BASES = ("all", "u5", "age5_14", "age15_64", "age65plus", "male")


def _event_frame(raw: pd.DataFrame) -> pd.DataFrame:
    date = pd.to_datetime(raw["deathdate"], errors="raise").dt.normalize()
    age = raw["age"].astype(float)
    gender = raw["gender"].astype(str).str.lower()
    return pd.DataFrame(
        {
            "date": date,
            "zone": raw["zone"].astype(str),
            "all": 1.0,
            "u5": (age < 5).astype(float),
            "age5_14": ((age >= 5) & (age < 15)).astype(float),
            "age15_64": ((age >= 15) & (age < 65)).astype(float),
            "age65plus": (age >= 65).astype(float),
            "male": gender.str.startswith("m").astype(float),
        },
        index=raw.index,
    )


def _daily_table(events: pd.DataFrame) -> pd.DataFrame:
    daily = events.groupby("date", sort=True)[list(CONTEXT_BASES)].sum()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_index, fill_value=0.0)


def _rolling_context(daily: pd.DataFrame, prefix: str, symmetric: bool) -> pd.DataFrame:
    features = pd.DataFrame(index=daily.index)
    for days in ROLLING_WINDOWS:
        trailing = daily[list(CONTEXT_BASES)].rolling(days, min_periods=1).sum()
        for base in CONTEXT_BASES:
            features[f"{prefix}_{base}_{days}d"] = trailing[base]

    if symmetric:
        for half_window in (7, 14, 28):
            width = 2 * half_window + 1
            centered = daily[list(CONTEXT_BASES)].rolling(
                width, min_periods=1, center=True
            ).sum()
            for base in CONTEXT_BASES:
                features[f"{prefix}_{base}_pm{half_window}d"] = centered[base]

    # Outbreak / acceleration signals. These compare recent event intensity with
    # the longer local temporal background and are scale-free.
    eps = 0.25
    for short, long in ((7, 28), (14, 56), (28, 84), (28, 168), (56, 168)):
        short_rate = features[f"{prefix}_all_{short}d"] / short
        long_rate = features[f"{prefix}_all_{long}d"] / long
        features[f"{prefix}_all_rate_ratio_{short}_{long}"] = (
            (short_rate + eps / short) / (long_rate + eps / long)
        )

    for days in (14, 28, 56, 84):
        denom = features[f"{prefix}_all_{days}d"].clip(lower=1.0)
        for base in ("u5", "age5_14", "age15_64", "age65plus", "male"):
            features[f"{prefix}_{base}_share_{days}d"] = (
                features[f"{prefix}_{base}_{days}d"] / denom
            )

    # Seasonal burst z-score: compare the 28-day event load with the same
    # week-of-year across other years. It intentionally uses no labels.
    roll28 = features[f"{prefix}_all_28d"]
    iso_week = pd.Index(features.index.isocalendar().week.astype(int), name="iso_week")
    temp = pd.DataFrame({"value": roll28.to_numpy(), "iso_week": iso_week.to_numpy()})
    week_mean = temp.groupby("iso_week")["value"].transform("mean").to_numpy()
    week_std = temp.groupby("iso_week")["value"].transform("std").fillna(0.0).to_numpy()
    features[f"{prefix}_all_28d_season_z"] = (roll28.to_numpy() - week_mean) / (
        week_std + 1.0
    )
    return features


def _subtract_self(
    mapped: pd.DataFrame,
    events: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    out = mapped.copy()
    for column in out.columns:
        if not column.startswith(prefix + "_"):
            continue
        for base in CONTEXT_BASES:
            marker = f"{prefix}_{base}_"
            if not column.startswith(marker):
                continue
            suffix = column[len(marker) :]
            is_direct_window = suffix.endswith("d") and suffix[:-1].isdigit()
            is_centered_window = (
                suffix.startswith("pm")
                and suffix.endswith("d")
                and suffix[2:-1].isdigit()
            )
            if is_direct_window or is_centered_window:
                out[column] = (out[column] - events[base].to_numpy()).clip(lower=0.0)
                break
    return out


def build_global_context(raw: pd.DataFrame, symmetric: bool) -> pd.DataFrame:
    events = _event_frame(raw)
    daily = _daily_table(events)
    table = _rolling_context(daily, "mortctx", symmetric=symmetric)
    mapped = table.reindex(events["date"].to_numpy()).reset_index(drop=True)
    mapped.index = raw.index
    return _subtract_self(mapped, events, "mortctx")


def build_zone_context(raw: pd.DataFrame) -> pd.DataFrame:
    events = _event_frame(raw)
    result = pd.DataFrame(index=raw.index)
    for zone, idx in events.groupby("zone", sort=False).groups.items():
        zone_events = events.loc[idx]
        daily = _daily_table(zone_events)
        # Zone context is intentionally leaner to limit site-identification risk.
        table = _rolling_context(daily, "zonectx", symmetric=False)
        keep = [
            c
            for c in table.columns
            if any(
                token in c
                for token in (
                    "_all_14d",
                    "_all_28d",
                    "_all_56d",
                    "_all_84d",
                    "_u5_28d",
                    "_age65plus_28d",
                    "_all_rate_ratio_",
                    "_all_28d_season_z",
                )
            )
        ]
        mapped = table[keep].reindex(zone_events["date"].to_numpy()).reset_index(drop=True)
        mapped.index = idx
        mapped = _subtract_self(mapped, zone_events, "zonectx")
        result.loc[idx, mapped.columns] = mapped
    return result.astype(float)


def _fit_predict(x_train, y_train, x_valid, y_valid, seed: int) -> np.ndarray:
    model = CatBoostClassifier(
        iterations=460,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        max_ctr_complexity=1,
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


def _target_cv(train: pd.DataFrame, x: pd.DataFrame, key: str, out_dir: Path):
    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()
    repeated_oof = []
    repeat_rows = []
    for repeat, split_seed in enumerate(SPLIT_SEEDS):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
        oof = np.zeros(len(train), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups)):
            oof[va_idx] = _fit_predict(
                x.iloc[tr_idx],
                y[tr_idx],
                x.iloc[va_idx],
                y[va_idx],
                seed=split_seed + fold,
            )
        metrics = official_metrics(y, oof)
        repeat_rows.append(
            {"config": key, "repeat": repeat, "split_seed": split_seed, **metrics}
        )
        repeated_oof.append(oof)

    mean_oof = np.vstack(repeated_oof).mean(axis=0)
    aggregate = official_metrics(y, mean_oof)
    repeat_df = pd.DataFrame(repeat_rows)
    pd.DataFrame(
        {ID_COL: train[ID_COL], "target": y, "oof_probability": mean_oof}
    ).to_csv(out_dir / f"{key}_oof.csv", index=False)
    return {
        **aggregate,
        "repeat_score_mean": float(repeat_df["score"].mean()),
        "repeat_score_std": float(repeat_df["score"].std(ddof=0)),
        "repeat_auc_std": float(repeat_df["auc"].std(ddof=0)),
        "repeat_f1_std": float(repeat_df["f1"].std(ddof=0)),
    }, repeat_rows


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
    out_dir = Path("reports/temporal_density_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    test_raw = test.reset_index(drop=True)
    universe = pd.concat([train_raw, test_raw], ignore_index=True)

    print("Building target-free mortality-event context from Train+Test covariates ...", flush=True)
    global_trailing = build_global_context(universe, symmetric=False)
    global_symmetric = build_global_context(universe, symmetric=True)
    zone_context = build_zone_context(universe)

    base_train = add_interactions(make_view(train, "demographics_time"), "all").reset_index(drop=True)
    test_dummy = test.copy()
    test_dummy[TARGET] = 0
    base_test = add_interactions(make_view(test_dummy, "demographics_time"), "all").reset_index(drop=True)

    n_train = len(train)
    context_sets = {
        "reference": (pd.DataFrame(index=range(n_train)), pd.DataFrame(index=range(len(test)))),
        "global_trailing": (
            global_trailing.iloc[:n_train].reset_index(drop=True),
            global_trailing.iloc[n_train:].reset_index(drop=True),
        ),
        "global_symmetric": (
            global_symmetric.iloc[:n_train].reset_index(drop=True),
            global_symmetric.iloc[n_train:].reset_index(drop=True),
        ),
        "global_zone": (
            pd.concat(
                [
                    global_trailing.iloc[:n_train].reset_index(drop=True),
                    zone_context.iloc[:n_train].reset_index(drop=True),
                ],
                axis=1,
            ),
            pd.concat(
                [
                    global_trailing.iloc[n_train:].reset_index(drop=True),
                    zone_context.iloc[n_train:].reset_index(drop=True),
                ],
                axis=1,
            ),
        ),
    }

    summaries = []
    all_repeats = []
    for key, (ctx_train, ctx_test) in context_sets.items():
        x_train = pd.concat([base_train, ctx_train], axis=1)
        x_test = pd.concat([base_test, ctx_test], axis=1)
        print(f"Temporal-density CV: {key} ({x_train.shape[1]} features) ...", flush=True)
        metrics, repeats = _target_cv(train, x_train, key, out_dir)
        shift = _shift_auc(x_train, x_test)
        row = {
            "config": key,
            "n_features": x_train.shape[1],
            "shift_auc": shift,
            **metrics,
        }
        print(row, flush=True)
        summaries.append(row)
        all_repeats.extend(repeats)

    summary = pd.DataFrame(summaries).sort_values(
        ["score", "shift_auc"], ascending=[False, True]
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(all_repeats).to_csv(out_dir / "repeat_scores.csv", index=False)
    print("\nTemporal-density summary:\n", summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
