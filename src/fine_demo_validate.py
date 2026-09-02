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


def add_fine_demographics(x: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = add_interactions(x, "all")
    if mode == "reference":
        return out

    age = out["age"].astype(float).clip(lower=0)
    year = out["year"].astype(int)

    if mode in {"fine_age", "rich"}:
        out["age_exact_cat"] = age.round().astype(int).astype(str)
        out["age_2yr_cat"] = (np.floor(age / 2.0) * 2).astype(int).astype(str)
        out["age_5yr_cat"] = (np.floor(age / 5.0) * 5).astype(int).astype(str)
        out["age_10yr_cat"] = (np.floor(age / 10.0) * 10).astype(int).astype(str)
        out["age_sqrt"] = np.sqrt(age)
        out["age_log2"] = np.log2(age + 1.0)

    if mode in {"cohort_calendar", "rich"}:
        birth_year = year - age.round().astype(int)
        out["birth_year"] = birth_year
        out["birth_5yr_cat"] = (np.floor(birth_year / 5.0) * 5).astype(int).astype(str)
        out["birth_10yr_cat"] = (np.floor(birth_year / 10.0) * 10).astype(int).astype(str)
        out["year_cat"] = year.astype(str)
        out["month_cat"] = out["month"].astype(int).astype(str)
        out["week_cat"] = out["week_of_year"].astype(int).astype(str)
        out["quarter_cat"] = (((out["month"].astype(int) - 1) // 3) + 1).astype(str)

        month = out["month"].astype(int)
        season = np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
            ["DJF", "MAM", "JJA"],
            default="SON",
        )
        out["season4"] = pd.Series(season, index=out.index, dtype="object")

    if mode == "rich":
        pairs = [
            ("age_5yr_cat", "year_cat"),
            ("age_5yr_cat", "zone"),
            ("age_5yr_cat", "season4"),
            ("age_5yr_cat", "gender"),
            ("birth_10yr_cat", "zone"),
            ("birth_10yr_cat", "gender"),
            ("age_band", "season4"),
        ]
        for left, right in pairs:
            out[f"{left}__x__{right}"] = out[left].astype(str) + "||" + out[right].astype(str)

    return out


def _fit_predict(x_train, y_train, x_valid, y_valid, seed: int):
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


def evaluate_target(train: pd.DataFrame, mode: str, out_dir: Path):
    x = add_fine_demographics(make_view(train, "demographics_time"), mode)
    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()
    repeated_oof = []
    repeat_rows = []

    for repeat, split_seed in enumerate(SPLIT_SEEDS):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
        oof = np.zeros(len(train), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups)):
            oof[va_idx] = _fit_predict(
                x.iloc[tr_idx], y[tr_idx], x.iloc[va_idx], y[va_idx], split_seed + fold
            )
        metrics = official_metrics(y, oof)
        repeat_rows.append({"config": mode, "repeat": repeat, "split_seed": split_seed, **metrics})
        repeated_oof.append(oof)

    mean_oof = np.vstack(repeated_oof).mean(axis=0)
    aggregate = official_metrics(y, mean_oof)
    repeat_df = pd.DataFrame(repeat_rows)
    pd.DataFrame({ID_COL: train[ID_COL], "target": y, "oof_probability": mean_oof}).to_csv(
        out_dir / f"{mode}_oof.csv", index=False
    )
    return {
        "config": mode,
        "n_features": x.shape[1],
        **aggregate,
        "repeat_score_mean": float(repeat_df["score"].mean()),
        "repeat_score_std": float(repeat_df["score"].std(ddof=0)),
        "repeat_auc_std": float(repeat_df["auc"].std(ddof=0)),
        "repeat_f1_std": float(repeat_df["f1"].std(ddof=0)),
    }, repeat_rows


def shift_auc(train: pd.DataFrame, test: pd.DataFrame, mode: str) -> float:
    train_x = add_fine_demographics(make_view(train, "demographics_time"), mode)
    test_dummy = test.copy()
    test_dummy[TARGET] = 0
    test_x = add_fine_demographics(make_view(test_dummy, "demographics_time"), mode)
    x = pd.concat([train_x, test_x], ignore_index=True)
    domain = np.concatenate([np.zeros(len(train_x), dtype=int), np.ones(len(test_x), dtype=int)])
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
    out_dir = Path("reports/fine_demo_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = ["reference", "fine_age", "cohort_calendar", "rich"]
    summaries, repeats = [], []
    for mode in configs:
        print(f"Fine demographic CV: {mode} ...", flush=True)
        summary, rows = evaluate_target(train, mode, out_dir)
        summary["shift_auc"] = shift_auc(train, test, mode)
        print(summary, flush=True)
        summaries.append(summary)
        repeats.extend(rows)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["score", "repeat_score_std"], ascending=[False, True]
    )
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(repeats).to_csv(out_dir / "repeat_scores.csv", index=False)
    print("\nFine demographic summary:\n", summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
