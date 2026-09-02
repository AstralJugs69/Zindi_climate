from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ablate_features import make_view
from features import ID_COL, TARGET, categorical_columns, load_competition_data
from metrics import official_metrics
from robust_validate import SPLIT_SEEDS


def add_interactions(x: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = x.copy()
    if mode == "base":
        return out

    if mode in {"categorical", "all"}:
        pairs = [
            ("age_band", "gender"),
            ("age_band", "zone"),
            ("age_band", "year"),
            ("age_band", "month"),
            ("zone", "year"),
            ("zone", "month"),
            ("gender", "year"),
            ("gender", "month"),
        ]
        for left, right in pairs:
            if left in out.columns and right in out.columns:
                out[f"{left}__x__{right}"] = (
                    out[left].astype(str) + "||" + out[right].astype(str)
                )

        if all(c in out.columns for c in ["age_band", "zone", "month"]):
            out["age_band__x__zone__x__month"] = (
                out["age_band"].astype(str)
                + "||"
                + out["zone"].astype(str)
                + "||"
                + out["month"].astype(str)
            )

    if mode in {"numeric", "all"}:
        year_center = out["year"].astype(float) - 2014.0
        age = out["age"].astype(float)
        out["year_center"] = year_center
        out["year_center_sq"] = year_center**2
        out["age_cubic_scaled"] = (age / 50.0) ** 3
        out["age_x_year_center"] = age * year_center
        out["under5_x_year_center"] = out["is_under5"].astype(float) * year_center
        out["child_x_year_center"] = out["is_child"].astype(float) * year_center
        out["elderly_x_year_center"] = out["is_elderly"].astype(float) * year_center
        for flag in ("is_infant", "is_under5", "is_child", "is_elderly"):
            for seasonal in ("month_sin", "month_cos", "doy_sin", "doy_cos"):
                if flag in out.columns and seasonal in out.columns:
                    out[f"{flag}_x_{seasonal}"] = (
                        out[flag].astype(float) * out[seasonal].astype(float)
                    )

    return out


def _fit_predict(x_train, y_train, x_valid, y_valid, seed: int, max_ctr_complexity: int):
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
        max_ctr_complexity=max_ctr_complexity,
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


def evaluate_target(train: pd.DataFrame, mode: str, max_ctr_complexity: int, out_dir: Path):
    x = add_interactions(make_view(train, "demographics_time"), mode)
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
                max_ctr_complexity=max_ctr_complexity,
            )
        metrics = official_metrics(y, oof)
        repeat_rows.append({
            "mode": mode,
            "max_ctr_complexity": max_ctr_complexity,
            "repeat": repeat,
            "split_seed": split_seed,
            **metrics,
        })
        repeated_oof.append(oof)

    mean_oof = np.vstack(repeated_oof).mean(axis=0)
    aggregate = official_metrics(y, mean_oof)
    repeat_df = pd.DataFrame(repeat_rows)
    key = f"{mode}_ctr{max_ctr_complexity}"
    pd.DataFrame({ID_COL: train[ID_COL], "target": y, "oof_probability": mean_oof}).to_csv(
        out_dir / f"{key}_oof.csv", index=False
    )
    return {
        "config": key,
        "mode": mode,
        "max_ctr_complexity": max_ctr_complexity,
        "n_features": x.shape[1],
        **aggregate,
        "repeat_score_mean": float(repeat_df["score"].mean()),
        "repeat_score_std": float(repeat_df["score"].std(ddof=0)),
        "repeat_auc_std": float(repeat_df["auc"].std(ddof=0)),
        "repeat_f1_std": float(repeat_df["f1"].std(ddof=0)),
    }, repeat_rows


def shift_auc(train: pd.DataFrame, test: pd.DataFrame, mode: str) -> float:
    train_x = add_interactions(make_view(train, "demographics_time"), mode)
    test_dummy = test.copy()
    test_dummy[TARGET] = 0
    test_x = add_interactions(make_view(test_dummy, "demographics_time"), mode)
    x = pd.concat([train_x, test_x], ignore_index=True)
    domain = np.concatenate([np.zeros(len(train_x), dtype=int), np.ones(len(test_x), dtype=int)])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros(len(x), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, domain)):
        model = CatBoostClassifier(
            iterations=220,
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
    out_dir = Path("reports/interaction_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("base", 1),
        ("categorical", 1),
        ("numeric", 1),
        ("all", 1),
        ("categorical", 2),
        ("all", 2),
    ]
    summaries, repeats = [], []
    shift_cache: dict[str, float] = {}
    for mode, ctr in configs:
        print(f"Interaction CV: mode={mode}, max_ctr_complexity={ctr} ...", flush=True)
        summary, rows = evaluate_target(train, mode, ctr, out_dir)
        if mode not in shift_cache:
            shift_cache[mode] = shift_auc(train, test, mode)
        summary["shift_auc"] = shift_cache[mode]
        print(summary, flush=True)
        summaries.append(summary)
        repeats.extend(rows)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["score", "repeat_score_std"], ascending=[False, True]
    )
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(repeats).to_csv(out_dir / "repeat_scores.csv", index=False)
    print("\nInteraction validation summary:\n", summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
