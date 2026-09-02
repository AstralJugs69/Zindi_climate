from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from features import ID_COL, TARGET, engineer_features, feature_columns, load_competition_data
from metrics import official_metrics


def make_model(seed: int, class_weights=None):
    return CatBoostClassifier(
        iterations=650,
        depth=6,
        learning_rate=0.035,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        class_weights=class_weights,
    )


def run_cv(train: pd.DataFrame, split_name: str, include_location: bool, class_weights=None):
    X = engineer_features(train.drop(columns=[TARGET]), include_location=include_location)
    y = train[TARGET].astype(int).to_numpy()
    cols = feature_columns(X)
    X = X[cols]
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if split_name == "stratified":
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
        folds = splitter.split(X, y)
    elif split_name == "location_group":
        groups = train["location"].astype(str).to_numpy()
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
        folds = splitter.split(X, y, groups=groups)
    else:
        raise ValueError(split_name)

    oof = np.zeros(len(train), dtype=float)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = make_model(seed=2026 + fold, class_weights=class_weights)
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            cat_features=cat_cols,
            eval_set=(X.iloc[va_idx], y[va_idx]),
            early_stopping_rounds=80,
            verbose=False,
        )
        pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = pred
        m = official_metrics(y[va_idx], pred)
        m.update({"fold": fold, "n": len(va_idx), "best_iteration": model.get_best_iteration()})
        rows.append(m)

    overall = official_metrics(y, oof)
    fold_df = pd.DataFrame(rows)
    return overall, fold_df, oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="reports/baselines")
    args = parser.parse_args()

    train, _, _ = load_competition_data(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("random_no_location", "stratified", False, None),
        ("random_with_location", "stratified", True, None),
        ("group_no_location", "location_group", False, None),
        ("group_with_location", "location_group", True, None),
        ("group_no_location_balanced", "location_group", False, [1.0, 1099 / 2047]),
    ]

    summary = []
    for name, split_name, include_location, class_weights in experiments:
        print(f"Running {name} ...", flush=True)
        overall, folds, oof = run_cv(train, split_name, include_location, class_weights)
        print(name, overall, flush=True)
        row = {"experiment": name, **overall, "fold_score_std": float(folds.score.std()), "fold_f1_std": float(folds.f1.std()), "fold_auc_std": float(folds.auc.std())}
        summary.append(row)
        folds.to_csv(out_dir / f"{name}_folds.csv", index=False)
        pd.DataFrame({ID_COL: train[ID_COL], "target": train[TARGET], "oof_probability": oof}).to_csv(out_dir / f"{name}_oof.csv", index=False)

    summary_df = pd.DataFrame(summary).sort_values("score", ascending=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary_df.to_dict(orient="records"), indent=2), encoding="utf-8")
    print("\n", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
