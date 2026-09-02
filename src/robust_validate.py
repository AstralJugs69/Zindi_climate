from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold

from ablate_features import make_view
from evaluate_model_suite import model_factory, preprocess_fold
from features import ID_COL, TARGET, categorical_columns, load_competition_data
from metrics import official_metrics


VIEWS = ("demographics_time", "no_spatial")
MODELS = ("catboost", "lightgbm", "xgboost", "logistic")
SPLIT_SEEDS = (2026, 73, 31415)


def _fit_predict(model_name, x_train, y_train, x_valid, y_valid, seed):
    if model_name == "catboost":
        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.035,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        cats = categorical_columns(x_train)
        model.fit(
            x_train,
            y_train,
            cat_features=cats,
            eval_set=(x_valid, y_valid),
            early_stopping_rounds=60,
            verbose=False,
        )
        return model.predict_proba(x_valid)[:, 1]

    z_train, z_valid = preprocess_fold(
        x_train,
        x_valid,
        scale_numeric=(model_name == "logistic"),
    )
    model = model_factory(model_name, seed)
    model.fit(z_train, y_train)
    return model.predict_proba(z_valid)[:, 1]


def evaluate_candidate(train, view_name, model_name, out_dir):
    x = make_view(train, view_name)
    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()
    repeated_oof = []
    repeat_rows = []

    for repeat_idx, split_seed in enumerate(SPLIT_SEEDS):
        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=split_seed,
        )
        oof = np.zeros(len(train), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups)):
            oof[va_idx] = _fit_predict(
                model_name,
                x.iloc[tr_idx],
                y[tr_idx],
                x.iloc[va_idx],
                y[va_idx],
                seed=split_seed + fold,
            )

        metrics = official_metrics(y, oof)
        repeat_rows.append(
            {
                "view": view_name,
                "model": model_name,
                "repeat": repeat_idx,
                "split_seed": split_seed,
                **metrics,
            }
        )
        repeated_oof.append(oof)

    repeated = np.vstack(repeated_oof)
    mean_oof = repeated.mean(axis=0)
    aggregate_metrics = official_metrics(y, mean_oof)
    repeat_df = pd.DataFrame(repeat_rows)

    summary = {
        "view": view_name,
        "model": model_name,
        "n_features": x.shape[1],
        **aggregate_metrics,
        "repeat_score_mean": float(repeat_df["score"].mean()),
        "repeat_score_std": float(repeat_df["score"].std(ddof=0)),
        "repeat_f1_std": float(repeat_df["f1"].std(ddof=0)),
        "repeat_auc_std": float(repeat_df["auc"].std(ddof=0)),
    }

    key = f"{view_name}__{model_name}"
    pd.DataFrame(
        {ID_COL: train[ID_COL], "target": y, "oof_probability": mean_oof}
    ).to_csv(out_dir / f"{key}_oof.csv", index=False)
    return summary, mean_oof, repeat_rows


def blend_search(y, probabilities):
    keys = sorted(probabilities)
    rows = []
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            for left_weight in (0.25, 0.50, 0.75):
                p = (
                    left_weight * probabilities[left]
                    + (1.0 - left_weight) * probabilities[right]
                )
                rows.append(
                    {
                        "blend": f"{left_weight:.2f}_{left}+{1-left_weight:.2f}_{right}",
                        **official_metrics(y, p),
                    }
                )
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def main():
    train, _, _ = load_competition_data()
    out_dir = Path("reports/robust_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    repeat_rows = []
    probabilities = {}

    for view_name in VIEWS:
        for model_name in MODELS:
            print(f"Repeated grouped CV: {view_name} / {model_name} ...", flush=True)
            summary, oof, rows = evaluate_candidate(
                train, view_name, model_name, out_dir
            )
            print(summary, flush=True)
            summaries.append(summary)
            repeat_rows.extend(rows)
            probabilities[f"{view_name}__{model_name}"] = oof

    summary_df = pd.DataFrame(summaries).sort_values(
        ["score", "repeat_score_std"], ascending=[False, True]
    )
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(out_dir / "repeat_scores.csv", index=False)

    y = train[TARGET].astype(int).to_numpy()
    blends = blend_search(y, probabilities)
    blends.to_csv(out_dir / "blend_search.csv", index=False)

    print("\nRobust validation summary:\n", summary_df.to_string(index=False), flush=True)
    print("\nTop robust blends:\n", blends.head(15).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
