from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold

from ablate_features import make_view
from features import ID_COL, TARGET, categorical_columns, load_competition_data
from interaction_validate import add_interactions
from metrics import official_metrics
from robust_validate import SPLIT_SEEDS


# Manual, auditable search over materially different CatBoost regimes. This is
# intentionally not AutoML: every configuration is declared up front and is
# evaluated under the same repeated location-group CV used elsewhere in the repo.
CONFIGS = {
    "plain_d5_reg": dict(
        iterations=850,
        depth=5,
        learning_rate=0.025,
        l2_leaf_reg=8.0,
        random_strength=0.5,
        bagging_temperature=0.5,
        boosting_type="Plain",
    ),
    "plain_d6_reference": dict(
        iterations=650,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=6.0,
        random_strength=1.0,
        bagging_temperature=1.0,
        boosting_type="Plain",
    ),
    "plain_d7_reg": dict(
        iterations=800,
        depth=7,
        learning_rate=0.02,
        l2_leaf_reg=12.0,
        random_strength=1.5,
        bagging_temperature=1.0,
        boosting_type="Plain",
    ),
    "ordered_d4_conservative": dict(
        iterations=1100,
        depth=4,
        learning_rate=0.02,
        l2_leaf_reg=12.0,
        random_strength=0.5,
        bagging_temperature=0.5,
        boosting_type="Ordered",
    ),
    "ordered_d5_reg": dict(
        iterations=900,
        depth=5,
        learning_rate=0.025,
        l2_leaf_reg=8.0,
        random_strength=0.5,
        bagging_temperature=0.5,
        boosting_type="Ordered",
    ),
    "ordered_d6_reg": dict(
        iterations=800,
        depth=6,
        learning_rate=0.025,
        l2_leaf_reg=10.0,
        random_strength=1.0,
        bagging_temperature=0.75,
        boosting_type="Ordered",
    ),
    "ordered_d7_reg": dict(
        iterations=750,
        depth=7,
        learning_rate=0.02,
        l2_leaf_reg=14.0,
        random_strength=1.0,
        bagging_temperature=1.0,
        boosting_type="Ordered",
    ),
}


def _fit_predict(x_train, y_train, x_valid, y_valid, seed: int, params: dict):
    model = CatBoostClassifier(
        **params,
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
        early_stopping_rounds=90,
        verbose=False,
    )
    return model.predict_proba(x_valid)[:, 1], int(model.get_best_iteration())


def evaluate(train: pd.DataFrame, name: str, params: dict, out_dir: Path):
    x = add_interactions(make_view(train, "demographics_time"), "all")
    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()

    repeat_oof = []
    repeat_rows = []
    fold_rows = []
    for repeat, split_seed in enumerate(SPLIT_SEEDS):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
        oof = np.zeros(len(train), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups)):
            p, best_iter = _fit_predict(
                x.iloc[tr_idx],
                y[tr_idx],
                x.iloc[va_idx],
                y[va_idx],
                split_seed + fold,
                params,
            )
            oof[va_idx] = p
            fold_rows.append(
                {
                    "config": name,
                    "repeat": repeat,
                    "split_seed": split_seed,
                    "fold": fold,
                    "best_iteration": best_iter,
                    **official_metrics(y[va_idx], p),
                }
            )
        metrics = official_metrics(y, oof)
        repeat_rows.append(
            {"config": name, "repeat": repeat, "split_seed": split_seed, **metrics}
        )
        repeat_oof.append(oof)

    mean_oof = np.vstack(repeat_oof).mean(axis=0)
    aggregate = official_metrics(y, mean_oof)
    repeats = pd.DataFrame(repeat_rows)
    folds = pd.DataFrame([r for r in fold_rows if r["config"] == name])
    pd.DataFrame(
        {ID_COL: train[ID_COL], "target": y, "oof_probability": mean_oof}
    ).to_csv(out_dir / f"{name}_oof.csv", index=False)

    return {
        "config": name,
        **aggregate,
        "repeat_score_mean": float(repeats["score"].mean()),
        "repeat_score_std": float(repeats["score"].std(ddof=0)),
        "repeat_auc_std": float(repeats["auc"].std(ddof=0)),
        "repeat_f1_std": float(repeats["f1"].std(ddof=0)),
        "best_iteration_median": float(folds["best_iteration"].median()),
        "best_iteration_mean": float(folds["best_iteration"].mean()),
    }, repeat_rows, fold_rows, mean_oof


def main():
    train, _, _ = load_competition_data()
    out_dir = Path("reports/catboost_diversity_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    y = train[TARGET].astype(int).to_numpy()
    summaries = []
    all_repeat_rows = []
    all_fold_rows = []
    predictions: dict[str, np.ndarray] = {}

    for name, params in CONFIGS.items():
        print(f"CatBoost diversity CV: {name} ...", flush=True)
        summary, repeat_rows, fold_rows, oof = evaluate(train, name, params, out_dir)
        print(summary, flush=True)
        summaries.append(summary)
        all_repeat_rows.extend(repeat_rows)
        all_fold_rows.extend(fold_rows)
        predictions[name] = oof

    summary_df = pd.DataFrame(summaries).sort_values(["score", "auc"], ascending=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(all_repeat_rows).to_csv(out_dir / "repeat_scores.csv", index=False)
    pd.DataFrame(all_fold_rows).to_csv(out_dir / "fold_scores.csv", index=False)

    blend_rows = []
    for name, p in predictions.items():
        blend_rows.append({"candidate": name, "kind": "single", **official_metrics(y, p)})
    for a, b in combinations(sorted(predictions), 2):
        for wa in (0.25, 0.50, 0.75):
            p = wa * predictions[a] + (1.0 - wa) * predictions[b]
            blend_rows.append(
                {
                    "candidate": f"{wa:.2f}_{a}+{1.0-wa:.2f}_{b}",
                    "kind": "pair",
                    **official_metrics(y, p),
                }
            )

    # A conservative three-model average often benefits AUC when the individual
    # tree depths/boosting regimes make different ranking errors.
    if {"plain_d5_reg", "ordered_d5_reg", "ordered_d6_reg"}.issubset(predictions):
        p = (
            predictions["plain_d5_reg"]
            + predictions["ordered_d5_reg"]
            + predictions["ordered_d6_reg"]
        ) / 3.0
        blend_rows.append(
            {
                "candidate": "equal_plain_d5+ordered_d5+ordered_d6",
                "kind": "triple",
                **official_metrics(y, p),
            }
        )

    blends = pd.DataFrame(blend_rows).sort_values(["score", "auc"], ascending=False)
    blends.to_csv(out_dir / "blend_search.csv", index=False)

    print("\nCatBoost diversity summary:\n", summary_df.to_string(index=False), flush=True)
    print("\nTop CatBoost diversity blends:\n", blends.head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
