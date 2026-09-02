from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ablate_features import make_view
from features import ID_COL, TARGET, categorical_columns, load_competition_data
from metrics import official_metrics
from robust_validate import SPLIT_SEEDS


MODELS = (
    "catboost",
    "spline_logistic",
    "extra_trees",
    "histgb",
    "knn100",
)


def _generic_preprocessor(x: pd.DataFrame, *, dense: bool) -> ColumnTransformer:
    cats = categorical_columns(x)
    nums = [c for c in x.columns if c not in cats]
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=not dense,
                ),
                cats,
            ),
            ("num", StandardScaler(), nums),
        ],
        remainder="drop",
    )


def _spline_logistic_pipeline(x: pd.DataFrame) -> Pipeline:
    cats = categorical_columns(x)
    special = {"age", "year"}
    nums = [c for c in x.columns if c not in cats and c not in special]
    pre = ColumnTransformer(
        [
            (
                "age_spline",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "spline",
                            SplineTransformer(
                                n_knots=8,
                                degree=3,
                                knots="quantile",
                                include_bias=False,
                            ),
                        ),
                    ]
                ),
                ["age"],
            ),
            (
                "year_spline",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "spline",
                            SplineTransformer(
                                n_knots=6,
                                degree=3,
                                knots="quantile",
                                include_bias=False,
                            ),
                        ),
                    ]
                ),
                ["year"],
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                cats,
            ),
            ("num", StandardScaler(), nums),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "model",
                LogisticRegression(
                    C=0.35,
                    max_iter=6000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _fit_predict(model_name, xtr, ytr, xva, yva, seed):
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
        model.fit(
            xtr,
            ytr,
            cat_features=categorical_columns(xtr),
            eval_set=(xva, yva),
            early_stopping_rounds=60,
            verbose=False,
        )
        return model.predict_proba(xva)[:, 1]

    if model_name == "spline_logistic":
        model = _spline_logistic_pipeline(xtr)
        model.fit(xtr, ytr)
        return model.predict_proba(xva)[:, 1]

    if model_name == "extra_trees":
        model = Pipeline(
            [
                ("pre", _generic_preprocessor(xtr, dense=False)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        max_features=0.75,
                        min_samples_leaf=10,
                        min_samples_split=16,
                        class_weight=None,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(xtr, ytr)
        return model.predict_proba(xva)[:, 1]

    if model_name == "histgb":
        model = Pipeline(
            [
                ("pre", _generic_preprocessor(xtr, dense=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        max_iter=350,
                        max_leaf_nodes=15,
                        min_samples_leaf=25,
                        l2_regularization=2.0,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(xtr, ytr)
        return model.predict_proba(xva)[:, 1]

    if model_name == "knn100":
        # A smooth non-parametric risk surface over the same low-shift demographic
        # and time variables. Large k deliberately favours stable age/year trends.
        model = Pipeline(
            [
                ("pre", _generic_preprocessor(xtr, dense=True)),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=100,
                        weights="distance",
                        p=2,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(xtr, ytr)
        return model.predict_proba(xva)[:, 1]

    raise ValueError(model_name)


def evaluate(train: pd.DataFrame, model_name: str, out_dir: Path):
    x = make_view(train, "demographics_time")
    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()
    repeated_oof = []
    repeat_rows = []

    for repeat, split_seed in enumerate(SPLIT_SEEDS):
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
                split_seed + fold,
            )

        metrics = official_metrics(y, oof)
        repeat_rows.append(
            {
                "model": model_name,
                "repeat": repeat,
                "split_seed": split_seed,
                **metrics,
            }
        )
        repeated_oof.append(oof)

    repeated = np.vstack(repeated_oof)
    mean_oof = repeated.mean(axis=0)
    aggregate = official_metrics(y, mean_oof)
    repeats = pd.DataFrame(repeat_rows)
    summary = {
        "model": model_name,
        **aggregate,
        "repeat_score_mean": float(repeats.score.mean()),
        "repeat_score_std": float(repeats.score.std(ddof=0)),
        "repeat_auc_std": float(repeats.auc.std(ddof=0)),
        "repeat_f1_std": float(repeats.f1.std(ddof=0)),
    }
    pd.DataFrame(
        {ID_COL: train[ID_COL], "target": y, "oof_probability": mean_oof}
    ).to_csv(out_dir / f"{model_name}_oof.csv", index=False)
    return summary, mean_oof, repeat_rows


def blend_search(y: np.ndarray, probabilities: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    keys = sorted(probabilities)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            for w in (0.25, 0.50, 0.75):
                p = w * probabilities[left] + (1.0 - w) * probabilities[right]
                rows.append(
                    {
                        "blend": f"{w:.2f}_{left}+{1-w:.2f}_{right}",
                        **official_metrics(y, p),
                    }
                )
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def main():
    train, _, _ = load_competition_data()
    out_dir = Path("reports/demographic_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_repeats = []
    probabilities = {}

    for model_name in MODELS:
        print(f"Demographic repeated CV: {model_name} ...", flush=True)
        summary, oof, repeats = evaluate(train, model_name, out_dir)
        print(summary, flush=True)
        summaries.append(summary)
        all_repeats.extend(repeats)
        probabilities[model_name] = oof

    summary_df = pd.DataFrame(summaries).sort_values(
        ["score", "repeat_score_std"], ascending=[False, True]
    )
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(all_repeats).to_csv(out_dir / "repeat_scores.csv", index=False)
    blends = blend_search(train[TARGET].astype(int).to_numpy(), probabilities)
    blends.to_csv(out_dir / "blend_search.csv", index=False)

    print("\nDemographic validation summary:\n", summary_df.to_string(index=False), flush=True)
    print("\nTop demographic blends:\n", blends.head(15).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
