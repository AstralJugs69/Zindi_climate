from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ablate_features import make_view
from evaluate_model_suite import model_factory
from features import ID_COL, TARGET, categorical_columns, load_competition_data
from robust_validate import SPLIT_SEEDS


def _catboost_predict(xtr, ytr, xva, yva, xtest, seed):
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
    cats = categorical_columns(xtr)
    model.fit(
        xtr,
        ytr,
        cat_features=cats,
        eval_set=(xva, yva),
        early_stopping_rounds=60,
        verbose=False,
    )
    return model.predict_proba(xtest)[:, 1]


def _tabular_predict(model_name, xtr, ytr, xtest, seed):
    cats = categorical_columns(xtr)
    nums = [c for c in xtr.columns if c not in cats]
    numeric_transform = StandardScaler() if model_name == "logistic" else "passthrough"
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cats),
            ("num", numeric_transform, nums),
        ],
        remainder="drop",
    )
    ztr = pre.fit_transform(xtr)
    ztest = pre.transform(xtest)
    model = model_factory(model_name, seed)
    model.fit(ztr, ytr)
    return model.predict_proba(ztest)[:, 1]


def cv_bag_test_predictions(train, test, view_name: str, model_name: str):
    x = make_view(train, view_name)
    test_with_dummy_target = test.copy()
    test_with_dummy_target[TARGET] = 0
    xt = make_view(test_with_dummy_target, view_name)
    xt = xt[x.columns]

    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()
    fold_predictions = []

    for split_seed in SPLIT_SEEDS:
        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=split_seed,
        )
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups)):
            seed = split_seed + fold
            xtr = x.iloc[tr_idx]
            ytr = y[tr_idx]
            if model_name == "catboost":
                pred = _catboost_predict(
                    xtr,
                    ytr,
                    x.iloc[va_idx],
                    y[va_idx],
                    xt,
                    seed,
                )
            else:
                pred = _tabular_predict(model_name, xtr, ytr, xt, seed)
            fold_predictions.append(pred)

    return np.vstack(fold_predictions).mean(axis=0)


def make_submission(sample, probability, path: Path):
    probability = np.asarray(probability, dtype=float)
    sub = sample[[ID_COL]].copy()
    sub["TargetF1"] = (probability >= 0.5).astype(int)
    sub["TargetRAUC"] = probability
    assert sub[ID_COL].is_unique
    assert len(sub) == len(probability)
    assert sub[["TargetF1", "TargetRAUC"]].notna().all().all()
    assert sub["TargetRAUC"].between(0, 1).all()
    sub.to_csv(path, index=False)
    return sub


def main():
    train, test, sample = load_competition_data()
    assert sample[ID_COL].tolist() == test[ID_COL].tolist()

    out_dir = Path("submissions")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path("reports/robust_candidates")
    report_dir.mkdir(parents=True, exist_ok=True)

    print("CV-bagging demographics_time / catboost (15 fold models) ...", flush=True)
    dt_cat = cv_bag_test_predictions(train, test, "demographics_time", "catboost")

    print("CV-bagging no_spatial / logistic (15 fold models) ...", flush=True)
    ns_log = cv_bag_test_predictions(train, test, "no_spatial", "logistic")

    print("CV-bagging no_spatial / xgboost (15 fold models) ...", flush=True)
    ns_xgb = cv_bag_test_predictions(train, test, "no_spatial", "xgboost")

    candidates = {
        "v101_robust_primary_cat50_log50.csv": 0.50 * dt_cat + 0.50 * ns_log,
        "v102_robust_diversity_cat75_xgb25.csv": 0.75 * dt_cat + 0.25 * ns_xgb,
        "v103_demographics_time_catboost.csv": dt_cat,
    }

    diagnostics = []
    for filename, probability in candidates.items():
        sub = make_submission(sample, probability, out_dir / filename)
        diagnostics.append(
            {
                "file": f"submissions/{filename}",
                "probability_mean": float(sub["TargetRAUC"].mean()),
                "probability_std": float(sub["TargetRAUC"].std()),
                "positive_rate_at_0_5": float(sub["TargetF1"].mean()),
                "probability_min": float(sub["TargetRAUC"].min()),
                "probability_max": float(sub["TargetRAUC"].max()),
            }
        )

    pd.DataFrame(
        {
            ID_COL: sample[ID_COL],
            "demographics_time_catboost": dt_cat,
            "no_spatial_logistic": ns_log,
            "no_spatial_xgboost": ns_xgb,
        }
    ).to_csv(report_dir / "test_component_predictions.csv", index=False)

    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df.to_csv(report_dir / "submission_diagnostics.csv", index=False)

    metadata = {
        "validation_reference": {
            "primary_blend": "0.50 demographics_time catboost + 0.50 no_spatial logistic",
            "observed_robust_oof_score": 0.818682,
            "diversity_blend": "0.75 demographics_time catboost + 0.25 no_spatial xgboost",
            "observed_robust_oof_score_diversity": 0.818663,
            "pure_demographics_time_catboost_score": 0.818729,
        },
        "inference": {
            "strategy": "3x5 StratifiedGroupKFold CV bagging by location",
            "split_seeds": list(SPLIT_SEEDS),
            "models_per_component": len(SPLIT_SEEDS) * 5,
            "fixed_classification_threshold": 0.5,
        },
    }
    (report_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSubmission diagnostics:\n", diagnostics_df.to_string(index=False), flush=True)
    print("\nGenerated candidates:")
    for filename in candidates:
        print(f"  submissions/{filename}")


if __name__ == "__main__":
    main()
