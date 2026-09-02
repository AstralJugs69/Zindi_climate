from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from features import ID_COL, TARGET, engineer_features, feature_columns, load_competition_data


PRIMARY_PARAMS = dict(
    iterations=120,
    depth=5,
    learning_rate=0.04,
    l2_leaf_reg=5.0,
    loss_function="Logloss",
)

AGE_EXPERT_PARAMS = dict(
    iterations=220,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=5.0,
    loss_function="Logloss",
)


def fit_seed_bag(X, y, X_test, cat_cols, params, seeds):
    preds = []
    importances = []
    for seed in seeds:
        model = CatBoostClassifier(
            **params,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=2,
        )
        model.fit(X, y, cat_features=cat_cols)
        preds.append(model.predict_proba(X_test)[:, 1])
        importances.append(model.get_feature_importance())
    return np.mean(preds, axis=0), np.mean(importances, axis=0)


def make_submission(sample, probability, path):
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
    train_ids = set(train[ID_COL])
    assert not train_ids.intersection(set(test[ID_COL]))
    assert sample[ID_COL].tolist() == test[ID_COL].tolist(), "SampleSubmission and Test ID order differ"

    X = engineer_features(train.drop(columns=[TARGET]), include_location=True)
    Xt = engineer_features(test, include_location=True)
    cols = feature_columns(X)
    X = X[cols]
    Xt = Xt[cols]
    y = train[TARGET].astype(int).to_numpy()
    cats = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()

    out_dir = Path("submissions")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    seeds = [2026, 2027, 2028, 2029, 2030]
    primary_p, primary_importance = fit_seed_bag(X, y, Xt, cats, PRIMARY_PARAMS, seeds)
    primary_sub = make_submission(sample, primary_p, out_dir / "v001_catboost_d5_i120_seedbag.csv")

    # Independent age-gated experts; age is observed at inference and is not target-derived.
    expert_preds = np.zeros(len(test), dtype=float)
    expert_importance = np.zeros(len(cols), dtype=float)
    expert_weight = 0.0
    for under5 in [True, False]:
        tr_mask = (train["age"].to_numpy() < 5) == under5
        te_mask = (test["age"].to_numpy() < 5) == under5
        p, imp = fit_seed_bag(
            X.loc[tr_mask], y[tr_mask], Xt.loc[te_mask], cats,
            AGE_EXPERT_PARAMS, seeds[:3],
        )
        expert_preds[te_mask] = p
        expert_importance += tr_mask.mean() * imp
        expert_weight += tr_mask.mean()
    expert_importance /= expert_weight

    # OOF research showed 75/25 primary/age-expert is a near-primary diversity candidate.
    diversity_p = 0.75 * primary_p + 0.25 * expert_preds
    diversity_sub = make_submission(sample, diversity_p, out_dir / "v002_catboost75_ageexpert25.csv")

    importance = pd.DataFrame({
        "feature": cols,
        "primary_importance": primary_importance,
        "age_expert_importance": expert_importance,
    }).sort_values("primary_importance", ascending=False)
    importance.to_csv(model_dir / "catboost_feature_importance.csv", index=False)

    metadata = {
        "primary": {
            "file": "submissions/v001_catboost_d5_i120_seedbag.csv",
            "params": PRIMARY_PARAMS,
            "seeds": seeds,
            "test_probability_mean": float(primary_sub.TargetRAUC.mean()),
            "test_positive_rate_at_0_5": float(primary_sub.TargetF1.mean()),
        },
        "diversity": {
            "file": "submissions/v002_catboost75_ageexpert25.csv",
            "blend": {"primary": 0.75, "age_expert": 0.25},
            "age_expert_params": AGE_EXPERT_PARAMS,
            "seeds": seeds[:3],
            "test_probability_mean": float(diversity_sub.TargetRAUC.mean()),
            "test_positive_rate_at_0_5": float(diversity_sub.TargetF1.mean()),
        },
    }
    (model_dir / "candidate_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print("\nTop primary features:\n", importance.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
