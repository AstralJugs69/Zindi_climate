from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold

from ablate_features import make_view
from features import ID_COL, TARGET, categorical_columns, load_competition_data
from fine_demo_validate import add_fine_demographics
from robust_validate import SPLIT_SEEDS


def _fit_fold_predict(x_train, y_train, x_valid, y_valid, x_test, seed: int):
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
    return model.predict_proba(x_test)[:, 1]


def cv_bag_component(train: pd.DataFrame, test: pd.DataFrame, mode: str) -> np.ndarray:
    x = add_fine_demographics(make_view(train, "demographics_time"), mode)
    test_dummy = test.copy()
    test_dummy[TARGET] = 0
    xt = add_fine_demographics(make_view(test_dummy, "demographics_time"), mode)
    xt = xt[x.columns]

    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()
    predictions = []

    for split_seed in SPLIT_SEEDS:
        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=split_seed,
        )
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups)):
            predictions.append(
                _fit_fold_predict(
                    x.iloc[tr_idx],
                    y[tr_idx],
                    x.iloc[va_idx],
                    y[va_idx],
                    xt,
                    seed=split_seed + fold,
                )
            )

    return np.vstack(predictions).mean(axis=0)


def make_submission(sample: pd.DataFrame, probability: np.ndarray, path: Path) -> pd.DataFrame:
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
    report_dir = Path("reports/cohort_candidates")
    report_dir.mkdir(parents=True, exist_ok=True)

    print("CV-bagging all_ctr1/reference component (15 fold models) ...", flush=True)
    all_ctr1 = cv_bag_component(train, test, "reference")
    print("CV-bagging cohort_calendar component (15 fold models) ...", flush=True)
    cohort = cv_bag_component(train, test, "cohort_calendar")

    candidates = {
        "v301_all25_cohort75.csv": 0.25 * all_ctr1 + 0.75 * cohort,
        "v302_all50_cohort50.csv": 0.50 * all_ctr1 + 0.50 * cohort,
        "v303_all75_cohort25.csv": 0.75 * all_ctr1 + 0.25 * cohort,
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
            "all_ctr1": all_ctr1,
            "cohort_calendar": cohort,
        }
    ).to_csv(report_dir / "test_component_predictions.csv", index=False)

    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df.to_csv(report_dir / "submission_diagnostics.csv", index=False)

    metadata = {
        "validation_reference": {
            "v301": {
                "blend": "0.25 all_ctr1 + 0.75 cohort_calendar",
                "oof_f1": 0.818008,
                "oof_auc": 0.826381,
                "oof_score": 0.821357,
            },
            "v302": {
                "blend": "0.50 all_ctr1 + 0.50 cohort_calendar",
                "oof_f1": 0.817746,
                "oof_auc": 0.826292,
                "oof_score": 0.821164,
            },
            "v303": {
                "blend": "0.75 all_ctr1 + 0.25 cohort_calendar",
                "oof_f1": 0.817287,
                "oof_auc": 0.826085,
                "oof_score": 0.820806,
            },
            "component_shift_auc": {
                "all_ctr1": 0.526968,
                "cohort_calendar": 0.535122,
            },
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
