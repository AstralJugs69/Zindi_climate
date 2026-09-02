from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold

from features import ID_COL, TARGET, engineer_features, feature_columns, load_competition_data
from metrics import official_metrics


CONFIGS = {
    "cat_d5_i120_lr04": dict(iterations=120, depth=5, learning_rate=0.04, l2_leaf_reg=5.0),
    "cat_d6_i220_lr03": dict(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0),
    "cat_d5_i350_lr025": dict(iterations=350, depth=5, learning_rate=0.025, l2_leaf_reg=6.0),
    "cat_d7_i180_lr03": dict(iterations=180, depth=7, learning_rate=0.03, l2_leaf_reg=6.0),
}


def main():
    train, _, _ = load_competition_data()
    X = engineer_features(train.drop(columns=[TARGET]), include_location=True)
    X = X[feature_columns(X)]
    y = train[TARGET].to_numpy()
    groups = train["location"].astype(str).to_numpy()
    cats = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    out_dir = Path("reports/catboost_tuning")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, params in CONFIGS.items():
        print("Running", name, flush=True)
        oof = np.zeros(len(train))
        fold_rows = []
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y, groups)):
            model = CatBoostClassifier(
                **params,
                loss_function="Logloss",
                random_seed=2026 + fold,
                verbose=False,
                allow_writing_files=False,
                thread_count=2,
            )
            model.fit(X.iloc[tr_idx], y[tr_idx], cat_features=cats)
            p = model.predict_proba(X.iloc[va_idx])[:, 1]
            oof[va_idx] = p
            fold_rows.append({"fold": fold, **official_metrics(y[va_idx], p)})
        metrics = official_metrics(y, oof)
        print(name, metrics, flush=True)
        rows.append({"config": name, **metrics})
        pd.DataFrame(fold_rows).to_csv(out_dir / f"{name}_folds.csv", index=False)
        pd.DataFrame({ID_COL: train[ID_COL], "target": y, "oof_probability": oof}).to_csv(out_dir / f"{name}_oof.csv", index=False)

    result = pd.DataFrame(rows).sort_values("score", ascending=False)
    result.to_csv(out_dir / "summary.csv", index=False)
    print("\n", result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
