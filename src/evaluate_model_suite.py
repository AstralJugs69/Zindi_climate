from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from features import ID_COL, TARGET, engineer_features, feature_columns, load_competition_data
from metrics import official_metrics


def preprocess_fold(x_train, x_valid, scale_numeric=False):
    cat_cols = x_train.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    num_cols = [c for c in x_train.columns if c not in cat_cols]
    numeric_transform = StandardScaler() if scale_numeric else "passthrough"
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cat_cols),
            ("num", numeric_transform, num_cols),
        ],
        remainder="drop",
    )
    return pre.fit_transform(x_train), pre.transform(x_valid)


def model_factory(name, seed):
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=650,
            learning_rate=0.025,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=25,
            subsample=0.90,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=2.0,
            random_state=seed,
            verbosity=-1,
            n_jobs=-1,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=650,
            learning_rate=0.025,
            max_depth=4,
            min_child_weight=6,
            subsample=0.90,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=3.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "logistic":
        return LogisticRegression(C=0.5, max_iter=5000, solver="liblinear")
    raise ValueError(name)


def evaluate(name, train, out_dir):
    X = engineer_features(train.drop(columns=[TARGET]), include_location=True)
    X = X[feature_columns(X)]
    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros(len(train))
    rows = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y, groups)):
        xtr, xva = X.iloc[tr_idx], X.iloc[va_idx]
        ztr, zva = preprocess_fold(xtr, xva, scale_numeric=(name == "logistic"))
        model = model_factory(name, 2026 + fold)
        model.fit(ztr, y[tr_idx])
        pred = model.predict_proba(zva)[:, 1]
        oof[va_idx] = pred
        m = official_metrics(y[va_idx], pred)
        m.update({"fold": fold, "n": len(va_idx)})
        rows.append(m)

    metrics = official_metrics(y, oof)
    pd.DataFrame(rows).to_csv(out_dir / f"{name}_group_folds.csv", index=False)
    pd.DataFrame({ID_COL: train[ID_COL], "target": y, "oof_probability": oof}).to_csv(out_dir / f"{name}_group_oof.csv", index=False)
    return metrics


def blend_search(train, out_dir, model_names):
    y = train[TARGET].to_numpy()
    probs = {}
    # CatBoost file is produced by evaluate_baselines.py.
    cat_path = Path("reports/baselines/group_with_location_oof.csv")
    if cat_path.exists():
        probs["catboost"] = pd.read_csv(cat_path)["oof_probability"].to_numpy()
    for name in model_names:
        probs[name] = pd.read_csv(out_dir / f"{name}_group_oof.csv")["oof_probability"].to_numpy()

    rows = []
    keys = list(probs)
    # Coarse, pre-declared blend grid. Avoids hyper-fine OOF weight overfitting.
    if len(keys) >= 2:
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                for wa in [0.25, 0.5, 0.75]:
                    p = wa * probs[a] + (1 - wa) * probs[b]
                    rows.append({"blend": f"{wa:.2f}_{a}+{1-wa:.2f}_{b}", **official_metrics(y, p)})

    if all(k in probs for k in ["catboost", "lightgbm", "xgboost"]):
        for wc, wl, wx in [(0.5, 0.3, 0.2), (0.5, 0.25, 0.25), (0.4, 0.4, 0.2), (0.4, 0.3, 0.3)]:
            p = wc * probs["catboost"] + wl * probs["lightgbm"] + wx * probs["xgboost"]
            rows.append({"blend": f"cat{wc}_lgb{wl}_xgb{wx}", **official_metrics(y, p)})

    result = pd.DataFrame(rows).sort_values("score", ascending=False)
    result.to_csv(out_dir / "blend_search.csv", index=False)
    return result


def main():
    train, _, _ = load_competition_data()
    out_dir = Path("reports/model_suite")
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["lightgbm", "xgboost", "logistic"]
    summary = []
    for name in names:
        print(f"Running {name} location-group CV ...", flush=True)
        m = evaluate(name, train, out_dir)
        print(name, m, flush=True)
        summary.append({"model": name, **m})
    pd.DataFrame(summary).sort_values("score", ascending=False).to_csv(out_dir / "summary.csv", index=False)

    blends = blend_search(train, out_dir, names)
    print("\nTop blends:\n", blends.head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
