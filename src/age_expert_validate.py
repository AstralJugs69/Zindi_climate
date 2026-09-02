from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold

from ablate_features import make_view
from features import ID_COL, TARGET, categorical_columns, load_competition_data
from interaction_validate import add_interactions, shift_auc
from metrics import official_metrics
from robust_validate import SPLIT_SEEDS


def _partition_labels(age: np.ndarray, name: str) -> np.ndarray:
    age = np.asarray(age, dtype=float)
    if name == "under5_2":
        return np.where(age < 5, "u5", "5plus")
    if name == "life3":
        return np.select(
            [age < 5, age < 65],
            ["u5", "5_64"],
            default="65plus",
        )
    if name == "life4":
        return np.select(
            [age < 5, age < 15, age < 65],
            ["u5", "5_14", "15_64"],
            default="65plus",
        )
    raise ValueError(name)


PARTITIONS = ("under5_2", "life3", "life4")
EXPERT_WEIGHTS = (0.25, 0.50, 0.75, 1.00)


def _fit_global(xtr, ytr, xva, yva, seed: int) -> np.ndarray:
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
        max_ctr_complexity=1,
    )
    model.fit(
        xtr,
        ytr,
        cat_features=categorical_columns(xtr),
        eval_set=(xva, yva),
        early_stopping_rounds=70,
        verbose=False,
    )
    return model.predict_proba(xva)[:, 1]


def _fit_expert(xtr, ytr, xva, yva, seed: int) -> np.ndarray:
    ytr = np.asarray(ytr, dtype=int)
    if len(np.unique(ytr)) < 2:
        return np.full(len(xva), float(np.mean(ytr)), dtype=float)

    model = CatBoostClassifier(
        iterations=320,
        depth=5,
        learning_rate=0.035,
        l2_leaf_reg=7.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        max_ctr_complexity=1,
    )
    fit_kwargs = {
        "cat_features": categorical_columns(xtr),
        "verbose": False,
    }
    yva = np.asarray(yva, dtype=int)
    if len(xva) >= 20 and len(np.unique(yva)) >= 2:
        fit_kwargs.update(
            {
                "eval_set": (xva, yva),
                "early_stopping_rounds": 60,
            }
        )
    model.fit(xtr, ytr, **fit_kwargs)
    return model.predict_proba(xva)[:, 1]


def _candidate_name(partition: str, expert_weight: float) -> str:
    if expert_weight == 1.0:
        return f"{partition}__expert_only"
    return f"{partition}__global{1-expert_weight:.2f}_expert{expert_weight:.2f}"


def main():
    train, test, _ = load_competition_data()
    out_dir = Path("reports/age_expert_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    x = add_interactions(make_view(train, "demographics_time"), "all")
    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()
    ages = train["age"].to_numpy(dtype=float)
    labels = {name: _partition_labels(ages, name) for name in PARTITIONS}

    count_rows = []
    for name in PARTITIONS:
        for stratum, count in pd.Series(labels[name]).value_counts().items():
            mask = labels[name] == stratum
            count_rows.append(
                {
                    "partition": name,
                    "stratum": stratum,
                    "n": int(count),
                    "target_rate": float(y[mask].mean()),
                }
            )
    pd.DataFrame(count_rows).to_csv(out_dir / "stratum_counts.csv", index=False)

    candidate_oofs: dict[str, list[np.ndarray]] = {}
    repeat_rows = []

    for repeat, split_seed in enumerate(SPLIT_SEEDS):
        print(
            f"Age-expert repeat {repeat + 1}/{len(SPLIT_SEEDS)} seed={split_seed} ...",
            flush=True,
        )
        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=split_seed,
        )
        folds = list(splitter.split(x, y, groups))
        global_oof = np.zeros(len(train), dtype=float)
        expert_oofs = {
            name: np.zeros(len(train), dtype=float) for name in PARTITIONS
        }

        for fold, (tr_idx, va_idx) in enumerate(folds):
            seed = split_seed + fold
            global_oof[va_idx] = _fit_global(
                x.iloc[tr_idx],
                y[tr_idx],
                x.iloc[va_idx],
                y[va_idx],
                seed,
            )

            for partition in PARTITIONS:
                fold_labels = labels[partition]
                for stratum in np.unique(fold_labels[va_idx]):
                    tr_sub = tr_idx[fold_labels[tr_idx] == stratum]
                    va_sub = va_idx[fold_labels[va_idx] == stratum]
                    if len(va_sub) == 0:
                        continue
                    if len(tr_sub) < 30:
                        expert_oofs[partition][va_sub] = global_oof[va_sub]
                        continue
                    expert_oofs[partition][va_sub] = _fit_expert(
                        x.iloc[tr_sub],
                        y[tr_sub],
                        x.iloc[va_sub],
                        y[va_sub],
                        seed + 1000 + sum(ord(c) for c in f"{partition}:{stratum}"),
                    )

        global_metrics = official_metrics(y, global_oof)
        repeat_rows.append(
            {
                "candidate": "global_all_ctr1",
                "repeat": repeat,
                "split_seed": split_seed,
                **global_metrics,
            }
        )
        candidate_oofs.setdefault("global_all_ctr1", []).append(global_oof)

        for partition in PARTITIONS:
            for expert_weight in EXPERT_WEIGHTS:
                name = _candidate_name(partition, expert_weight)
                pred = (
                    (1.0 - expert_weight) * global_oof
                    + expert_weight * expert_oofs[partition]
                )
                metrics = official_metrics(y, pred)
                repeat_rows.append(
                    {
                        "candidate": name,
                        "repeat": repeat,
                        "split_seed": split_seed,
                        **metrics,
                    }
                )
                candidate_oofs.setdefault(name, []).append(pred)

    repeat_df = pd.DataFrame(repeat_rows)
    repeat_df.to_csv(out_dir / "repeat_scores.csv", index=False)

    common_shift_auc = shift_auc(train, test, "all")
    summaries = []
    for name, repeated in candidate_oofs.items():
        mean_oof = np.vstack(repeated).mean(axis=0)
        metrics = official_metrics(y, mean_oof)
        rows = repeat_df.loc[repeat_df["candidate"] == name]
        summaries.append(
            {
                "candidate": name,
                **metrics,
                "repeat_score_mean": float(rows["score"].mean()),
                "repeat_score_std": float(rows["score"].std(ddof=0)),
                "repeat_auc_std": float(rows["auc"].std(ddof=0)),
                "repeat_f1_std": float(rows["f1"].std(ddof=0)),
                "feature_shift_auc": common_shift_auc,
            }
        )
        pd.DataFrame(
            {ID_COL: train[ID_COL], "target": y, "oof_probability": mean_oof}
        ).to_csv(out_dir / f"{name}_oof.csv", index=False)

    summary = pd.DataFrame(summaries).sort_values(
        ["score", "auc", "repeat_score_std"],
        ascending=[False, False, True],
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("\nAge-expert validation summary:\n", summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
