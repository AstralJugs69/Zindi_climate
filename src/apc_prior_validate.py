from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from features import ID_COL, TARGET, load_competition_data
from metrics import official_metrics
from robust_validate import SPLIT_SEEDS


REFERENCE_SOURCES = {
    "all_ctr1": Path("reports/interaction_validation/all_ctr1_oof.csv"),
    "cohort_calendar": Path("reports/fine_demo_validation/cohort_calendar_oof.csv"),
}


def _meta(frame: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(frame["deathdate"], errors="raise")
    age = frame["age"].astype(float).clip(lower=0)
    year = dt.dt.year.astype(float)
    out = pd.DataFrame(index=frame.index)
    out["age"] = age
    out["year"] = year
    out["birth_year"] = year - age
    out["gender"] = frame["gender"].astype(str).str.lower()
    out["zone"] = frame["zone"].astype(str)
    return out


def _kernel_prior(
    train_meta: pd.DataFrame,
    apply_meta: pd.DataFrame,
    y_train: np.ndarray,
    *,
    age_bw: float,
    year_bw: float | None = None,
    cohort_bw: float | None = None,
    gender_soft: float | None = None,
    zone_soft: float | None = None,
    alpha: float = 12.0,
) -> np.ndarray:
    """Smooth empirical-Bayes risk surface built only from the fit fold.

    The prior is intentionally continuous rather than a categorical target encoding.
    Validation locations are completely excluded by StratifiedGroupKFold before this
    function is called.
    """
    tr_age = train_meta["age"].to_numpy(dtype=float)
    ap_age = apply_meta["age"].to_numpy(dtype=float)
    logw = -0.5 * ((ap_age[:, None] - tr_age[None, :]) / age_bw) ** 2

    if year_bw is not None:
        tr = train_meta["year"].to_numpy(dtype=float)
        ap = apply_meta["year"].to_numpy(dtype=float)
        logw += -0.5 * ((ap[:, None] - tr[None, :]) / year_bw) ** 2
    if cohort_bw is not None:
        tr = train_meta["birth_year"].to_numpy(dtype=float)
        ap = apply_meta["birth_year"].to_numpy(dtype=float)
        logw += -0.5 * ((ap[:, None] - tr[None, :]) / cohort_bw) ** 2

    # Stabilise before exponentiating. Categorical mismatches are soft penalties,
    # not hard cells, so sparse strata still borrow strength from the full cohort.
    logw -= np.max(logw, axis=1, keepdims=True)
    weights = np.exp(logw)
    if gender_soft is not None:
        same = (
            apply_meta["gender"].to_numpy()[:, None]
            == train_meta["gender"].to_numpy()[None, :]
        )
        weights *= np.where(same, 1.0, gender_soft)
    if zone_soft is not None:
        same = (
            apply_meta["zone"].to_numpy()[:, None]
            == train_meta["zone"].to_numpy()[None, :]
        )
        weights *= np.where(same, 1.0, zone_soft)

    y = np.asarray(y_train, dtype=float)
    global_mean = float(y.mean())
    numerator = weights @ y + alpha * global_mean
    denominator = weights.sum(axis=1) + alpha
    return numerator / np.maximum(denominator, 1e-12)


CONFIGS = {
    "age_bw2": dict(age_bw=2.0, alpha=12.0),
    "age_bw5": dict(age_bw=5.0, alpha=12.0),
    "age_year_3_2": dict(age_bw=3.0, year_bw=2.0, alpha=10.0),
    "age_year_5_3": dict(age_bw=5.0, year_bw=3.0, alpha=12.0),
    "age_cohort_4_8": dict(age_bw=4.0, cohort_bw=8.0, alpha=12.0),
    "age_year_gender": dict(age_bw=4.0, year_bw=2.5, gender_soft=0.45, alpha=12.0),
    "age_year_zone": dict(age_bw=4.0, year_bw=2.5, zone_soft=0.55, alpha=12.0),
    "age_year_gender_zone": dict(
        age_bw=4.0,
        year_bw=2.5,
        gender_soft=0.55,
        zone_soft=0.65,
        alpha=14.0,
    ),
}


def _load_reference(train: pd.DataFrame):
    refs: dict[str, np.ndarray] = {}
    y = train[TARGET].astype(int).to_numpy()
    ids = train[ID_COL].astype(str).to_numpy()
    for name, path in REFERENCE_SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run interaction-validate and fine-demo-validate first."
            )
        frame = pd.read_csv(path)
        if not np.array_equal(ids, frame[ID_COL].astype(str).to_numpy()):
            raise ValueError(f"ID order mismatch in {path}")
        if not np.array_equal(y, frame["target"].astype(int).to_numpy()):
            raise ValueError(f"Target order mismatch in {path}")
        refs[name] = frame["oof_probability"].astype(float).to_numpy()
    return y, refs


def main():
    train, _, _ = load_competition_data()
    out_dir = Path("reports/apc_prior_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _meta(train)
    y, refs = _load_reference(train)
    groups = train["location"].astype(str).to_numpy()

    aggregate_priors: dict[str, list[np.ndarray]] = {k: [] for k in CONFIGS}
    repeat_rows = []
    for repeat, split_seed in enumerate(SPLIT_SEEDS):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
        repeat_oof = {k: np.zeros(len(train), dtype=float) for k in CONFIGS}
        for tr_idx, va_idx in splitter.split(meta, y, groups):
            tr_meta = meta.iloc[tr_idx]
            va_meta = meta.iloc[va_idx]
            for name, kwargs in CONFIGS.items():
                repeat_oof[name][va_idx] = _kernel_prior(
                    tr_meta, va_meta, y[tr_idx], **kwargs
                )
        for name, p in repeat_oof.items():
            repeat_rows.append(
                {
                    "candidate": name,
                    "repeat": repeat,
                    "split_seed": split_seed,
                    **official_metrics(y, p),
                }
            )
            aggregate_priors[name].append(p)

    priors = {name: np.vstack(values).mean(axis=0) for name, values in aggregate_priors.items()}
    rows = []
    for name, p in priors.items():
        rows.append({"candidate": name, "kind": "smooth_prior", **official_metrics(y, p)})

    for ref_name, ref in refs.items():
        rows.append({"candidate": ref_name, "kind": "reference", **official_metrics(y, ref)})
        for prior_name, prior in priors.items():
            for w_prior in (0.10, 0.20, 0.30, 0.40):
                p = (1.0 - w_prior) * ref + w_prior * prior
                rows.append(
                    {
                        "candidate": f"{1-w_prior:.2f}_{ref_name}+{w_prior:.2f}_{prior_name}",
                        "kind": "blend",
                        **official_metrics(y, p),
                    }
                )

    # Cross the best two low-shift model families with a modest smooth prior weight.
    for prior_name, prior in priors.items():
        for w_prior in (0.10, 0.20):
            remaining = 1.0 - w_prior
            for all_share in (0.25, 0.50, 0.75):
                p = (
                    remaining * all_share * refs["all_ctr1"]
                    + remaining * (1.0 - all_share) * refs["cohort_calendar"]
                    + w_prior * prior
                )
                rows.append(
                    {
                        "candidate": (
                            f"{remaining*all_share:.3f}_all_ctr1+"
                            f"{remaining*(1-all_share):.3f}_cohort_calendar+"
                            f"{w_prior:.2f}_{prior_name}"
                        ),
                        "kind": "triple",
                        **official_metrics(y, p),
                    }
                )

    result = pd.DataFrame(rows).sort_values(["score", "auc"], ascending=False)
    result.to_csv(out_dir / "blend_search.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(out_dir / "repeat_scores.csv", index=False)
    pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            "target": y,
            **{f"prior_{name}": p for name, p in priors.items()},
        }
    ).to_csv(out_dir / "prior_oof.csv", index=False)
    print("\nTop smooth APC-prior candidates:\n", result.head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
