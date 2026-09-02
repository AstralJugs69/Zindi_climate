from __future__ import annotations

from io import StringIO
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ablate_features import make_view
from features import TARGET, categorical_columns, load_competition_data
from fine_demo_validate import add_fine_demographics
from interaction_validate import add_interactions
from metrics import official_metrics
from robust_validate import SPLIT_SEEDS


DMI_URL = "https://psl.noaa.gov/data/timeseries/month/data/dmi.had.long.csv"
NINO34_URL = "https://psl.noaa.gov/data/correlation/nina34.anom.csv"


def _download_text(url: str, path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=60, headers={"User-Agent": "zindi-climate-research/1.0"})
            response.raise_for_status()
            text = response.text
            if len(text) < 100:
                raise RuntimeError("NOAA response unexpectedly short")
            path.write_text(text, encoding="utf-8")
            return text
        except Exception as exc:
            last_exc = exc
            if attempt < 4:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"NOAA climate-index download failed for {url}: {last_exc}")


def _monthly_series(text: str, name: str) -> pd.DataFrame:
    """Parse NOAA PSL monthly CSVs in either long or year-by-month layout."""
    frame = pd.read_csv(StringIO(text))
    frame.columns = [str(c).strip() for c in frame.columns]
    lower = {c.lower(): c for c in frame.columns}

    date_col = next((c for c in frame.columns if "date" in c.lower()), None)
    if date_col is not None:
        value_cols = [c for c in frame.columns if c != date_col]
        for value_col in value_cols:
            values = pd.to_numeric(frame[value_col], errors="coerce")
            dates = pd.to_datetime(frame[date_col], errors="coerce")
            good = dates.notna() & values.notna()
            if good.sum() > 100:
                return pd.DataFrame({"month": dates[good].dt.to_period("M"), name: values[good]})

    year_col = next((lower[k] for k in lower if k in {"year", "yr"}), None)
    month_col = next((lower[k] for k in lower if k in {"month", "mon", "mn"}), None)
    if year_col is not None and month_col is not None:
        value_cols = [c for c in frame.columns if c not in {year_col, month_col}]
        for value_col in value_cols:
            vals = pd.to_numeric(frame[value_col], errors="coerce")
            years = pd.to_numeric(frame[year_col], errors="coerce")
            months = pd.to_numeric(frame[month_col], errors="coerce")
            good = vals.notna() & years.notna() & months.notna()
            if good.sum() > 100:
                dates = pd.to_datetime(
                    {"year": years[good].astype(int), "month": months[good].astype(int), "day": 1}
                )
                return pd.DataFrame({"month": dates.dt.to_period("M"), name: vals[good]})

    # Wide PSL format: first column is year, remaining columns are Jan..Dec or 1..12.
    first = frame.columns[0]
    years = pd.to_numeric(frame[first], errors="coerce")
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    rows = []
    for col in frame.columns[1:]:
        token = col.strip().lower()[:3]
        month = month_map.get(token)
        if month is None:
            try:
                month = int(float(col))
            except Exception:
                continue
        if not 1 <= month <= 12:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        good = years.notna() & vals.notna() & (vals > -90)
        for y, v in zip(years[good].astype(int), vals[good].astype(float)):
            rows.append((pd.Period(year=int(y), month=month, freq="M"), float(v)))
    if len(rows) < 100:
        raise RuntimeError(f"Could not parse NOAA monthly series {name}; columns={list(frame.columns)}")
    return pd.DataFrame(rows, columns=["month", name]).drop_duplicates("month").sort_values("month")


def _load_indices(out_dir: Path) -> pd.DataFrame:
    dmi = _monthly_series(_download_text(DMI_URL, out_dir / "dmi.csv"), "dmi")
    nino = _monthly_series(_download_text(NINO34_URL, out_dir / "nino34.csv"), "nino34")
    merged = dmi.merge(nino, on="month", how="inner").sort_values("month").reset_index(drop=True)
    if len(merged) < 500:
        raise RuntimeError("Merged NOAA teleconnection history is unexpectedly short")
    return merged


def _tele_features(raw: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    lookup = indices.set_index("month")
    dates = pd.to_datetime(raw["deathdate"], errors="raise")
    periods = dates.dt.to_period("M")
    out = pd.DataFrame(index=raw.index)
    for name in ("dmi", "nino34"):
        for lag in range(7):
            out[f"tele_{name}_lag{lag}"] = [
                float(lookup.at[p - lag, name]) if (p - lag) in lookup.index else np.nan
                for p in periods
            ]
        out[f"tele_{name}_mean0_2"] = out[[f"tele_{name}_lag{i}" for i in range(3)]].mean(axis=1)
        out[f"tele_{name}_mean1_3"] = out[[f"tele_{name}_lag{i}" for i in range(1, 4)]].mean(axis=1)
        out[f"tele_{name}_mean0_5"] = out[[f"tele_{name}_lag{i}" for i in range(6)]].mean(axis=1)
        out[f"tele_{name}_trend0_2"] = out[f"tele_{name}_lag0"] - out[f"tele_{name}_lag2"]

    out["tele_iod_positive"] = (out["tele_dmi_mean0_2"] > 0.4).astype(int)
    out["tele_iod_strong_positive"] = (out["tele_dmi_mean0_2"] > 0.8).astype(int)
    out["tele_iod_negative"] = (out["tele_dmi_mean0_2"] < -0.4).astype(int)
    out["tele_el_nino"] = (out["tele_nino34_mean0_2"] > 0.5).astype(int)
    out["tele_la_nina"] = (out["tele_nino34_mean0_2"] < -0.5).astype(int)
    out["tele_dmi_x_nino"] = out["tele_dmi_mean0_2"] * out["tele_nino34_mean0_2"]

    age = raw["age"].astype(float)
    under5 = (age < 5).astype(float)
    age5_14 = ((age >= 5) & (age < 15)).astype(float)
    male = raw["gender"].astype(str).str.lower().str.startswith("m").astype(float)
    for col in ("tele_dmi_lag1", "tele_dmi_lag2", "tele_dmi_mean0_2", "tele_nino34_mean0_2"):
        out[f"under5_x_{col}"] = under5 * out[col]
        out[f"age5_14_x_{col}"] = age5_14 * out[col]
    out["male_5_14_x_dmi"] = male * age5_14 * out["tele_dmi_mean0_2"]
    return out


def _fit_predict(x_train, y_train, x_valid, y_valid, seed: int) -> np.ndarray:
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
    return model.predict_proba(x_valid)[:, 1]


def _cv(train: pd.DataFrame, x: pd.DataFrame, key: str, out_dir: Path):
    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()
    repeated = []
    repeat_rows = []
    for repeat, seed in enumerate(SPLIT_SEEDS):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(train), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups)):
            oof[va_idx] = _fit_predict(
                x.iloc[tr_idx], y[tr_idx], x.iloc[va_idx], y[va_idx], seed + fold
            )
        repeat_rows.append({"config": key, "repeat": repeat, "split_seed": seed, **official_metrics(y, oof)})
        repeated.append(oof)
    mean_oof = np.vstack(repeated).mean(axis=0)
    metrics = official_metrics(y, mean_oof)
    rep = pd.DataFrame(repeat_rows)
    pd.DataFrame({"ID": train["ID"], "target": y, "oof_probability": mean_oof}).to_csv(
        out_dir / f"{key}_oof.csv", index=False
    )
    return {
        **metrics,
        "repeat_score_mean": float(rep["score"].mean()),
        "repeat_score_std": float(rep["score"].std(ddof=0)),
        "repeat_auc_std": float(rep["auc"].std(ddof=0)),
        "repeat_f1_std": float(rep["f1"].std(ddof=0)),
    }, repeat_rows


def _shift_auc(train_x: pd.DataFrame, test_x: pd.DataFrame) -> float:
    x = pd.concat([train_x, test_x], ignore_index=True)
    domain = np.concatenate([np.zeros(len(train_x), dtype=int), np.ones(len(test_x), dtype=int)])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros(len(x), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, domain)):
        model = CatBoostClassifier(
            iterations=220, depth=5, learning_rate=0.05, loss_function="Logloss",
            eval_metric="AUC", random_seed=2026 + fold, verbose=False,
            allow_writing_files=False, thread_count=-1,
        )
        model.fit(
            x.iloc[tr_idx], domain[tr_idx], cat_features=categorical_columns(x),
            eval_set=(x.iloc[va_idx], domain[va_idx]), early_stopping_rounds=50, verbose=False,
        )
        oof[va_idx] = model.predict_proba(x.iloc[va_idx])[:, 1]
    return float(roc_auc_score(domain, oof))


def main():
    train, test, _ = load_competition_data()
    out_dir = Path("reports/teleconnection_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = _load_indices(out_dir)

    raw_train = train.drop(columns=[TARGET]).reset_index(drop=True)
    raw_test = test.reset_index(drop=True)
    tele_train = _tele_features(raw_train, indices).reset_index(drop=True)
    tele_test = _tele_features(raw_test, indices).reset_index(drop=True)
    if tele_train.isna().any().any() or tele_test.isna().any().any():
        raise RuntimeError("NOAA teleconnection table does not fully cover competition dates")

    base_all_train = add_interactions(make_view(train, "demographics_time"), "all").reset_index(drop=True)
    dummy = test.copy()
    dummy[TARGET] = 0
    base_all_test = add_interactions(make_view(dummy, "demographics_time"), "all").reset_index(drop=True)
    base_cohort_train = add_fine_demographics(make_view(train, "demographics_time"), "cohort_calendar").reset_index(drop=True)
    base_cohort_test = add_fine_demographics(make_view(dummy, "demographics_time"), "cohort_calendar").reset_index(drop=True)

    dmi_cols = [c for c in tele_train if "dmi" in c and "nino" not in c]
    compact_cols = [
        c for c in tele_train
        if c in {
            "tele_dmi_lag0", "tele_dmi_lag1", "tele_dmi_lag2", "tele_dmi_lag3",
            "tele_dmi_mean0_2", "tele_dmi_mean1_3", "tele_dmi_trend0_2",
            "tele_nino34_lag0", "tele_nino34_lag1", "tele_nino34_lag2",
            "tele_nino34_mean0_2", "tele_nino34_mean1_3", "tele_nino34_trend0_2",
            "tele_iod_positive", "tele_iod_strong_positive", "tele_iod_negative",
            "tele_el_nino", "tele_la_nina", "tele_dmi_x_nino",
        }
    ]
    modes = {
        "reference_all": (base_all_train, base_all_test, []),
        "reference_cohort": (base_cohort_train, base_cohort_test, []),
        "cohort_dmi": (base_cohort_train, base_cohort_test, dmi_cols),
        "cohort_tele_compact": (base_cohort_train, base_cohort_test, compact_cols),
        "cohort_tele_all": (base_cohort_train, base_cohort_test, list(tele_train.columns)),
    }

    summaries, repeats = [], []
    for key, (base_tr, base_te, cols) in modes.items():
        xtr = pd.concat([base_tr, tele_train[cols]], axis=1)
        xte = pd.concat([base_te, tele_test[cols]], axis=1)
        print(f"Teleconnection CV: {key} ({xtr.shape[1]} features) ...", flush=True)
        metrics, rows = _cv(train, xtr, key, out_dir)
        summary = {
            "config": key,
            "n_features": xtr.shape[1],
            "shift_auc": _shift_auc(xtr, xte),
            **metrics,
        }
        summaries.append(summary)
        repeats.extend(rows)
        print(summary, flush=True)

    summary_df = pd.DataFrame(summaries).sort_values(["score", "auc"], ascending=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(repeats).to_csv(out_dir / "repeat_scores.csv", index=False)
    print("\nTeleconnection summary:\n", summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
