from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ablate_features import make_view
from features import ID_COL, TARGET, categorical_columns, load_competition_data
from metrics import official_metrics
from robust_validate import SPLIT_SEEDS


POWER_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
MET_PARAMS = (
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "T2MDEW",
    "T2MWET",
    "RH2M",
    "WS10M",
    "PS",
    "ALLSKY_SFC_SW_DWN",
)
RAIN_CANDIDATES = ("PRECTOTCORR", "PRECTOT")
ROLLING_WINDOWS = (14, 30, 56, 84, 180, 365)
RAIN_EVENT_WINDOWS = (30, 84, 180)


def _power_cell(values: pd.Series) -> np.ndarray:
    x = values.to_numpy(dtype=float)
    return np.floor(x * 2.0 + 0.5) / 2.0


def _api_json(latitude: float, longitude: float, start: str, end: str, params) -> dict:
    query = urlencode(
        {
            "parameters": ",".join(params),
            "community": "AG",
            "longitude": f"{longitude:.4f}",
            "latitude": f"{latitude:.4f}",
            "start": start,
            "end": end,
            "format": "JSON",
            "time-standard": "UTC",
        }
    )
    url = f"{POWER_ENDPOINT}?{query}"
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            req = Request(url, headers={"User-Agent": "zindi-climate-research/1.0"})
            with urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(2.0 ** attempt)
    raise RuntimeError(
        f"NASA POWER request failed for ({latitude}, {longitude}) params={params}: {last_error}"
    )


def _parameter_frame(payload: dict) -> pd.DataFrame:
    parameters = payload.get("properties", {}).get("parameter", {})
    if not parameters:
        raise RuntimeError("NASA POWER response contained no parameter data")
    series = {}
    for name, mapping in parameters.items():
        s = pd.Series(mapping, dtype="float64")
        s.index = pd.to_datetime(s.index, format="%Y%m%d", errors="coerce")
        series[name] = s
    out = pd.DataFrame(series).sort_index()
    return out.mask(out <= -900.0)


def _download_cell(latitude, longitude, start, end, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"lat_{latitude:+06.2f}_lon_{longitude:+07.2f}".replace("+", "p").replace("-", "m")
    cache_path = cache_dir / f"{key}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path).set_index("date")

    print(f"POWER download: cell ({latitude:.2f}, {longitude:.2f})", flush=True)
    try:
        met = _parameter_frame(_api_json(latitude, longitude, start, end, MET_PARAMS))
    except Exception as exc:
        # Be resilient to a POWER catalog change in an optional parameter. The
        # meteorological core is documented broadly across POWER communities;
        # humidity/solar are then attempted independently.
        print(f"  full meteorology request failed, using fallback groups: {exc}", flush=True)
        core = ("T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "T2MWET", "WS10M", "PS")
        met = _parameter_frame(_api_json(latitude, longitude, start, end, core))
        for optional in ("RH2M", "ALLSKY_SFC_SW_DWN"):
            try:
                opt = _parameter_frame(_api_json(latitude, longitude, start, end, (optional,)))
                met[optional] = opt[optional]
            except Exception as optional_exc:
                print(f"  optional POWER parameter {optional} unavailable: {optional_exc}", flush=True)
                met[optional] = np.nan
    rain = None
    for candidate in RAIN_CANDIDATES:
        try:
            candidate_frame = _parameter_frame(
                _api_json(latitude, longitude, start, end, (candidate,))
            )
            if candidate in candidate_frame.columns:
                rain = candidate_frame[candidate]
                break
        except Exception as exc:
            print(f"  rainfall parameter {candidate} unavailable: {exc}", flush=True)
    if rain is None:
        raise RuntimeError("No NASA POWER precipitation parameter could be retrieved")
    met["RAIN"] = rain
    met.reset_index(names="date").to_parquet(cache_path, index=False)
    time.sleep(0.35)
    return met


def _slice(history, death_date, days):
    start = death_date - pd.Timedelta(days=days)
    end = death_date - pd.Timedelta(days=1)
    return history.loc[(history.index >= start) & (history.index <= end)]


def _safe_mean(frame, column):
    if column not in frame or frame[column].notna().sum() == 0:
        return np.nan
    return float(frame[column].mean())


def _safe_sum(frame, column):
    if column not in frame or frame[column].notna().sum() == 0:
        return np.nan
    return float(frame[column].sum())


def _record_features(history: pd.DataFrame, death_date: pd.Timestamp) -> dict[str, float]:
    f: dict[str, float] = {}
    means = (
        "T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "T2MWET",
        "RH2M", "WS10M", "PS", "ALLSKY_SFC_SW_DWN",
    )
    windows = {}
    for days in ROLLING_WINDOWS:
        w = _slice(history, death_date, days)
        windows[days] = w
        for column in means:
            f[f"power_{column.lower()}_mean_{days}d"] = _safe_mean(w, column)
        f[f"power_rain_sum_{days}d"] = _safe_sum(w, "RAIN")

    for days in RAIN_EVENT_WINDOWS:
        w = windows[days]
        rain = w["RAIN"].dropna()
        tmax = w["T2M_MAX"].dropna()
        rh = w["RH2M"].dropna()
        f[f"power_rain_days_gt1_{days}d"] = float((rain > 1).sum())
        f[f"power_rain_days_gt5_{days}d"] = float((rain > 5).sum())
        f[f"power_rain_max_{days}d"] = float(rain.max()) if len(rain) else np.nan
        f[f"power_hot_days_gt30_{days}d"] = float((tmax > 30).sum())
        f[f"power_hot_days_gt32_{days}d"] = float((tmax > 32).sum())
        f[f"power_humid_days_gt80_{days}d"] = float((rh > 80).sum())

    for column in ("T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "T2MWET", "RH2M", "WS10M"):
        c = column.lower()
        f[f"power_{c}_delta_14_180"] = f[f"power_{c}_mean_14d"] - f[f"power_{c}_mean_180d"]
        f[f"power_{c}_delta_30_365"] = f[f"power_{c}_mean_30d"] - f[f"power_{c}_mean_365d"]
        f[f"power_{c}_delta_84_365"] = f[f"power_{c}_mean_84d"] - f[f"power_{c}_mean_365d"]

    rain365 = f["power_rain_sum_365d"]
    for days in (30, 56, 84, 180):
        expected = rain365 * days / 365.0 if pd.notna(rain365) else np.nan
        observed = f[f"power_rain_sum_{days}d"]
        f[f"power_rain_anom_{days}_vs365"] = observed - expected
        f[f"power_rain_ratio_{days}_vs365"] = (
            observed / (expected + 1e-3)
            if pd.notna(observed) and pd.notna(expected)
            else np.nan
        )

    for days in (30, 84, 180, 365):
        f[f"power_dewpoint_depression_{days}d"] = (
            f[f"power_t2m_mean_{days}d"] - f[f"power_t2mdew_mean_{days}d"]
        )
        f[f"power_temp_range_{days}d"] = (
            f[f"power_t2m_max_mean_{days}d"] - f[f"power_t2m_min_mean_{days}d"]
        )
    return f


def build_power_features(train, test, out_dir: Path) -> pd.DataFrame:
    feature_path = out_dir / "power_features.csv"
    expected_ids = set(train[ID_COL]) | set(test[ID_COL])
    if feature_path.exists():
        cached = pd.read_csv(feature_path)
        if set(cached[ID_COL]) == expected_ids:
            print("Using cached NASA POWER feature table", flush=True)
            return cached

    raw = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True, sort=False)
    raw["deathdate"] = pd.to_datetime(raw["deathdate"], errors="raise")
    raw["power_lat"] = _power_cell(raw["latitude"])
    raw["power_lon"] = _power_cell(raw["longitude"])
    start = (raw.deathdate.min() - pd.Timedelta(days=400)).strftime("%Y%m%d")
    end = raw.deathdate.max().strftime("%Y%m%d")

    histories = {}
    cells = raw[["power_lat", "power_lon"]].drop_duplicates().sort_values(["power_lat", "power_lon"])
    print(f"NASA POWER cells to hydrate: {len(cells)}", flush=True)
    for row in cells.itertuples(index=False):
        key = (float(row.power_lat), float(row.power_lon))
        histories[key] = _download_cell(key[0], key[1], start, end, out_dir / "cache")

    records = []
    for i, row in enumerate(raw.itertuples(index=False), start=1):
        values = _record_features(
            histories[(float(row.power_lat), float(row.power_lon))],
            pd.Timestamp(row.deathdate),
        )
        values[ID_COL] = getattr(row, ID_COL)
        records.append(values)
        if i % 500 == 0:
            print(f"  engineered POWER features for {i}/{len(raw)} rows", flush=True)
    features = pd.DataFrame(records)
    features.to_csv(feature_path, index=False)
    return features


def _power_columns(power: pd.DataFrame, mode: str) -> list[str]:
    columns = [c for c in power.columns if c != ID_COL]
    if mode == "all":
        return columns
    if mode == "relative":
        needles = (
            "_delta_", "_anom_", "_ratio_", "dewpoint_depression",
            "temp_range", "rain_days_", "hot_days_", "humid_days_",
        )
        return [c for c in columns if any(n in c for n in needles)]
    raise ValueError(mode)


def _prepare_view(raw, power, base_view, mode):
    model_raw = raw.copy()
    if TARGET not in model_raw.columns:
        model_raw[TARGET] = 0
    x = make_view(model_raw, base_view).reset_index(drop=True)
    joined = raw[[ID_COL]].reset_index(drop=True).merge(
        power, on=ID_COL, how="left", validate="one_to_one"
    )
    for column in _power_columns(power, mode):
        x[column] = joined[column].to_numpy()
    return x


def _fit_predict(xtr, ytr, xva, yva, seed):
    model = CatBoostClassifier(
        iterations=360,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    model.fit(
        xtr, ytr,
        cat_features=categorical_columns(xtr),
        eval_set=(xva, yva),
        early_stopping_rounds=70,
        verbose=False,
    )
    return model.predict_proba(xva)[:, 1]


def repeated_target_cv(train, x):
    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()
    oofs, repeat_rows = [], []
    for repeat, split_seed in enumerate(SPLIT_SEEDS):
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=split_seed)
        oof = np.zeros(len(train), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups)):
            oof[va_idx] = _fit_predict(
                x.iloc[tr_idx], y[tr_idx], x.iloc[va_idx], y[va_idx], split_seed + fold
            )
        metrics = official_metrics(y, oof)
        repeat_rows.append({"repeat": repeat, "split_seed": split_seed, **metrics})
        oofs.append(oof)
    mean_oof = np.vstack(oofs).mean(axis=0)
    aggregate = official_metrics(y, mean_oof)
    repeats = pd.DataFrame(repeat_rows)
    aggregate.update(
        repeat_score_mean=float(repeats.score.mean()),
        repeat_score_std=float(repeats.score.std(ddof=0)),
        repeat_auc_std=float(repeats.auc.std(ddof=0)),
        repeat_f1_std=float(repeats.f1.std(ddof=0)),
    )
    return aggregate, repeat_rows


def adversarial_shift_auc(x_train, x_test):
    x = pd.concat([x_train, x_test], ignore_index=True)
    y = np.r_[np.zeros(len(x_train), dtype=int), np.ones(len(x_test), dtype=int)]
    splitter = StratifiedKFold(5, shuffle=True, random_state=2026)
    oof = np.zeros(len(x), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y)):
        model = CatBoostClassifier(
            iterations=220, depth=5, learning_rate=0.04, l2_leaf_reg=5.0,
            loss_function="Logloss", eval_metric="AUC", random_seed=9000 + fold,
            verbose=False, allow_writing_files=False, thread_count=-1,
        )
        model.fit(
            x.iloc[tr_idx], y[tr_idx],
            cat_features=categorical_columns(x),
            eval_set=(x.iloc[va_idx], y[va_idx]),
            early_stopping_rounds=40, verbose=False,
        )
        oof[va_idx] = model.predict_proba(x.iloc[va_idx])[:, 1]
    return float(roc_auc_score(y, oof))


CONFIGS = {
    "demo_power_relative": ("demographics_time", "relative"),
    "demo_power_all": ("demographics_time", "all"),
    "no_spatial_power_relative": ("no_spatial", "relative"),
    "no_spatial_power_all": ("no_spatial", "all"),
}


def main():
    train, test, _ = load_competition_data()
    out_dir = Path("reports/power_climate")
    out_dir.mkdir(parents=True, exist_ok=True)
    power = build_power_features(train, test, out_dir)
    train_power = power[power[ID_COL].isin(set(train[ID_COL]))].copy()
    test_power = power[power[ID_COL].isin(set(test[ID_COL]))].copy()

    summaries, repeats_all = [], []
    for config, (base_view, mode) in CONFIGS.items():
        print(f"POWER repeated CV: {config} ...", flush=True)
        x_train = _prepare_view(train, train_power, base_view, mode)
        x_test = _prepare_view(test, test_power, base_view, mode)
        metrics, repeats = repeated_target_cv(train, x_train)
        summary = {
            "config": config,
            "base_view": base_view,
            "power_mode": mode,
            "n_features": x_train.shape[1],
            "shift_auc": adversarial_shift_auc(x_train, x_test),
            **metrics,
        }
        print(summary, flush=True)
        summaries.append(summary)
        repeats_all.extend({"config": config, **r} for r in repeats)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["score", "shift_auc"], ascending=[False, True]
    )
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(repeats_all).to_csv(out_dir / "repeat_scores.csv", index=False)
    print("\nNASA POWER validation summary:\n", summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
