from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ablate_features import make_view
from features import ID_COL, TARGET, categorical_columns, load_competition_data
from metrics import official_metrics
from robust_validate import SPLIT_SEEDS


API_ROOT = "https://climateserv.servirglobal.net/api"
WINDOWS = (7, 14, 30, 56, 84, 90, 120, 180, 365)
MAX_WORKERS = 3


def _coord_key(lat: float, lon: float) -> str:
    return f"{lat:.5f}_{lon:.5f}".replace("-", "m").replace(".", "p")


def _square_geometry(lat: float, lon: float, half_size: float = 0.03) -> dict:
    x0, x1 = lon - half_size, lon + half_size
    y0, y1 = lat - half_size, lat + half_size
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]]],
    }


def _json_get(url: str, *, params: dict, timeout: int = 90, retries: int = 5):
    last_exc = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            if attempt + 1 == retries:
                break
            time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"Request failed after {retries} attempts: {url}: {last_exc}")


def _submit_job(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp) -> str:
    payload = _json_get(
        f"{API_ROOT}/submitDataRequest/",
        params={
            "datatype": 0,
            "begintime": start.strftime("%m/%d/%Y"),
            "endtime": end.strftime("%m/%d/%Y"),
            "intervaltype": 0,
            "operationtype": 5,
            "dateType_Category": "default",
            "isZip_CurrentDataType": "false",
            "geometry": json.dumps(_square_geometry(lat, lon), separators=(",", ":")),
        },
    )
    if isinstance(payload, list) and payload:
        return str(payload[0])
    if isinstance(payload, str):
        return payload
    raise RuntimeError(f"Unexpected ClimateSERV submit response: {payload!r}")


def _progress(job_id: str) -> float:
    payload = _json_get(
        f"{API_ROOT}/getDataRequestProgress/",
        params={"id": job_id},
        timeout=60,
    )
    if isinstance(payload, list) and payload:
        return float(payload[0])
    return float(payload)


def _fetch_job(job_id: str) -> pd.DataFrame:
    payload = _json_get(
        f"{API_ROOT}/getDataFromRequest/",
        params={"id": job_id},
        timeout=120,
    )
    data = payload.get("data", []) if isinstance(payload, dict) else []
    rows = []
    for item in data:
        value_obj = item.get("value", {}) if isinstance(item, dict) else {}
        if isinstance(value_obj, dict):
            value = value_obj.get("avg")
            if value is None and value_obj:
                value = next(iter(value_obj.values()))
        else:
            value = value_obj
        rows.append({"date": item.get("date"), "rain_mm": value})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"ClimateSERV job {job_id} returned no data")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["rain_mm"] = pd.to_numeric(frame["rain_mm"], errors="coerce")
    frame.loc[frame["rain_mm"] <= -9990, "rain_mm"] = np.nan
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    return frame


def _download_one(
    lat: float,
    lon: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_coord_key(lat, lon)}.csv"
    if path.exists():
        return path

    job_id = _submit_job(lat, lon, start, end)
    deadline = time.time() + 20 * 60
    while True:
        progress = _progress(job_id)
        if progress >= 100.0:
            break
        if progress < 0:
            raise RuntimeError(f"ClimateSERV job failed: {job_id}")
        if time.time() > deadline:
            raise TimeoutError(f"ClimateSERV job timed out: {job_id}")
        time.sleep(3)

    frame = _fetch_job(job_id)
    frame.to_csv(path, index=False)
    return path


def ensure_chirps_cache(raw: pd.DataFrame, cache_dir: Path) -> dict[tuple[float, float], Path]:
    dates = pd.to_datetime(raw["deathdate"], errors="raise")
    # Two years of context allow current 365-day history plus a matched prior-year baseline.
    start = (dates.min() - pd.Timedelta(days=740)).normalize()
    end = (dates.max() - pd.Timedelta(days=1)).normalize()

    coords = (
        raw[["latitude", "longitude"]]
        .drop_duplicates()
        .astype(float)
        .itertuples(index=False, name=None)
    )
    coords = list(coords)
    print(
        f"CHIRPS cache: {len(coords)} unique coordinates, {start.date()} to {end.date()}",
        flush=True,
    )

    result: dict[tuple[float, float], Path] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_download_one, lat, lon, start, end, cache_dir): (lat, lon)
            for lat, lon in coords
        }
        done = 0
        for future in as_completed(futures):
            coord = futures[future]
            result[coord] = future.result()
            done += 1
            print(f"  CHIRPS {done}/{len(coords)} cached", flush=True)
    return result


def _window_values(series: pd.DataFrame, end: pd.Timestamp, days: int) -> np.ndarray:
    start = end - pd.Timedelta(days=days - 1)
    values = series.loc[(series["date"] >= start) & (series["date"] <= end), "rain_mm"]
    return values.to_numpy(dtype=float)


def _summaries(values: np.ndarray, days: int) -> dict[str, float]:
    valid = values[np.isfinite(values)]
    prefix = f"chirps_{days}d"
    if valid.size == 0:
        return {
            f"{prefix}_sum": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_wetfrac": np.nan,
            f"{prefix}_heavy10frac": np.nan,
            f"{prefix}_heavy20frac": np.nan,
        }
    return {
        f"{prefix}_sum": float(valid.sum()),
        f"{prefix}_mean": float(valid.mean()),
        f"{prefix}_max": float(valid.max()),
        f"{prefix}_std": float(valid.std(ddof=0)),
        f"{prefix}_wetfrac": float(np.mean(valid >= 1.0)),
        f"{prefix}_heavy10frac": float(np.mean(valid >= 10.0)),
        f"{prefix}_heavy20frac": float(np.mean(valid >= 20.0)),
    }


def build_chirps_features(raw: pd.DataFrame, cache_paths: dict[tuple[float, float], Path]) -> pd.DataFrame:
    series_cache: dict[tuple[float, float], pd.DataFrame] = {}
    for coord, path in cache_paths.items():
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame["rain_mm"] = pd.to_numeric(frame["rain_mm"], errors="coerce")
        series_cache[coord] = frame

    rows = []
    for row in raw.itertuples(index=False):
        coord = (float(row.latitude), float(row.longitude))
        series = series_cache[coord]
        death = pd.Timestamp(row.deathdate)
        end = death - pd.Timedelta(days=1)
        features: dict[str, float] = {}
        for window in WINDOWS:
            features.update(_summaries(_window_values(series, end, window), window))

        # Same-season rainfall one year earlier. This is a direct anomaly reference
        # rather than a broad geographic climatology, so it should transfer better.
        prior_end = end - pd.Timedelta(days=365)
        for window in (30, 84, 180, 365):
            prior = _window_values(series, prior_end, window)
            current = _window_values(series, end, window)
            prior_mean = float(np.nanmean(prior)) if np.isfinite(prior).any() else np.nan
            current_mean = float(np.nanmean(current)) if np.isfinite(current).any() else np.nan
            features[f"chirps_{window}d_prior_year_mean"] = prior_mean
            features[f"chirps_{window}d_yoy_mean_ratio"] = (
                current_mean / (prior_mean + 0.05)
                if np.isfinite(current_mean) and np.isfinite(prior_mean)
                else np.nan
            )
            features[f"chirps_{window}d_yoy_mean_diff"] = (
                current_mean - prior_mean
                if np.isfinite(current_mean) and np.isfinite(prior_mean)
                else np.nan
            )

        # Relative recent-vs-background features; these are the lower-shift candidates.
        for short, long in ((7, 90), (14, 180), (30, 180), (30, 365), (56, 365), (84, 365), (180, 365)):
            s = features[f"chirps_{short}d_mean"]
            l = features[f"chirps_{long}d_mean"]
            features[f"chirps_mean_ratio_{short}_{long}"] = (
                s / (l + 0.05) if np.isfinite(s) and np.isfinite(l) else np.nan
            )
            features[f"chirps_mean_diff_{short}_{long}"] = (
                s - l if np.isfinite(s) and np.isfinite(l) else np.nan
            )
        rows.append(features)
    result = pd.DataFrame(rows, index=raw.index)
    return result


def _relative_columns(frame: pd.DataFrame) -> list[str]:
    tokens = (
        "_ratio_",
        "_diff_",
        "_yoy_",
        "wetfrac",
        "heavy10frac",
        "heavy20frac",
    )
    return [c for c in frame.columns if any(token in c for token in tokens)]


def _fit_target_cv(train: pd.DataFrame, x: pd.DataFrame):
    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()
    repeated_oof = []
    repeat_rows = []
    for repeat, split_seed in enumerate(SPLIT_SEEDS):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
        oof = np.zeros(len(train), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups)):
            model = CatBoostClassifier(
                iterations=420,
                depth=6,
                learning_rate=0.03,
                l2_leaf_reg=6.0,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=split_seed + fold,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )
            model.fit(
                x.iloc[tr_idx],
                y[tr_idx],
                cat_features=categorical_columns(x),
                eval_set=(x.iloc[va_idx], y[va_idx]),
                early_stopping_rounds=70,
                verbose=False,
            )
            oof[va_idx] = model.predict_proba(x.iloc[va_idx])[:, 1]
        metrics = official_metrics(y, oof)
        repeat_rows.append({"repeat": repeat, "split_seed": split_seed, **metrics})
        repeated_oof.append(oof)

    mean_oof = np.vstack(repeated_oof).mean(axis=0)
    aggregate = official_metrics(y, mean_oof)
    repeats = pd.DataFrame(repeat_rows)
    return {
        **aggregate,
        "repeat_score_mean": float(repeats["score"].mean()),
        "repeat_score_std": float(repeats["score"].std(ddof=0)),
        "repeat_auc_std": float(repeats["auc"].std(ddof=0)),
        "repeat_f1_std": float(repeats["f1"].std(ddof=0)),
    }, repeat_rows, mean_oof


def _shift_auc(train_x: pd.DataFrame, test_x: pd.DataFrame) -> float:
    x = pd.concat([train_x, test_x], ignore_index=True)
    domain = np.concatenate([np.zeros(len(train_x), dtype=int), np.ones(len(test_x), dtype=int)])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros(len(x), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, domain)):
        model = CatBoostClassifier(
            iterations=260,
            depth=5,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=2026 + fold,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(
            x.iloc[tr_idx],
            domain[tr_idx],
            cat_features=categorical_columns(x),
            eval_set=(x.iloc[va_idx], domain[va_idx]),
            early_stopping_rounds=50,
            verbose=False,
        )
        oof[va_idx] = model.predict_proba(x.iloc[va_idx])[:, 1]
    return float(roc_auc_score(domain, oof))


def main():
    train, test, _ = load_competition_data()
    report_dir = Path("reports/chirps")
    cache_dir = report_dir / "cache"
    report_dir.mkdir(parents=True, exist_ok=True)

    raw_train = train.drop(columns=[TARGET]).copy()
    raw_all = pd.concat([raw_train, test], ignore_index=True)
    cache_paths = ensure_chirps_cache(raw_all, cache_dir)
    chirps_all = build_chirps_features(raw_all, cache_paths)
    chirps_train = chirps_all.iloc[: len(train)].reset_index(drop=True)
    chirps_test = chirps_all.iloc[len(train) :].reset_index(drop=True)
    chirps_all.to_csv(report_dir / "chirps_features.csv", index=False)

    base_train = make_view(train, "demographics_time").reset_index(drop=True)
    test_dummy = test.copy()
    test_dummy[TARGET] = 0
    base_test = make_view(test_dummy, "demographics_time").reset_index(drop=True)

    relative = _relative_columns(chirps_train)
    modes = {
        "relative": relative,
        "all": list(chirps_train.columns),
    }

    summaries = []
    repeats_all = []
    for mode, columns in modes.items():
        x_train = pd.concat([base_train, chirps_train[columns]], axis=1)
        x_test = pd.concat([base_test, chirps_test[columns]], axis=1)
        print(f"CHIRPS validation: {mode} ({x_train.shape[1]} features) ...", flush=True)
        metrics, repeats, oof = _fit_target_cv(train, x_train)
        shift = _shift_auc(x_train, x_test)
        summary = {
            "config": f"demo_chirps_{mode}",
            "n_features": x_train.shape[1],
            "shift_auc": shift,
            **metrics,
        }
        print(summary, flush=True)
        summaries.append(summary)
        for row in repeats:
            repeats_all.append({"config": f"demo_chirps_{mode}", **row})
        pd.DataFrame(
            {ID_COL: train[ID_COL], "target": train[TARGET], "oof_probability": oof}
        ).to_csv(report_dir / f"demo_chirps_{mode}_oof.csv", index=False)

    summary_df = pd.DataFrame(summaries).sort_values("score", ascending=False)
    summary_df.to_csv(report_dir / "summary.csv", index=False)
    pd.DataFrame(repeats_all).to_csv(report_dir / "repeat_scores.csv", index=False)
    print("\nCHIRPS summary:\n", summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
