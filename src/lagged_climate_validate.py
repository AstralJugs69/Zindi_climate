from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ablate_features import make_view
from features import ID_COL, TARGET, categorical_columns, load_competition_data
from interaction_validate import add_interactions
from metrics import official_metrics
from robust_validate import SPLIT_SEEDS


API_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = (
    "temperature_2m_max",
    "temperature_2m_mean",
    "temperature_2m_min",
    "precipitation_sum",
)
LAGS = tuple(range(13))


def _download_site_weather(raw: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        frame = pd.read_csv(cache_path)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        return frame

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lat = float(raw["latitude"].astype(float).median())
    lon = float(raw["longitude"].astype(float).median())
    dates = pd.to_datetime(raw["deathdate"], errors="raise")
    start = (dates.min() - pd.Timedelta(days=100)).date().isoformat()
    end = (dates.max() - pd.Timedelta(days=1)).date().isoformat()
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "models": "era5_land",
        "timezone": "Africa/Kampala",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "cell_selection": "nearest",
        # Avoid row-specific elevation/downscaling; this is deliberately one common
        # site-wide climate signal so village geography cannot identify Train/Test.
        "elevation": "nan",
    }

    last_exc = None
    for attempt in range(5):
        try:
            response = requests.get(API_URL, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()
            daily = payload["daily"]
            frame = pd.DataFrame({"date": pd.to_datetime(daily["time"], errors="raise")})
            for var in DAILY_VARS:
                frame[var] = pd.to_numeric(daily[var], errors="coerce")
            if frame[list(DAILY_VARS)].isna().mean().max() > 0.05:
                raise RuntimeError("Open-Meteo returned excessive missing daily values")
            frame.to_csv(cache_path, index=False)
            print(
                f"Cached ERA5-Land site series at requested centroid lat={lat:.5f}, lon={lon:.5f}; "
                f"resolved grid lat={payload.get('latitude')}, lon={payload.get('longitude')}",
                flush=True,
            )
            return frame
        except Exception as exc:
            last_exc = exc
            if attempt == 4:
                break
            time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"Open-Meteo ERA5-Land request failed: {last_exc}")


def _slice(weather: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return weather.loc[(weather["date"] >= start) & (weather["date"] <= end)]


def _weekly_features(raw: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.sort_values("date").reset_index(drop=True)
    tmax95 = float(weather["temperature_2m_max"].quantile(0.95))
    tmax05 = float(weather["temperature_2m_max"].quantile(0.05))
    rain95 = float(weather["precipitation_sum"].quantile(0.95))

    # Day-of-year climate normals are computed only from external climate data.
    clim = weather.copy()
    clim["doy"] = clim["date"].dt.dayofyear.clip(upper=365)
    normals = clim.groupby("doy").agg(
        tmax_mean=("temperature_2m_max", "mean"),
        tmax_std=("temperature_2m_max", "std"),
        rain_mean=("precipitation_sum", "mean"),
        rain_std=("precipitation_sum", "std"),
    )

    rows = []
    for row in raw.itertuples(index=False):
        death = pd.Timestamp(row.deathdate).normalize()
        features: dict[str, float] = {}
        weekly_tmax = []
        weekly_tmean = []
        weekly_tmin = []
        weekly_rain = []
        for lag in LAGS:
            end = death - pd.Timedelta(days=7 * lag + 1)
            start = end - pd.Timedelta(days=6)
            block = _slice(weather, start, end)
            tmax = float(block["temperature_2m_max"].mean())
            tmean = float(block["temperature_2m_mean"].mean())
            tmin = float(block["temperature_2m_min"].mean())
            rain = float(block["precipitation_sum"].sum())
            weekly_tmax.append(tmax)
            weekly_tmean.append(tmean)
            weekly_tmin.append(tmin)
            weekly_rain.append(rain)
            features[f"era5lag_tmax_w{lag}"] = tmax
            features[f"era5lag_tmean_w{lag}"] = tmean
            features[f"era5lag_tmin_w{lag}"] = tmin
            features[f"era5lag_rain_w{lag}"] = rain

        tmax_arr = np.asarray(weekly_tmax)
        rain_arr = np.asarray(weekly_rain)

        # Same-site published malaria-mortality windows.
        features["dlnm_tmax_5_11_mean"] = float(np.mean(tmax_arr[5:12]))
        features["dlnm_tmax_5_11_max"] = float(np.max(tmax_arr[5:12]))
        features["dlnm_tmax_5_11_riskfrac"] = float(
            np.mean((tmax_arr[5:12] >= 25.2) & (tmax_arr[5:12] <= 29.9))
        )
        features["dlnm_rain_2_8_sum"] = float(np.sum(rain_arr[2:9]))
        features["dlnm_rain_2_8_maxweek"] = float(np.max(rain_arr[2:9]))
        features["dlnm_rain_2_8_gt200_count"] = float(np.sum(rain_arr[2:9] > 200.0))
        features["dlnm_rain_lag4"] = float(rain_arr[4])
        features["dlnm_rain_lag4_excess270"] = float(max(rain_arr[4] - 270.0, 0.0))
        features["dlnm_rain_11_12_max"] = float(np.max(rain_arr[11:13]))
        features["dlnm_rain_11_12_gt646"] = float(np.max(rain_arr[11:13]) > 646.0)

        # Published age/sex effect modification.
        is_u5 = float(row.age < 5)
        is_5_14 = float(5 <= row.age < 15)
        is_male = float(str(row.gender).lower().startswith("m"))
        lag8 = float(tmax_arr[8])
        band4_8 = tmax_arr[4:9]
        features["dlnm_u5_tmax_lag8"] = is_u5 * lag8
        features["dlnm_u5_tmax_lag8_risk"] = is_u5 * float(25.6 <= lag8 <= 30.6)
        features["dlnm_5_14_tmax_4_8_mean"] = is_5_14 * float(np.mean(band4_8))
        features["dlnm_5_14_tmax_4_8_riskfrac"] = is_5_14 * float(
            np.mean((band4_8 >= 25.1) & (band4_8 <= 29.6))
        )
        features["dlnm_male_5_14_tmax_4_8_mean"] = (
            is_male * is_5_14 * float(np.mean(band4_8))
        )

        # Acute weather pathway: same-day/short-lag heat and heavy rain.
        recent7 = _slice(weather, death - pd.Timedelta(days=7), death - pd.Timedelta(days=1))
        recent3 = _slice(weather, death - pd.Timedelta(days=3), death - pd.Timedelta(days=1))
        recent1 = _slice(weather, death - pd.Timedelta(days=1), death - pd.Timedelta(days=1))
        features["acute_tmax_1d"] = float(recent1["temperature_2m_max"].mean())
        features["acute_tmax_3d_max"] = float(recent3["temperature_2m_max"].max())
        features["acute_tmax_7d_max"] = float(recent7["temperature_2m_max"].max())
        features["acute_tmax_7d_gt95_count"] = float(
            np.sum(recent7["temperature_2m_max"].to_numpy() > tmax95)
        )
        features["acute_tmax_7d_lt05_count"] = float(
            np.sum(recent7["temperature_2m_max"].to_numpy() < tmax05)
        )
        features["acute_rain_7d_gt95_count"] = float(
            np.sum(recent7["precipitation_sum"].to_numpy() > rain95)
        )

        doy = min(int((death - pd.Timedelta(days=1)).dayofyear), 365)
        normal = normals.loc[doy]
        features["acute_tmax_1d_clim_z"] = (
            features["acute_tmax_1d"] - float(normal["tmax_mean"])
        ) / (float(normal["tmax_std"]) + 0.25)
        features["acute_rain_7d_vs_clim"] = (
            float(recent7["precipitation_sum"].mean()) - float(normal["rain_mean"])
        ) / (float(normal["rain_std"]) + 0.25)
        rows.append(features)

    return pd.DataFrame(rows, index=raw.index)


def _core_columns(frame: pd.DataFrame) -> list[str]:
    return [
        c
        for c in frame.columns
        if c.startswith("dlnm_") or c.startswith("acute_")
    ]


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


def _target_cv(train: pd.DataFrame, x: pd.DataFrame, key: str, out_dir: Path):
    y = train[TARGET].astype(int).to_numpy()
    groups = train["location"].astype(str).to_numpy()
    repeated_oof = []
    repeat_rows = []
    for repeat, split_seed in enumerate(SPLIT_SEEDS):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
        oof = np.zeros(len(train), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups)):
            oof[va_idx] = _fit_predict(
                x.iloc[tr_idx], y[tr_idx], x.iloc[va_idx], y[va_idx], split_seed + fold
            )
        metrics = official_metrics(y, oof)
        repeat_rows.append({"config": key, "repeat": repeat, "split_seed": split_seed, **metrics})
        repeated_oof.append(oof)

    mean_oof = np.vstack(repeated_oof).mean(axis=0)
    aggregate = official_metrics(y, mean_oof)
    repeats = pd.DataFrame(repeat_rows)
    pd.DataFrame(
        {ID_COL: train[ID_COL], "target": y, "oof_probability": mean_oof}
    ).to_csv(out_dir / f"{key}_oof.csv", index=False)
    return {
        **aggregate,
        "repeat_score_mean": float(repeats["score"].mean()),
        "repeat_score_std": float(repeats["score"].std(ddof=0)),
        "repeat_auc_std": float(repeats["auc"].std(ddof=0)),
        "repeat_f1_std": float(repeats["f1"].std(ddof=0)),
    }, repeat_rows


def _shift_auc(train_x: pd.DataFrame, test_x: pd.DataFrame) -> float:
    x = pd.concat([train_x, test_x], ignore_index=True)
    domain = np.concatenate([np.zeros(len(train_x), dtype=int), np.ones(len(test_x), dtype=int)])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros(len(x), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, domain)):
        model = CatBoostClassifier(
            iterations=240,
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
    out_dir = Path("reports/lagged_climate_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_train = train.drop(columns=[TARGET]).reset_index(drop=True)
    raw_all = pd.concat([raw_train, test.reset_index(drop=True)], ignore_index=True)
    weather = _download_site_weather(raw_all, out_dir / "openmeteo_era5_land_site_daily.csv")
    lagged = _weekly_features(raw_all, weather).reset_index(drop=True)
    lagged.to_csv(out_dir / "lagged_climate_features.csv", index=False)
    lag_train = lagged.iloc[: len(train)].reset_index(drop=True)
    lag_test = lagged.iloc[len(train) :].reset_index(drop=True)

    base_train = add_interactions(make_view(train, "demographics_time"), "all").reset_index(drop=True)
    test_dummy = test.copy()
    test_dummy[TARGET] = 0
    base_test = add_interactions(make_view(test_dummy, "demographics_time"), "all").reset_index(drop=True)

    core = _core_columns(lag_train)
    weekly_temp = [c for c in lag_train.columns if c.startswith("era5lag_tmax_")]
    weekly_rain = [c for c in lag_train.columns if c.startswith("era5lag_rain_")]
    modes = {
        "reference": [],
        "dlnm_core": core,
        "dlnm_core_tmax": core + weekly_temp,
        "dlnm_core_rain": core + weekly_rain,
        "dlnm_full_weekly": list(lag_train.columns),
    }

    summaries, repeat_rows = [], []
    for key, cols in modes.items():
        x_train = pd.concat([base_train, lag_train[cols]], axis=1)
        x_test = pd.concat([base_test, lag_test[cols]], axis=1)
        print(f"Lagged climate CV: {key} ({x_train.shape[1]} features) ...", flush=True)
        metrics, repeats = _target_cv(train, x_train, key, out_dir)
        shift = _shift_auc(x_train, x_test)
        row = {
            "config": key,
            "n_features": x_train.shape[1],
            "shift_auc": shift,
            **metrics,
        }
        print(row, flush=True)
        summaries.append(row)
        repeat_rows.extend(repeats)

    summary = pd.DataFrame(summaries).sort_values(
        ["score", "shift_auc"], ascending=[False, True]
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(out_dir / "repeat_scores.csv", index=False)
    print("\nLagged climate summary:\n", summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
