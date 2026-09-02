from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "is_climate_sensitive"
ID_COL = "ID"


def load_competition_data(raw_dir: str | Path = "data/raw"):
    raw_dir = Path(raw_dir)
    train = pd.read_csv(raw_dir / "Train.csv")
    test = pd.read_csv(raw_dir / "Test.csv")
    climate = pd.read_csv(raw_dir / "climate_features.csv")
    sample = pd.read_csv(raw_dir / "SampleSubmission.csv")

    climate = climate.drop(columns=["deathdate"], errors="ignore")
    train = train.merge(climate, on=ID_COL, how="left", validate="one_to_one")
    test = test.merge(climate, on=ID_COL, how="left", validate="one_to_one")
    return train, test, sample


def _safe_ratio(a, b):
    return a / (b.abs() + 1e-3)


def engineer_features(df: pd.DataFrame, include_location: bool = True) -> pd.DataFrame:
    """Target-free feature engineering used identically for train and test."""
    out = df.copy()
    dt = pd.to_datetime(out["deathdate"], errors="raise")

    out["year"] = dt.dt.year.astype(int)
    out["month"] = dt.dt.month.astype(int)
    out["day_of_year"] = dt.dt.dayofyear.astype(int)
    out["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)

    age_bins = [-1, 0, 1, 2, 4, 9, 14, 24, 44, 64, 200]
    age_labels = ["age0", "age1", "age2", "age3_4", "age5_9", "age10_14", "age15_24", "age25_44", "age45_64", "age65plus"]
    out["age_band"] = pd.cut(out["age"], bins=age_bins, labels=age_labels).astype(str)
    out["age_log1p"] = np.log1p(out["age"].clip(lower=0))
    out["age_sq"] = out["age"] ** 2
    out["is_infant"] = (out["age"] < 1).astype(int)
    out["is_under5"] = (out["age"] < 5).astype(int)
    out["is_child"] = (out["age"] < 15).astype(int)
    out["is_elderly"] = (out["age"] >= 65).astype(int)

    out["temperature_range"] = out["max_temperature"] - out["min_temperature"]
    out["is_rainy_day_current"] = (out["precipitation"] > 0).astype(int)
    out["precip_log1p"] = np.log1p(out["precipitation"].clip(lower=0))

    # Multi-window changes: short-term conditions relative to recent background.
    out["tavg_7_minus_30"] = out["tavg_7d"] - out["tavg_30d"]
    out["tavg_30_minus_90"] = out["tavg_30d"] - out["tavg_90d"]
    out["rain_7_share_30"] = _safe_ratio(out["rain_sum_7d"], out["rain_sum_30d"])
    out["rain_30_share_90"] = _safe_ratio(out["rain_sum_30d"], out["rain_sum_90d"])
    out["rain_intensity_30"] = _safe_ratio(out["rain_sum_30d"], out["rain_days_30d"])
    out["ndvi_30_minus_90"] = out["ndvi_30d"] - out["ndvi_90d"]

    # Epidemiologically motivated effect modification. These are target-free.
    for climate_col in ["tavg_7d", "tavg_30d", "tavg_90d", "rain_sum_7d", "rain_sum_30d", "rain_sum_90d", "max_daily_rain_30d", "ndvi_30d", "ndvi_90d"]:
        out[f"age_x_{climate_col}"] = out["age"] * out[climate_col]
        out[f"under5_x_{climate_col}"] = out["is_under5"] * out[climate_col]
        out[f"elderly_x_{climate_col}"] = out["is_elderly"] * out[climate_col]

    # Coarse spatial cells can capture broad environmental regime without exact village memorisation.
    out["lat_bin_025"] = (out["latitude"] / 0.25).round().astype(int).astype(str)
    out["lon_bin_025"] = (out["longitude"] / 0.25).round().astype(int).astype(str)
    out["spatial_cell_025"] = out["lat_bin_025"] + "_" + out["lon_bin_025"]

    out = out.drop(columns=["deathdate"])
    if not include_location:
        out = out.drop(columns=["location"], errors="ignore")
    return out


def feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in {TARGET, ID_COL}]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return categorical/string columns in a pandas-version-safe way."""
    cols: list[str] = []
    for column in df.columns:
        dtype = df[column].dtype
        if (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        ):
            cols.append(column)
    return cols
