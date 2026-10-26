"""Leakage-free construction of per-city forecasting datasets.

Every model sees exactly the same information: the last WINDOW days of AQI and
pollutant readings, plus calendar features. Anything estimated from data
(which pollutants to keep, fill-in medians, scalers) uses the training period only,
and gap filling is causal (forward fill), so no future value leaks into an input.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


def load_city_day() -> pd.DataFrame:
    df = pd.read_csv(C.DATA_FILE, parse_dates=["Date"])
    return df.drop(columns=["AQI_Bucket"])


def load_weather() -> pd.DataFrame:
    w = pd.read_csv(C.WEATHER_FILE)
    w["Date"] = pd.to_datetime(w["Date"], format="%d-%m-%Y")
    return w[["Date"] + C.WEATHER_VARS].set_index("Date")


def eligible_cities(df: pd.DataFrame) -> list[str]:
    counts = df.dropna(subset=["AQI"]).groupby("City").size()
    return sorted(counts[counts >= C.MIN_VALID_DAYS].index)


@dataclass
class CityDataset:
    city: str
    horizon: int
    variables: list[str]              # per-day input variables (AQI first unless excluded)
    dropped: list[str]                # pollutants dropped for sparsity
    X_seq: dict = field(default_factory=dict)    # split -> (n, WINDOW, n_vars + 4 calendar)
    X_tab: dict = field(default_factory=dict)    # split -> (n, WINDOW * n_vars + 4 calendar)
    y: dict = field(default_factory=dict)        # split -> (n,) AQI at t + horizon
    persistence: dict = field(default_factory=dict)  # split -> (n,) AQI at t
    dates: dict = field(default_factory=dict)    # split -> target dates
    tab_names: list[str] = field(default_factory=list)


def _calendar(dates: pd.DatetimeIndex) -> np.ndarray:
    dow = dates.dayofweek.values
    doy = dates.dayofyear.values
    return np.column_stack([
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
    ]).astype(np.float32)


def build_city_dataset(df: pd.DataFrame, city: str, horizon: int = 1,
                       window: int = C.WINDOW, feature_set: str = "full",
                       weather: pd.DataFrame | None = None) -> CityDataset:
    """feature_set: 'full' (AQI + pollutants), 'aqi_only', or 'pollutants_only'."""
    c = df[df["City"] == city].set_index("Date").sort_index()
    c = c.reindex(pd.date_range(c.index.min(), c.index.max(), freq="D"))
    if weather is not None:
        c = c.join(weather, how="left")

    aqi_obs = c["AQI"].values
    n_days = len(c)

    # Forecast origins t: AQI reported on day t and on the target day t + h, full window available.
    origins = np.array([t for t in range(window - 1, n_days - horizon)
                        if not np.isnan(aqi_obs[t]) and not np.isnan(aqi_obs[t + horizon])])
    n = len(origins)
    n_tr, n_va = int(C.SPLIT[0] * n), int(C.SPLIT[1] * n)
    split_idx = {"train": origins[:n_tr], "val": origins[n_tr:n_tr + n_va], "test": origins[n_tr + n_va:]}

    # Everything up to the last training target date counts as "training period".
    train_end = origins[n_tr - 1] + horizon
    train_rows = c.iloc[: train_end + 1]

    miss = train_rows[C.POLLUTANTS].isna().mean()
    kept = [p for p in C.POLLUTANTS if miss[p] <= C.MISSING_THRESHOLD]
    kept = [p for p in kept if train_rows[p].std(skipna=True) > 1e-6]
    dropped = [p for p in C.POLLUTANTS if p not in kept]

    if feature_set == "full":
        variables = ["AQI"] + kept
    elif feature_set == "aqi_only":
        variables = ["AQI"]
    elif feature_set == "pollutants_only":
        variables = kept
    else:
        raise ValueError(feature_set)
    if weather is not None:
        variables = variables + C.WEATHER_VARS

    filled = c[variables].ffill(limit=C.FFILL_LIMIT)
    medians = train_rows[variables].median()
    filled = filled.fillna(medians).fillna(0.0)
    values = filled.values.astype(np.float32)
    cal = _calendar(c.index)

    ds = CityDataset(city=city, horizon=horizon, variables=variables, dropped=dropped)
    lag_names = [f"{v}_lag{lag}" for lag in range(window - 1, -1, -1) for v in variables]
    ds.tab_names = lag_names + ["dow_sin", "dow_cos", "doy_sin", "doy_cos"]
    for split, idx in split_idx.items():
        win = np.stack([values[t - window + 1: t + 1] for t in idx])            # (n, W, V)
        win_cal = np.stack([cal[t - window + 1: t + 1] for t in idx])           # (n, W, 4)
        ds.X_seq[split] = np.concatenate([win, win_cal], axis=2)
        ds.X_tab[split] = np.concatenate([win.reshape(len(idx), -1), cal[idx + horizon]], axis=1)
        ds.y[split] = aqi_obs[idx + horizon].astype(np.float32)
        ds.persistence[split] = aqi_obs[idx].astype(np.float32)
        ds.dates[split] = c.index[idx + horizon]
    return ds
