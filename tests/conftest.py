import numpy as np
import pandas as pd
import pytest

from src import config as C


def make_city_frame(city="Testville", n_days=400, seed=0, gaps=True):
    """Synthetic city-day table with the same columns as the Kaggle file."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    base = 120 + 60 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 10, n_days)
    df = pd.DataFrame({"City": city, "Date": dates})
    for i, p in enumerate(C.POLLUTANTS):
        df[p] = np.clip(base * (0.2 + 0.05 * i) + rng.normal(0, 5, n_days), 0.1, None)
    df["AQI"] = np.round(np.clip(base + rng.normal(0, 5, n_days), 10, None))
    if gaps:
        df.loc[rng.choice(n_days, 25, replace=False), "AQI"] = np.nan
        df.loc[rng.choice(n_days, 40, replace=False), "PM2.5"] = np.nan
        df["Xylene"] = np.where(rng.random(n_days) < 0.8, np.nan, df["Xylene"])   # mostly missing
    return df


@pytest.fixture
def city_df():
    return make_city_frame()
