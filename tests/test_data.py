"""The data pipeline must never let information from after the forecast origin into the inputs."""
import numpy as np
import pandas as pd

from src import config as C
from src.data import build_city_dataset, eligible_cities
from tests.conftest import make_city_frame


def test_chronological_split_and_sizes(city_df):
    ds = build_city_dataset(city_df, "Testville", horizon=1)
    n = sum(len(ds.y[s]) for s in ("train", "val", "test"))
    assert abs(len(ds.y["train"]) / n - C.SPLIT[0]) < 0.01
    assert ds.dates["train"].max() < ds.dates["val"].min()
    assert ds.dates["val"].max() < ds.dates["test"].min()


def test_target_and_persistence_are_observed_aqi(city_df):
    h = 3
    ds = build_city_dataset(city_df, "Testville", horizon=h)
    aqi = city_df.set_index("Date")["AQI"]
    for split in ("train", "test"):
        target_dates = ds.dates[split]
        assert np.allclose(ds.y[split], aqi.loc[target_dates].values)
        origin_dates = target_dates - pd.Timedelta(days=h)
        assert np.allclose(ds.persistence[split], aqi.loc[origin_dates].values)
        assert not np.isnan(ds.y[split]).any()


def test_inputs_do_not_change_when_the_future_changes(city_df):
    """Perturbing every value after a forecast origin must leave that origin's inputs untouched."""
    ds = build_city_dataset(city_df, "Testville", horizon=1)
    origin = ds.dates["test"][10] - pd.Timedelta(days=1)
    future = city_df.copy()
    after = future["Date"] > origin
    future.loc[after, C.POLLUTANTS + ["AQI"]] = future.loc[after, C.POLLUTANTS + ["AQI"]] * 5 + 100
    ds2 = build_city_dataset(future, "Testville", horizon=1)
    i = list(ds.dates["test"]).index(origin + pd.Timedelta(days=1))
    assert np.allclose(ds.X_tab["test"][i], ds2.X_tab["test"][i])
    assert np.allclose(ds.X_seq["test"][i], ds2.X_seq["test"][i])


def test_sparse_pollutants_are_dropped_using_training_period_only(city_df):
    ds = build_city_dataset(city_df, "Testville", horizon=1)
    assert "Xylene" in ds.dropped and "Xylene" not in ds.variables
    # Making xylene complete only in the test period must not bring it back.
    late = city_df.copy()
    test_start = ds.dates["test"].min()
    late.loc[late["Date"] >= test_start, "Xylene"] = 1.0
    assert "Xylene" in build_city_dataset(late, "Testville", horizon=1).dropped


def test_window_shapes_and_feature_names(city_df):
    ds = build_city_dataset(city_df, "Testville", horizon=1)
    n_vars = len(ds.variables)
    assert ds.X_seq["train"].shape[1:] == (C.WINDOW, n_vars + 4)
    assert ds.X_tab["train"].shape[1] == C.WINDOW * n_vars + 4 == len(ds.tab_names)
    assert not np.isnan(ds.X_tab["train"]).any()


def test_feature_set_options(city_df):
    aqi_only = build_city_dataset(city_df, "Testville", feature_set="aqi_only")
    pollutants = build_city_dataset(city_df, "Testville", feature_set="pollutants_only")
    assert aqi_only.variables == ["AQI"]
    assert "AQI" not in pollutants.variables


def test_eligible_cities_threshold():
    df = pd.concat([make_city_frame("Big", n_days=400, gaps=False), make_city_frame("Small", n_days=100, gaps=False)])
    assert eligible_cities(df) == ["Big"]
