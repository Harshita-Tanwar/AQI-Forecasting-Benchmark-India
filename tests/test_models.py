"""Smoke tests: every forecaster trains on a small synthetic city and returns sensible test predictions."""
import numpy as np
import pytest

from src.data import build_city_dataset
from src.models import RUNNERS, run_model
from tests.conftest import make_city_frame


@pytest.fixture(scope="module")
def ds():
    # Two years, so the training period covers every season (as it does for the real cities).
    return build_city_dataset(make_city_frame(n_days=760, gaps=False), "Testville", horizon=1)


@pytest.mark.parametrize("name", ["Persistence", "MLR", "SVR", "RF", "XGBoost", "LSTM", "GRU"])
def test_each_model_runs(ds, name):
    preds, params, secs = run_model(name, ds, seeds=[0])
    assert 0 in preds
    p = np.asarray(preds[0])
    assert p.shape == ds.y["test"].shape
    assert np.isfinite(p).all()
    # On a smooth seasonal series every model should be in the right ballpark.
    assert np.sqrt(np.mean((p - ds.y["test"]) ** 2)) < 3 * ds.y["test"].std() + 1


def test_persistence_returns_today(ds):
    preds, _, _ = run_model("Persistence", ds, seeds=[0])
    assert np.array_equal(preds[0], ds.persistence["test"])


def test_stochastic_models_are_reproducible(ds):
    a, _, _ = run_model("XGBoost", ds, seeds=[0])
    b, _, _ = run_model("XGBoost", ds, seeds=[0])
    assert np.allclose(a[0], b[0])


def test_all_runners_registered():
    assert set(RUNNERS) == {"Persistence", "MLR", "SVR", "RF", "XGBoost", "LSTM", "GRU"}
