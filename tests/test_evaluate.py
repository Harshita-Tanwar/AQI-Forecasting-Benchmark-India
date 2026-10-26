import numpy as np
import pandas as pd
import pytest

from src.evaluate import aqi_category, diebold_mariano, friedman_nemenyi, metrics, wilcoxon_holm


def test_metrics_on_known_values():
    y = np.array([100.0, 200.0, 300.0])
    m = metrics(y, y + 10, persistence=y + 20)
    assert m["rmse"] == pytest.approx(10.0)
    assert m["mae"] == pytest.approx(10.0)
    assert m["skill"] == pytest.approx(0.5)          # half the RMSE of persistence


def test_persistence_has_zero_skill():
    y = np.array([50.0, 80.0, 120.0, 90.0])
    p = np.array([40.0, 50.0, 80.0, 120.0])
    assert metrics(y, p, persistence=p)["skill"] == pytest.approx(0.0)


def test_cpcb_categories():
    values = [10, 50, 51, 100, 150, 250, 350, 450]
    assert list(aqi_category(values)) == [0, 0, 1, 1, 2, 3, 4, 5]


def test_diebold_mariano():
    rng = np.random.default_rng(1)
    loss = rng.random(200)
    assert diebold_mariano(loss, loss) == (0.0, 1.0)            # identical losses: no difference
    stat, p = diebold_mariano(loss, loss + 1.0 + 0.1 * rng.random(200))
    assert stat < 0 and p < 1e-6                                # model A clearly better


def test_friedman_nemenyi_critical_difference():
    rng = np.random.default_rng(2)
    table = pd.DataFrame(rng.random((20, 7)), columns=list("ABCDEFG"))
    table["A"] -= 1.0                                           # A is best everywhere
    ranks, chi2, p, cd = friedman_nemenyi(table)
    assert ranks.index[0] == "A" and ranks.iloc[0] == pytest.approx(1.0)
    assert cd == pytest.approx(2.949 * np.sqrt(7 * 8 / (6 * 20)))
    assert p < 0.001


def test_wilcoxon_holm_adjusted_p_values_are_monotone():
    rng = np.random.default_rng(3)
    table = pd.DataFrame(rng.random((20, 4)), columns=["best", "b", "c", "d"])
    table["best"] -= 0.3
    out = wilcoxon_holm(table, "best")
    assert set(out.model) == {"b", "c", "d"}
    assert (out.p_holm.diff().dropna() >= 0).all()
    assert (out.p_holm >= out.p_raw).all()
