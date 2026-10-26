"""Metrics and statistical tests."""
import numpy as np
import pandas as pd
from scipy import stats

from . import config as C


def aqi_category(values):
    return pd.cut(np.asarray(values), bins=C.AQI_BINS, labels=False)


def metrics(y, p, persistence):
    y, p, persistence = map(np.asarray, (y, p, persistence))
    err = y - p
    rmse = float(np.sqrt(np.mean(err ** 2)))
    rmse_persist = float(np.sqrt(np.mean((y - persistence) ** 2)))
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
        "mape": float(np.mean(np.abs(err) / np.maximum(y, 1.0)) * 100),
        "skill": 1 - rmse / rmse_persist,             # >0 means better than "tomorrow = today"
        "cat_acc": float(np.mean(aqi_category(y) == aqi_category(p))),
    }


def diebold_mariano(loss_a, loss_b, h=1):
    """Two-sided DM test with the Harvey-Leybourne-Newbold small-sample correction.
    Negative statistic means model A has lower loss."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    d_mean = d.mean()
    gamma = [np.mean((d[k:] - d_mean) * (d[:n - k] - d_mean)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return 0.0, 1.0
    dm = d_mean / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)


def friedman_nemenyi(score_table: pd.DataFrame):
    """score_table: rows = cities, columns = models, lower is better."""
    ranks = score_table.rank(axis=1, method="average")
    chi2, p = stats.friedmanchisquare(*[score_table[c].values for c in score_table.columns])
    k, n = score_table.shape[1], score_table.shape[0]
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}[k]
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    return ranks.mean().sort_values(), float(chi2), float(p), float(cd)


def wilcoxon_holm(score_table: pd.DataFrame, reference: str):
    """Paired Wilcoxon signed-rank test of `reference` against every other model, Holm-corrected."""
    rows = []
    for m in score_table.columns:
        if m == reference:
            continue
        stat, p = stats.wilcoxon(score_table[reference], score_table[m])
        rows.append({"model": m, "W": stat, "p_raw": p,
                     "median_diff": float(np.median(score_table[reference] - score_table[m]))})
    out = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    k = len(out)
    adj, running = [], 0.0
    for i, p in enumerate(out["p_raw"]):
        running = max(running, min(1.0, (k - i) * p))
        adj.append(running)
    out["p_holm"] = adj
    return out
