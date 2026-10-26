"""Collect every number quoted in the manuscript into paper/paper_numbers.json.

python -m src.export_paper_numbers
"""
import json

import numpy as np
import pandas as pd

from . import config as C
from .analyze import city_means, dm_counts, lockdown_split, overall_table, rank_tests
from .evaluate import friedman_nemenyi, wilcoxon_holm
from .data import build_city_dataset, eligible_cities, load_city_day

OUT = C.ROOT / "paper" / "paper_numbers.json"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def ablation_skill(tag):
    path = C.RESULTS_DIR / f"{tag}_metrics.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path)
    cm = m.groupby(["model", "city"])[["rmse", "skill", "r2"]].mean().groupby("model").mean()
    return cm.to_dict(orient="index")


def data_numbers():
    raw = load_city_day()
    cities = eligible_cities(raw)
    n = {
        "raw_rows": len(raw), "raw_cities": raw.City.nunique(),
        "valid_aqi_rows": int(raw.AQI.notna().sum()),
        "date_min": str(raw.Date.min().date()), "date_max": str(raw.Date.max().date()),
        "n_cities": len(cities), "cities": cities,
        "aqi_mean": float(raw.AQI.mean()), "aqi_median": float(raw.AQI.median()),
        "aqi_max": float(raw.AQI.max()), "aqi_min": float(raw.AQI.min()),
        "missing_pct_valid": (raw[raw.AQI.notna()][C.POLLUTANTS].isna().mean() * 100).round(1).to_dict(),
    }
    ds_path = C.RESULTS_DIR / "main_datasets.csv"
    if ds_path.exists():
        d1 = pd.read_csv(ds_path)
        d1 = d1[d1.horizon == 1]
    else:  # results not ready yet: rebuild the split summaries directly (cheap)
        rows = []
        for c in cities:
            d = build_city_dataset(raw, c, horizon=1)
            rows.append({"city": c, "horizon": 1, "n_train": len(d.y["train"]), "n_val": len(d.y["val"]),
                         "n_test": len(d.y["test"]), "variables": "|".join(d.variables),
                         "dropped": "|".join(d.dropped), "test_start": str(d.dates["test"][0].date()),
                         "test_end": str(d.dates["test"][-1].date())})
        d1 = pd.DataFrame(rows)
    n["datasets_h1"] = d1.to_dict(orient="records")
    n["samples_h1"] = {k: int(d1[k].sum()) for k in ["n_train", "n_val", "n_test"]}
    n["test_start_min"], n["test_end_max"] = d1.test_start.min(), d1.test_end.max()
    n["lockdown_cities_in_test"] = int((pd.to_datetime(d1.test_end) >= "2020-03-25").sum())
    lock_frac = [float((pd.date_range(r.test_start, r.test_end) >= "2020-03-25").mean()) for _, r in d1.iterrows()]
    n["lockdown_test_fraction_mean"] = float(np.mean(lock_frac))
    return n


def main():
    n = data_numbers()
    if not (C.RESULTS_DIR / "main_metrics.csv").exists():
        n["draft"] = True
        OUT.write_text(json.dumps(_clean(n), indent=1))
        print("wrote", OUT, "(data only: results not ready)")
        return
    met = pd.read_csv(C.RESULTS_DIR / "main_metrics.csv")
    ds = pd.read_csv(C.RESULTS_DIR / "main_datasets.csv")
    pred = pd.read_csv(C.RESULTS_DIR / "main_predictions.csv.gz", parse_dates=["date"])
    overall = overall_table(met)
    tests = rank_tests(met)
    _, dm_sum = dm_counts(pred)
    lock = lockdown_split(pred)
    cm = city_means(met)
    n["overall"] = {int(h): g.set_index("model").drop(columns="horizon").to_dict(orient="index")
                    for h, g in overall.groupby("horizon", observed=True)}
    n["ranks"] = {int(h): r["ranks"].to_dict() for h, r in tests.items()}
    n["friedman"] = {int(h): {"chi2": r["chi2"], "p": r["p"], "cd": r["cd"], "best": r["best"]} for h, r in tests.items()}
    n["wilcoxon"] = {int(h): r["wilcoxon"].to_dict(orient="records") for h, r in tests.items()}
    n["dm"] = {int(h): g.set_index("model")[["better", "worse"]].to_dict(orient="index")
               for h, g in dm_sum.groupby("horizon")}
    n["lockdown"] = lock.groupby(["model", "period"]).skill.mean().unstack().to_dict(orient="index")
    n["lockdown_rmse"] = lock.groupby(["model", "period"]).rmse.mean().unstack().to_dict(orient="index")
    n["lockdown_n_cities"] = int(lock.city.nunique())
    pre = lock[lock.period == "pre-lockdown"].pivot(index="city", columns="model", values="rmse")[C.MODEL_ORDER]
    ranks, chi2, p, cd = friedman_nemenyi(pre)
    n["pre_lockdown_tests"] = {"ranks": ranks.to_dict(), "chi2": chi2, "p": p, "cd": cd, "best": ranks.index[0],
                               "n_cities": len(pre),
                               "wilcoxon": wilcoxon_holm(pre, ranks.index[0]).to_dict(orient="records")}
    n["lockdown_wins"] = (lock.pivot_table(index=["city"], columns=["period", "model"], values="skill") > 0).sum().unstack(0).to_dict()
    n["per_city_skill"] = {int(h): g.pivot(index="city", columns="model", values="skill").to_dict(orient="index")
                           for h, g in cm.groupby("horizon")}
    n["per_city_rmse"] = {int(h): g.pivot(index="city", columns="model", values="rmse").to_dict(orient="index")
                          for h, g in cm.groupby("horizon")}
    n["runtime_s"] = met.groupby("model").seconds.mean().to_dict()
    # Does the gap between recurrent and tree models depend on how much training data a city has?
    from scipy.stats import spearmanr
    sk = cm[cm.horizon == 1].pivot(index="city", columns="model", values="skill")
    ntr = ds[ds.horizon == 1].set_index("city").n_train.reindex(sk.index)
    n["data_size"] = {
        "rho_ntrain_lstm": spearmanr(ntr, sk.LSTM).statistic, "p_ntrain_lstm": spearmanr(ntr, sk.LSTM).pvalue,
        "rho_ntrain_gap": spearmanr(ntr, sk.LSTM - sk.XGBoost).statistic,
        "p_ntrain_gap": spearmanr(ntr, sk.LSTM - sk.XGBoost).pvalue,
        "rho_ntrain_xgb": spearmanr(ntr, sk.XGBoost).statistic, "p_ntrain_xgb": spearmanr(ntr, sk.XGBoost).pvalue,
    }

    n["ablation"] = {
        "level_full": ablation_skill("ablation_level"),
        "level_aqi_only": ablation_skill("ablation_aqi_only"),
        "level_pollutants_only": ablation_skill("ablation_pollutants_only"),
        "delta_full_h1": cm[cm.horizon == 1].groupby("model")[["rmse", "skill", "r2"]].mean().to_dict(orient="index"),
    }
    for name in ["shap_by_variable", "shap_by_lag"]:
        p = C.RESULTS_DIR / f"{name}.csv"
        n[name] = pd.read_csv(p, index_col=0).mean().to_dict() if p.exists() else None

    params = met[met.seed == 0][["horizon", "model", "params"]]
    n["params_examples"] = {m: params[params.model == m].params.value_counts().head(3).to_dict()
                            for m in C.MODEL_ORDER[1:]}

    OUT.write_text(json.dumps(_clean(n), indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
