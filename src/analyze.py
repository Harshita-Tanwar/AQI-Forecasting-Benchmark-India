"""Aggregate benchmark results into the tables and statistics reported in the paper.

python -m src.analyze            # writes results/tables/*.csv and results/summary.md
"""
import json

import numpy as np
import pandas as pd

from . import config as C
from .evaluate import diebold_mariano, friedman_nemenyi, metrics, wilcoxon_holm

TABLES = C.RESULTS_DIR / "tables"
LOCKDOWN_START = pd.Timestamp("2020-03-25")   # nationwide COVID-19 lockdown in India


def load(tag="main"):
    met = pd.read_csv(C.RESULTS_DIR / f"{tag}_metrics.csv")
    pred = pd.read_csv(C.RESULTS_DIR / f"{tag}_predictions.csv.gz", parse_dates=["date"])
    return met, pred


def city_means(met):
    return met.groupby(["horizon", "model", "city"], as_index=False)[
        ["rmse", "mae", "r2", "mape", "skill", "cat_acc"]].mean()


def overall_table(met):
    cm = city_means(met)
    rows = []
    for (h, m), g in cm.groupby(["horizon", "model"]):
        seed_means = met[(met.horizon == h) & (met.model == m)].groupby("seed")[["rmse", "skill"]].mean()
        rows.append({
            "horizon": h, "model": m,
            "rmse": g.rmse.mean(), "rmse_sd_seeds": seed_means.rmse.std(ddof=1) if len(seed_means) > 1 else 0.0,
            "mae": g.mae.mean(), "r2_mean": g.r2.mean(), "r2_median": g.r2.median(), "mape": g.mape.mean(),
            "skill": g.skill.mean(), "skill_sd_seeds": seed_means.skill.std(ddof=1) if len(seed_means) > 1 else 0.0,
            "skill_median": g.skill.median(), "cat_acc": g.cat_acc.mean(),
            "cities_beating_persistence": int((g.skill > 0).sum()) if m != "Persistence" else np.nan,
        })
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out.model, C.MODEL_ORDER, ordered=True)
    return out.sort_values(["horizon", "model"]).reset_index(drop=True)


def rank_tests(met):
    cm = city_means(met)
    res = {}
    for h, g in cm.groupby("horizon"):
        table = g.pivot(index="city", columns="model", values="rmse")[C.MODEL_ORDER]
        ranks, chi2, p, cd = friedman_nemenyi(table)
        best = ranks.index[0]
        wil = wilcoxon_holm(table, best)
        res[h] = {"ranks": ranks, "chi2": chi2, "p": p, "cd": cd, "best": best, "wilcoxon": wil}
    return res


def dm_counts(pred):
    """Per city: DM test (squared error, seed-averaged) of each model against persistence."""
    rows = []
    for (h, city), g in pred.groupby(["horizon", "city"]):
        g = g.copy()
        g["se"] = (g.y_true - g.y_pred) ** 2
        se = g.groupby(["model", "date"]).se.mean().unstack(0)
        for m in C.MODEL_ORDER[1:]:
            stat, p = diebold_mariano(se[m].values, se["Persistence"].values, h=h)
            rows.append({"horizon": h, "city": city, "model": m, "dm": stat, "p": p,
                         "better": stat < 0 and p < 0.05, "worse": stat > 0 and p < 0.05})
    df = pd.DataFrame(rows)
    summary = df.groupby(["horizon", "model"])[["better", "worse"]].sum().astype(int).reset_index()
    return df, summary


def lockdown_split(pred, horizon=1, min_days=30):
    """Skill against persistence before vs during the COVID-19 lockdown (test period only)."""
    rows = []
    p = pred[pred.horizon == horizon]
    for (city, model), g in p.groupby(["city", "model"]):
        g = g.groupby("date")[["y_true", "y_pred", "persistence"]].mean().reset_index()
        for period, mask in [("pre-lockdown", g.date < LOCKDOWN_START), ("lockdown", g.date >= LOCKDOWN_START)]:
            s = g[mask]
            if len(s) < min_days:
                continue
            m = metrics(s.y_true, s.y_pred, s.persistence)
            rows.append({"city": city, "model": model, "period": period, "n": len(s), **m})
    df = pd.DataFrame(rows)
    both = df.groupby("city").period.nunique()
    df = df[df.city.isin(both[both == 2].index)]
    return df


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    met, pred = load("main")
    datasets = pd.read_csv(C.RESULTS_DIR / "main_datasets.csv")

    overall = overall_table(met)
    overall.round(4).to_csv(TABLES / "overall_by_horizon.csv", index=False)

    cm = city_means(met)
    for h in sorted(cm.horizon.unique()):
        cm[cm.horizon == h].pivot(index="city", columns="model", values="rmse")[C.MODEL_ORDER] \
            .round(2).to_csv(TABLES / f"per_city_rmse_h{h}.csv")
        cm[cm.horizon == h].pivot(index="city", columns="model", values="skill")[C.MODEL_ORDER] \
            .round(4).to_csv(TABLES / f"per_city_skill_h{h}.csv")

    tests = rank_tests(met)
    rank_rows, wil_rows = [], []
    for h, r in tests.items():
        for m, v in r["ranks"].items():
            rank_rows.append({"horizon": h, "model": m, "mean_rank": v, "friedman_chi2": r["chi2"],
                              "friedman_p": r["p"], "nemenyi_cd": r["cd"]})
        w = r["wilcoxon"].copy()
        w.insert(0, "reference", r["best"])
        w.insert(0, "horizon", h)
        wil_rows.append(w)
    pd.DataFrame(rank_rows).round(4).to_csv(TABLES / "friedman_ranks.csv", index=False)
    pd.concat(wil_rows).round(5).to_csv(TABLES / "wilcoxon_holm.csv", index=False)

    dm, dm_sum = dm_counts(pred)
    dm.round(5).to_csv(TABLES / "dm_vs_persistence_per_city.csv", index=False)
    dm_sum.to_csv(TABLES / "dm_vs_persistence_summary.csv", index=False)

    lock = lockdown_split(pred)
    lock.round(4).to_csv(TABLES / "lockdown_split_h1.csv", index=False)
    lock_sum = lock.groupby(["model", "period"])[["rmse", "skill", "r2"]].mean().unstack("period").round(3)

    params = met[met.seed == 0][["horizon", "model", "city", "params"]]
    params.to_csv(TABLES / "selected_hyperparameters.csv", index=False)

    with open(C.RESULTS_DIR / "summary.md", "w") as f:
        f.write("# Benchmark summary (generated by src/analyze.py)\n\n")
        f.write(f"Cities: {datasets.city.nunique()}  |  test windows: "
                f"{datasets.test_start.min()} to {datasets.test_end.max()}\n\n")
        for h in sorted(overall.horizon.unique()):
            f.write(f"## Horizon t+{h}\n\n")
            f.write(overall[overall.horizon == h].drop(columns="horizon").round(3).to_markdown(index=False))
            r = tests[h]
            f.write(f"\n\nFriedman chi2={r['chi2']:.2f}, p={r['p']:.2e}, Nemenyi CD={r['cd']:.2f}\n\n")
            f.write(r["ranks"].round(2).to_frame("mean rank").to_markdown())
            f.write(f"\n\nWilcoxon (Holm) vs best = {r['best']}\n\n")
            f.write(r["wilcoxon"].round(4).to_markdown(index=False))
            f.write("\n\nDM test vs persistence (cities significantly better / worse, p<0.05)\n\n")
            f.write(dm_sum[dm_sum.horizon == h].drop(columns="horizon").to_markdown(index=False))
            f.write("\n\n")
        f.write("## Lockdown split (h=1)\n\n")
        f.write(f"Cities with >=30 test days in both periods: {lock.city.nunique()}\n\n")
        f.write(lock_sum.to_markdown())
        f.write("\n")
    print(open(C.RESULTS_DIR / "summary.md").read())


if __name__ == "__main__":
    main()
