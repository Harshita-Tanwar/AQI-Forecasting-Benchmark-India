"""Run the forecasting benchmark for every city, horizon and model.

Examples
--------
python -m src.run_benchmark                                   # main study (h = 1, 3, 7)
python -m src.run_benchmark --tag ablation_aqi --feature-set aqi_only --horizons 1 --models Persistence MLR RF XGBoost LSTM
python -m src.run_benchmark --tag weather --cities Bengaluru --weather --horizons 1
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as C
from .data import build_city_dataset, eligible_cities, load_city_day, load_weather
from .evaluate import metrics
from .models import run_model


def run_task(city, horizon, models, seeds, feature_set, use_weather, target="level"):
    import torch
    torch.set_num_threads(1)
    df = load_city_day()
    weather = load_weather() if use_weather else None
    ds = build_city_dataset(df, city, horizon=horizon, feature_set=feature_set, weather=weather)
    info = {"city": city, "horizon": horizon, "n_train": len(ds.y["train"]), "n_val": len(ds.y["val"]),
            "n_test": len(ds.y["test"]), "variables": "|".join(ds.variables), "dropped": "|".join(ds.dropped),
            "test_start": str(ds.dates["test"][0].date()), "test_end": str(ds.dates["test"][-1].date())}
    info["target"] = target
    metric_rows, pred_frames = [], []
    for name in models:
        if target == "delta" and name != "Persistence":
            # Learn the change from today's AQI; add today's AQI back afterwards.
            dsd = copy.deepcopy(ds)
            for split in dsd.y:
                dsd.y[split] = ds.y[split] - ds.persistence[split]
            preds, params, secs = run_model(name, dsd, seeds)
            preds = {s: p + ds.persistence["test"] for s, p in preds.items()}
        else:
            preds, params, secs = run_model(name, ds, seeds)
        for seed, p in preds.items():
            m = metrics(ds.y["test"], p, ds.persistence["test"])
            metric_rows.append({"city": city, "horizon": horizon, "model": name, "seed": seed, **m,
                                "params": json.dumps(params), "seconds": secs / max(len(preds), 1)})
            pred_frames.append(pd.DataFrame({
                "city": city, "horizon": horizon, "model": name, "seed": seed,
                "date": ds.dates["test"], "y_true": ds.y["test"], "y_pred": np.round(p, 3),
                "persistence": ds.persistence["test"]}))
    return info, metric_rows, pd.concat(pred_frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--horizons", type=int, nargs="+", default=C.HORIZONS)
    ap.add_argument("--models", nargs="+", default=C.MODEL_ORDER)
    ap.add_argument("--cities", nargs="+", default=None)
    ap.add_argument("--n-seeds", type=int, default=len(C.SEEDS))
    ap.add_argument("--feature-set", default="full", choices=["full", "aqi_only", "pollutants_only"])
    ap.add_argument("--weather", action="store_true")
    ap.add_argument("--target", default="level", choices=["level", "delta"],
                    help="level: predict AQI directly; delta: predict AQI(t+h) - AQI(t)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    cities = args.cities or eligible_cities(load_city_day())
    seeds = C.SEEDS[: args.n_seeds]
    tasks = [(c, h) for h in args.horizons for c in cities]
    print(f"[{args.tag}] {len(cities)} cities x {len(args.horizons)} horizons, models={args.models}, seeds={seeds}")
    t0 = time.time()
    out = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_task)(c, h, args.models, seeds, args.feature_set, args.weather, args.target) for c, h in tasks)

    C.RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame([o[0] for o in out]).to_csv(C.RESULTS_DIR / f"{args.tag}_datasets.csv", index=False)
    met = pd.DataFrame([r for o in out for r in o[1]])
    met.to_csv(C.RESULTS_DIR / f"{args.tag}_metrics.csv", index=False)
    pd.concat([o[2] for o in out]).to_csv(C.RESULTS_DIR / f"{args.tag}_predictions.csv.gz", index=False)
    print(f"done in {time.time() - t0:.0f}s")
    summary = (met.groupby(["horizon", "model", "city"])[["rmse", "mae", "r2", "skill", "cat_acc"]].mean()
               .groupby(["horizon", "model"]).mean().round(3))
    print(summary.to_string())


if __name__ == "__main__":
    main()
