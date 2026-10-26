"""SHAP attribution for the XGBoost forecaster (t+1, change-from-today target).

python -m src.run_shap      # writes results/shap_by_variable.csv and results/shap_by_lag.csv
"""
import json

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed

from . import config as C
from .data import build_city_dataset, eligible_cities, load_city_day
from xgboost import XGBRegressor


def city_shap(city, params):
    df = load_city_day()
    ds = build_city_dataset(df, city, horizon=1)
    y = {k: ds.y[k] - ds.persistence[k] for k in ds.y}
    m = XGBRegressor(n_estimators=2000, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                     early_stopping_rounds=100, tree_method="hist", n_jobs=1, random_state=0, **params)
    m.fit(ds.X_tab["train"], y["train"], eval_set=[(ds.X_tab["val"], y["val"])], verbose=False)
    sv = shap.TreeExplainer(m).shap_values(ds.X_tab["test"])
    imp = pd.Series(np.abs(sv).mean(axis=0), index=ds.tab_names)
    imp = imp / imp.sum()                                       # share of total attribution in this city
    var = imp.groupby(lambda n: n.rsplit("_lag", 1)[0] if "_lag" in n else "Calendar").sum()
    lag = imp[[n for n in imp.index if "_lag" in n]].groupby(lambda n: int(n.rsplit("_lag", 1)[1])).sum()
    return city, var, lag


def main():
    met = pd.read_csv(C.RESULTS_DIR / "main_metrics.csv")
    sel = met[(met.model == "XGBoost") & (met.horizon == 1) & (met.seed == 0)].set_index("city").params
    cities = eligible_cities(load_city_day())
    out = Parallel(n_jobs=9)(delayed(city_shap)(c, json.loads(sel[c])) for c in cities)
    var = pd.DataFrame({c: v for c, v, _ in out}).T.fillna(0.0)
    lag = pd.DataFrame({c: l for c, _, l in out}).T.fillna(0.0)
    var.round(5).to_csv(C.RESULTS_DIR / "shap_by_variable.csv")
    lag.round(5).to_csv(C.RESULTS_DIR / "shap_by_lag.csv")
    print(var.mean().sort_values(ascending=False).round(3))
    print(lag.mean().round(3))


if __name__ == "__main__":
    main()
