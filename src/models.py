"""Forecasting models. Each runner tunes on the validation split, trains on the
training split only, and returns test predictions for every seed."""
import itertools
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from .data import CityDataset


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _scaled(ds: CityDataset):
    sx = StandardScaler().fit(ds.X_tab["train"])
    return {k: sx.transform(v) for k, v in ds.X_tab.items()}


def _select(grid, fit_fn, ds):
    """Fit every config on train, keep the one with the lowest validation RMSE."""
    best = None
    for params in grid:
        model, pred_val = fit_fn(params, seed=0)
        score = _rmse(ds.y["val"], pred_val)
        if best is None or score < best[0]:
            best = (score, params)
    return best[1]


def run_persistence(ds: CityDataset, seeds):
    return {s: ds.persistence["test"].copy() for s in seeds[:1]}, {}


def run_mlr(ds: CityDataset, seeds):
    X = _scaled(ds)
    grid = [{"alpha": a} for a in [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]]

    def fit(p, seed):
        m = Ridge(alpha=p["alpha"]).fit(X["train"], ds.y["train"])
        return m, m.predict(X["val"])

    best = _select(grid, fit, ds)
    m, _ = fit(best, 0)
    return {s: m.predict(X["test"]) for s in seeds[:1]}, best


def run_svr(ds: CityDataset, seeds):
    X = _scaled(ds)
    sy = StandardScaler().fit(ds.y["train"].reshape(-1, 1))
    ytr = sy.transform(ds.y["train"].reshape(-1, 1)).ravel()
    inv = lambda z: sy.inverse_transform(z.reshape(-1, 1)).ravel()
    grid = [{"C": c, "epsilon": e} for c, e in itertools.product([1.0, 10.0, 100.0], [0.05, 0.1])]

    def fit(p, seed):
        m = SVR(kernel="rbf", gamma="scale", C=p["C"], epsilon=p["epsilon"]).fit(X["train"], ytr)
        return m, inv(m.predict(X["val"]))

    best = _select(grid, fit, ds)
    m, _ = fit(best, 0)
    return {s: inv(m.predict(X["test"])) for s in seeds[:1]}, best


def run_rf(ds: CityDataset, seeds):
    grid = [{"max_depth": d, "min_samples_leaf": l, "max_features": f}
            for d, l, f in itertools.product([None, 15], [1, 5], [0.33, 1.0])]

    def fit(p, seed):
        m = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1, **p)
        m.fit(ds.X_tab["train"], ds.y["train"])
        return m, m.predict(ds.X_tab["val"])

    best = _select(grid, fit, ds)
    preds = {}
    for s in seeds:
        m, _ = fit(best, s)
        preds[s] = m.predict(ds.X_tab["test"])
    return preds, best


def run_xgb(ds: CityDataset, seeds, return_model=False):
    grid = [{"max_depth": d, "min_child_weight": w} for d, w in itertools.product([3, 5, 7], [1, 5])]

    def fit(p, seed):
        m = XGBRegressor(n_estimators=2000, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                         early_stopping_rounds=100, tree_method="hist", n_jobs=1,
                         random_state=seed, **p)
        m.fit(ds.X_tab["train"], ds.y["train"], eval_set=[(ds.X_tab["val"], ds.y["val"])], verbose=False)
        return m, m.predict(ds.X_tab["val"])

    best = _select(grid, fit, ds)
    preds, models = {}, {}
    for s in seeds:
        m, _ = fit(best, s)
        preds[s] = m.predict(ds.X_tab["test"])
        models[s] = m
    if return_model:
        return preds, best, models
    return preds, best


class RecurrentRegressor(nn.Module):
    def __init__(self, cell, n_in, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        rnn = nn.LSTM if cell == "LSTM" else nn.GRU
        self.rnn = rnn(n_in, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _train_rnn(cell, ds, Xs, ys, hidden, seed, max_epochs=300, patience=30, batch=32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RecurrentRegressor(cell, Xs["train"].shape[2], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    loss_fn = nn.MSELoss()
    Xtr, ytr = torch.from_numpy(Xs["train"]), torch.from_numpy(ys)
    Xva = torch.from_numpy(Xs["val"])
    yva = ds.y["val"]
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad = np.inf, None, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva).numpy()
        val_rmse = _rmse(yva, Xs["inv"](pv))
        sched.step(val_rmse)
        if val_rmse < best - 1e-4:
            best, bad = val_rmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience and epoch >= 20:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, epoch + 1


def run_rnn(cell, ds: CityDataset, seeds):
    n, w, f = ds.X_seq["train"].shape
    sx = StandardScaler().fit(ds.X_seq["train"].reshape(-1, f))
    Xs = {k: sx.transform(v.reshape(-1, f)).reshape(v.shape).astype(np.float32) for k, v in ds.X_seq.items()}
    sy = StandardScaler().fit(ds.y["train"].reshape(-1, 1))
    ys = sy.transform(ds.y["train"].reshape(-1, 1)).ravel().astype(np.float32)
    Xs["inv"] = lambda z: sy.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel()

    # Pick the hidden size on validation with seed 0, then train all seeds.
    best_hidden, best_val = None, np.inf
    cache = {}
    for hidden in [32, 64]:
        model, val, _ = _train_rnn(cell, ds, Xs, ys, hidden, seed=0)
        cache[hidden] = model
        if val < best_val:
            best_hidden, best_val = hidden, val
    preds = {}
    for s in seeds:
        model = cache[best_hidden] if s == 0 else _train_rnn(cell, ds, Xs, ys, best_hidden, seed=s)[0]
        with torch.no_grad():
            preds[s] = Xs["inv"](model(torch.from_numpy(Xs["test"])).numpy())
    return preds, {"hidden": best_hidden}


RUNNERS = {
    "Persistence": run_persistence,
    "MLR": run_mlr,
    "SVR": run_svr,
    "RF": run_rf,
    "XGBoost": run_xgb,
    "LSTM": lambda ds, seeds: run_rnn("LSTM", ds, seeds),
    "GRU": lambda ds, seeds: run_rnn("GRU", ds, seeds),
}


def run_model(name, ds, seeds):
    t0 = time.time()
    preds, params = RUNNERS[name](ds, seeds)
    return preds, params, time.time() - t0
