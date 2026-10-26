"""Paper figures (300 dpi PNG, sized for one IEEE column).

python -m src.make_figures
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

from . import config as C
from .analyze import city_means, lockdown_split, overall_table, rank_tests
from .data import eligible_cities, load_city_day

COL_W = 3.45                     # inches, one IEEE column
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
HORIZON_COLORS = {1: "#2a78d6", 3: "#eb6834", 7: "#1baf7a"}   # validated categorical slots 1-3
HORIZON_HATCH = {1: "", 3: "////", 7: "...."}
SEQ = "#2a78d6"
DIVERGING = LinearSegmentedColormap.from_list(
    "skill", ["#c43b3a", "#e88c8b", "#f0efec", "#86b6ef", "#1c5cab"])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "grid.color": GRID, "grid.linewidth": 0.5, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "hatch.linewidth": 0.5, "hatch.color": "white",
})
LABEL = {**C.MODEL_LABELS, "MLR": "MLR", "SVR": "SVR", "RF": "RF"}


def save(fig, name):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES_DIR / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def fig_pipeline():
    steps = [
        ("Data", "CPCB city-day records (Kaggle): 26 cities, 2015 to 2020"),
        ("City filter", "≥ 300 days with a reported AQI  →  20 cities"),
        ("Chronological split", "70% train | 10% validation | 20% test (per city)"),
        ("Train-only preprocessing", "drop pollutants >50% missing; causal forward-fill (≤ 7 d);\ntraining medians; scalers fitted on training data only"),
        ("Inputs", "last 7 days of AQI + pollutants, calendar terms"),
        ("Models (target: AQI change)", "Persistence | MLR | SVR | RF | XGBoost | LSTM | GRU\nvalidation-tuned, 5 seeds"),
        ("Evaluation (t+1, t+3, t+7)", "RMSE, MAE, R², skill vs persistence, AQI category accuracy;\nFriedman and Nemenyi, Wilcoxon (Holm), Diebold-Mariano, SHAP"),
    ]
    fig, ax = plt.subplots(figsize=(COL_W, 3.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(steps))
    ax.axis("off")
    for i, (title, body) in enumerate(steps):
        y = len(steps) - i - 0.5
        accent = i == 5
        box = FancyBboxPatch((0.03, y - 0.33), 0.94, 0.66, boxstyle="round,pad=0.01,rounding_size=0.03",
                             facecolor="#cde2fb" if accent else "#f4f3f0", edgecolor=SEQ if accent else MUTED,
                             linewidth=0.8)
        ax.add_patch(box)
        two = "\n" in body
        ax.text(0.5, y + (0.2 if two else 0.14), title, ha="center", va="center", fontsize=7.2, fontweight="bold", color=INK)
        ax.text(0.5, y - (0.1 if two else 0.12), body, ha="center", va="center", fontsize=6.2, color=INK_2, linespacing=1.1)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.66), xytext=(0.5, y - 0.345),
                        arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.8, mutation_scale=7))
    save(fig, "fig1_pipeline")


def fig_data_overview():
    df = load_city_day()
    cities = eligible_cities(df)
    d = df[df.City.isin(cities)].dropna(subset=["AQI"])
    order = d.groupby("City").AQI.median().sort_values().index
    fig, ax = plt.subplots(figsize=(COL_W, 3.6))
    data = [d[d.City == c].AQI.values for c in order]
    bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=1.5, markerfacecolor=MUTED, markeredgecolor="none"),
                    medianprops=dict(color=INK, linewidth=0.9), whiskerprops=dict(color=MUTED, linewidth=0.6),
                    capprops=dict(color=MUTED, linewidth=0.6), boxprops=dict(linewidth=0.5, edgecolor=SEQ))
    for b in bp["boxes"]:
        b.set_facecolor("#b7d3f6")
    ax.set_yticks(range(1, len(order) + 1), [f"{c} ({len(x)})" for c, x in zip(order, data)])
    ax.set_xscale("log")
    ax.set_xticks([25, 50, 100, 200, 400, 1000], ["25", "50", "100", "200", "400", "1000"])
    for edge in [50, 100, 200, 300, 400]:
        ax.axvline(edge, color=GRID, linewidth=0.6, zorder=0)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Daily AQI (log scale); grid lines mark CPCB category edges")
    ax.tick_params(axis="y", length=0)
    save(fig, "fig2_aqi_distribution")


def fig_skill_by_horizon(overall):
    models = C.MODEL_ORDER[1:]
    horizons = sorted(overall.horizon.unique())
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    width = 0.8 / len(horizons)
    x = np.arange(len(models))
    for j, h in enumerate(horizons):
        o = overall[overall.horizon == h].set_index("model").loc[models]
        ax.bar(x + (j - 1) * width, o.skill, width * 0.92, yerr=o.skill_sd_seeds, color=HORIZON_COLORS[h],
               hatch=HORIZON_HATCH[h], edgecolor="white", linewidth=0.3, label=f"t+{h}",
               error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor=INK_2))
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.text(len(models) - 0.5, 0.004, "persistence = 0", ha="right", va="bottom", fontsize=6.5, color=INK_2)
    ax.set_xticks(x, [LABEL[m] for m in models])
    ax.set_ylabel("Mean skill vs persistence")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(title="Horizon", ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.18),
              title_fontsize=7, handlelength=1.4, columnspacing=1.0)
    save(fig, "fig3_skill_by_horizon")


def fig_city_heatmap(met, horizon=1):
    cm = city_means(met)
    t = cm[cm.horizon == horizon].pivot(index="city", columns="model", values="skill")[C.MODEL_ORDER[1:]]
    t = t.loc[t.max(axis=1).sort_values(ascending=False).index]
    lim = 0.4
    fig, ax = plt.subplots(figsize=(COL_W, 3.6))
    im = ax.imshow(t.clip(-lim, lim).values, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            ax.text(j, i, f"{v:.2f}".replace("-0.", "-.").replace("0.", "."), ha="center", va="center",
                    fontsize=5.6, color="white" if abs(v) > 0.28 else INK)
    ax.set_xticks(range(t.shape[1]), [LABEL[m] for m in t.columns])
    ax.set_yticks(range(t.shape[0]), t.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(t.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(t.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(f"Skill vs persistence (t+{horizon}); clipped at +/-{lim}", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    save(fig, f"fig4_city_skill_h{horizon}")


def fig_ranks(tests, pre=None):
    horizons = sorted(tests)
    order = tests[horizons[0]]["ranks"].index.tolist()          # models sorted by t+1 rank
    markers = {1: "o", 3: "s", 7: "D"}
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    y = np.arange(len(order))[::-1]
    for k, h in enumerate(horizons):
        r = tests[h]["ranks"].reindex(order)
        ax.scatter(r.values, y + (1 - k) * 0.22, s=22, marker=markers[h], color=HORIZON_COLORS[h],
                   edgecolor="white", linewidth=0.6, zorder=3,
                   label=f"t+{h} (Friedman p = {tests[h]['p']:.2f})")
    if pre is not None:
        r = pre["ranks"].reindex(order)
        ax.scatter(r.values, y, s=34, marker="o", facecolor="none", edgecolor=INK, linewidth=0.9, zorder=4,
                   label=f"t+1, pre-lockdown only (p = {pre['p']:.3f})")
    ax.set_yticks(y, [LABEL[m] for m in order])
    ax.set_xlim(1, 7)
    ax.set_ylim(-0.7, len(order) - 0.1)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    cd = tests[horizons[0]]["cd"]
    cx, cy = 5.85, len(order) - 1
    ax.plot([cx - cd / 2, cx + cd / 2], [cy, cy], color=INK, linewidth=1.4)
    ax.plot([cx - cd / 2] * 2, [cy - 0.15, cy + 0.15], color=INK, linewidth=1.0)
    ax.plot([cx + cd / 2] * 2, [cy - 0.15, cy + 0.15], color=INK, linewidth=1.0)
    ax.text(cx, cy - 0.25, f"Nemenyi CD = {cd:.2f}", ha="center", va="top", fontsize=6.5)
    ax.set_xlabel("Mean rank across 20 cities (RMSE; 1 = best)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=6.3,
              handletextpad=0.2, columnspacing=0.8, borderaxespad=0.2)
    save(fig, "fig5_mean_ranks")


def fig_timeseries(pred, city="Delhi", horizon=1, start="2019-10-01", end="2020-01-31", models=("XGBoost", "LSTM")):
    p = pred[(pred.city == city) & (pred.horizon == horizon)]
    p = p[(p.date >= start) & (p.date <= end)]
    mean = p.groupby(["model", "date"]).y_pred.mean().unstack(0)
    truth = p.groupby("date").y_true.first()
    pers = p.groupby("date").persistence.first()
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    ax.plot(truth.index, truth.values, color=INK, linewidth=1.3, label="Observed")
    ax.plot(pers.index, pers.values, color=HORIZON_COLORS[3], linewidth=1.0, linestyle=(0, (3, 1.5)),
            label="Persistence")
    styles = {models[0]: dict(color=HORIZON_COLORS[1], linewidth=1.1),
              models[1]: dict(color=HORIZON_COLORS[7], linewidth=1.0, linestyle=(0, (1, 1)))}
    for m in models:
        ax.plot(mean.index, mean[m].values, label=LABEL[m], **styles[m])
    ax.set_ylabel("AQI")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.2), handlelength=1.8, columnspacing=0.9)
    fig.autofmt_xdate(rotation=0, ha="center")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    save(fig, f"fig6_timeseries_{city.lower()}")


def fig_shap():
    try:
        var = pd.read_csv(C.RESULTS_DIR / "shap_by_variable.csv", index_col=0)
        lag = pd.read_csv(C.RESULTS_DIR / "shap_by_lag.csv", index_col=0)
    except FileNotFoundError:
        print("skip SHAP figure (run src.run_shap first)")
        return
    v = var.mean().sort_values(ascending=True).tail(10)
    lag.columns = lag.columns.astype(int)
    l = lag.mean().sort_index()
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 2.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
    a.barh(v.index, v.values * 100, color=SEQ, height=0.65)
    a.set_xlabel("Share of |SHAP| (%)")
    a.set_title("(a) by input variable", loc="left", fontsize=7.5)
    a.tick_params(axis="y", length=0)
    a.xaxis.grid(True)
    a.set_axisbelow(True)
    b.bar([f"t-{k}" if k else "t" for k in l.index], l.values * 100, color=SEQ, width=0.65)
    b.set_ylabel("Share of |SHAP| (%)")
    b.set_title("(b) by day in window", loc="left", fontsize=7.5)
    b.tick_params(axis="x", labelsize=6.3)
    b.yaxis.grid(True)
    b.set_axisbelow(True)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig7_shap")


def fig_lockdown(pred):
    lock = lockdown_split(pred)
    s = lock.groupby(["model", "period"]).skill.mean().unstack()
    models = C.MODEL_ORDER[1:]
    s = s.loc[models]
    n = lock.city.nunique()
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    x = np.arange(len(models))
    w = 0.38
    for j, (period, color, hatch) in enumerate([("pre-lockdown", HORIZON_COLORS[1], ""),
                                                ("lockdown", HORIZON_COLORS[3], "////")]):
        ax.bar(x + (j - 0.5) * w, s[period], w * 0.92, color=color, hatch=hatch, edgecolor="white",
               linewidth=0.3, label=period.capitalize())
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, [LABEL[m] for m in models])
    ax.set_ylabel("Mean skill vs persistence (t+1)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.16))
    save(fig, "fig8_lockdown")


def fig_data_size(met):
    from scipy.stats import spearmanr
    cm = city_means(met)
    sk = cm[cm.horizon == 1].pivot(index="city", columns="model", values="skill")
    ds = pd.read_csv(C.RESULTS_DIR / "main_datasets.csv")
    ntr = ds[ds.horizon == 1].set_index("city").n_train.reindex(sk.index)
    gap = sk.LSTM - sk.XGBoost
    rho, p = spearmanr(ntr, gap)
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.scatter(ntr, gap, s=18, color=SEQ, edgecolor="white", linewidth=0.5, zorder=3)
    label = {"Coimbatore", "Guwahati", "Amaravati", "Delhi", "Hyderabad", "Bengaluru", "Lucknow", "Talcher", "Mumbai"}
    offsets = {"Delhi": (0, 6, "center"), "Bengaluru": (-6, -2, "right"), "Lucknow": (5, -6, "left")}
    for c in sk.index:
        if c in label:
            dx, dy, ha = offsets.get(c, (4, 2, "left"))
            ax.annotate(c, (ntr[c], gap[c]), xytext=(dx, dy), textcoords="offset points", fontsize=6.2,
                        color=INK_2, ha=ha)
    ax.set_xlabel("Training forecasts per city (t+1)")
    ax.set_ylabel("Skill difference\n(LSTM minus XGBoost)")
    ax.text(0.98, 0.06, f"Spearman \u03c1 = {rho:.2f} (p = {p:.3f})", transform=ax.transAxes, ha="right", fontsize=6.8)
    ax.text(0.02, 0.97, "\u2191 LSTM better than XGBoost", transform=ax.transAxes, fontsize=6.3, color=INK_2, va="top")
    ax.set_ylim(top=0.22)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    save(fig, "fig9_data_size")


def main():
    met = pd.read_csv(C.RESULTS_DIR / "main_metrics.csv")
    pred = pd.read_csv(C.RESULTS_DIR / "main_predictions.csv.gz", parse_dates=["date"])
    overall = overall_table(met)
    tests = rank_tests(met)
    fig_pipeline()
    fig_data_overview()
    fig_skill_by_horizon(overall)
    fig_city_heatmap(met, 1)
    lock = lockdown_split(pred)
    pre_tab = lock[lock.period == "pre-lockdown"].pivot(index="city", columns="model", values="rmse")[C.MODEL_ORDER]
    from .evaluate import friedman_nemenyi
    ranks, chi2, pval, cd = friedman_nemenyi(pre_tab)
    fig_ranks(tests, {"ranks": ranks, "p": pval})
    fig_timeseries(pred)
    fig_shap()
    fig_lockdown(pred)
    fig_data_size(met)


if __name__ == "__main__":
    main()
