// Abstract, results, discussion and conclusion. Every number is read from paper_numbers.json;
// statements whose direction depends on the data are computed, and show a [placeholder] until results exist.
const { MODELS, NAME, f, pct, sgn, fmtInt, nm, pv, DRAFT } = require("./content");

const NONP = MODELS.slice(1);
const X = "[[XX]]";
const LOCKDOWN = "2020-03-25";
const SHORT = { Persistence: "Persistence", MLR: "MLR", SVR: "SVR", RF: "RF", XGBoost: "XGBoost", LSTM: "LSTM", GRU: "GRU" };
const NA = "n/a";

function bestOf(o, models, key = "skill") {
  if (!o || !o[models[0]] || o[models[0]][key] === undefined) return null;
  return models.reduce((a, m) => (o[m][key] > o[a][key] ? m : a), models[0]);
}
const pctAbs = (x, d = 1) => (x === null || x === undefined ? X : `${Math.abs(100 * x).toFixed(d)}%`);
const list = (a) => (a.length < 2 ? a.join("") : `${a.slice(0, -1).join(", ")} and ${a[a.length - 1]}`);
const lockFrac = (d) => {
  const s = new Date(d.test_start), e = new Date(d.test_end), l = new Date(LOCKDOWN);
  if (e < l) return 0;
  if (s >= l) return 1;
  return (e - l) / (e - s);
};

function abstract(N) {
  if (DRAFT(N)) return "[[Abstract written from final results.]]";
  const pre = N.pre_lockdown_tests, lk = N.lockdown, ds = N.data_size;
  const wPers = pre.wilcoxon.find((r) => r.model === "Persistence");
  return `A city can only prepare for a high-pollution day if someone forecasts it, and recurrent neural networks are often proposed for forecasting the daily Air Quality Index (AQI). Many comparisons in this area, however, estimate same-day AQI from same-day pollutant levels, fit preprocessing before splitting the data, or report no persistence baseline. We compared persistence, ridge regression, support vector regression, Random Forest, XGBoost, LSTM and GRU on CPCB data from 20 Indian cities (2015 to 2020) at lead times of one, three and seven days. Every model saw the same seven-day input window, preprocessing used training data only, and stochastic models were run with five seeds. XGBoost had the best mean rank at all three lead times. The LSTM ranked last one day ahead and was significantly worse than XGBoost. Before the COVID-19 lockdown of March 2020, XGBoost cut the one-day-ahead RMSE of persistence by ${pctAbs(lk.XGBoost["pre-lockdown"])} (Friedman p ${pv(pre.p)}; Holm-corrected Wilcoxon p ${pv(wPers.p_holm)}), while the recurrent networks improved on it by ${pctAbs(lk.GRU["pre-lockdown"])} at most. No model beat persistence during the lockdown. The LSTM's shortfall against XGBoost shrank in cities with longer records (Spearman ρ = ${f(ds.rho_ntrain_gap)}, p ${pv(ds.p_ntrain_gap)}), which suggests that about four years of daily data per city is too little for it. For daily city-level AQI we would start from a tuned gradient-boosted tree model and always check it against persistence.`;
}

function results({ L, N, cite, FIG, o1, o3, o7, fr, dm1, dm7 }) {
  const draft = DRAFT(N);
  const out = [];
  const add = (...x) => out.push(...x.flat());
  const pre = N.pre_lockdown_tests || { ranks: {}, wilcoxon: [] };
  const lk = N.lockdown || new Proxy({}, { get: () => ({}) });
  const lw = N.lockdown_wins || { lockdown: {}, "pre-lockdown": {} };
  const ds = N.data_size || {};
  const ab = N.ablation || {};
  const wins = (h, m) => (N.overall && N.overall[h][m] ? N.overall[h][m].cities_beating_persistence : X);
  const minRmse = (o) => (draft ? null : NONP.concat(["Persistence"]).reduce((a, m) => (o[m].rmse < o[a].rmse ? m : a), "Persistence"));

  add(L.H1("V", "Results"));

  // ---------- A. One-day-ahead ----------
  add(L.H2("A", "One-Day-Ahead Accuracy"));
  const bestRmse1 = minRmse(o1);
  const rfW = wins("1", "RF"), xgW = wins("1", "XGBoost");
  const treeWins = rfW === xgW ? `each beat persistence in ${rfW} of the 20 cities` : `beat persistence in ${rfW} and ${xgW} of the 20 cities`;
  add(L.P(`One day ahead, persistence on its own already reaches a mean R^2^ of ${f(o1.Persistence?.r2_mean)} (median ${f(o1.Persistence?.r2_median)}; ${L.tab("III")}), so a large part of tomorrow's AQI is already in today's value. Only the two tree ensembles have a positive mean skill. ${nm(bestRmse1)} has the lowest mean RMSE (${f(o1[bestRmse1]?.rmse, 1)} against ${f(o1.Persistence?.rmse, 1)} for persistence), and Random Forest and XGBoost ${treeWins}. On average the recurrent networks do worse than persistence (LSTM ${sgn(o1.LSTM?.skill)}, GRU ${sgn(o1.GRU?.skill)}), and so does ridge regression. The median skill is positive for every model, though, so the means are dragged down by large errors in a handful of cities (${L.sec("V", "D")}). Persistence still has the best category accuracy (${pct(o1.Persistence?.cat_acc, 0)}). The learned models cut large errors, but now and then they push a forecast over a category boundary that persistence would not have crossed.`));
  add(L.table("Mean one-day-ahead test performance over 20 cities",
    ["Model", "RMSE", "MAE", "R²", "Skill", "Med. skill", "Pre-LD skill", "Cat. acc.", "Wins"],
    MODELS.map((m) => {
      const o = o1[m] || {};
      const b = (s, best) => (!draft && best ? `**${s}**` : s);
      const isP = m === "Persistence";
      const preBest = bestOf(Object.fromEntries(NONP.map((k) => [k, { skill: lk[k]?.["pre-lockdown"] }])), NONP);
      return [SHORT[m],
        b(f(o.rmse, 1), m === bestRmse1), f(o.mae, 1), f(o.r2_mean),
        isP ? "0" : b(sgn(o.skill, 2), m === bestOf(o1, NONP)),
        isP ? "0" : sgn(o.skill_median, 2),
        isP ? "0" : b(sgn(lk[m]?.["pre-lockdown"], 2), m === preBest),
        pct(o.cat_acc, 0), isP ? NA : String(o.cities_beating_persistence ?? X)];
    }),
    [800, 520, 470, 420, 520, 540, 560, 500, 428],
    { fontSize: 7.5, pad: 25, note: `Skill is 1 − RMSE/RMSE~persistence~, so positive values mean better than persistence. Pre-LD refers to test days before the 25 March 2020 lockdown (${N.lockdown_n_cities ?? X} cities). Wins is the number of cities in which the model's RMSE is below that of persistence. Values are averaged over cities and, for RF, XGBoost, LSTM and GRU, over five seeds. MLR denotes ridge-regularised linear regression. The best value in each column is in bold.` }));

  // ---------- B. Lockdown ----------
  add(L.H2("B", "Before and During the COVID-19 Lockdown"));
  add(L.P(`The lockdown that began on 25 March 2020 cut vehicular and industrial emissions sharply ${cite("mahato")}, and it falls inside every test period. ${L.figRef("fig8_lockdown")} splits the test days of the ${N.lockdown_n_cities ?? X} cities that have at least 30 days on each side. Before the lockdown, XGBoost and Random Forest lower the RMSE of persistence by ${pctAbs(lk.XGBoost?.["pre-lockdown"])} and ${pctAbs(lk.RF?.["pre-lockdown"])} on average, and XGBoost beats persistence in ${lw["pre-lockdown"].XGBoost ?? X} of the ${N.lockdown_n_cities ?? X} cities. The GRU, ridge regression and LSTM stay very close to persistence (${sgn(lk.GRU?.["pre-lockdown"])}, ${sgn(lk.MLR?.["pre-lockdown"])} and ${sgn(lk.LSTM?.["pre-lockdown"])}). During the lockdown every model is worse than persistence. XGBoost loses the least (${sgn(lk.XGBoost?.lockdown)}) and ridge regression the most (${sgn(lk.MLR?.lockdown)}). Our reading is that models trained on earlier years expect pollution to drift back to its usual seasonal level. That did not happen while emissions stayed low, and persistence, which assumes nothing about the season, adjusted at once.`));
  add(L.figure(FIG("fig8_lockdown"), `One-day-ahead skill relative to persistence on test days before and during the 2020 lockdown, averaged over the ${N.lockdown_n_cities ?? X} cities that have at least 30 test days in each period.`));

  // ---------- C. Lead time ----------
  add(L.H2("C", "Effect of Lead Time"));
  add(L.P(`${L.figRef("fig3_skill_by_horizon")} and ${L.tab("IV")} add the three- and seven-day forecasts. Persistence gets worse quickly as the lead time grows (mean RMSE ${f(o1.Persistence?.rmse, 1)}, ${f(o3.Persistence?.rmse, 1)} and ${f(o7.Persistence?.rmse, 1)}), but so do the learned models: their inputs say nothing about the weather that will control dispersion over the coming week. XGBoost has the lowest mean RMSE at both longer horizons (${f(o3.XGBoost?.rmse, 1)} and ${f(o7.XGBoost?.rmse, 1)}) and beats persistence in ${wins("3", "XGBoost")} and ${wins("7", "XGBoost")} cities, with mean skills of ${sgn(o3.XGBoost?.skill)} and ${sgn(o7.XGBoost?.skill)}. The recurrent networks do not catch up at longer horizons (LSTM ${sgn(o3.LSTM?.skill)} and ${sgn(o7.LSTM?.skill)}). The spread between seeds is small next to the differences between models.`));
  add(L.figure(FIG("fig3_skill_by_horizon"), "Mean skill relative to persistence over 20 cities at lead times of one, three and seven days (all test days). Error bars show the standard deviation of the 20-city mean across five seeds; MLR and SVR are deterministic."));
  add(L.table("RMSE and Diebold-Mariano outcomes by lead time",
    ["Model", "RMSE t+1", "RMSE t+3", "RMSE t+7", "DM t+1", "DM t+3", "DM t+7"],
    MODELS.map((m) => {
      const dm3 = N.dm ? N.dm["3"] : {};
      const isP = m === "Persistence";
      const d = (x) => (isP ? NA : `${x?.better ?? X} / ${x?.worse ?? X}`);
      return [SHORT[m], f(o1[m]?.rmse, 1), f(o3[m]?.rmse, 1), f(o7[m]?.rmse, 1), d(dm1[m]), d(dm3?.[m]), d(dm7[m])];
    }),
    [880, 620, 620, 620, 670, 670, 670],
    { fontSize: 7.5, pad: 30, note: "DM gives the number of cities in which the model is significantly better / worse than persistence (Diebold-Mariano test with the HLN correction, p < 0.05)." }));

  // ---------- D. City level + data size ----------
  add(L.H2("D", "City-Level Results and the Role of Data Size"));
  const pcs = N.per_city_skill ? N.per_city_skill["1"] : null;
  let cityText = "[[city-level summary]]";
  if (pcs) {
    const cities = Object.keys(pcs);
    const best = cities.map((c) => [c, Math.max(...NONP.map((m) => pcs[c][m]))]).sort((a, b) => b[1] - a[1]);
    const anyBeat = best.filter((x) => x[1] > 0).length;
    const losers = best.filter((x) => x[1] <= 0).map((x) => x[0]);
    const dsBy = Object.fromEntries(N.datasets_h1.map((d) => [d.city, d]));
    const L_ = losers.map((c) => dsBy[c]);
    const nMin = Math.min(...L_.map((d) => d.n_train)), nMax = Math.max(...L_.map((d) => d.n_train));
    const fMin = Math.min(...L_.map(lockFrac)), fMax = Math.max(...L_.map(lockFrac));
    const minCity = losers.reduce((a, c) => (dsBy[c].n_train < dsBy[a].n_train ? c : a), losers[0]);
    cityText = `Results differ a lot between cities (${L.figRef("fig4_city_skill_h1")}). In ${anyBeat} of the 20 cities, at least one model does better than persistence. The best case is ${best[0][0]}, where the top model reaches a skill of ${sgn(best[0][1], 2)}, and ${best[1][0]} and ${best[2][0]} come next (${sgn(best[1][1], 2) === sgn(best[2][1], 2) ? `both ${sgn(best[1][1], 2)}` : `${sgn(best[1][1], 2)} and ${sgn(best[2][1], 2)}`}). In the other ${losers.length === 4 ? "four" : losers.length} cities, ${list(losers)}, nothing we tried beat persistence. None of them has more than ${nMax} training days, ${minCity} has only ${nMin}, and at least ${pct(fMin, 0)} of each one's test days come after the lockdown started. These cities are the main reason the mean skills in ${L.tab("III")} are so low.`;
  }
  add(L.P(cityText));
  add(L.figure(FIG("fig4_city_skill_h1"), "One-day-ahead skill relative to persistence for every city (rows, ordered by best skill) and model (columns). Blue cells beat persistence and red cells are worse. The colour scale is clipped at ±0.4, but the printed values are not."));
  add(L.P(`How much training data a city has turns out to matter a great deal for the recurrent networks and hardly at all for the tree ensembles. Across cities, the LSTM's skill rises with the number of training forecasts (Spearman ρ = ${f(ds.rho_ntrain_lstm)}, p ${pv(ds.p_ntrain_lstm)}), and so does its skill relative to XGBoost (ρ = ${f(ds.rho_ntrain_gap)}, p ${pv(ds.p_ntrain_gap)}; ${L.figRef("fig9_data_size")}). XGBoost shows no significant dependence on data size (ρ = ${f(ds.rho_ntrain_xgb)}, p ${pv(ds.p_ntrain_xgb)}). In Delhi and Bengaluru, which have the longest records, the LSTM is slightly better than XGBoost (in Delhi ${sgn(pcs?.Delhi?.LSTM, 2)} against ${sgn(pcs?.Delhi?.XGBoost, 2)}). In most cities with fewer than 600 training days it falls well behind.`));
  add(L.figure(FIG("fig9_data_size"), "Difference in one-day-ahead skill between the LSTM and XGBoost, plotted against the number of training forecasts in each city. Positive values favour the LSTM."));
  if (L.include("fig6_timeseries_delhi")) {
    add(L.figure(FIG("fig6_timeseries_delhi"), "Observed and one-day-ahead forecast AQI in Delhi during the 2019-20 winter pollution season (test period). The XGBoost and LSTM forecasts are averaged over five seeds."));
    add(L.P(`${L.figRef("fig6_timeseries_delhi")} shows the Delhi forecasts for the winter of 2019-20. Every forecast, persistence included, tends to react to the steepest rises about a day late. The learned models mostly differ from persistence when an episode ends, because they do not carry the previous day's peak forward.`));
  }

  // ---------- E. Significance ----------
  add(L.H2("E", "Statistical Significance"));
  const w1 = N.wilcoxon ? N.wilcoxon["1"] : [];
  const wf = (arr, m) => arr.find((r) => r.model === m) || {};
  const preSig = pre.wilcoxon.filter((r) => r.p_holm < 0.05).map((r) => nm(r.model).toLowerCase());
  add(L.P(`Taken over all test days, the Friedman test gives only weak evidence that the models differ at *t*+1 (χ^2^ = ${f(fr["1"].chi2, 1)}, p ${pv(fr["1"].p)}) and none at *t*+3 (p ${pv(fr["3"].p)}) or *t*+7 (p ${pv(fr["7"].p)}). XGBoost still has the best mean rank at every lead time (${L.figRef("fig5_mean_ranks")}). It is significantly more accurate than the LSTM at *t*+1 (Holm p ${pv(wf(w1, "LSTM").p_holm)}), but not than persistence (p ${pv(wf(w1, "Persistence").p_holm)}) or the other learners. On the pre-lockdown test days of the ${pre.n_cities ?? X} cities the differences are significant (χ^2^ = ${f(pre.chi2, 1)}, p ${pv(pre.p)}). XGBoost ranks first there (${f(pre.ranks.XGBoost)}) and persistence last (${f(pre.ranks.Persistence)}), and XGBoost is significantly better than ${preSig.length ? list(preSig) : "none of the other models"} (Holm p ${pv(wf(pre.wilcoxon, "Persistence").p_holm)}), though we cannot separate it statistically from Random Forest or the other learners. City by city, the DM test finds XGBoost significantly better than persistence in ${dm1.XGBoost?.better ?? X} cities and worse in ${dm1.XGBoost?.worse ?? X}; for the LSTM the counts are ${dm1.LSTM?.better ?? X} and ${dm1.LSTM?.worse ?? X} (${L.tab("IV")}).`));
  add(L.figure(FIG("fig5_mean_ranks"), "Mean rank of each model across the 20 cities (RMSE, 1 = best) at each lead time, and on pre-lockdown test days only (hollow circles). Models whose ranks differ by less than the Nemenyi critical difference are not significantly different (α = 0.05)."));

  // ---------- F. Ablations ----------
  add(L.H2("F", "Effect of the Learning Target and the Inputs"));
  const lv = ab.level_full, dl = ab.delta_full_h1, aq = ab.level_aqi_only, po = ab.level_pollutants_only;
  const abModels = ["MLR", "SVR", "XGBoost", "LSTM", "GRU"];
  let abText = "[[ablation summary]]";
  if (lv && dl && aq && po) {
    const worse = abModels.filter((m) => lv[m] && dl[m].skill - lv[m].skill > 0.02);
    const same = abModels.filter((m) => lv[m] && Math.abs(dl[m].skill - lv[m].skill) <= 0.02);
    const anyPos = abModels.some((m) => lv[m] && lv[m].skill > 0);
    const arrow = (m) => `${sgn(dl[m].skill)} to ${sgn(lv[m].skill)}`;
    abText = `${L.tab("V")} looks at two design choices at *t*+1. The first is the target. Most earlier studies train models to predict the AQI level. When we do the same, mean skill drops for ${list(worse.map((m) => `${SHORT[m]} (${arrow(m)})`))}${same.length ? `; only the ${list(same.map((m) => SHORT[m]))} barely changes (${list(same.map(arrow))})` : ""}. ${anyPos ? "" : "No model beats persistence with the level target, because each one has to reproduce an absolute pollution level that drifts between the training and test periods. "}The second choice is the input set, which we vary with the level target. XGBoost does worse with past AQI alone (${sgn(aq.XGBoost.skill)}) or pollutants alone (${sgn(po.XGBoost.skill)}) than with both (${sgn(lv.XGBoost.skill)}). Ridge regression is the opposite case: it does better with AQI history alone (${sgn(aq.MLR.skill)} against ${sgn(lv.MLR.skill)}), probably because about 90 correlated lag features are too many for a linear model fitted on a few hundred days.`;
  }
  add(L.P(abText));
  add(L.table("Ablation at t+1: mean skill relative to persistence",
    ["Model", "Change target, all inputs", "Level target, all inputs", "Level, AQI history only", "Level, pollutants only"],
    abModels.map((m) => [SHORT[m], sgn(dl?.[m]?.skill), sgn(lv?.[m]?.skill), aq?.[m] ? sgn(aq[m].skill) : NA, po?.[m] ? sgn(po[m].skill) : NA]),
    [900, 1000, 1000, 930, 928],
    { fontSize: 7.5, note: "The first column is the main protocol (five seeds); the ablations use three seeds. RF was left out of the ablations to save computation time. n/a: not run." }));

  // ---------- G. SHAP ----------
  add(L.H2("G", "What the Models Rely On"));
  const sv = Object.entries(N.shap_by_variable || {}).sort((a, b) => b[1] - a[1]);
  const sl = N.shap_by_lag || {};
  const sub = (k) => k.replace("PM2.5", "PM~2.5~").replace("PM10", "PM~10~").replace("NO2", "NO~2~").replace("NOx", "NO~x~").replace("SO2", "SO~2~").replace("O3", "O~3~").replace("NH3", "NH~3~");
  let shapText = "[[SHAP summary]]";
  if (sv.length) {
    const [a, b, c] = sv;
    const older = [2, 3, 4, 5, 6].map((k) => sl[String(k)]).filter((v) => v !== undefined);
    shapText = `TreeSHAP gives a fairly simple picture of the one-day-ahead XGBoost model (${L.figRef("fig7_shap")}). Past ${sub(a[0])} gets the largest share of the attribution (${pct(a[1], 0)}), ${sub(b[0])} gets ${pct(b[1], 0)} and ${sub(c[0])} ${pct(c[1], 0)}, while each of the remaining inputs gets ${pct(sv[3][1], 0)} or less. Looking at the days in the window, ${pct(sl["0"], 0)} sits on the most recent day and ${pct(sl["1"], 0)} on the day before; the five older days get between ${pct(Math.min(...older), 0)} and ${pct(Math.max(...older), 0)} each. The model is therefore mostly adjusting today's value from the latest AQI and particulate readings. It hardly uses the longer patterns that recurrent networks are designed for.`;
  }
  add(L.P(shapText));
  add(L.figure(FIG("fig7_shap"), "Mean share of absolute SHAP attribution for the one-day-ahead XGBoost forecasts, averaged over 20 cities: (a) by input variable, summed over the seven days of the window; (b) by day in the window, summed over variables."));

  // ---------- VI. Discussion ----------
  add(L.H1("VI", "Discussion"));
  add(L.H2("A", "Is Deep Learning Necessary?"));
  add(L.P(`For daily city-level AQI forecasting from the CPCB records, our results say no. With identical inputs and no leakage, the LSTM and GRU did not beat a tuned gradient-boosted tree model on average at any lead time, and the LSTM was significantly worse than XGBoost. They fall furthest behind where data are scarce (${L.figRef("fig9_data_size")}), so longer records, hourly data or training pooled across cities might change the picture. Earlier reports of high accuracy for deep models ${cite("janarthanan", "sarkar", "mishra")} used different protocols and cannot be compared directly with ours. Our results add multi-city evidence from India to the question Kalantari et al. raised ${cite("kalantari")}.`));
  add(L.H2("B", "Why Persistence Is Hard to Beat"));
  add(L.P(`Daily AQI is strongly autocorrelated. It is computed from 24-hour averages ${cite("cpcb")}, and the weather that traps or disperses pollutants usually lasts several days. A model without weather information can only look backwards, and most of what it can learn about tomorrow is already in today's value. Whatever advantage is left is fragile, as the lockdown showed. This is also why R^2^ is a poor yardstick here: persistence scores ${f(N.overall?.["1"]?.Persistence?.r2_mean)} without learning anything.`));
  add(L.H2("C", "Recommendations for Practice"));
  add(L.P(`A few practical points follow from this study. Persistence, and the skill relative to it, should be reported next to R^2^. Preprocessing should be fitted on training data alone, gaps filled from past values only, and the test set kept away from both tuning and early stopping ${cite("kapoor")}. Models should predict the change in AQI, because predicting the level made almost every model worse (${L.tab("V")}). If we had to pick one model to start with, it would be a tuned gradient-boosted tree. We would switch to a recurrent network only if it won on held-out data, and after a shock like the 2020 lockdown we would retrain, or simply fall back to persistence until enough new data had come in.`));
  add(L.H2("D", "Limitations"));
  add(L.P(`The whole study rests on one public collection, and it stops in mid-2020. Some test periods are therefore short, and every one of them overlaps the lockdown. Each city had one chronological split rather than a rolling-origin evaluation, and each was trained on its own data only. We did not use meteorological data, because they were not available for all cities. Our conclusions may not hold for station-level or hourly forecasting.`));

  // ---------- VII. Conclusion ----------
  add(L.H1("VII", "Conclusion"));
  add(L.P(draft ? "[[Conclusion written from final results.]]" :
    `We compared seven forecasters for daily AQI in 20 Indian cities under a leakage-free protocol, with identical inputs, three lead times, five seeds and significance tests. Persistence was a strong baseline. Only the tree ensembles, and XGBoost in particular, improved on it consistently, by about ${pctAbs(lk.XGBoost["pre-lockdown"], 0)} in RMSE before the 2020 lockdown. The recurrent networks cost more to train without being more accurate, and they did worst where little training data was available. During the lockdown no model beat persistence; what the models had learned from earlier years no longer applied once emissions dropped. Next, we plan to add meteorological forecasts, test on data after 2020 with rolling origins, and pool training data across cities, which may be what deep models need to become useful here.`));

  // ---------- VIII. Acknowledgment ----------
  add(L.ack("VIII", `The authors thank [[Professor Name, Designation, Department, Institution]] for guidance and support throughout this work. We also thank the Central Pollution Control Board for making its monitoring data public, and the curator of the Kaggle collection used in this study. An AI assistant (Claude, Anthropic) helped write the analysis code, run the experiments, and draft and edit this manuscript. The authors reviewed all code, results and text and take full responsibility for the content of this paper.`));
  return out;
}

module.exports = { abstract, results };
