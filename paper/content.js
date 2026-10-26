// Manuscript text. Numbers come from paper_numbers.json (N); citations from references.js (cite).
const MODELS = ["Persistence", "MLR", "SVR", "RF", "XGBoost", "LSTM", "GRU"];
const NAME = { Persistence: "Persistence", MLR: "MLR (ridge)", SVR: "SVR", RF: "Random Forest", XGBoost: "XGBoost", LSTM: "LSTM", GRU: "GRU" };
const X = "[[XX]]";
const nil = (x) => x === null || x === undefined || Number.isNaN(x);
const f = (x, d = 2) => (nil(x) ? X : Number(x).toFixed(d).replace(/^-/, "−"));
const pct = (x, d = 1) => (nil(x) ? X : `${(100 * x).toFixed(d)}%`);
const sgn = (x, d = 3) => (nil(x) ? X : (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d));
const fmtInt = (x) => (nil(x) ? X : Number(x).toLocaleString("en-US"));
const nm = (m) => (m ? NAME[m] : "[[model]]");
const pv = (p) => (nil(p) ? X : p < 0.001 ? "< 0.001" : `= ${p.toFixed(3)}`);
const DRAFT = (N) => !N.overall;

const content = {
  title: "Is Deep Learning Necessary for Short-Term AQI Forecasting? A Leakage-Free Benchmark of Classical and Recurrent Models across 20 Indian Cities",
  refsNumeral: "IX",

  body({ L, N, cite, FIG }) {
    const empty = new Proxy({}, { get: () => ({}) });
    const ov = N.overall || { 1: empty, 3: empty, 7: empty };
    const o1 = ov["1"], o3 = ov["3"], o7 = ov["7"];
    const fr = N.friedman || { 1: {}, 3: {}, 7: {} };
    const best1 = fr["1"].best, best7 = fr["7"].best;
    const dm = N.dm || { 1: empty, 3: empty, 7: empty };
    const dm1 = dm["1"], dm7 = dm["7"];
    const nonPers = MODELS.slice(1);
    const bestSkill = (h) => (N.overall ? nonPers.reduce((a, m) => (ov[h][m].skill > ov[h][a].skill ? m : a), nonPers[0]) : null);
    const bs1 = bestSkill("1"), bs3 = bestSkill("3"), bs7 = bestSkill("7");
    const rnnBest1 = N.overall ? (o1.LSTM.skill > o1.GRU.skill ? "LSTM" : "GRU") : null;
    const treeBest1 = N.overall ? (o1.XGBoost.skill > o1.RF.skill ? "XGBoost" : "RF") : null;
    const shapVar = Object.entries(N.shap_by_variable || {}).sort((a, b) => b[1] - a[1]);
    const shapLag = N.shap_by_lag || {};
    const ab = N.ablation || {};
    const lock = N.lockdown || empty;
    const out = [];
    const add = (...x) => out.push(...x.flat());

    // ---------------- Abstract ----------------
    add(L.abstract(content.abstract(N)));
    add(L.keywords(["air quality index", "time-series forecasting", "LSTM", "GRU", "XGBoost", "persistence baseline", "data leakage", "India"]));

    // ---------------- I. Introduction ----------------
    add(L.H1("I", "Introduction"));
    add(L.P(`In 2017, air pollution was linked to about 1.24 million deaths in India, roughly one in eight deaths in the country ${cite("gbd")}. The Central Pollution Control Board (CPCB) reports a daily Air Quality Index (AQI) that combines up to eight pollutant concentrations into six categories, from *Good* to *Severe* ${cite("cpcb")}. If the next day's AQI could be forecast reliably, hospitals and city agencies would have a day to prepare for a *Poor* or *Very Poor* day.`));
    add(L.P(`Many groups have applied machine learning (ML) to this problem, from support vector regression and tree ensembles to recurrent neural networks such as the Long Short-Term Memory (LSTM) network ${cite("zhu", "ameer", "gupta", "janarthanan", "sarkar")}, and the reported accuracies are often very high. Some of these numbers are hard to interpret. A model that estimates today's AQI from today's pollutant levels mostly relearns the CPCB formula, since the index is computed from those levels ${cite("cpcb")}. When interpolation, scaling or outlier removal is applied before the data are split, or early stopping uses the test set, information from the test period leaks into training ${cite("kapoor")}. Few studies report persistence (tomorrow's AQI equals today's), which is the usual reference for short-range forecasts ${cite("hyndman")}.`));
    add(L.P(`We wanted to know whether recurrent networks still beat classical models once these problems are removed. Using public CPCB records for 20 Indian cities ${cite("kaggle")}, we built a benchmark in which every model sees the same seven-day input window, all preprocessing is fitted on the training period, gaps are filled from past values only and hyperparameters are chosen on a separate validation period. We compare persistence, ridge regression (MLR), support vector regression (SVR), Random Forest (RF), XGBoost, LSTM and GRU at lead times of one, three and seven days, run each stochastic model with five seeds, and test the differences statistically across cities. We also look at which inputs the models rely on, using SHAP, and at how they behaved during the COVID-19 lockdown of March 2020, which falls inside our test period.`));
    add(L.P(`${L.sec("II")} reviews related work and ${L.sec("III")} describes the data. The methods are in ${L.sec("IV")}, the results in ${L.sec("V")} and the discussion in ${L.sec("VI")}.`));

    // ---------------- II. Related work ----------------
    add(L.H1("II", "Literature Review"));
    add(L.H2("A", "Classical Machine Learning Approaches"));
    add(L.P(`Zhu et al. proposed hybrid models for daily AQI forecasting in Chinese cities and found that they outperformed single models ${cite("zhu")}. Ameer et al. compared decision trees, Random Forest, gradient boosting and multilayer perceptrons for pollution prediction in smart cities ${cite("ameer")}, and Castelli et al. applied support vector regression in California ${cite("castelli")}. For India, Gupta et al. compared SVR, Random Forest and CatBoost on CPCB data ${cite("gupta")}, Ravindiran et al. evaluated boosting and bagging ensembles for Visakhapatnam ${cite("ravindiran")}, and Pant et al. compared several supervised learners ${cite("pant")}. In most of these studies the tree ensembles did well and were cheap to train.`));
    add(L.H2("B", "Deep and Hybrid Recurrent Models"));
    add(L.P(`The LSTM ${cite("lstm")} and the simpler Gated Recurrent Unit (GRU) ${cite("gru")} keep information over long sequences through gated memory cells. Janarthanan et al. combined SVR and LSTM for AQI prediction in Chennai ${cite("janarthanan")}, Seng et al. proposed a spatiotemporal LSTM ${cite("seng")}, and both Yan et al. ${cite("yan")} and Zhang and Li ${cite("zhang")} reported gains from CNN-LSTM models in Beijing. Drewil and Al-Bahadili tuned LSTMs with metaheuristic search ${cite("drewil")}. Hybrid LSTM-GRU models have been proposed for Delhi ${cite("sarkar")} and for daily AQI in general ${cite("hossain")}, and Barthwal and Goel combined convolutional and LSTM layers to classify AQI in urban India ${cite("barthwal")}.`));
    add(L.H2("C", "Comparative Studies and Evaluation Practice"));
    add(L.P(`Model families have been compared directly by Mishra and Gupta ${cite("mishra")}, Aram et al. ${cite("aram")} and Özüpak et al. ${cite("ozupak")}, and Sidhu et al. modelled AQI across many Indian cities and states ${cite("sidhu")}. Kalantari et al. asked whether AQI forecasting needs shallow or deep learning ${cite("kalantari")}; we ask the same question for India. CPCB data have large gaps ${cite("sharma")}, and Rao et al. ${cite("rao")} and Lawrence and Bhathmanabhan ${cite("lawrence")} studied how imputation choices affect AQI models for Indian cities. Outside air quality, Kapoor and Narayanan traced many irreproducible ML results to data leakage ${cite("kapoor")}, and the forecasting literature recommends comparing every method with simple benchmarks such as persistence ${cite("hyndman")}.`));
    add(L.P(`We have not found a study on Indian AQI that compares recurrent networks with properly implemented classical learners on identical, leakage-free inputs, at several lead times, across many cities, against persistence and with significance tests. This paper reports such a comparison.`));

    // ---------------- III. Data ----------------
    add(L.H1("III", "Data and Study Area"));
    add(L.H2("A", "Source and Coverage"));
    const mp = N.missing_pct_valid;
    add(L.P(`We use the public *Air Quality Data in India (2015 to 2020)* collection ${cite("kaggle")}, which compiles daily city-level averages from CPCB continuous monitoring stations. Its city-day table has ${fmtInt(N.raw_rows)} records for ${N.raw_cities} cities between ${N.date_min} and ${N.date_max}. Each record gives the concentrations of twelve pollutants (PM~2.5~, PM~10~, NO, NO~2~, NO~x~, NH~3~, CO, SO~2~, O~3~, benzene, toluene and xylene) and the AQI computed with the CPCB definition ${cite("cpcb")}. The AQI itself is reported on ${fmtInt(N.valid_aqi_rows)} city-days. On those days xylene is missing ${f(mp.Xylene, 1)}% of the time, PM~10~ ${f(mp.PM10, 1)}% and NH~3~ ${f(mp.NH3, 1)}%, in line with the data quality problems reported for this network ${cite("sharma")}.`));
    add(L.P(`We keep the ${N.n_cities} cities with at least 300 days of reported AQI. They are quite different${L.include("fig2_aqi_distribution") ? ` (${L.figRef("fig2_aqi_distribution")})` : ""}: the median daily AQI is below 70 in Thiruvananthapuram and above 250 in Delhi and Ahmedabad.`));
    if (L.include("fig2_aqi_distribution")) add(L.figure(FIG("fig2_aqi_distribution"), "Distribution of daily AQI in the 20 study cities (number of reported days in brackets), ordered by median. Grid lines mark the CPCB category boundaries at 50, 100, 200, 300 and 400."));
    add(L.H2("B", "Forecasting Task"));
    add(L.P(`A forecast is issued on every day *t* with a reported AQI and targets day *t*+*h*, where *h* is 1, 3 or 7 days. We score it only if the AQI was also reported on the target day. The forecast origins of each city are split in time order into training (first 70%), validation (next 10%) and test (last 20%) periods, which gives ${fmtInt(N.samples_h1.n_train)} training, ${fmtInt(N.samples_h1.n_val)} validation and ${fmtInt(N.samples_h1.n_test)} test forecasts across the 20 cities for *h* = 1. All test periods end in mid-2020. On average ${pct(N.lockdown_test_fraction_mean, 0)} of each one falls after the nationwide COVID-19 lockdown of 25 March 2020 ${cite("mahato")}. ${L.tab("I")} gives the numbers for each city.`));
    const ds = N.datasets_h1;
    add(L.table("Per-city forecasting data (h = 1)",
      ["City", "Train / val / test", "Test period", "Inputs"],
      ds.map((d) => [d.city, `${d.n_train} / ${d.n_val} / ${d.n_test}`,
        `${d.test_start.slice(2, 7)} to ${d.test_end.slice(2, 7)}`, String(d.variables.split("|").length)]),
      [1450, 1300, 1250, 750],
      { fontSize: 8, note: "Inputs: AQI plus the pollutants retained for that city (pollutants with more than 50% missing values in the training period are dropped). Dates are given as yy-mm." }));

    // ---------------- IV. Methodology ----------------
    add(L.H1("IV", "Methodology"));
    add(L.P(`${L.figRef("fig1_pipeline")} shows the pipeline. Everything that depends on the data is estimated from the training period of each city, and no input ever comes from after the forecast origin.`));
    add(L.H2("A", "Leakage-Free Preprocessing"));
    add(L.P(`For each city, a pollutant is dropped when more than half of its training-period values are missing or when it never changes. If a monitor misses a day, we reuse its last reading for up to a week; after that, the gap gets the training-period median. We deliberately avoided interpolation. Interpolating across a gap needs the reading that comes after it, and nobody has that reading when the forecast is made. Days with extreme AQI stay in every split, since those are the days a warning system is meant to catch and an interquartile-range filter would throw many of them away. Inputs are standardised to zero mean and unit variance using statistics from the training set.`));
    add(L.figure(FIG("fig1_pipeline"), "Leakage-free forecasting pipeline used for every city, lead time and model."));
    add(L.H2("B", "Inputs and Learning Target"));
    add(L.P(`Every model gets the AQI and the retained pollutant readings for the seven days up to and including day *t*, plus four calendar terms for the target day (sine and cosine of the day of the week and of the day of the year). The tabular models see this window as one flat vector, and the recurrent models see it as a sequence of seven steps. Each model predicts the *change* in AQI over the lead time, so the forecast is`));
    add(L.equation([L.M.sub("ŷ", "t+h"), L.M.r(" = "), L.M.sub("y", "t"), L.M.r(" + "), L.M.r("f("), L.M.sub("x", "t"), L.M.r(")")], 1));
    add(L.P(`where *y*~t~ is the AQI on the forecast origin and *f* is the learned model. Persistence is the case *f* ≡ 0, so every model is a learned correction to persistence. ${L.sec("V", "F")} tests how much this choice matters.`, { indent: { firstLine: 0 } }));
    add(L.H2("C", "Models"));
    add(L.P(`We picked seven forecasters so that each model family used in earlier work is represented. Persistence simply repeats today's value (*ŷ*~t+h~ = *y*~t~). MLR is least squares with an L2 (ridge) penalty, which we added because the pollutant lags are strongly correlated. SVR uses a radial basis function kernel ${cite("cortes")}, and Random Forest ${cite("breiman")} and XGBoost ${cite("xgb")} stand for bagged and boosted trees. The two recurrent models, an LSTM ${cite("lstm")} and a GRU ${cite("gru")}, have two layers each, and their last hidden state goes through a small fully connected head (two layers, dropout 0.2). The search space for every model is given in ${L.tab("II")}.`));
    add(L.table("Hyperparameter search spaces (selected on validation RMSE)",
      ["Model", "Search space / settings"],
      [
        ["MLR", "α ∈ {10^−3^, 10^−2^, 0.1, 1, 10, 100}"],
        ["SVR", "C ∈ {1, 10, 100}; ε ∈ {0.05, 0.1}; γ = 1/(d·Var(x))"],
        ["RF", "300 trees; max depth ∈ {none, 15}; min leaf ∈ {1, 5}; max features ∈ {0.33, 1.0}"],
        ["XGBoost", "depth ∈ {3, 5, 7}; min child weight ∈ {1, 5}; learning rate 0.03; subsample 0.8; early stopping (100 rounds)"],
        ["LSTM / GRU", "2 layers; hidden units ∈ {32, 64}; dropout 0.2; Adam (lr 10^−3^, weight decay 10^−4^); batch 32; ≤ 300 epochs; early stopping (patience 30)"],
      ],
      [1150, 3600], { align: [require("docx").AlignmentType.LEFT, require("docx").AlignmentType.LEFT] }));
    add(L.H2("D", "Training Protocol"));
    add(L.P(`For every city and lead time we train each configuration in ${L.tab("II")} on the training period and keep the one with the lowest validation RMSE. The neural networks are trained with Adam ${cite("adam")} on the mean squared error. Their learning rate is halved after ten epochs without validation improvement, gradients are clipped at a norm of 1, and the best validation checkpoint is restored at the end. Random Forest, XGBoost, LSTM and GRU are then retrained with five seeds and their test metrics averaged (MLR and SVR are deterministic). The test period is used once, for the final evaluation. Everything ran on the CPU of an Apple M3 Pro laptop with scikit-learn ${cite("sklearn")}, XGBoost ${cite("xgb")} and PyTorch ${cite("pytorch")}. The code and every individual forecast are released with the paper.`));
    add(L.H2("E", "Evaluation Metrics"));
    add(L.P(`For *n* test forecasts with observations *y*~i~ and forecasts *ŷ*~i~, we compute the root mean squared error and the mean absolute error,`));
    add(L.equation([L.M.r("RMSE = "), L.M.sqrt([L.M.frac([L.M.r("1")], [L.M.r("n")]), L.M.sum([L.M.sup([L.M.r("("), L.M.sub("y", "i"), L.M.r("−"), L.M.sub("ŷ", "i"), L.M.r(")")], "2")])])], 2));
    add(L.equation([L.M.r("MAE = "), L.M.frac([L.M.r("1")], [L.M.r("n")]), L.M.sum([L.M.r("|"), L.M.sub("y", "i"), L.M.r("−"), L.M.sub("ŷ", "i"), L.M.r("|")])], 3));
    add(L.P(`together with R^2^ and the mean absolute percentage error. RMSE values cannot be compared directly between cities with very different pollution levels, so our main metric is the *skill score* relative to persistence,`, { indent: { firstLine: 0 } }));
    add(L.equation([L.M.r("Skill = 1 − "), L.M.frac([L.M.sub("RMSE", "model")], [L.M.sub("RMSE", "persistence")])], 4));
    add(L.P(`which is positive when a model beats persistence and comparable across cities ${cite("hyndman")}. The public sees the AQI as a category, so we also report *category accuracy*: the fraction of test days on which the forecast lands in the same CPCB category as the observed AQI ${cite("cpcb")}.`, { indent: { firstLine: 0 } }));
    add(L.H2("F", "Statistical Testing and Interpretation"));
    add(L.P(`Following Demšar ${cite("demsar")}, we rank the models by RMSE within each city and compare the ranks with the Friedman test ${cite("friedman")}. Two models whose mean ranks differ by less than the Nemenyi critical difference (CD) are not significantly different at the 5% level. We compare the best-ranked model with each of the others using Wilcoxon signed-rank tests over cities ${cite("wilcoxon")} with Holm correction ${cite("holm")}. Inside each city, the Diebold-Mariano (DM) test ${cite("dm")} with the correction of Harvey et al. ${cite("hln")} checks whether a model's squared errors differ from those of persistence. To see which inputs the XGBoost forecasts depend on, we use TreeSHAP ${cite("shap", "treeshap")}.`));

    // Remaining sections are generated from results.
    add(content.results({ L, N, cite, FIG, o1, o3, o7, fr, dm1, dm7, bs1, bs3, bs7, best1, best7, rnnBest1, treeBest1, shapVar, shapLag, ab, lock }));
    return out;
  },
};

module.exports = { content, MODELS, NAME, f, pct, sgn, fmtInt, nm, pv, DRAFT };

// Abstract and results live in content_results.js (loaded after the exports above, which it imports).
Object.assign(content, require("./content_results"));
