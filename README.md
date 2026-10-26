# Is Deep Learning Necessary for Short-Term AQI Forecasting?

Code, tests and manuscript for a leakage-free benchmark of classical and recurrent models that forecast the daily Air Quality Index (AQI) of 20 Indian cities 1, 3 and 7 days ahead, using CPCB city-day records (2015 to 2020).

We compare seven forecasters on identical inputs: **persistence** (tomorrow = today), **ridge regression (MLR)**, **SVR**, **Random Forest**, **XGBoost**, **LSTM** and **GRU**. Preprocessing is fitted on training data only, stochastic models are run with five seeds, and the differences are tested statistically.

**Paper:** [`paper/AQI_Forecasting_Paper.docx`](paper/AQI_Forecasting_Paper.docx) (IEEE two-column format), with a PDF copy in [`paper/AQI_Forecasting_Paper.pdf`](paper/AQI_Forecasting_Paper.pdf).

## Main findings

- XGBoost has the best mean rank at every lead time. Before the March 2020 COVID-19 lockdown it reduces one-day-ahead RMSE by about 10% relative to persistence (Friedman p = 0.004; Holm-corrected Wilcoxon p < 0.001).
- The LSTM ranks last one day ahead and is significantly worse than XGBoost. The recurrent networks improve on persistence by at most 1% before the lockdown.
- The LSTM's shortfall shrinks in cities with more training data (Spearman ρ = 0.56, p = 0.010). Deep models need more data than one city's daily record provides.
- No model beats persistence during the lockdown.
- Predicting the change from today's AQI, instead of the AQI level, improves almost every model.

## Repository layout

```
src/                      benchmark code
  config.py               settings (window, splits, seeds, horizons, CPCB categories)
  data.py                 leakage-free per-city datasets
  models.py               persistence, ridge MLR, SVR, Random Forest, XGBoost, LSTM, GRU
  evaluate.py             metrics, Diebold-Mariano, Friedman-Nemenyi, Wilcoxon-Holm
  run_benchmark.py        runs every city x horizon x model x seed in parallel
  run_shap.py             TreeSHAP attributions for XGBoost
  analyze.py              aggregate tables and statistical tests
  make_figures.py         paper figures
  export_paper_numbers.py collects every number quoted in the paper
tests/                    pytest suite (data leakage, metrics, statistics, model smoke tests)
paper/                    manuscript generator (Node.js) and the IEEE manuscript
legacy/                   original course-project notebooks (outputs stripped) and report
city_day_AQI_pred_dataset.csv   input data (Kaggle city_day.csv)
```

Generated outputs (`results/`, `figures/`, logs) are not stored in the repository. Every one of them can be rebuilt with the commands below.

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest
```

The suite takes about a minute on a laptop CPU, and CI runs it on every push. It checks the following:
- **No leakage.** Changing any value after a forecast origin leaves that origin's inputs unchanged.
- **Data handling.** Pollutants are dropped based on training-period missingness only, and targets and persistence match the observed AQI.
- **Statistics.** Metrics, CPCB categories and the statistical tests return known values.
- **Models.** All seven models train and predict, and the seeded models are reproducible.

## Reproduce the paper

```bash
# 1. Main benchmark: 20 cities x 3 lead times x 7 models x 5 seeds (~20 min on an Apple M3 Pro)
python -m src.run_benchmark --tag main --target delta

# 2. Ablations and SHAP
python -m src.run_benchmark --tag ablation_level --target level --horizons 1 --models Persistence MLR SVR XGBoost LSTM GRU --n-seeds 3
python -m src.run_benchmark --tag ablation_aqi_only --target level --feature-set aqi_only --horizons 1 --models Persistence MLR XGBoost LSTM --n-seeds 3
python -m src.run_benchmark --tag ablation_pollutants_only --target level --feature-set pollutants_only --horizons 1 --models Persistence MLR XGBoost LSTM --n-seeds 3
python -m src.run_shap

# 3. Tables, statistics and figures
python -m src.analyze
python -m src.make_figures

# 4. Manuscript (needs Node.js)
python -m src.export_paper_numbers
cd paper && npm install && node build_paper.js
```

## Method in brief

- **Task:** forecast the AQI on day *t+h* (h = 1, 3, 7) from the last seven days of AQI and pollutant readings plus calendar terms.
- **Split:** chronological 70 / 10 / 20 per city (train / validation / test). Hyperparameters are chosen on validation, and the test set is used once.
- **No leakage:** pollutant selection, fill-in medians and scalers come from the training period only. Gaps are filled forward for at most seven days.
- **Target:** every model learns the change from today's AQI, so each forecast is a learned correction to persistence.
- **Outliers are kept,** because severe pollution days are the events a forecast has to catch.

## About `legacy/`

The original course-project notebooks are kept for reference, with their outputs stripped. Those experiments had problems that the code in `src/` fixes:
- The "Gradient Boosting", "Random Forest" and "SVR" models were neural networks imitating those methods.
- Early stopping used the test set.
- The LSTM forecast the next day while the other models estimated same-day AQI.
- Preprocessing was fitted on the full series.

Install `requirements-legacy.txt` to open the notebooks.

## Data

R. Rao (Vopani), *Air Quality Data in India (2015-2020)*, Kaggle, compiled from Central Pollution Control Board records: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india. `city_day_AQI_pred_dataset.csv` is the `city_day.csv` file from that collection.

`AQI_prediction_dataset.csv` (Bengaluru pollutants joined with weather data from an unverified source) is not used in the study.

## AI assistance

An AI assistant (Claude, Anthropic) helped write the analysis code, run the experiments, and draft and edit the manuscript. The authors reviewed all code, results and text.
