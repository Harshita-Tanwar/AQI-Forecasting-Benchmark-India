"""Shared settings for the AQI forecasting benchmark."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "city_day_AQI_pred_dataset.csv"      # Kaggle "Air Quality Data in India (2015-2020)", city_day.csv
WEATHER_FILE = ROOT / "AQI_prediction_dataset.csv"      # Bengaluru pollutants + weather (2016-2020)
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures" / "paper"

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
              "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
WEATHER_VARS = ["temp", "max_temp", "min_temp", "humid", "visible", "wind"]

MIN_VALID_DAYS = 300        # cities need at least this many days with a reported AQI
MISSING_THRESHOLD = 0.5     # drop a pollutant if >50% missing in the city's training period
FFILL_LIMIT = 7             # causal gap filling: carry the last reading forward at most 7 days
WINDOW = 7                  # look-back window (days) given to every model
SPLIT = (0.7, 0.1, 0.2)     # chronological train / validation / test
SEEDS = [0, 1, 2, 3, 4]
HORIZONS = [1, 3, 7]        # forecast lead time in days

MODEL_ORDER = ["Persistence", "MLR", "SVR", "RF", "XGBoost", "LSTM", "GRU"]
MODEL_LABELS = {
    "Persistence": "Persistence",
    "MLR": "MLR (ridge)",
    "SVR": "SVR (RBF)",
    "RF": "Random Forest",
    "XGBoost": "XGBoost",
    "LSTM": "LSTM",
    "GRU": "GRU",
}

# CPCB National AQI categories
AQI_BINS = [-np.inf, 50, 100, 200, 300, 400, np.inf]
AQI_CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
