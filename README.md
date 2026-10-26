# AQI Prediction: LSTM & Machine Learning Comparative Study

Comparative analysis of **LSTM, Gradient Boosting, Random Forest, SVR, and Multiple Linear Regression** for Air Quality Index prediction across 20 Indian cities. This study evaluates deep learning vs classical ML approaches for time-series forecasting on real CPCB (Central Pollution Control Board) data.

## Key Results

| Model | R² | RMSE | MAE | Accuracy |
|-------|-----|------|-----|--------|
| **LSTM** | 0.57 | 37.3 | 25.5 | 70% |
| **Gradient Boosting** | 0.67 | 33.6 | 22.2 | 72% |
| **Random Forest** | 0.67 | 33.1 | 23.3 | 73% |
| **SVR** | 0.65 | 35.0 | 24.9 | 71% |
| **MLR** | 0.47 | 36.8 | 27.0 | 69% |
| **GRU** | 0.58 | 36.2 | 25.1 | 71% |
| **Persistence** | 0.64 | 35.7 | 22.5 | 75% |

## Project Structure

```
AQI-Forecasting-Benchmark-India/
├── .github/
├── legacy/
│   ├── .ipynb_checkpoints/
│       ├── AQI_Prediction_Report.md
│       ├── GradientBoosting.ipynb
│       ├── LSTM.ipynb
│       ├── MLR.ipynb
│       ├── RandomForest.ipynb
│       └── SVR.ipynb
├── paper/
├── src/
├── tests/
├── .gitignore
├── AQI_prediction_dataset.csv
├── city_day_AQI_pred_dataset.csv
├── pytest.ini
├── README.md
├── requirements-legacy.txt
└── requirements.txt
```

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run a model
jupyter notebook legacy/LSTM.ipynb
# Then: Kernel → Restart & Run All
```


## Data

R. Rao (Vopani), *Air Quality Data in India (2015-2020)*, Kaggle, compiled from Central Pollution Control Board records: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india. `city_day_AQI_pred_dataset.csv` is the `city_day.csv` file from that collection.

## Models Overview

- **LSTM**: 2-layer recurrent network; captures temporal dependencies
- **Gradient Boosting**: Sequential ensemble; fastest training relative to accuracy
- **Random Forest**: Parallel ensemble; robust to outliers
- **SVR**: Non-linear kernel; effective for high-dimensional data
- **MLR**: Linear baseline; interpretable reference

## Results

See `figures/` for visualizations:
- `fig5_avg_r2_comparison.png` - Model accuracy comparison
- `fig6_avg_rmse_comparison.png` - Prediction error analysis
- `fig7_percity_r2_heatmap.png` - Geographic performance distribution
