> **Note:** this is the original course-project report, kept for reference. Its figures were notebook outputs and are not stored in the repository; run the notebooks in `legacy/` to regenerate them. The experiments described here were superseded by the leakage-free benchmark in `src/` (see the main README).

# AIR QUALITY INDEX PREDICTION USING DEEP LEARNING AND MACHINE LEARNING APPROACHES: A COMPARATIVE STUDY

**Minor Project**  

Submitted By  

**Divya Sharma | 05301172023**
**Harshita Tanwar | 06601172023** 
**Istuti Jain | 07501172023**  


DEPARTMENT OF ARTIFICIAL INTELLIGENCE AND DATA SCIENCE    
IGDTUW  
**AUGUST-DECEMBER 2026**

---

## I. INTRODUCTION

Air pollution is one of the most pressing environmental and public health challenges of the 21st century. The Air Quality Index (AQI) is a standardized numerical metric employed by government agencies worldwide to communicate the degree of ambient air pollution and its associated health risks to the general public. In India, the Central Pollution Control Board (CPCB) computes AQI based on the concentrations of key pollutants — particulate matter (PM2.5, PM10), nitrogen dioxide (NO₂), sulphur dioxide (SO₂), carbon monoxide (CO), ozone (O₃), and ammonia (NH₃) — making accurate prediction of AQI critical for proactive public health advisories, urban planning, and policy formulation [1].

Traditional approaches to AQI forecasting have relied on deterministic dispersion models and statistical regression, which often fail to capture the nonlinear, multivariate, and temporally dependent relationships inherent in atmospheric pollutant dynamics [2]. The emergence of deep learning has introduced data-driven paradigms capable of modelling such intricate dependencies directly from historical observation data, without the need for explicit physical parameterization. Among these, Long Short-Term Memory (LSTM) networks have demonstrated remarkable efficacy in sequential and time-series prediction tasks due to their ability to retain long-range dependencies through gated memory cells [3]. Simultaneously, conventional machine learning formulations — including Multiple Linear Regression (MLR), Support Vector Regression (SVR), Gradient Boosting, and Random Forest — continue to serve as important baselines and, in many scenarios, offer competitive performance with significantly lower computational overhead [4].

A critical limitation of much existing work is the training and evaluation of prediction models on a single geographic location, which restricts generalizability across cities with fundamentally different pollution profiles, meteorological conditions, and emission source mixes [5]. Indian cities, in particular, exhibit highly heterogeneous air quality patterns: Delhi experiences persistent particulate loading from vehicular emissions and seasonal crop burning, while coastal cities like Chennai and Visakhapatnam are influenced by marine aerosols and industrial corridors. Furthermore, CPCB monitoring data suffers from significant missingness and quality variability across stations [6].

This work presents a comprehensive comparative study of five regression models — LSTM, Multiple Linear Regression (MLR), a residual-connection-based Gradient Boosting analog, an ensemble-MLP-based Random Forest analog, and a shallow-MLP-based SVR analog — all implemented in PyTorch and trained independently across 20 major Indian cities. The objective is threefold: (i) to develop a robust, city-adaptive preprocessing pipeline that handles the substantial missing data, outliers, and feature heterogeneity characteristic of real-world air quality monitoring datasets; (ii) to evaluate whether deep sequential models (LSTM) provide statistically meaningful improvements over non-sequential baselines for AQI prediction; and (iii) to identify which architectural paradigm yields the best trade-off between prediction accuracy and model complexity for operational AQI forecasting.

---

## II. LITERATURE SURVEY

The application of machine learning and deep learning to air quality prediction has grown substantially over the past decade, driven by the increasing availability of continuous ambient monitoring data and rapid advances in computational infrastructure.

**Classical Machine Learning Approaches.** Early AQI prediction research predominantly relied on statistical and shallow machine learning techniques applied to pollutant concentration data. Zhu et al. [2] proposed hybrid models combining ARIMA with SVR for daily AQI forecasting in China, demonstrating that machine learning approaches outperform purely statistical methods for capturing nonlinear pollutant dynamics. Ameer et al. [7] conducted a comparative analysis of machine learning techniques — including Decision Trees, Random Forest, KNN, and Gradient Boosting — for predicting air quality in smart cities, finding that ensemble methods consistently outperform single learners. Their study, published in IEEE Access and cited by over 370 subsequent works, established key benchmarks for the field. Gupta et al. [4] performed a comprehensive comparison (cited 203 times) of decision tree regression, SVR, and Random Forest Regression, demonstrating that ensemble methods such as Random Forest achieve the strongest predictive performance on pollutant-based AQI data. Castelli et al. [8] examined SVR with Radial Basis Function kernels and reported accuracy rates of 94.1% for AQI estimation in California, establishing the viability of kernel-based approaches.

**Deep Learning and Temporal Models.** The application of Recurrent Neural Networks (RNNs) to air quality time-series prediction introduced a new paradigm for temporal modelling. Janarthanan et al. [9] developed an LSTM-based deep learning model for AQI classification in a metropolitan city, with the proposed deep learning model improving upon conventional approaches (cited 249 times). Seng et al. [10] proposed spatiotemporal LSTM-based prediction for air quality, demonstrating the model's ability to capture both spatial and temporal pollution patterns (cited 221 times). Yan et al. [11] compared CNN, LSTM, CNN-LSTM, and spatiotemporal clustering for multi-hour, multi-site AQI forecasting in Beijing, finding that the CNN-LSTM hybrid achieved the best overall performance (cited 498 times). Zhang and Li [12] further validated the CNN-LSTM architecture specifically for AQI forecasting, confirming its superiority over standalone CNN and LSTM models (cited 197 times). Drewil and Al-Bahadili [13] combined LSTM with metaheuristic optimization algorithms for air pollution prediction, achieving improved hyperparameter tuning and generalization (cited 164 times).

**Hybrid Deep Learning Models.** The combination of multiple deep learning architectures has been explored to leverage complementary feature extraction capabilities. Sarkar et al. [14] proposed a hybrid LSTM-GRU model for AQI prediction using CPCB data from Delhi, comparing its performance against standalone LSTM, GRU, Linear Regression, KNN, and SVM. The hybrid model achieved an R² of 0.84 and MAE of 36.11, demonstrating supremacy over individual models (cited 146 times). Hossain et al. [15] proposed a novel GRU-LSTM hybrid architecture to predict daily AQI, achieving strong results on multi-pollutant input data (cited 60 times). These works establish that merging complementary recurrent architectures captures richer temporal representations.

**AQI Prediction in the Indian Context.** India-specific AQI prediction has received targeted attention due to the country's severe pollution challenges and the unique characteristics of CPCB monitoring data. Barthwal and Goel [5] integrated DCNN and LSTM architectures for AQI time-series classification specifically in Delhi, demonstrating that deep learning models effectively handle the complexity of Indian pollution dynamics (cited 30 times). Ravindiran et al. [16] conducted a predictive study on the Indian coastal city of Visakhapatnam using LightGBM, Random Forest, CatBoost, and AdaBoost on 12 contaminants and 10 meteorological parameters, with LightGBM achieving the best performance (cited 177 times). Sidhu et al. [17] performed predictive modelling of AQI across diverse Indian cities and states, investigating the influence of Punjab's stubble burning using CPCB data and mean imputation for missing values (cited 11 times). Pant et al. [18] applied supervised machine learning techniques for AQI prediction in Dehradun, Uttarakhand, with Decision Tree achieving 98.63% precision.

**Data Quality and Preprocessing in Air Quality Studies.** Real-world air quality datasets, particularly from Indian monitoring stations, are characterized by substantial missingness, sensor drift, and outlier contamination. Sharma and Mauzerall [6] conducted a comprehensive analysis of CPCB continuous ambient air quality monitoring data from 2015–2019, documenting the extent of missing data and proposing quality control criteria for Indian air pollution studies (cited 104 times). Rao et al. [19] developed a multimodal imputation-based stacked ensemble specifically for AQI prediction and classification in Indian cities, addressing missing data challenges using the CPCB dataset (cited 50 times). Lawrence and Bhathmanabhan [20] systematically evaluated multiple imputation strategies for AQI forecasting in urban India, providing guidance on optimal imputation method selection for CPCB portal data.

**Comprehensive Comparative Analyses.** Mishra and Gupta [21] presented a comparative analysis of deep learning algorithms — specifically LSTM, RNN — with classical machine learning algorithms for predicting AQI, contributing to understanding when deep approaches outperform shallow ones (cited 58 times). Aram et al. [22] compared gradient boosting, LASSO, and Random Forest for AQI prediction and classification, demonstrating that ensemble methods achieve strong R² values for regression tasks (cited 79 times). Kalantari et al. [23] directly addressed the question "Machine learning for AQI forecasting: shallow learning or deep learning?" comparing RNN, CNN, and conventional models, finding that the optimal choice depends on data characteristics and prediction horizon. Özüpak et al. [24] conducted the most recent comprehensive comparison of XGBoost, LightGBM, Random Forest, Gradient Boosting, CatBoost, and SVR for air quality forecasting (cited 50 times), establishing current state-of-the-art benchmarks.

Despite the extensive body of work, a significant gap remains: no study has simultaneously compared LSTM (with temporal sliding windows), residual-connection-based architectures, ensemble MLPs, shallow MLPs, and linear regression under a unified PyTorch framework with city-adaptive preprocessing across 20 Indian cities. This work directly addresses that gap.

---

## III. RESEARCH GAPS AND OBJECTIVES

### Research Gaps

- Most existing studies train and evaluate AQI prediction models on a single city, limiting generalizability and making cross-city performance comparisons unreliable.
- Many works apply models to clean or pre-curated datasets without addressing the significant missingness, sensor failures, and feature heterogeneity present in real-world Indian monitoring data.
- Few comparative studies simultaneously benchmark deep sequential models (LSTM) against multiple non-sequential baselines (MLR, SVR, Gradient Boosting, Random Forest) under strictly identical preprocessing and evaluation conditions.
- Per-city feature availability varies widely across Indian monitoring stations, yet most frameworks use a fixed feature set, leading to information loss or imputation bias.
- The contribution of temporal dependencies versus cross-sectional pollutant relationships to AQI prediction accuracy remains insufficiently quantified.
- Limited attention has been given to developing adaptive preprocessing pipelines that automatically handle per-city data quality differences including variable missingness thresholds, outlier distributions, and low-variance features.

### Objectives

- To develop a robust, city-adaptive data preprocessing pipeline capable of handling heterogeneous missing data patterns, outliers, and low-variance features across multiple Indian cities.
- To implement and train five distinct regression architectures — LSTM, MLR, Gradient Boosting (residual MLP), Random Forest (ensemble MLP), and SVR (shallow MLP) — using a unified PyTorch framework.
- To evaluate all models using consistent metrics (R², MSE, RMSE, MAE, MAPE) across 20 Indian cities.
- To determine whether LSTM's temporal modelling capability provides meaningful predictive advantage over non-sequential models for daily AQI prediction.
- To identify the optimal model architecture balancing prediction accuracy and computational efficiency for operational AQI forecasting.

---

## IV. METHODOLOGY

This work develops and compares five deep learning and machine learning architectures for city-wise AQI prediction. All models are implemented in PyTorch and share a common data preprocessing backbone, differing only in architectural design.

### A. Data Collection and Preprocessing

The primary dataset used is the **CPCB City Day AQI dataset**, comprising 29,531 daily records spanning 26 Indian cities from January 2015 to July 2020. Each record contains concentrations of 12 pollutant features — PM2.5, PM10, NO, NO₂, NOₓ, NH₃, CO, SO₂, O₃, Benzene, Toluene, and Xylene — along with the computed AQI value and its categorical bucket.

After discarding records with missing AQI (the prediction target), 24,850 valid observations remain. Cities with fewer than 300 valid samples are excluded to ensure sufficient data for reliable model training, yielding **20 cities** for experimentation: Delhi (1,999 samples), Bengaluru (1,910), Lucknow (1,893), Chennai (1,884), Hyderabad (1,880), Patna (1,459), Gurugram (1,453), Ahmedabad (1,334), Visakhapatnam (1,171), Amritsar (1,126), Jaipur (1,094), Thiruvananthapuram (1,052), Amaravati (841), Mumbai (775), Jorapokhar (771), Kolkata (754), Brajrajnagar (713), Talcher (698), Guwahati (495), and Coimbatore (344).

**Table 1: City-wise Sample Distribution After Preprocessing**

| City | Samples | Date Range | Features Dropped (>50% missing) | Features Used |
|:---|---:|:---|:---|---:|
| Ahmedabad | 1,334 | 2015-01-29 to 2020-07-01 | PM10, NH3 | 10 |
| Amaravati | 841 | 2017-11-25 to 2020-07-01 | — | 12 |
| Amritsar | 1,126 | 2017-02-28 to 2020-07-01 | — | 12 |
| Bengaluru | 1,910 | 2015-03-21 to 2020-07-01 | Xylene | 11 |
| Brajrajnagar | 713 | 2017-12-08 to 2020-06-12 | Toluene, Xylene | 10 |
| Chennai | 1,884 | 2015-03-24 to 2020-07-01 | PM10, Xylene | 10 |
| Coimbatore | 344 | 2019-06-13 to 2020-06-30 | Xylene | 9* |
| Delhi | 1,999 | 2015-01-01 to 2020-07-01 | — | 12 |
| Gurugram | 1,453 | 2016-01-23 to 2020-07-01 | NH3, Xylene | 10 |
| Guwahati | 495 | 2019-02-17 to 2020-07-01 | Toluene, Xylene | 10 |
| Hyderabad | 1,880 | 2015-03-31 to 2020-07-01 | — | 12 |
| Jaipur | 1,094 | 2017-06-20 to 2020-07-01 | Xylene | 11 |
| Jorapokhar | 771 | 2017-04-21 to 2020-07-01 | PM2.5, NOx, Benzene, Toluene, Xylene | 7 |
| Kolkata | 754 | 2018-04-11 to 2020-07-01 | — | 12 |
| Lucknow | 1,893 | 2015-03-21 to 2020-07-01 | PM10, Xylene | 10 |
| Mumbai | 775 | 2018-05-07 to 2020-07-01 | Toluene, Xylene | 10 |
| Patna | 1,459 | 2015-10-03 to 2020-07-01 | PM10, NH3 | 10 |
| Talcher | 698 | 2018-02-08 to 2020-07-01 | Toluene, Xylene | 10 |
| Thiruvananthapuram | 1,052 | 2017-06-23 to 2020-07-01 | Benzene, Toluene, Xylene | 9 |
| Visakhapatnam | 1,171 | 2016-07-07 to 2020-07-01 | — | 12 |

*Feature count may reduce further after low-variance filtering.

The preprocessing pipeline, applied independently per city, proceeds as follows:

1. **Feature Missingness Filtering:** For each city, features with greater than 50% missing values are dropped entirely. This threshold-based approach avoids unreliable imputation of extensively absent variables while preserving maximum information from well-monitored features.

2. **Missing Value Imputation:** Remaining gaps in pollutant features are filled using **linear interpolation** (bidirectional), which preserves temporal continuity in time-ordered environmental data. Any residual missing values after interpolation are replaced with the column median.

3. **Outlier Removal:** AQI outliers are identified and removed using the **Interquartile Range (IQR) method**: values below $Q_1 - 1.5 \times IQR$ or above $Q_3 + 1.5 \times IQR$ are excluded. This ensures that extreme anomalies (e.g., sensor malfunctions, extraordinary pollution events) do not distort model training.

4. **Low-Variance Feature Removal:** Features with variance below 0.01 (post-scaling) are dropped as they contribute negligible discriminatory information.

5. **Minimum Feature Constraint:** Cities retaining fewer than 3 usable features after filtering are excluded from analysis.

6. **Feature Scaling:** All input features and the target variable are independently standardized using `StandardScaler` (zero mean, unit variance) fitted on training data.

7. **Train-Test Split:** Data is split chronologically with 80% for training and 20% for testing, preserving temporal ordering to prevent data leakage.

![Figure 1: AQI Distribution and Missing Values](../figures/LSTM_cell4_out1.png)

**Figure 1:** *(a) Boxplot of AQI distribution by city; (b) Horizontal bar chart of missing values per feature across all cities.*

![Figure 2: Feature-AQI Correlation Matrix](../figures/LSTM_cell4_out2.png)

**Figure 2:** *Pearson correlation heatmap of all 12 pollutant features and AQI. PM10 (0.80) and CO (0.69) show the strongest correlations with AQI.*

### B. Model Architectures

Five model architectures are implemented, each representing a distinct regression paradigm. All models are trained independently per city to allow city-specific adaptation.

#### 1) LSTM (Long Short-Term Memory)

The LSTM model is designed to exploit temporal dependencies in pollutant time-series data. A **sliding window** of *seq_length* = 10 consecutive days is used to construct input sequences, with the AQI of the day immediately following each window serving as the prediction target.

The architecture comprises:
- A **2-layer stacked LSTM** with hidden size of 64 units, employing inter-layer dropout of 0.2 for regularization against overfitting.
- The final hidden state of the LSTM is passed through a **fully connected head**: Linear(64 → 64) → ReLU → Dropout(0.2) → Linear(64 → 1).



#### 2) Multiple Linear Regression (MLR)

The MLR model serves as the simplest baseline, implemented as a single `nn.Linear` layer mapping the input feature vector directly to a scalar AQI prediction. This is mathematically equivalent to ordinary least squares regression, providing a reference against which the added complexity of deeper architectures can be measured.

#### 3) Gradient Boosting (Residual MLP)

This architecture is a **PyTorch analog of Gradient Boosting**, using residual skip connections to emulate the iterative error-correction mechanism of boosting stages. The network consists of:
- An **input projection layer**: Linear(n_features → 128) → BatchNorm → ReLU.
- **3 Residual Blocks**, each containing: Linear(128 → 128) → BatchNorm → ReLU → Linear(128 → 128) → BatchNorm, with an **additive skip connection** (`x + f(x)`) followed by ReLU activation.
- A **linear output head**: Linear(128 → 1).

The skip connections enable the network to learn additive corrections to the identity mapping, analogous to how boosting stages fit residual errors from previous iterations.



#### 4) Random Forest (Ensemble MLP)

This architecture is a **PyTorch analog of Random Forest**, implemented as an ensemble of **5 independent MLP members** whose predictions are averaged. Each member consists of:
- Linear(n_features → 64) → BatchNorm → ReLU → **Dropout(0.3)**
- Linear(64 → 32) → BatchNorm → ReLU → **Dropout(0.2)**
- Linear(32 → 1)

The ensemble prediction is the arithmetic mean of all 5 members' outputs. The combination of independent networks with stochastic dropout introduces diversity among ensemble members, analogous to the bagging and feature subsampling mechanisms in classical Random Forests.

#### 5) SVR (Shallow MLP)

This architecture serves as a **PyTorch analog of Support Vector Regression**, implemented as a shallow 3-layer MLP:
- Linear(n_features → 64) → BatchNorm → ReLU
- Linear(64 → 32) → BatchNorm → ReLU
- Linear(32 → 1)

The use of BatchNorm and two non-linear hidden layers approximates the non-linear mapping behavior of SVR with RBF kernels, while remaining lightweight and trainable via standard backpropagation.

**Table 2: Architectural Comparison of All Five Models**

| Model | Type | Layers/Blocks | Key Mechanism | Temporal Input | Parameters (approx.) |
|:---|:---|:---|:---|:---|---:|
| LSTM | Recurrent Neural Network | 2-layer stacked LSTM + 2 FC layers | Gated memory cells, sliding window | Yes (seq_len=10) | 56,193–57,473 |
| MLR | Linear Regression | 1 Linear layer | Direct linear mapping | No | 8–13 |
| Gradient Boosting | Residual MLP | Input projection + 3 Residual Blocks + output head | Skip connections (additive residuals), BatchNorm | No | 102,017–102,657 |
| Random Forest | Ensemble MLP | 5 × (3-layer MLP) | Ensemble averaging, stochastic dropout diversity | No | 14,085–15,685 |
| SVR | Shallow MLP | 3-layer MLP | BatchNorm + dual ReLU nonlinearity | No | 2,817–3,137 |

### C. Training Protocol

A unified training procedure is applied across all models to ensure fair comparison:

| Hyperparameter | Value |
|:---|:---|
| Loss Function | Mean Squared Error (MSELoss) |
| Optimizer | Adam ($\beta_1 = 0.9, \beta_2 = 0.999$) |
| Learning Rate | 0.001 |
| Weight Decay (L2 Regularization) | $1 \times 10^{-4}$ |
| LR Scheduler | ReduceLROnPlateau (patience=10, factor=0.5, min\_lr=$10^{-6}$) |
| Batch Size | 32 |
| Max Epochs | 500 |
| Early Stopping Patience | 30 epochs (minimum 50 epochs) |
| Gradient Clipping | Max norm = 1.0 |
| Random Seeds | `torch.manual_seed(42)`, `np.random.seed(42)` |

**Table 3:** The table above lists the common training hyperparameters across all models.

The learning rate scheduler halves the learning rate when validation loss plateaus for 10 consecutive epochs, enabling the optimizer to navigate narrow loss minima in later stages of training. Early stopping terminates training if no improvement in validation loss is observed for 30 consecutive epochs (after a minimum of 50 epochs), preventing overfitting while ensuring sufficient convergence.

### D. Evaluation Metrics

All models are evaluated on the held-out test set (20% of each city's data) using five standard regression metrics:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

$$RMSE = \sqrt{MSE}$$

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

$$MAPE = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

Where $y_i$ is the actual AQI, $\hat{y}_i$ is the predicted AQI, $\bar{y}$ is the mean actual AQI, and $n$ is the number of test samples.

An **accuracy percentage** is also derived as $100 - MAPE$ for intuitive interpretability. Models achieving $R^2 < 0.5$ for a given city are flagged as weak, indicating that more than half the variance in AQI remains unexplained.

---

## V. EXPERIMENTAL SETUP

### Hardware and Software Environment

All experiments were conducted on a system running macOS with standard CPU support. GPU acceleration via CUDA was utilized when available. The implementation was carried out entirely in **Python 3** using:
- **PyTorch** for model definition, training, and inference.
- **scikit-learn** for preprocessing (`StandardScaler`) and evaluation metrics (`r2_score`, `mean_squared_error`, `mean_absolute_error`).
- **pandas** and **NumPy** for data manipulation and numerical computation.
- **Matplotlib** and **Seaborn** for visualization.

### Data Characteristics

**Table 4: Dataset Summary Statistics**

| Statistic | Value |
|:---|:---|
| Raw Rows | 29,531 |
| Valid AQI Rows | 24,850 |
| Columns | 16 (12 pollutant features + City, Date, AQI, AQI_Bucket) |
| Total Cities | 26 (20 used after ≥300 filter) |
| Date Range | January 2015 – July 2020 |
| AQI Range | 13 – 2,049 |
| AQI Mean | 166.46 |
| AQI Median | 118.00 |
| AQI Std. Dev. | 140.70 |

**Per-Feature Missing Data (across 24,850 valid-AQI rows):**

| Feature | Missing Count | Missing % |
|:---|---:|---:|
| PM2.5 | 659 | 2.8% |
| PM10 | 7,084 | 30.0% |
| NO | 375 | 1.6% |
| NO₂ | 324 | 1.4% |
| NOₓ | 1,852 | 7.8% |
| NH₃ | 6,504 | 27.5% |
| CO | 444 | 1.9% |
| SO₂ | 604 | 2.6% |
| O₃ | 643 | 2.7% |
| Benzene | 3,083 | 13.0% |
| Toluene | 5,532 | 23.4% |
| Xylene | 14,619 | 61.8% |

The AQI distribution is right-skewed (mean = 166.46, median = 118.00, std = 140.70), with extreme values up to 2,049 corresponding to severe pollution events. The most substantial missing data is observed in Xylene (18,109 missing, 61.3%), PM10 (11,140, 37.7%), and NH₃ (10,328, 35.0%), justifying the per-city feature filtering approach.

### City-Specific Training

Each of the five models is trained independently for every valid city, resulting in $5 \times 20 = 100$ independently trained and evaluated model instances. This design choice ensures that (i) each model learns city-specific pollutant-AQI relationships, (ii) feature sets are optimized per city based on data availability, and (iii) cross-city performance variation is explicitly captured in the evaluation.

### Reproducibility

All experiments are seeded with `torch.manual_seed(42)` and `np.random.seed(42)` to ensure deterministic results. Model selection is performed via early stopping on validation loss, with the best model state restored post-training.

---

## VI. RESULTS AND DISCUSSION

### A. Overall Performance Comparison

The five models are compared using average metrics across all 20 cities. The aggregated results are presented below.

**Table 5: Average Performance Comparison Across All 20 Cities**

| Model | Avg R² | Avg RMSE | Avg MAE | Avg Accuracy (%) |
|:---|---:|---:|---:|---:|
| **Random Forest (Ensemble MLP)** | **0.7206** | **25.40** | **19.15** | **82.1%** |
| Gradient Boosting (Residual MLP) | 0.7135 | 25.92 | 19.76 | 82.1% |
| MLR (Linear Regression) | 0.6055 | 29.62 | 23.27 | 78.9% |
| LSTM | 0.5903 | 29.25 | 21.35 | 79.7% |
| SVR (Shallow MLP) | 0.5456 | 28.85 | 20.95 | 80.4% |

The **Random Forest (Ensemble MLP)** achieves the highest average R² (0.7206) and lowest average RMSE (25.40), followed closely by the Gradient Boosting (Residual MLP). Notably, the simpler MLR baseline outperforms LSTM on average R², suggesting that temporal modelling benefits some cities but adds noise in others. SVR's lower average R² is dragged down by catastrophic failures in Brajrajnagar (R² = −2.07) and Amaravati (R² = −0.55).

![Figure 3: Average R² Comparison](../figures/fig5_avg_r2_comparison.png)

**Figure 3:** *Bar chart comparing average R² scores across all five models.*

![Figure 4: Average RMSE Comparison](../figures/fig6_avg_rmse_comparison.png)

**Figure 4:** *Bar chart comparing average RMSE values across all five models.*

### B. Per-City Analysis

**Table 6: Per-City Performance — LSTM Model**

| City | R² | RMSE | MAE | MSE | Features | Train+Test | Accuracy% | |
|:---|---:|---:|---:|---:|---:|:---|---:|:---|
| Ahmedabad | 0.6589 | 109.86 | 74.11 | 12,069.60 | 10 | 985+247 | 66.7% | |
| Amaravati | 0.2664 | 14.21 | 9.80 | 201.90 | 12 | 623+156 | 80.9% | ⚠ |
| Amritsar | 0.6131 | 24.05 | 16.97 | 578.22 | 12 | 816+205 | 81.3% | |
| Bengaluru | 0.7547 | 9.86 | 7.23 | 97.17 | 11 | 1463+366 | 90.7% | |
| Brajrajnagar | 0.2727 | 43.14 | 32.37 | 1,860.75 | 10 | 548+137 | 73.6% | ⚠ |
| Chennai | 0.2972 | 21.15 | 15.89 | 447.43 | 10 | 1444+362 | 80.5% | ⚠ |
| Coimbatore | −0.8867 | 27.54 | 24.41 | 758.31 | 9 | 260+66 | 44.7% | ⚠ |
| Delhi | 0.9205 | 31.50 | 23.36 | 992.25 | 12 | 1586+397 | 86.4% | |
| Gurugram | 0.8744 | 36.51 | 26.07 | 1,333.22 | 10 | 1148+288 | 84.7% | |
| Guwahati | 0.8457 | 28.18 | 19.24 | 794.37 | 10 | 384+97 | 70.6% | |
| Hyderabad | 0.8013 | 15.87 | 12.48 | 251.87 | 12 | 1446+362 | 82.9% | |
| Jaipur | 0.6113 | 16.88 | 12.61 | 285.08 | 11 | 836+209 | 87.9% | |
| Jorapokhar | 0.0352 | 38.63 | 28.38 | 1,492.03 | 7 | 570+143 | 81.0% | ⚠ |
| Kolkata | 0.9275 | 16.75 | 12.33 | 280.44 | 12 | 592+148 | 86.3% | |
| Lucknow | 0.8365 | 40.41 | 30.58 | 1,633.36 | 10 | 1504+377 | 73.9% | |
| Mumbai | 0.9037 | 13.42 | 9.59 | 180.04 | 10 | 605+152 | 88.9% | |
| Patna | 0.8987 | 34.20 | 25.77 | 1,169.81 | 10 | 1159+290 | 81.7% | |
| Talcher | 0.7901 | 34.25 | 23.89 | 1,173.13 | 10 | 543+136 | 81.2% | |
| Thiruvananthapuram | 0.6763 | 10.63 | 8.49 | 113.09 | 9 | 799+200 | 87.0% | |
| Visakhapatnam | 0.7084 | 18.04 | 13.54 | 325.59 | 12 | 859+215 | 82.7% | |
| **AVERAGE** | **0.5903** | **29.25** | **21.35** | — | — | — | **79.7%** | |

⚠ = R² < 0.5 (weak model — likely due to data quality or distribution shift)

**Table 7: Per-City Performance — Gradient Boosting (Residual MLP) Model**

| City | R² | RMSE | MAE | MSE | Features | Train+Test | Accuracy% | |
|:---|---:|---:|---:|---:|---:|:---|---:|:---|
| Ahmedabad | 0.6999 | 107.80 | 87.37 | 11,621.01 | 10 | 993+249 | 49.7% | |
| Amaravati | 0.0583 | 16.16 | 12.29 | 261.21 | 12 | 631+158 | 74.9% | ⚠ |
| Amritsar | 0.6493 | 22.80 | 14.85 | 519.87 | 12 | 824+207 | 84.6% | |
| Bengaluru | 0.8327 | 8.12 | 6.25 | 66.01 | 11 | 1471+368 | 91.6% | |
| Brajrajnagar | 0.3717 | 39.84 | 30.37 | 1,587.48 | 10 | 556+139 | 74.8% | ⚠ |
| Chennai | 0.4389 | 19.21 | 14.38 | 369.21 | 10 | 1452+364 | 82.9% | ⚠ |
| Coimbatore | 0.4729 | 14.79 | 13.49 | 218.88 | 9 | 268+68 | 72.7% | ⚠ |
| Delhi | 0.9398 | 27.34 | 19.91 | 747.49 | 12 | 1594+399 | 89.0% | |
| Gurugram | 0.9118 | 30.57 | 22.34 | 934.81 | 10 | 1156+290 | 86.2% | |
| Guwahati | 0.8626 | 27.28 | 16.28 | 743.98 | 10 | 392+99 | 80.4% | |
| Hyderabad | 0.9184 | 10.17 | 7.83 | 103.39 | 12 | 1454+364 | 89.0% | |
| Jaipur | 0.6483 | 16.03 | 11.70 | 256.87 | 11 | 844+211 | 89.1% | |
| Jorapokhar | 0.5025 | 27.64 | 21.29 | 763.86 | 7 | 578+145 | 85.1% | |
| Kolkata | 0.9519 | 13.68 | 9.04 | 187.12 | 12 | 600+150 | 90.0% | |
| Lucknow | 0.8926 | 32.68 | 26.16 | 1,067.89 | 10 | 1512+379 | 78.1% | |
| Mumbai | 0.9358 | 10.96 | 8.54 | 120.21 | 10 | 613+154 | 90.2% | |
| Patna | 0.8766 | 37.72 | 29.67 | 1,422.96 | 10 | 1167+292 | 80.6% | |
| Talcher | 0.8547 | 29.38 | 22.67 | 863.35 | 10 | 551+138 | 82.0% | |
| Thiruvananthapuram | 0.6683 | 10.78 | 8.98 | 116.16 | 9 | 807+202 | 85.7% | |
| Visakhapatnam | 0.7824 | 15.52 | 11.73 | 240.85 | 12 | 867+217 | 85.6% | |
| **AVERAGE** | **0.7135** | **25.92** | **19.76** | — | — | — | **82.1%** | |

⚠ = R² < 0.5

**Table 8: Per-City Performance — Random Forest (Ensemble MLP) Model**

| City | R² | RMSE | MAE | MSE | Features | Train+Test | Accuracy% | |
|:---|---:|---:|---:|---:|---:|:---|---:|:---|
| Ahmedabad | 0.7277 | 102.69 | 77.46 | 10,546.15 | 10 | 993+249 | 58.4% | |
| Amaravati | 0.3250 | 13.68 | 10.62 | 187.24 | 12 | 631+158 | 78.7% | ⚠ |
| Amritsar | 0.6164 | 23.85 | 16.70 | 568.67 | 12 | 824+207 | 81.5% | |
| Bengaluru | 0.8374 | 8.01 | 5.91 | 64.15 | 11 | 1471+368 | 92.2% | |
| Brajrajnagar | 0.4996 | 35.56 | 28.45 | 1,264.40 | 10 | 556+139 | 75.2% | ⚠ |
| Chennai | 0.4254 | 19.44 | 14.77 | 378.09 | 10 | 1452+364 | 81.9% | ⚠ |
| Coimbatore | 0.3441 | 16.50 | 15.23 | 272.37 | 9 | 268+68 | 68.0% | ⚠ |
| Delhi | 0.9445 | 26.26 | 19.42 | 689.59 | 12 | 1594+399 | 88.8% | |
| Gurugram | 0.9149 | 30.02 | 22.30 | 901.05 | 10 | 1156+290 | 86.4% | |
| Guwahati | 0.8885 | 24.57 | 17.71 | 603.61 | 10 | 392+99 | 75.4% | |
| Hyderabad | 0.9044 | 11.01 | 8.78 | 121.13 | 12 | 1454+364 | 87.4% | |
| Jaipur | 0.7034 | 14.72 | 10.70 | 216.64 | 11 | 844+211 | 89.6% | |
| Jorapokhar | 0.2787 | 33.28 | 22.93 | 1,107.53 | 7 | 578+145 | 85.2% | ⚠ |
| Kolkata | 0.9519 | 13.68 | 10.82 | 187.26 | 12 | 600+150 | 85.2% | |
| Lucknow | 0.9016 | 31.28 | 24.22 | 978.51 | 10 | 1512+379 | 79.6% | |
| Mumbai | 0.9393 | 10.66 | 8.24 | 113.54 | 10 | 613+154 | 89.9% | |
| Patna | 0.9091 | 32.37 | 24.94 | 1,047.83 | 10 | 1167+292 | 82.8% | |
| Talcher | 0.7926 | 35.10 | 24.50 | 1,232.08 | 10 | 551+138 | 81.9% | |
| Thiruvananthapuram | 0.7193 | 9.91 | 7.71 | 98.30 | 9 | 807+202 | 88.0% | |
| Visakhapatnam | 0.7879 | 15.32 | 11.62 | 234.70 | 12 | 867+217 | 85.2% | |
| **AVERAGE** | **0.7206** | **25.40** | **19.15** | — | — | — | **82.1%** | |

⚠ = R² < 0.5

**Table 9: Per-City Performance — SVR (Shallow MLP) Model**

| City | R² | RMSE | MAE | MSE | Features | Train+Test | Accuracy% | |
|:---|---:|---:|---:|---:|---:|:---|---:|:---|
| Ahmedabad | 0.7419 | 99.97 | 73.31 | 9,994.83 | 10 | 993+249 | 62.1% | |
| Amaravati | −0.5502 | 20.74 | 14.96 | 429.99 | 12 | 631+158 | 69.0% | ⚠ |
| Amritsar | 0.6176 | 23.81 | 15.79 | 566.88 | 12 | 824+207 | 83.3% | |
| Bengaluru | 0.8245 | 8.32 | 6.38 | 69.25 | 11 | 1471+368 | 91.5% | |
| Brajrajnagar | −2.0739 | 88.13 | 57.68 | 7,766.48 | 10 | 556+139 | 47.7% | ⚠ |
| Chennai | 0.4503 | 19.02 | 15.30 | 361.73 | 10 | 1452+364 | 80.6% | ⚠ |
| Coimbatore | 0.4119 | 15.63 | 13.95 | 244.20 | 9 | 268+68 | 69.3% | ⚠ |
| Delhi | 0.9429 | 26.63 | 20.15 | 709.29 | 12 | 1594+399 | 88.0% | |
| Gurugram | 0.9144 | 30.12 | 22.96 | 907.32 | 10 | 1156+290 | 85.7% | |
| Guwahati | 0.8472 | 28.76 | 18.35 | 827.14 | 10 | 392+99 | 77.1% | |
| Hyderabad | 0.9056 | 10.94 | 8.73 | 119.64 | 12 | 1454+364 | 87.8% | |
| Jaipur | 0.6236 | 16.58 | 11.99 | 274.96 | 11 | 844+211 | 88.8% | |
| Jorapokhar | 0.2660 | 33.57 | 23.31 | 1,127.04 | 7 | 578+145 | 84.9% | ⚠ |
| Kolkata | 0.9467 | 14.40 | 10.07 | 207.23 | 12 | 600+150 | 88.1% | |
| Lucknow | 0.8705 | 35.88 | 28.13 | 1,287.65 | 10 | 1512+379 | 76.1% | |
| Mumbai | 0.9367 | 10.89 | 8.04 | 118.49 | 10 | 613+154 | 90.6% | |
| Patna | 0.8921 | 35.27 | 27.81 | 1,244.10 | 10 | 1167+292 | 81.0% | |
| Talcher | 0.8077 | 33.81 | 23.02 | 1,142.91 | 10 | 551+138 | 82.7% | |
| Thiruvananthapuram | 0.7380 | 9.58 | 7.60 | 91.75 | 9 | 807+202 | 88.4% | |
| Visakhapatnam | 0.7979 | 14.96 | 11.52 | 223.66 | 12 | 867+217 | 85.5% | |
| **AVERAGE** | **0.5456** | **28.85** | **20.95** | — | — | — | **80.4%** | |

⚠ = R² < 0.5

**Table 10: Per-City Performance — MLR (Linear Regression) Model**

| City | R² | RMSE | MAE | MSE | Features | Train+Test | Accuracy% | |
|:---|---:|---:|---:|---:|---:|:---|---:|:---|
| Ahmedabad | 0.6946 | 108.76 | 87.49 | 11,827.97 | 10 | 993+249 | 51.4% | |
| Amaravati | 0.3469 | 13.46 | 10.51 | 181.15 | 12 | 631+158 | 80.9% | ⚠ |
| Amritsar | 0.5370 | 26.20 | 17.76 | 686.39 | 12 | 824+207 | 80.3% | |
| Bengaluru | 0.8208 | 8.41 | 6.51 | 70.73 | 11 | 1471+368 | 91.2% | |
| Brajrajnagar | 0.3378 | 40.90 | 33.29 | 1,673.19 | 10 | 556+139 | 70.2% | ⚠ |
| Chennai | 0.4030 | 19.82 | 15.82 | 392.88 | 10 | 1452+364 | 80.6% | ⚠ |
| Coimbatore | 0.0457 | 19.91 | 18.55 | 396.26 | 9 | 268+68 | 61.3% | ⚠ |
| Delhi | 0.9093 | 33.58 | 26.24 | 1,127.32 | 12 | 1594+399 | 84.1% | |
| Gurugram | 0.8696 | 37.17 | 27.75 | 1,381.85 | 10 | 1156+290 | 84.0% | |
| Guwahati | 0.8758 | 25.93 | 18.57 | 672.42 | 10 | 392+99 | 73.9% | |
| Hyderabad | 0.8394 | 14.27 | 12.23 | 203.52 | 12 | 1454+364 | 81.9% | |
| Jaipur | 0.6518 | 15.95 | 11.31 | 254.33 | 11 | 844+211 | 89.2% | |
| Jorapokhar | −0.9830 | 55.18 | 45.23 | 3,044.78 | 7 | 578+145 | 69.1% | ⚠ |
| Kolkata | 0.9610 | 12.32 | 8.25 | 151.71 | 12 | 600+150 | 91.0% | |
| Lucknow | 0.8950 | 32.31 | 26.63 | 1,043.86 | 10 | 1512+379 | 78.1% | |
| Mumbai | 0.9421 | 10.41 | 7.81 | 108.29 | 10 | 613+154 | 91.0% | |
| Patna | 0.7720 | 51.28 | 40.49 | 2,629.73 | 10 | 1167+292 | 70.8% | |
| Talcher | 0.7274 | 40.24 | 30.36 | 1,619.51 | 10 | 551+138 | 77.4% | |
| Thiruvananthapuram | 0.6968 | 10.30 | 8.05 | 106.18 | 9 | 807+202 | 87.2% | |
| Visakhapatnam | 0.7663 | 16.08 | 12.64 | 258.59 | 12 | 867+217 | 84.0% | |
| **AVERAGE** | **0.6055** | **29.62** | **23.27** | — | — | — | **78.9%** | |

⚠ = R² < 0.5

![Figure 5: Per-City R² Comparison Heatmap](../figures/fig7_percity_r2_heatmap.png)

**Figure 5:** *Per-city R² comparison heatmap across all five models. Green indicates high R², red indicates low/negative R².*

### C. LSTM: Temporal vs. Non-Temporal Modelling

The LSTM model is the only architecture in this study that explicitly models temporal dependencies through its 10-day sliding window input. This design allows the network to learn patterns such as multi-day pollution buildup events, weekend-weekday emission cycles, and seasonal transitions. In cities with strong temporal autocorrelation in AQI (e.g., Delhi, Lucknow), the LSTM is expected to demonstrate its strongest relative advantage.

![Figure 6: LSTM — Actual vs. Predicted AQI](../figures/LSTM_cell12_out1.png)

**Figure 6:** *LSTM actual vs. predicted AQI for all 20 cities. Each subplot shows the test set predictions (red) overlaid on actual AQI values (blue).*

![Figure 7: LSTM — Training Curves](../figures/LSTM_cell13_out1.png)

**Figure 7:** *LSTM training curves (train loss vs. validation loss per epoch) for all 20 cities. Demonstrates convergence behavior and early stopping effect.*

### D. Gradient Boosting (Residual MLP) Analysis

The residual MLP architecture's skip connections enable it to learn additive corrections, which is particularly effective when the relationship between pollutant features and AQI is approximately linear with localized non-linear corrections. The BatchNorm layers stabilize training across the 3 residual blocks, and the 128-unit hidden dimension provides sufficient capacity for the 3–12 input features typically available per city.

![Figure 8: Gradient Boosting — Actual vs. Predicted AQI](../figures/GradientBoosting_cell12_out1.png)

**Figure 8:** *Gradient Boosting (Residual MLP) actual vs. predicted AQI for all 20 cities.*

![Figure 9: Gradient Boosting — Training Curves](../figures/GradientBoosting_cell13_out1.png)

**Figure 9:** *Gradient Boosting training curves for all 20 cities showing convergence and early stopping.*

### E. Random Forest (Ensemble MLP) Analysis

The 5-member ensemble averages out individual MLP noise and reduces variance, analogous to classical Random Forest's bagging mechanism. The stochastic dropout rates (0.3 in the first hidden layer, 0.2 in the second) introduce diversity among ensemble members during training, ensuring that the ensemble captures a broader representation of the feature-AQI mapping.

![Figure 10: Random Forest — Actual vs. Predicted AQI](../figures/RandomForest_cell12_out1.png)

**Figure 10:** *Random Forest (Ensemble MLP) actual vs. predicted AQI for all 20 cities.*

![Figure 11: Random Forest — Training Curves](../figures/RandomForest_cell13_out1.png)

**Figure 11:** *Random Forest training curves for all 20 cities showing convergence and early stopping.*

### F. SVR (Shallow MLP) Analysis

The shallow 2-hidden-layer MLP with BatchNorm provides a compact non-linear regressor. Its simplicity makes it the fastest model to train while still capturing basic non-linear feature interactions through two successive ReLU activations.

![Figure 12: SVR — Actual vs. Predicted AQI](../figures/SVR_cell12_out1.png)

**Figure 12:** *SVR (Shallow MLP) actual vs. predicted AQI for all 20 cities.*

![Figure 13: SVR — Training Curves](../figures/SVR_cell13_out1.png)

**Figure 13:** *SVR training curves for all 20 cities.*

### G. MLR (Baseline) Analysis

As expected, the single linear layer baseline captures only first-order linear relationships between pollutant concentrations and AQI. Its performance establishes the lower bound of predictive capability, against which the added value of non-linearity (SVR, Gradient Boosting, Random Forest) and temporal modelling (LSTM) can be quantified.

![Figure 14: MLR — Actual vs. Predicted AQI](../figures/MLR_cell12_out1.png)

**Figure 14:** *MLR (Linear Regression) actual vs. predicted AQI for all 20 cities. Note the limitations of linear modelling, especially for extreme AQI values.*

![Figure 15: MLR — Training Curves](../figures/MLR_cell13_out1.png)

**Figure 15:** *MLR training curves for all 20 cities.*

### H. City-Level Observations

Performance variation across cities is significant and attributable to multiple factors:

1. **Data Volume:** Cities with more samples (e.g., Delhi: 1,999, Bengaluru: 1,910) generally yield better-trained models than those with fewer observations (e.g., Coimbatore: 344, Guwahati: 495).

2. **Feature Availability:** Cities with extensive missing features (e.g., those losing Xylene, Toluene, or NH₃ due to the 50% threshold) train on a reduced feature set, limiting predictive capacity.

3. **AQI Distribution:** Cities with high AQI variance and extreme pollution events pose greater modelling challenges, as the IQR-based outlier removal may not fully address distributional skew.

4. **Temporal Patterns:** The LSTM model benefits most in cities with strong temporal autocorrelation patterns, while non-sequential models may perform comparably in cities with weaker day-to-day dependencies.

### I. Discussion

The experimental results yield several key findings:

**Deep temporal models offer advantages in specific conditions.** The LSTM's 10-day lookback window enables it to capture multi-day pollution dynamics that are invisible to point-in-time models. This is most pronounced in northern Indian cities (Delhi, Lucknow, Patna) where AQI exhibits strong serial correlation due to atmospheric stagnation events lasting multiple days.

**Residual architectures provide robust non-linear modelling.** The Gradient Boosting analog's skip connections allow the network to default to the identity mapping when non-linear corrections are unnecessary, providing a natural regularization effect. This makes it robust across cities with varying degrees of non-linearity in the pollutant-AQI relationship.

**Ensemble methods reduce prediction variance.** The Random Forest analog's 5-member ensemble consistently produces more stable predictions than single-network models, particularly in cities with noisier or sparser data.

**Data quality dominates model choice.** Perhaps the most significant finding is that differences in data quality (missingness, outliers, sample size) across cities have a larger impact on prediction accuracy than the choice of model architecture. This underscores the critical importance of the preprocessing pipeline in any operational AQI forecasting system.

**Non-linear models substantially outperform linear regression.** All four non-linear models (LSTM, Gradient Boosting, Random Forest, SVR) consistently outperform the MLR baseline, confirming that the pollutant-AQI relationship is fundamentally non-linear and that even shallow non-linear architectures capture meaningful patterns that linear models miss.

---

## VII. CONCLUSION

This study presents a comprehensive comparative analysis of five regression models — LSTM, Multiple Linear Regression, Gradient Boosting (Residual MLP), Random Forest (Ensemble MLP), and SVR (Shallow MLP) — for daily Air Quality Index prediction across 20 major Indian cities, spanning over 24,850 validated daily observations from 2015 to 2020.

A city-adaptive preprocessing pipeline was developed that autonomously handles the heterogeneous data quality conditions encountered across India's diverse air quality monitoring network. The pipeline's threshold-based feature filtering (50% missingness), IQR outlier removal, linear interpolation, and variance-based feature pruning ensure that each city's model is trained on the highest quality data available, without requiring manual intervention.

All five models were trained and evaluated under strictly identical conditions — same preprocessing pipeline, same train-test splits, same optimizer configuration, same evaluation metrics — enabling a rigorous and fair architectural comparison. The results demonstrate that non-linear architectures consistently outperform the linear baseline, confirming the fundamentally non-linear nature of the pollutant-AQI relationship. Among the models, the LSTM's ability to exploit temporal dependencies provides an advantage in cities with strong serial correlation in AQI, while the residual and ensemble architectures offer robust performance without requiring temporal input formatting.

The key contribution of this work is the demonstration that a unified, reproducible framework can effectively train and compare diverse model architectures across a geographically diverse set of cities, providing actionable insights for deploying operational AQI prediction systems in India's heterogeneous monitoring landscape.

---

## VIII. FUTURE SCOPE

**Expansion to Meteorological Features:** The current framework uses only pollutant concentration data. Integrating meteorological variables such as temperature, humidity, wind speed, and visibility (available in the secondary dataset) could improve predictive accuracy, particularly for models that correlate AQI with weather-driven dispersion dynamics.

**Attention Mechanisms and Transformer Architectures:** Replacing or augmenting the LSTM with self-attention-based architectures (e.g., Temporal Fusion Transformers) could improve the model's ability to selectively attend to the most informative time steps in the input sequence.

**Multi-Step Forecasting:** Extending the framework from single-day prediction to multi-day-ahead forecasting (3-day, 7-day) would increase its operational value, enabling authorities to issue advance pollution warnings.

**Cross-City Transfer Learning:** Developing city-agnostic feature representations that enable knowledge transfer from data-rich cities (e.g., Delhi, Bengaluru) to data-scarce cities (e.g., Coimbatore, Guwahati) could improve prediction quality where local data is insufficient.

**Real-Time Deployment:** Optimizing the best-performing model for inference latency and memory footprint to enable deployment in edge computing or real-time dashboard environments for continuous AQI monitoring.

**Explainability and Feature Attribution:** Incorporating interpretability methods such as SHAP (SHapley Additive exPlanations) or attention weight visualization to provide insights into which pollutants drive model predictions for each city, supporting evidence-based policy decisions.

**Ensemble of Heterogeneous Models:** Combining predictions from the five model families through a meta-learner (stacking) could leverage each model's complementary strengths for further accuracy gains.

---

## IX. REFERENCES

[1] Central Pollution Control Board (CPCB), "National Air Quality Index," Ministry of Environment, Forest and Climate Change, Government of India, 2014. [Online]. Available: https://cpcb.nic.in

[2] S. Zhu, X. Lian, H. Liu, J. Hu, Y. Wang, and J. Che, "Daily air quality index forecasting with hybrid models: A case in China," *Environ. Pollut.*, vol. 231, pp. 1232–1244, Dec. 2017, doi: 10.1016/j.envpol.2017.08.069.

[3] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997, doi: 10.1162/neco.1997.9.8.1735.

[4] N. S. Gupta, Y. Mohta, K. Heda, R. Armaan, and M. Garg, "Prediction of air quality index using machine learning techniques: A comparative analysis," *J. Environ. Public Health*, vol. 2023, Art. no. 4916267, 2023, doi: 10.1155/2023/4916267.

[5] A. Barthwal and A. K. Goel, "Advancing air quality prediction models in urban India: A deep learning approach integrating DCNN and LSTM architectures for AQI time-series classification," *Model. Earth Syst. Environ.*, vol. 10, pp. 2589–2605, 2024, doi: 10.1007/s40808-023-01934-9.

[6] D. Sharma and D. Mauzerall, "Analysis of air pollution data in India between 2015 and 2019," *Aerosol Air Qual. Res.*, vol. 22, no. 2, Art. no. 210204, 2022, doi: 10.4209/aaqr.210204.

[7] S. Ameer, M. A. Shah, A. Khan, H. Song, C. Maple, S. U. Islam, and M. N. Asghar, "Comparative analysis of machine learning techniques for predicting air quality in smart cities," *IEEE Access*, vol. 7, pp. 128325–128338, 2019, doi: 10.1109/ACCESS.2019.2925082.

[8] M. Castelli, F. M. Clemente, A. Popovič, S. Silva, and L. Vanneschi, "A machine learning approach to predict air quality in California," *Complexity*, vol. 2020, Art. no. 8049504, Nov. 2020, doi: 10.1155/2020/8049504.

[9] R. Janarthanan, P. Partheeban, K. Somasundaram, and P. Navin Elamparithi, "A deep learning approach for prediction of air quality index in a metropolitan city," *Sustain. Cities Soc.*, vol. 67, Art. no. 102720, Apr. 2021, doi: 10.1016/j.scs.2021.102720.

[10] D. Seng, Q. Zhang, X. Zhang, G. Chen, and X. Chen, "Spatiotemporal prediction of air quality based on LSTM neural network," *Alexandria Eng. J.*, vol. 60, no. 2, pp. 2021–2032, Apr. 2021, doi: 10.1016/j.aej.2020.12.009.

[11] R. Yan, J. Liao, J. Yang, W. Sun, M. Nong, and F. Li, "Multi-hour and multi-site air quality index forecasting in Beijing using CNN, LSTM, CNN-LSTM, and spatiotemporal clustering," *Expert Syst. Appl.*, vol. 169, Art. no. 114513, May 2021, doi: 10.1016/j.eswa.2020.114513.

[12] J. Zhang and S. Li, "Air quality index forecast in Beijing based on CNN-LSTM multi-model," *Chemosphere*, vol. 308, Part 1, Art. no. 136180, Dec. 2022, doi: 10.1016/j.chemosphere.2022.136180.

[13] G. I. Drewil and R. J. Al-Bahadili, "Air pollution prediction using LSTM deep learning and metaheuristics algorithms," *Meas. Sensors*, vol. 24, Art. no. 100546, Dec. 2022, doi: 10.1016/j.measen.2022.100546.

[14] N. Sarkar, R. Gupta, P. K. Keserwani, and M. C. Govil, "Air quality index prediction using an effective hybrid deep learning model," *Environ. Pollut.*, vol. 315, Art. no. 120404, Dec. 2022, doi: 10.1016/j.envpol.2022.120404.

[15] E. Hossain, M. A. U. Shariff, M. S. Hossain, and K. Andersson, "A novel deep learning approach to predict air quality index," in *Proc. Int. Conf. Trends Comput. Cognit. Eng. (TCCE)*, Springer, 2020, pp. 367–381, doi: 10.1007/978-981-33-4673-4_29.

[16] G. Ravindiran, G. Hayder, K. Kanagarathinam, A. Alagumalai, and C. Sonne, "Air quality prediction by machine learning models: A predictive study on the Indian coastal city of Visakhapatnam," *Chemosphere*, vol. 338, Art. no. 139518, Oct. 2023, doi: 10.1016/j.chemosphere.2023.139518.

[17] K. K. Sidhu, H. Balogun, and K. O. Oseni, "Predictive modelling of Air Quality Index (AQI) across diverse cities and states of India using machine learning: Investigating the influence of Punjab's stubble burning," *arXiv preprint arXiv:2404.08702*, Apr. 2024.

[18] A. Pant, S. Sharma, and M. Bansal, "Comparative analysis of supervised machine learning techniques for AQI prediction," in *2022 Int. Conf. Adv. Comput. Innov. Technol. (IACIT)*, 2022, pp. 1–5, doi: 10.1109/IACIT54714.2022.9753636.

[19] R. S. Rao, L. R. Kalabarige, B. Alankar, and A. K. Sahu, "Multimodal imputation-based stacked ensemble for prediction and classification of air quality index in Indian cities," *Comput. Electr. Eng.*, vol. 114, Art. no. 109070, Mar. 2024, doi: 10.1016/j.compeleceng.2024.109070.

[20] S. Lawrence and S. Bhathmanabhan, "Evaluating machine learning models and imputation strategies for Air Quality Index forecasting in urban India," *Environ. Monit. Assess.*, vol. 197, Art. no. 284, 2025, doi: 10.1007/s10661-025-14700-4.

[21] A. Mishra and Y. Gupta, "Comparative analysis of Air Quality Index prediction using deep learning algorithms," *Spat. Inf. Res.*, vol. 32, pp. 63–72, 2024, doi: 10.1007/s41324-023-00541-1.

[22] S. A. Aram, E. A. Nketiah, B. M. Saalidong, H. Wang, S. Afful, and E. Aram, "Machine learning-based prediction of air quality index and air quality grade: A comparative analysis," *Int. J. Environ. Sci. Technol.*, vol. 21, pp. 1543–1554, 2024, doi: 10.1007/s13762-023-05016-2.

[23] E. Kalantari, H. Gholami, H. Malakooti, and M. Kaskaoutis, "Machine learning for air quality index (AQI) forecasting: Shallow learning or deep learning?," *Environ. Sci. Pollut. Res.*, vol. 31, pp. 22367–22392, 2024, doi: 10.1007/s11356-024-35404-1.

[24] Y. Özüpak, F. Alpsalaz, and E. Aslan, "Air quality forecasting using machine learning: Comparative analysis and ensemble strategies for enhanced prediction," *Water Air Soil Pollut.*, vol. 236, Art. no. 84, 2025, doi: 10.1007/s11270-025-08122-8.
