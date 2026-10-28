# AI-Driven Crop-Type Classification, Moisture Stress Detection and Irrigation Advisory

**Multi-Sensor Satellite Fusion (ISRO Optical, Sentinel-1 SAR, NASA POWER Weather) with Random Forest and XGBoost Ensembles**  
Team 37 · BCSE306 Artificial Intelligence · Final Project Review  

---

## Executive Summary & System Overview

This repository implements an end-to-end AI system for precision agriculture that combines multi-spectral optical satellite imagery (ISRO AWiFS and LISS-4), synthetic aperture radar (Sentinel-1 C-band SAR), land surface temperature, and NASA POWER meteorological data to solve three coupled agronomic problems:
1. **Crop-Type Classification**: Identifying dominant crop categories across agricultural landscapes using a tuned 300-tree `RandomForestClassifier`.
2. **Crop Water Stress Index (CWSI) Regression**: Accurately estimating continuous moisture stress values using `RandomForestRegressor` and `XGBRegressor` ensembles.
3. **Moisture Stress Level Classification**: Categorizing field stress into actionable urgency tiers (`Healthy / Well-Irrigated`, `Moderate Stress`, `Severe Water Stress`) via an `XGBClassifier`.
4. **Actionable Irrigation Advisory Engine**: Integrating FAO-56 Penman-Monteith / Hargreaves crop evapotranspiration ($ET_c = K_c \times ET_0$) with a 3-tier decision alert matrix (`CRITICAL / RED`, `WARNING / YELLOW`, `OPTIMAL / GREEN`).

---

## Key Performance Results

All models are trained and benchmarked based on `Crop_type_detection_and_msm.ipynb` using 2,100 calibrated multi-sensor sample points across 23 predictor features with stratified 80/20 train/test evaluation (420 test points):

### 1. Crop-Type Classification (Random Forest Classifier, 300 Trees)
* **Overall Test Accuracy**: **89.76%** (~90%)
* **Macro Average F1-Score**: **0.91** (Macro Precision: 0.93, Macro Recall: 0.89)
* **Weighted Average F1-Score**: **0.90**
* **$R^2$ Score**: **0.8520** | **MSE**: **0.4952**

| Crop Class | Precision | Recall (Sensitivity) | F1-Score | Test Support |
|:---|:---:|:---:|:---:|:---:|
| **Oilseeds** | 1.00 | 1.00 | **1.00** | 30 |
| **Paddy** | 0.92 | 0.97 | **0.95** | 150 |
| **Pulses** | 1.00 | 0.97 | **0.98** | 30 |
| **Soybean** | 1.00 | 0.83 | **0.91** | 30 |
| **Sugarcane** | 0.84 | 0.68 | **0.75** | 60 |
| **nan / Unmapped** | 0.82 | 0.88 | **0.85** | 120 |
| **Overall / Macro** | **0.93** | **0.89** | **0.91** | **420** |

### 2. Moisture Stress Index (CWSI) Continuous Regression
Predicting continuous `Crop_Water_Stress_Index_CWSI`:

| Model | Train $R^2$ | Test $R^2$ | Test RMSE | Test MAE | Accuracy ($100 - \text{MAPE}$) | Within $\pm10\%$ Tolerance |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **RandomForestRegressor** (150 trees, depth 10) | 0.9999 | **0.9996** | **0.0021** | **0.0011** | — | — |
| **XGBRegressor** (200 trees, $\eta=0.05$) | 0.9999 | **0.9995** | **0.0022** | **0.0014** | **99.68%** | **100.00%** |

### 3. Moisture Stress Level Classification (XGBoost Classifier)
Categorizing stress into 3 operational management tiers:
* **Test Accuracy**: **99.05%** | **Macro-F1**: **0.98** | **Weighted-F1**: **0.99**

| Stress Category | Precision | Recall | F1-Score | Test Support | True Class Acc. (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Healthy / Well-Irrigated** | 1.00 | 0.96 | **0.98** | 71 | 95.8% |
| **Moderate Stress** | 0.99 | 1.00 | **0.99** | 336 | 100.0% |
| **Severe Water Stress** | 0.93 | 1.00 | **0.96** | 13 | 100.0% |

### 4. Irrigation Advisory Decision Support Matrix

$$\text{Alert Level} = \begin{cases} 
\textbf{CRITICAL / RED} \ (\text{Immediate Irrigation } < 24\text{h}) & \text{if Stress} = \text{'Severe Water Stress'} \lor \text{CWSI} \ge 0.55 \\ 
\textbf{WARNING / YELLOW} \ (\text{Scheduled Irrigation } 48\text{--}72\text{h}) & \text{if Stress} = \text{'Moderate Stress'} \lor \text{CWSI} \ge 0.35 \\ 
\textbf{OPTIMAL / GREEN} \ (\text{No Irrigation Required}) & \text{otherwise} 
\end{cases}$$

---

## Visualizations & Experimental Graphs

The primary figures generated from the experimental evaluation in `Crop_type_detection_and_msm.ipynb` are saved in [`figures/`](figures/):

| Figure File | Description |
|:---|:---|
| [`figures/fig_system_pipeline.png`](figures/fig_system_pipeline.png) | End-to-end system architecture (Inputs $\to$ Feature Engineering $\to$ Models $\to$ Advisory) |
| [`figures/fig_crop_confusion_matrix.png`](figures/fig_crop_confusion_matrix.png) | Normalized confusion matrix heatmap for Random Forest Crop Classification |
| [`figures/fig_crop_feature_importance.png`](figures/fig_crop_feature_importance.png) | Feature importance bar chart for Random Forest Crop Classification |
| [`figures/fig_ndvi_sar_scatter.png`](figures/fig_ndvi_sar_scatter.png) | Optical NDVI vs Sentinel-1 SAR VH Backscatter (dB) scatter plot |
| [`figures/fig_cwsi_feature_importance.png`](figures/fig_cwsi_feature_importance.png) | XGBoost feature importance for continuous moisture stress (CWSI) detection |
| [`figures/fig_stress_confusion_matrix.png`](figures/fig_stress_confusion_matrix.png) | XGBoost normalized confusion matrix (%) for 3-class stress detection |
| [`figures/fig_stress_feature_importance.png`](figures/fig_stress_feature_importance.png) | Top 10 feature importances (Split Gain) for XGBoost stress classification |

---

## Feature Engineering & Input Modalities

The system processes 23 fused predictors:
1. **Optical Reflectance**: `Reflectance_Green`, `Reflectance_Red`, `Reflectance_NIR`, `Reflectance_SWIR` (calibrated surface reflectance).
2. **Vegetation & Water Indices**: `Optical_NDVI`, `Optical_EVI2`, `Moisture_NDWI`, `NDVI_NDWI_Ratio` ($= \text{NDVI} / (\text{NDWI} + 10^{-5})$).
3. **Synthetic Aperture Radar (SAR)**: `SAR_VV_Backscatter_dB`, `SAR_VH_Backscatter_dB`, `SAR_VH_VV_Ratio` ($= \text{VH} - \text{VV}$ in dB).
4. **Thermal & Rainfall**: `Surface_Temperature_C`, `Cumulative_Rainfall_7Day_mm`.
5. **Meteorological Weather (NASA POWER)**: `T2M`, `T2M_MAX`, `T2M_MIN`, `PRECTOTCORR`, `RH2M`, `GWETTOP`, `GWETROOT`, `WS2M`, `ALLSKY_SFC_SW_DWN`, and derived reference evapotranspiration `ET0_mm_day`.

---

## Interactive Web Application (Localhost)

An interactive Streamlit decision advisory platform is provided in `app/streamlit_app.py`.

### How to Run Locally:
```bash
streamlit run app/streamlit_app.py
```
The app opens automatically at `http://localhost:8501`.

### Application Features:
* **Advisory Tab**: Select a satellite scene point or input custom sensor readings. Predicts the crop class, outputs class probabilities, and calculates FAO-56 crop water demand ($ET_0, K_c, ET_c$, Net Irrigation Requirement).
* **Leakage Demo Tab**: Compare in-scene vs. held-out scene performance to demonstrate generalization.
* **Model Comparison Tab**: Explore the benchmark results, metrics tables, and diagnostic figures.
* **Usability Survey Tab**: Standard 10-item System Usability Scale (SUS) questionnaire for evaluator feedback.

---

## Repository Structure

```
├── Crop_type_detection_and_msm.ipynb   # Main Jupyter notebook with all model training & figures
├── Team37_BCSE306_FinalReview_DRAFT.docx # Comprehensive Final Project Review Report
├── repo/
│   ├── app/
│   │   └── streamlit_app.py            # Streamlit web app
│   ├── data/
│   │   ├── README.md                   # Data documentation & sources
│   │   ├── processed/master_dataset.csv
│   │   └── raw/                        # ISRO table & NASA POWER CSVs
│   ├── figures/                        # Generated figures and charts
│   │   ├── fig_system_pipeline.png
│   │   ├── fig_crop_confusion_matrix.png
│   │   ├── fig_crop_feature_importance.png
│   │   ├── fig_ndvi_sar_scatter.png
│   │   ├── fig_cwsi_feature_importance.png
│   │   ├── fig_stress_confusion_matrix.png
│   │   └── fig_stress_feature_importance.png
│   ├── results/
│   │   └── RESULTS.md                  # Comprehensive results & metrics tables
│   ├── src/
│   │   ├── common.py                   # Data loaders, model configurations, evaluation metrics
│   │   ├── advisory.py                 # FAO-56 evapotranspiration & decision rules
│   │   ├── build_dataset.py            # Dataset merge and feature engineering
│   │   └── compare_classification.py   # Benchmark scripts
│   ├── tests/
│   │   ├── test_advisory.py            # Unit tests for advisory & FAO-56 calculations
│   │   └── test_app.py                 # Streamlit headless integration tests
│   └── requirements.txt
```

---

## Verification & Testing

Run the automated test suite:
```bash
python tests/test_advisory.py
python tests/test_app.py
```
All unit tests verify ET₀ formulas, Kc assignments, water demand rules, model output formats, and interface elements.
