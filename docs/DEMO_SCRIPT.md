# Demo Video Script & Presentation Guide (Target 4–5 Minutes)

**Pre-Recording Setup**:
1. Ensure the web app is running: `streamlit run app/streamlit_app.py` (accessible at `http://localhost:8501`).
2. Have `Crop_type_detection_and_msm.ipynb` open to showcase notebook execution, feature importances, and confusion matrix plots.
3. Open the project report `Team37_BCSE306_FinalReview_DRAFT.docx`.
4. Use standard 1080p recording software (OBS or Windows Game Bar `Win + G`).

---

## Chronological Presentation Walkthrough

| Timestamp | Screen View | Speaker & Talking Points |
|:---|:---|:---|
| **0:00–0:40** | Title Slide / README / Architecture Diagram | **Introduction & Problem Scope**: Team 37, BCSE306 Artificial Intelligence. Present the multi-sensor satellite fusion pipeline (`fig_system_pipeline.png`). Explain how combining ISRO optical bands, Sentinel-1 dual-pol SAR, and NASA POWER weather tackles crop classification and irrigation water stress estimation. |
| **0:40–1:40** | **Crop_type_detection_and_msm.ipynb** (Cells 1–15) | **Crop Classification & Feature Synergy**: Highlight the tuned 300-tree `RandomForestClassifier` achieving **89.76% accuracy** and **0.91 macro-F1**. Show the confusion matrix heatmap (`fig_crop_confusion_matrix.png`) and the feature importance bar chart (`fig_crop_feature_importance.png`) demonstrating that `Optical_NDVI` (14.4%), `Moisture_NDWI` (13.1%), and `SAR_VH_Backscatter_dB` (13.0%) drive predictive accuracy. Point out the NDVI vs. SAR VH scatter plot (`fig_ndvi_sar_scatter.png`). |
| **1:40–2:40** | **Crop_type_detection_and_msm.ipynb** (Cells 16–27) | **Moisture Stress Regression & Stress Classification**: Show CWSI regression where `RandomForestRegressor` and `XGBRegressor` achieve $R^2 = 0.9996$ and $0.9995$ with **100% within $\pm10\%$ tolerance**. Show the `XGBClassifier` achieving **99.05% accuracy** on 3 stress classes (`fig_stress_confusion_matrix.png`) and explain the split gain feature importance (`fig_stress_feature_importance.png`). |
| **2:40–3:45** | **Streamlit Web App** (`http://localhost:8501`) | **Live Interactive Advisory & Decision Logic**: Switch to the live browser on localhost. Test a sample point in the **Advisory Tab**. Demonstrate real-time crop prediction, class probability breakdown, and FAO-56 evapotranspiration calculation ($ET_0, K_c, ET_c$). Demonstrate the 3-tier automated decision rule: triggering `CRITICAL / RED` (Immediate <24h irrigation), `WARNING / YELLOW` (48–72h), or `OPTIMAL / GREEN`. Walk through the **Leakage Demo** and **Model Comparison** tabs. |
| **3:45–4:30** | Final Report / Conclusion | **Summary & Engineering Rigor**: Highlight key project takeaways: multi-sensor synergy overcomes single-sensor limitations, tree ensembles provide low-latency edge capability (~1.5–13 ms), and automated alert thresholds give actionable guidance to farmers. |
