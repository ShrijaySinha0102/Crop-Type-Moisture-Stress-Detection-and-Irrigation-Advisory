# Dataset Documentation

This directory contains the multi-sensor dataset files used for crop-type classification, moisture stress estimation, and irrigation advisory in `Crop_type_detection_and_msm.ipynb`.

| File Path | Description | Sensor / Source | Format & Dimensions |
|:---|:---|:---|:---|
| `processed/master_dataset.csv` | Master calibrated dataset containing aligned optical reflectance, radar backscatter, surface temperature, and weather variables. | Multi-sensor fusion (ISRO, Sentinel-1, NASA POWER) | 2,100 sample points × 51+ columns |
| `raw/isro_complete_master_all_scenes_dataset.csv` | Raw per-point extraction across 14 satellite scenes. | ISRO AWiFS (56 m) & LISS-4 (5 m) | 2,100 points (150 per scene) |
| `raw/NASA_POWER_*.csv` | Ground-incident solar radiation, air temperatures (mean/max/min), relative humidity, precipitation, wind speed, and soil wetness profiles. | NASA POWER MERRA-2 & CERES satellite reanalysis | Daily gridded observations (2025-09-23 to 2025-10-26) |

## Calibrated Feature Groups (23 Predictors)

1. **Optical Reflectance Bands**: `Reflectance_Green`, `Reflectance_Red`, `Reflectance_NIR`, `Reflectance_SWIR`
2. **Vegetation & Moisture Indices**: `Optical_NDVI`, `Optical_EVI2`, `Moisture_NDWI`, `NDVI_NDWI_Ratio` ($= \text{NDVI} / (\text{NDWI} + 10^{-5})$)
3. **Synthetic Aperture Radar (SAR)**: `SAR_VV_Backscatter_dB`, `SAR_VH_Backscatter_dB`, `SAR_VH_VV_Ratio` ($= \text{VH} - \text{VV}$)
4. **Thermal & Rainfall**: `Surface_Temperature_C`, `Cumulative_Rainfall_7Day_mm`
5. **Meteorological Predictors (NASA POWER)**: `T2M`, `T2M_MAX`, `T2M_MIN`, `PRECTOTCORR`, `RH2M`, `GWETTOP`, `GWETROOT`, `WS2M`, `ALLSKY_SFC_SW_DWN`, `ET0_mm_day`

## Target Variables

* **Crop Category**: `Standard_Crop` (Paddy, Sugarcane, Soybean, Pulses, Oilseeds, Maize, nan/unmapped)
* **Moisture Stress Index**: `Crop_Water_Stress_Index_CWSI` $\in [0, 1]$
* **Stress Category**: `Stress_Classification` (`Healthy / Well-Irrigated`, `Moderate Stress`, `Severe Water Stress`)
