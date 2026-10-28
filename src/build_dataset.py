"""Build the master dataset: ISRO per-point table + NASA POWER weather.

Fixes versus the Review 1/2 script (master_dataset_script.py):
  * Weather is joined to the NEAREST grid cell on each product's own grid.
    MERRA-2 variables use a 0.5 x 0.625 deg grid; ALLSKY_SFC_SW_DWN (CERES)
    uses a 1 x 1 deg grid. The old script snapped longitude to 0.5 deg and
    inner-merged all 9 files, which collapsed the weather to a single row.
  * No ffill()/bfill() over the whole table. The script aborts if any row
    has no weather match.
  * Dates are parsed with an explicit ISO format.

Output: data/processed/master_dataset.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, '..', 'data', 'raw')
OUT = os.path.join(HERE, '..', 'data', 'processed', 'master_dataset.csv')
MERRA = ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'GWETTOP', 'GWETROOT', 'RH2M', 'WS2M']
CERES = ['ALLSKY_SFC_SW_DWN']

# Same 11 -> 6 harmonisation as random_forest.py (Review 1)
CLASS_MAPPING = {
    'Paddy (Rice)': 'Paddy', 'Paddy (Aman Dhan)': 'Paddy', 'Paddy (Rice Bowl)': 'Paddy',
    'Paddy / Pulses': 'Paddy', 'Guava / Paddy / Pulses': 'Paddy',
    'Sugarcane / Paddy': 'Sugarcane', 'Sugarcane': 'Sugarcane',
    'Soybean': 'Soybean', 'Pulses (Arhar/Gram)': 'Pulses',
    'Oilseeds (Mustard/Sesame)': 'Oilseeds', 'Maize / Paddy': 'Maize'}

# FAO-56 style single-value crop coefficients (as in Review 1)
FAO_KC = {'Soybean': 1.05, 'Paddy (Rice)': 1.20, 'Paddy (Aman Dhan)': 1.20, 'Paddy (Rice Bowl)': 1.20,
          'Sugarcane': 1.25, 'Sugarcane / Paddy': 1.22, 'Pulses (Arhar/Gram)': 1.05,
          'Paddy / Pulses': 1.10, 'Oilseeds (Mustard/Sesame)': 1.00, 'Maize / Paddy': 1.15,
          'Guava / Paddy / Pulses': 0.95}


def load_power(param):
    path = os.path.join(RAW, f'NASA_POWER_{param}.csv')
    with open(path, encoding='utf-8') as fh:
        lines = fh.read().splitlines()
    k = [i for i, line in enumerate(lines) if '-END HEADER-' in line][0]
    df = pd.read_csv(path, skiprows=k + 1)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={df.columns[-1]: param})
    return df[df[param] != -999]


def nearest(values, grid):
    grid = np.array(sorted(grid))
    return grid[np.abs(grid[:, None] - np.asarray(values)[None, :]).argmin(0)]


def main():
    iso = pd.read_csv(os.path.join(RAW, 'isro_complete_master_all_scenes_dataset.csv'))
    iso.columns = iso.columns.str.strip()
    d = pd.to_datetime(iso['Acquisition_Date'], format='%Y-%m-%d')
    iso['YEAR'], iso['DOY'] = d.dt.year, d.dt.dayofyear

    # MERRA-2 (0.5 x 0.625 deg): merge the 8 variables that share one grid
    w = load_power(MERRA[0])
    for p in MERRA[1:]:
        w = w.merge(load_power(p), on=['LAT', 'LON', 'YEAR', 'DOY'], how='inner')
    iso['M_LAT'] = nearest(iso['Latitude'].values, w['LAT'].unique())
    iso['M_LON'] = nearest(iso['Longitude'].values, w['LON'].unique())
    iso = iso.merge(w, left_on=['M_LAT', 'M_LON', 'YEAR', 'DOY'],
                    right_on=['LAT', 'LON', 'YEAR', 'DOY'], how='left').drop(columns=['LAT', 'LON'])

    # CERES (1 x 1 deg): solar radiation lives on its own, coarser grid
    c = load_power(CERES[0]).rename(columns={'LAT': 'C_LAT', 'LON': 'C_LON'})
    iso['C_LAT'] = nearest(iso['Latitude'].values, c['C_LAT'].unique())
    iso['C_LON'] = nearest(iso['Longitude'].values, c['C_LON'].unique())
    iso = iso.merge(c, on=['C_LAT', 'C_LON', 'YEAR', 'DOY'], how='left')

    missing = int(iso[MERRA + CERES].isna().any(axis=1).sum())
    if missing:
        sys.exit(f'ABORT: {missing} rows have no weather match (no silent filling).')

    # Derived irrigation quantities. They depend on the crop label, so they are
    # NEVER used as classifier features (that was the 1.0000 accuracy leak).
    t_range = np.maximum(iso['T2M_MAX'] - iso['T2M_MIN'], 0.1)
    # Hargreaves-Samani form with surface solar radiation (Rs) in place of Ra -> approximation
    iso['ET0_mm_day'] = np.maximum(
        0.0023 * iso['ALLSKY_SFC_SW_DWN'] * 0.408 * (iso['T2M'] + 17.8) * np.sqrt(t_range), 0)
    iso['Crop_Kc'] = iso['Dominant_Crop'].map(FAO_KC).fillna(1.0)
    iso['ETc_mm_day'] = iso['ET0_mm_day'] * iso['Crop_Kc']
    iso['Net_Irrigation_Requirement_mm'] = np.maximum(iso['ETc_mm_day'] - iso['PRECTOTCORR'], 0)

    # Engineered remote-sensing ratios + harmonised label
    iso['SAR_VH_VV_Ratio'] = iso['SAR_VH_Backscatter_dB'] - iso['SAR_VV_Backscatter_dB']
    iso['NDVI_NDWI_Ratio'] = iso['Optical_NDVI'] / (iso['Moisture_NDWI'] + 1e-5)
    iso['Crop'] = iso['Dominant_Crop'].map(CLASS_MAPPING)
    assert iso['Crop'].notna().all(), 'unmapped crop label'

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    iso.to_csv(OUT, index=False)
    print(f'Wrote {os.path.normpath(OUT)}  shape={iso.shape}')
    cells = iso.groupby('ISRO_ProductID')[['M_LAT', 'M_LON']].apply(lambda g: len(g.drop_duplicates()))
    print('Scenes:', iso['ISRO_ProductID'].nunique(),
          '| weather cells per scene (min/max):', int(cells.min()), int(cells.max()))
    print(iso.groupby('Crop')['ISRO_ProductID'].nunique().rename('scenes_per_class').to_string())


if __name__ == '__main__':
    main()
