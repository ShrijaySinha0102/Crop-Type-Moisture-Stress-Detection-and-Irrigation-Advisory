"""Regression on Crop_Water_Stress_Index_CWSI: Random Forest vs XGBoost vs AdaBoost.

Finding that motivates the two feature sets: in this dataset CWSI == 0.88 * (1 - NDVI) exactly
(linear fit R^2 = 1.0000), and the thermal column is uncorrelated with it (r = -0.007). So
  * "All features"  contains NDVI (or Red+NIR, EVI2 that reconstruct it) -> a formula-recovery test.
  * "NDVI-free"     removes NDVI, EVI2, Red, NIR and NDVI_NDWI_Ratio -> asks whether anything else
                    predicts the target.
Protocols: A = random 80/20 (5 seeds), B = leave-one-scene-out.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (FEATURE_SETS, LEARNERS, MAIN_SET, RBASELINES, RESULTS, cfg_str, fit_predict_reg,
                    load_data, reg_metrics, tune_reg)

TARGET = 'Crop_Water_Stress_Index_CWSI'
NDVI_DERIVED = ['Optical_NDVI', 'Optical_EVI2', 'Reflectance_Red', 'Reflectance_NIR', 'NDVI_NDWI_Ratio']
REG_SETS = {'All features': FEATURE_SETS[MAIN_SET],
            'NDVI-free': [f for f in FEATURE_SETS[MAIN_SET] if f not in NDVI_DERIVED]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--only', default=None, help='run just this feature set and merge into existing CSVs')
    a = ap.parse_args()
    df = load_data()
    t0 = time.time()
    log = lambda m: print(f'[{time.time() - t0:6.0f}s] {m}', flush=True)
    y, g = df[TARGET].values, df['ISRO_ProductID'].values
    rows_a, rows_b, scene_rows = [], [], []
    for set_name, feats in REG_SETS.items():
        if a.only and set_name != a.only:
            continue
        X = df[feats + (['Optical_NDVI'] if 'Optical_NDVI' not in feats else [])]
        names = LEARNERS + RBASELINES
        if set_name == 'NDVI-free':
            names = LEARNERS + ['Mean predictor']        # the NDVI baseline needs NDVI
        # ---- protocol A
        for seed in range(a.seeds):
            tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=seed)
            for name in names:
                cfg = tune_reg(name, X[feats].iloc[tr], y[tr], None, 'rand', seed) if name in LEARNERS else {}
                Xs = X if name == 'Linear regression on NDVI' else X[feats]
                p, tf, tp, _ = fit_predict_reg(name, cfg, Xs.iloc[tr], y[tr], Xs.iloc[te], seed)
                rows_a.append(dict(feature_set=set_name, seed=seed, model=name, cfg=cfg_str(cfg), fit_s=tf, **reg_metrics(y[te], p)))
            log(f'A {set_name} seed {seed}')
        # ---- protocol B
        pooled = {n: np.zeros(len(df)) for n in names}
        for s in sorted(set(g)):
            te, tr = np.where(g == s)[0], np.where(g != s)[0]
            for name in names:
                cfg = tune_reg(name, X[feats].iloc[tr], y[tr], g[tr], 'group', 0) if name in LEARNERS else {}
                Xs = X if name == 'Linear regression on NDVI' else X[feats]
                p, *_ = fit_predict_reg(name, cfg, Xs.iloc[tr], y[tr], Xs.iloc[te], 0)
                pooled[name][te] = p
            log(f'B {set_name} held-out scene {s}')
        for name in names:
            rows_b.append(dict(feature_set=set_name, model=name, **reg_metrics(y, pooled[name])))
            for s in sorted(set(g)):
                idx = np.where(g == s)[0]
                scene_rows.append(dict(feature_set=set_name, model=name, scene=s,
                                       MAE=float(np.abs(y[idx] - pooled[name][idx]).mean())))
        for fn, rows in (('reg_A_runs.csv', rows_a), ('reg_B_pooled.csv', rows_b), ('reg_B_per_scene.csv', scene_rows)):
            new_df, path = pd.DataFrame(rows), f'{RESULTS}/{fn}'
            if a.only and os.path.exists(path):
                old_df = pd.read_csv(path)
                new_df = pd.concat([old_df[old_df.feature_set != a.only], new_df], ignore_index=True)
            new_df.to_csv(path, index=False)
    log('done')


if __name__ == '__main__':
    main()
