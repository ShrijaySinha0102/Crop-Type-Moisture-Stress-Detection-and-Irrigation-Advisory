"""Shared utilities for the AdaBoost vs XGBoost vs Random Forest comparison."""
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'processed', 'master_dataset.csv')
RESULTS = os.path.join(HERE, '..', 'results')
FIGS = os.path.join(HERE, '..', 'figures')

CLASSES = np.array(['Maize', 'Oilseeds', 'Paddy', 'Pulses', 'Soybean', 'Sugarcane'])
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------- feature groups
# Reflectance_SWIR is excluded: LISS-4 (7 of 14 scenes) has no SWIR band, so the column is 50% empty.
# Median-imputing it (Reviews 1-2) turns it into a sensor flag (AWiFS vs LISS-4).
OPTICAL = ['Reflectance_Green', 'Reflectance_Red', 'Reflectance_NIR',
           'Optical_NDVI', 'Optical_EVI2', 'Moisture_NDWI', 'NDVI_NDWI_Ratio']
SAR = ['SAR_VV_Backscatter_dB', 'SAR_VH_Backscatter_dB', 'SAR_VH_VV_Ratio']
THERMAL_RAIN = ['Surface_Temperature_C', 'Cumulative_Rainfall_7Day_mm']
WEATHER = ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M', 'GWETTOP', 'GWETROOT',
           'WS2M', 'ALLSKY_SFC_SW_DWN', 'ET0_mm_day']

FEATURE_SETS = {
    'Optical only': OPTICAL,
    'SAR only': SAR,
    'Optical + SAR': OPTICAL + SAR,
    'Optical + SAR + thermal/rain': OPTICAL + SAR + THERMAL_RAIN,
    'Weather only (NASA POWER)': WEATHER,
    'All features': OPTICAL + SAR + THERMAL_RAIN + WEATHER,
}
MAIN_SET = 'All features'          # the Review 1 / Review 2 feature whitelist (23 columns)
HONEST_SET = 'Optical + SAR + thermal/rain'   # no scene-constant weather


def load_data():
    df = pd.read_csv(DATA)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- model grids
# Same tuning budget for every learner: 4 configurations each, 3-fold inner CV, macro-F1.
GRIDS = {
    'Random Forest': [dict(n_estimators=200, max_depth=8, min_samples_leaf=2),
                      dict(n_estimators=200, max_depth=16, min_samples_leaf=2),
                      dict(n_estimators=200, max_depth=None, min_samples_leaf=1),
                      dict(n_estimators=200, max_depth=12, min_samples_leaf=5)],
    'XGBoost': [dict(n_estimators=150, max_depth=3, learning_rate=0.1),
                dict(n_estimators=150, max_depth=5, learning_rate=0.1),
                dict(n_estimators=150, max_depth=3, learning_rate=0.3),
                dict(n_estimators=150, max_depth=5, learning_rate=0.3)],
    'AdaBoost': [dict(n_estimators=100, max_depth=2, learning_rate=0.5),
                 dict(n_estimators=100, max_depth=3, learning_rate=1.0),
                 dict(n_estimators=200, max_depth=2, learning_rate=1.0),
                 dict(n_estimators=200, max_depth=3, learning_rate=0.5)],
}
# Fixed "reference" configuration used for ablation / noise / latency (no tuning)
REFERENCE = {'Random Forest': GRIDS['Random Forest'][1],
             'XGBoost': GRIDS['XGBoost'][0],
             'AdaBoost': GRIDS['AdaBoost'][0]}
LEARNERS = ['Random Forest', 'XGBoost', 'AdaBoost']
BASELINES = ['Majority class', 'NDVI rule (depth-2 tree)']


def build_clf(name, cfg, seed):
    if name == 'Random Forest':
        return RandomForestClassifier(max_features='sqrt', random_state=seed, n_jobs=1, **cfg)
    if name == 'XGBoost':
        return XGBClassifier(subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method='hist',
                             objective='multi:softprob', random_state=seed, n_jobs=1, verbosity=0, **cfg)
    if name == 'AdaBoost':
        c = dict(cfg)
        depth = c.pop('max_depth')
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth, random_state=seed),
                                  random_state=seed, **c)
    if name == 'Majority class':
        return DummyClassifier(strategy='prior')
    if name == 'NDVI rule (depth-2 tree)':
        return DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=seed)
    raise ValueError(name)


def fit_predict_clf(name, cfg, Xtr, ytr, Xte, seed=0):
    """Fit on (Xtr, ytr) and return (proba[n_test, 6], fit_seconds, predict_seconds, fitted_model).
    Handles classes missing from the training fold and balanced sample weights for every learner."""
    ytr = np.asarray(ytr)
    present = np.array(sorted(set(ytr)))
    enc = {c: i for i, c in enumerate(present)}
    ytr_e = np.array([enc[c] for c in ytr])
    if name == 'NDVI rule (depth-2 tree)':
        Xtr, Xte = Xtr[['Optical_NDVI']], Xte[['Optical_NDVI']]
    model = build_clf(name, cfg, seed)
    t0 = time.perf_counter()
    if name == 'Majority class':
        model.fit(Xtr, ytr_e)
    elif name == 'NDVI rule (depth-2 tree)':
        model.fit(Xtr, ytr_e)
    else:
        model.fit(Xtr, ytr_e, sample_weight=compute_sample_weight('balanced', ytr_e))
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = model.predict_proba(Xte)
    t_pred = time.perf_counter() - t0
    full = np.zeros((len(Xte), len(CLASSES)))
    for i, c in enumerate(present):
        full[:, CLASS_IDX[c]] = p[:, i]
    return full, t_fit, t_pred, model


def macro_f1_present(y_true, y_pred):
    labels = sorted(set(y_true))
    return f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)


def tune_clf(name, X, y, groups, inner, seed):
    """Pick the best configuration of `name` by inner-CV macro-F1 on the TRAINING data only."""
    y = np.asarray(y)
    if inner == 'group':
        splits = list(GroupKFold(n_splits=3).split(X, y, groups))
    else:
        splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(X, y))
    scores = []
    for cfg in GRIDS[name]:
        s = []
        for tr, va in splits:
            proba, *_ = fit_predict_clf(name, cfg, X.iloc[tr], y[tr], X.iloc[va], seed)
            s.append(macro_f1_present(y[va], CLASSES[proba.argmax(1)]))
        scores.append(np.mean(s))
    return GRIDS[name][int(np.argmax(scores))], float(np.max(scores))


def clf_metrics(y_true, proba):
    """Overall + per-class metrics from pooled predictions (proba has 6 columns in CLASSES order)."""
    y_true = np.asarray(y_true)
    pred = CLASSES[proba.argmax(1)]
    cm = confusion_matrix(y_true, pred, labels=CLASSES)
    out = dict(accuracy=accuracy_score(y_true, pred),
               macro_f1=f1_score(y_true, pred, labels=CLASSES, average='macro', zero_division=0),
               weighted_f1=f1_score(y_true, pred, labels=CLASSES, average='weighted', zero_division=0),
               kappa=cohen_kappa_score(y_true, pred))
    try:
        present = [i for i, c in enumerate(CLASSES) if (y_true == c).any()]
        yb = np.stack([(y_true == CLASSES[i]).astype(int) for i in present], 1)
        out['roc_auc_ovr_macro'] = float(np.mean([roc_auc_score(yb[:, j], proba[:, i])
                                                  for j, i in enumerate(present)]))
    except ValueError:
        out['roc_auc_ovr_macro'] = np.nan
    rows = []
    n = cm.sum()
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp; tn = n - tp - fp - fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else np.nan          # sensitivity
        spec = tn / (tn + fp) if tn + fp else np.nan
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and not np.isnan(rec) else 0.0
        rows.append(dict(cls=c, precision=prec, recall=rec, specificity=spec, f1=f1, support=int(cm[i].sum())))
    return out, pd.DataFrame(rows), cm


# ---------------------------------------------------------------- regression
RGRIDS = {
    'Random Forest': [dict(n_estimators=200, max_depth=8, min_samples_leaf=2),
                      dict(n_estimators=200, max_depth=16, min_samples_leaf=2),
                      dict(n_estimators=200, max_depth=None, min_samples_leaf=1),
                      dict(n_estimators=200, max_depth=12, min_samples_leaf=5)],
    'XGBoost': GRIDS['XGBoost'],
    'AdaBoost': [dict(n_estimators=100, max_depth=3, learning_rate=0.5),
                 dict(n_estimators=100, max_depth=5, learning_rate=1.0),
                 dict(n_estimators=200, max_depth=3, learning_rate=1.0),
                 dict(n_estimators=200, max_depth=5, learning_rate=0.5)],
}
RBASELINES = ['Mean predictor', 'Linear regression on NDVI']


def build_reg(name, cfg, seed):
    if name == 'Random Forest':
        return RandomForestRegressor(max_features=0.5, random_state=seed, n_jobs=1, **cfg)
    if name == 'XGBoost':
        return XGBRegressor(subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method='hist',
                            random_state=seed, n_jobs=1, verbosity=0, **cfg)
    if name == 'AdaBoost':
        c = dict(cfg)
        depth = c.pop('max_depth')
        return AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=depth, random_state=seed),
                                 random_state=seed, **c)
    if name == 'Mean predictor':
        return DummyRegressor(strategy='mean')
    if name == 'Linear regression on NDVI':
        return LinearRegression()
    raise ValueError(name)


def fit_predict_reg(name, cfg, Xtr, ytr, Xte, seed=0):
    if name == 'Linear regression on NDVI':
        Xtr, Xte = Xtr[['Optical_NDVI']], Xte[['Optical_NDVI']]
    m = build_reg(name, cfg, seed)
    t0 = time.perf_counter(); m.fit(Xtr, ytr); t_fit = time.perf_counter() - t0
    t0 = time.perf_counter(); p = m.predict(Xte); t_pred = time.perf_counter() - t0
    return p, t_fit, t_pred, m


def tune_reg(name, X, y, groups, inner, seed):
    y = np.asarray(y)
    splits = (list(GroupKFold(n_splits=3).split(X, y, groups)) if inner == 'group'
              else list(KFold(n_splits=3, shuffle=True, random_state=seed).split(X)))
    scores = []
    for cfg in RGRIDS[name]:
        s = []
        for tr, va in splits:
            p, *_ = fit_predict_reg(name, cfg, X.iloc[tr], y[tr], X.iloc[va], seed)
            s.append(-mean_squared_error(y[va], p))
        scores.append(np.mean(s))
    return RGRIDS[name][int(np.argmax(scores))]


def reg_metrics(y_true, pred):
    return dict(MAE=mean_absolute_error(y_true, pred), RMSE=float(np.sqrt(mean_squared_error(y_true, pred))),
                R2=r2_score(y_true, pred))


def cfg_str(cfg):
    return ';'.join(f'{k}={v}' for k, v in cfg.items())
