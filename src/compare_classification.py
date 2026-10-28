"""Crop classification: Random Forest vs XGBoost vs AdaBoost (+ 2 baselines).

Protocol A  - random stratified 80/20 split, repeated over seeds (what Reviews 1-2 did).
Protocol B  - leave-one-scene-out (every scene is a held-out "new field site").
Hyper-parameters are tuned INSIDE each training set (inner 3-fold CV, macro-F1), with the
same budget (4 configurations) for every learner. Balanced sample weights for every learner.

Usage:  python src/compare_classification.py --features "All features" --seeds 10
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (BASELINES, CLASSES, FEATURE_SETS, LEARNERS, RESULTS, cfg_str, clf_metrics,
                    fit_predict_clf, load_data, tune_clf)


def slug(s):
    return s.lower().replace(' + ', '_').replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')


def run_protocol_a(df, feats, seeds, log):
    X, y = df[feats], df['Crop'].values
    rows, per_class, preds = [], [], {}
    for seed in range(seeds):
        tr, te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=seed, stratify=y)
        for name in LEARNERS + BASELINES:
            if name in LEARNERS:
                cfg, cv = tune_clf(name, X.iloc[tr], y[tr], None, 'strat', seed)
            else:
                cfg, cv = {}, np.nan
            proba, tf, tp, _ = fit_predict_clf(name, cfg, X.iloc[tr], y[tr], X.iloc[te], seed)
            m, pc, _ = clf_metrics(y[te], proba)
            rows.append(dict(seed=seed, model=name, cfg=cfg_str(cfg), inner_cv_f1=cv, fit_s=tf, predict_s=tp, **m))
            pc['seed'], pc['model'] = seed, name
            per_class.append(pc)
            if seed == 0:
                preds[name] = dict(idx=te, y_true=y[te], proba=proba)
            log(f'A seed={seed} {name:26s} acc={m["accuracy"]:.3f} macroF1={m["macro_f1"]:.3f}')
    return pd.DataFrame(rows), pd.concat(per_class), preds


def run_protocol_b(df, feats, log):
    X, y, g = df[feats], df['Crop'].values, df['ISRO_ProductID'].values
    scenes = sorted(set(g))
    pooled = {n: np.zeros((len(df), len(CLASSES))) for n in LEARNERS + BASELINES}
    cfgs, fit_t = [], {n: [] for n in LEARNERS + BASELINES}
    for k, s in enumerate(scenes):
        te = np.where(g == s)[0]
        tr = np.where(g != s)[0]
        for name in LEARNERS + BASELINES:
            if name in LEARNERS:
                cfg, _ = tune_clf(name, X.iloc[tr], y[tr], g[tr], 'group', 0)
            else:
                cfg = {}
            proba, tf, _, _ = fit_predict_clf(name, cfg, X.iloc[tr], y[tr], X.iloc[te], 0)
            pooled[name][te] = proba
            fit_t[name].append(tf)
            cfgs.append(dict(scene=s, model=name, cfg=cfg_str(cfg)))
        log(f'B held-out scene {k + 1}/{len(scenes)} {s} ({y[te][0]})')
    rows, per_class, per_scene = [], [], []
    for name in LEARNERS + BASELINES:
        m, pc, cm = clf_metrics(y, pooled[name])
        rows.append(dict(model=name, fit_s=np.mean(fit_t[name]), **m))
        pc['model'] = name
        per_class.append(pc)
        pred = CLASSES[pooled[name].argmax(1)]
        for s in scenes:
            idx = np.where(g == s)[0]
            per_scene.append(dict(scene=s, crop=y[idx][0], model=name, accuracy=float((pred[idx] == y[idx]).mean())))
    return (pd.DataFrame(rows), pd.concat(per_class), pd.DataFrame(per_scene), pd.DataFrame(cfgs),
            dict(y_true=y, groups=g, proba=pooled))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='All features', choices=list(FEATURE_SETS))
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--skip-b', action='store_true')
    ap.add_argument('--skip-a', action='store_true')
    a = ap.parse_args()
    feats, tag = FEATURE_SETS[a.features], slug(a.features)
    df = load_data()
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f'[{time.time() - t0:6.0f}s] {msg}', flush=True)

    log(f'Feature set "{a.features}" ({len(feats)} columns)')
    if not a.skip_a:
        ra, pca, pa = run_protocol_a(df, feats, a.seeds, log)
        ra.to_csv(f'{RESULTS}/cls_A_{tag}_runs.csv', index=False)
        pca.to_csv(f'{RESULTS}/cls_A_{tag}_per_class.csv', index=False)
        pickle.dump(pa, open(f'{RESULTS}/cls_A_{tag}_preds_seed0.pkl', 'wb'))
    if not a.skip_b:
        rb, pcb, psb, cfb, pb = run_protocol_b(df, feats, log)
        rb.to_csv(f'{RESULTS}/cls_B_{tag}_pooled.csv', index=False)
        pcb.to_csv(f'{RESULTS}/cls_B_{tag}_per_class.csv', index=False)
        psb.to_csv(f'{RESULTS}/cls_B_{tag}_per_scene.csv', index=False)
        cfb.to_csv(f'{RESULTS}/cls_B_{tag}_chosen_configs.csv', index=False)
        pickle.dump(pb, open(f'{RESULTS}/cls_B_{tag}_preds.pkl', 'wb'))
    log('done')


if __name__ == '__main__':
    main()
