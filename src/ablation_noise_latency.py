"""Modality ablation, noise robustness and latency for the three learners.

All three learners use their fixed REFERENCE configuration here (no tuning), so the
comparison isolates the effect of the feature group / noise level, not of the search.

Protocols:  A = random stratified 80/20 (5 seeds)   B = leave-one-scene-out.
Noise: Gaussian noise added to the TEST features only, sigma = k * (training SD of that feature),
k in {0, 0.1, 0.25, 0.5, 1.0}, 3 noise draws per level.
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (CLASSES, FEATURE_SETS, HONEST_SET, LEARNERS, MAIN_SET, REFERENCE, RESULTS,
                    clf_metrics, fit_predict_clf, load_data)

NOISE = [0.0, 0.1, 0.25, 0.5, 1.0]
DRAWS = 3


def noisy(X, sd, k, rng):
    if k == 0:
        return X
    return X + rng.normal(0.0, 1.0, X.shape) * (k * sd.values)


def eval_sets(df, set_name, feats, noise_levels, seeds, log):
    y, g = df['Crop'].values, df['ISRO_ProductID'].values
    X = df[feats]
    rows = []
    # ---- protocol A
    for name in LEARNERS:
        for seed in range(seeds):
            tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=seed, stratify=y)
            sd = X.iloc[tr].std()
            rng = np.random.default_rng(seed)
            _, _, _, model = fit_predict_clf(name, REFERENCE[name], X.iloc[tr], y[tr], X.iloc[te].iloc[:2], seed)
            for k in noise_levels:
                for d in range(DRAWS if k else 1):
                    # re-use the fitted model: predict on noisy test data
                    Xn = noisy(X.iloc[te], sd, k, rng)
                    proba = _predict_full(model, name, Xn, y[tr])
                    m, _, _ = clf_metrics(y[te], proba)
                    rows.append(dict(protocol='A', feature_set=set_name, model=name, noise_k=k, rep=seed * 10 + d,
                                     accuracy=m['accuracy'], macro_f1=m['macro_f1']))
        log(f'A  {set_name:32s} {name}')
    # ---- protocol B (leave-one-scene-out), pooled over scenes
    for name in LEARNERS:
        pooled = {(k, d): np.zeros((len(df), len(CLASSES))) for k in noise_levels for d in range(DRAWS if k else 1)}
        rng = np.random.default_rng(0)
        for s in sorted(set(g)):
            te, tr = np.where(g == s)[0], np.where(g != s)[0]
            sd = X.iloc[tr].std()
            _, _, _, model = fit_predict_clf(name, REFERENCE[name], X.iloc[tr], y[tr], X.iloc[te].iloc[:2], 0)
            for (k, d) in pooled:
                pooled[(k, d)][te] = _predict_full(model, name, noisy(X.iloc[te], sd, k, rng), y[tr])
        for (k, d), proba in pooled.items():
            m, _, _ = clf_metrics(y, proba)
            rows.append(dict(protocol='B', feature_set=set_name, model=name, noise_k=k, rep=d,
                             accuracy=m['accuracy'], macro_f1=m['macro_f1']))
        log(f'B  {set_name:32s} {name}')
    return rows


def _predict_full(model, name, X, ytr):
    present = np.array(sorted(set(ytr)))
    p = model.predict_proba(X)
    full = np.zeros((len(X), len(CLASSES)))
    for i, c in enumerate(present):
        full[:, list(CLASSES).index(c)] = p[:, i]
    return full


def latency(df, reps=5):
    feats = FEATURE_SETS[MAIN_SET]
    y = df['Crop'].values
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=0, stratify=y)
    rows = []
    for name in LEARNERS:
        fits, batch = [], []
        for r in range(reps):
            _, tf, tp, model = fit_predict_clf(name, REFERENCE[name], df[feats].iloc[tr], y[tr], df[feats].iloc[te], r)
            fits.append(tf); batch.append(tp)
        one = df[feats].iloc[te[:1]]
        t = []
        for _ in range(200):
            t0 = time.perf_counter(); model.predict_proba(one); t.append((time.perf_counter() - t0) * 1000)
        rows.append(dict(model=name, fit_s_mean=np.mean(fits), fit_s_sd=np.std(fits, ddof=1),
                         batch420_ms_mean=np.mean(batch) * 1000, batch420_ms_sd=np.std(batch, ddof=1) * 1000,
                         single_row_ms_mean=np.mean(t), single_row_ms_p95=np.percentile(t, 95),
                         model_size_kb=len(pickle.dumps(model)) / 1024))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    a = ap.parse_args()
    df = load_data()
    t0 = time.time()
    log = lambda m: print(f'[{time.time() - t0:6.0f}s] {m}', flush=True)
    raw = f'{RESULTS}/cls_ablation_noise_raw.csv'
    rows = pd.read_csv(raw).to_dict('records') if os.path.exists(raw) else []
    done = {r['feature_set'] for r in rows}
    for set_name, feats in FEATURE_SETS.items():
        if set_name in done:
            continue
        levels = NOISE if set_name in (MAIN_SET, HONEST_SET) else [0.0]
        rows += eval_sets(df, set_name, feats, levels, a.seeds, log)
        pd.DataFrame(rows).to_csv(f'{RESULTS}/cls_ablation_noise_raw.csv', index=False)   # checkpoint
    latency(df).to_csv(f'{RESULTS}/cls_latency.csv', index=False)
    log('done')


if __name__ == '__main__':
    main()
