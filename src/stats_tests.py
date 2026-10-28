"""Summary statistics and significance tests for the classification comparison.

* Protocol A: mean +/- SD over seeds, 95% bootstrap CI over seeds; paired Wilcoxon on per-seed
  macro-F1; exact McNemar on the seed-0 test predictions.
* Protocol B: pooled metrics, 95% CLUSTER bootstrap CI (resampling whole scenes, because the 150
  points of a scene are not independent); paired Wilcoxon on per-scene accuracy (n = 14);
  exact McNemar on pooled predictions (anti-conservative: points within a scene are correlated,
  so the Wilcoxon-over-scenes result is the one to quote).
* Holm correction over the 3 pairwise comparisons within each family.
"""
import itertools
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import f1_score

from common import CLASSES, LEARNERS, RESULTS, BASELINES

TAGS = {'All features': 'all_features', 'Optical + SAR + thermal/rain': 'optical_sar_thermal_rain'}
rng = np.random.default_rng(0)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def mcnemar_exact(y, pa, pb):
    a_ok, b_ok = pa == y, pb == y
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    return 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue, b, c


def safe_wilcoxon(x, y):
    d = np.asarray(x) - np.asarray(y)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(x, y, zero_method='wilcox').pvalue)


def boot_ci(vals, n=5000):
    vals = np.asarray(vals)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def main():
    summ_a, summ_b, tests = [], [], []
    for fs, tag in TAGS.items():
        ra = pd.read_csv(f'{RESULTS}/cls_A_{tag}_runs.csv')
        for m, d in ra.groupby('model'):
            lo, hi = boot_ci(d['macro_f1'])
            summ_a.append(dict(feature_set=fs, model=m, n_seeds=len(d),
                               accuracy=f"{d.accuracy.mean():.3f} ± {d.accuracy.std():.3f}",
                               macro_f1=f"{d.macro_f1.mean():.3f} ± {d.macro_f1.std():.3f}",
                               macro_f1_ci95=f"[{lo:.3f}, {hi:.3f}]",
                               weighted_f1=f"{d.weighted_f1.mean():.3f} ± {d.weighted_f1.std():.3f}",
                               kappa=f"{d.kappa.mean():.3f} ± {d.kappa.std():.3f}",
                               roc_auc=f"{d.roc_auc_ovr_macro.mean():.3f} ± {d.roc_auc_ovr_macro.std():.3f}",
                               fit_s=f"{d.fit_s.mean():.2f}"))
        pa = pickle.load(open(f'{RESULTS}/cls_A_{tag}_preds_seed0.pkl', 'rb'))
        fam = []
        for m1, m2 in itertools.combinations(LEARNERS, 2):
            x = ra[ra.model == m1].sort_values('seed').macro_f1.values
            y = ra[ra.model == m2].sort_values('seed').macro_f1.values
            p_w = safe_wilcoxon(x, y)
            p_m, b, c = mcnemar_exact(pa[m1]['y_true'], CLASSES[pa[m1]['proba'].argmax(1)], CLASSES[pa[m2]['proba'].argmax(1)])
            fam.append(dict(protocol='A', feature_set=fs, pair=f'{m1} vs {m2}', mean_diff_macro_f1=float((x - y).mean()),
                            wilcoxon_p=p_w, mcnemar_p=p_m, only_first_correct=b, only_second_correct=c))
        for key in ('wilcoxon_p', 'mcnemar_p'):
            adj = holm([f[key] for f in fam])
            for f, v in zip(fam, adj):
                f[key + '_holm'] = v
        tests += fam

        # ---- protocol B
        rb = pd.read_csv(f'{RESULTS}/cls_B_{tag}_pooled.csv')
        ps = pd.read_csv(f'{RESULTS}/cls_B_{tag}_per_scene.csv')
        pb = pickle.load(open(f'{RESULTS}/cls_B_{tag}_preds.pkl', 'rb'))
        y, g = pb['y_true'], pb['groups']
        scenes = sorted(set(g))
        idx_by_scene = {s: np.where(g == s)[0] for s in scenes}
        for m in LEARNERS + BASELINES:
            pred = CLASSES[pb['proba'][m].argmax(1)]
            accs, f1s = [], []
            for _ in range(1000):
                pick = rng.choice(scenes, len(scenes), replace=True)
                idx = np.concatenate([idx_by_scene[s] for s in pick])
                accs.append((pred[idx] == y[idx]).mean())
                f1s.append(f1_score(y[idx], pred[idx], labels=CLASSES, average='macro', zero_division=0))
            r = rb[rb.model == m].iloc[0]
            summ_b.append(dict(feature_set=fs, model=m, accuracy=f'{r.accuracy:.3f}',
                               accuracy_ci95=f'[{np.percentile(accs, 2.5):.3f}, {np.percentile(accs, 97.5):.3f}]',
                               macro_f1=f'{r.macro_f1:.3f}',
                               macro_f1_ci95=f'[{np.percentile(f1s, 2.5):.3f}, {np.percentile(f1s, 97.5):.3f}]',
                               kappa=f'{r.kappa:.3f}', roc_auc=f'{r.roc_auc_ovr_macro:.3f}'))
        fam = []
        for m1, m2 in itertools.combinations(LEARNERS, 2):
            x = ps[ps.model == m1].sort_values('scene').accuracy.values
            yy = ps[ps.model == m2].sort_values('scene').accuracy.values
            p_m, b, c = mcnemar_exact(y, CLASSES[pb['proba'][m1].argmax(1)], CLASSES[pb['proba'][m2].argmax(1)])
            fam.append(dict(protocol='B', feature_set=fs, pair=f'{m1} vs {m2}', mean_diff_scene_accuracy=float((x - yy).mean()),
                            wilcoxon_p=safe_wilcoxon(x, yy), mcnemar_p=p_m, only_first_correct=b, only_second_correct=c))
        for key in ('wilcoxon_p', 'mcnemar_p'):
            adj = holm([f[key] for f in fam])
            for f, v in zip(fam, adj):
                f[key + '_holm'] = v
        tests += fam
    pd.DataFrame(summ_a).to_csv(f'{RESULTS}/summary_A.csv', index=False)
    pd.DataFrame(summ_b).to_csv(f'{RESULTS}/summary_B.csv', index=False)
    pd.DataFrame(tests).to_csv(f'{RESULTS}/significance_tests.csv', index=False)
    print(pd.DataFrame(summ_a).to_string()); print(); print(pd.DataFrame(summ_b).to_string()); print(); print(pd.DataFrame(tests).round(4).to_string())


if __name__ == '__main__':
    main()
