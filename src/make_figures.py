"""Generate report figures from results/*.csv (skips any figure whose inputs are missing)."""
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from common import CLASSES, FIGS, LEARNERS, RESULTS, BASELINES

os.makedirs(FIGS, exist_ok=True)
TAGS = {'All features': 'all_features', 'Optical + SAR + thermal/rain': 'optical_sar_thermal_rain'}
COL = {'Random Forest': '#2a7f62', 'XGBoost': '#c8553d', 'AdaBoost': '#3d5a80',
       'Majority class': '#aaaaaa', 'NDVI rule (depth-2 tree)': '#cccccc'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})


def exists(*p):
    return all(os.path.exists(f'{RESULTS}/{x}') for x in p)


def fig_main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    models = LEARNERS + BASELINES
    for ax, (fs, tag) in zip(axes, TAGS.items()):
        if not exists(f'cls_A_{tag}_runs.csv', f'cls_B_{tag}_pooled.csv'):
            continue
        ra = pd.read_csv(f'{RESULTS}/cls_A_{tag}_runs.csv')
        rb = pd.read_csv(f'{RESULTS}/cls_B_{tag}_pooled.csv')
        x = np.arange(len(models)); w = 0.38
        a = [ra[ra.model == m].macro_f1.mean() for m in models]
        sd = [ra[ra.model == m].macro_f1.std() for m in models]
        b = [rb[rb.model == m].macro_f1.iloc[0] for m in models]
        ax.bar(x - w / 2, a, w, yerr=sd, capsize=3, color=[COL[m] for m in models], label='A: random 80/20 split (mean ± SD)')
        ax.bar(x + w / 2, b, w, color=[COL[m] for m in models], hatch='//', edgecolor='white',
               label='B: leave-one-scene-out')
        ax.set_xticks(x); ax.set_xticklabels([m.replace(' (depth-2 tree)', '').replace(' ', '\n', 1) for m in models], fontsize=8)
        ax.set_title(fs); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel('Macro-F1 (6 classes)')
    axes[0].legend(fontsize=7, loc='upper right', frameon=False)
    fig.tight_layout(); fig.savefig(f'{FIGS}/fig_main_comparison.png'); plt.close(fig)


def fig_confusion():
    for fs, tag in TAGS.items():
        if not exists(f'cls_B_{tag}_preds.pkl'):
            continue
        pb = pickle.load(open(f'{RESULTS}/cls_B_{tag}_preds.pkl', 'rb'))
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for ax, m in zip(axes, LEARNERS):
            pred = CLASSES[pb['proba'][m].argmax(1)]
            cm = confusion_matrix(pb['y_true'], pred, labels=CLASSES).astype(float)
            cmn = cm / cm.sum(1, keepdims=True)
            ax.imshow(cmn, cmap='Greens', vmin=0, vmax=1)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f'{cmn[i, j]:.2f}', ha='center', va='center', fontsize=7,
                            color='white' if cmn[i, j] > .6 else 'black')
            ax.set_xticks(range(6)); ax.set_yticks(range(6))
            ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=7); ax.set_yticklabels(CLASSES, fontsize=7)
            ax.set_title(f'{m}'); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        fig.suptitle(f'Leave-one-scene-out, row-normalised confusion matrices — {fs}', fontsize=10)
        fig.tight_layout(); fig.savefig(f'{FIGS}/fig_confusion_B_{tag}.png'); plt.close(fig)


def fig_per_scene():
    tag = TAGS['All features']
    if not exists(f'cls_B_{tag}_per_scene.csv'):
        return
    ps = pd.read_csv(f'{RESULTS}/cls_B_{tag}_per_scene.csv')
    ps['label'] = ps.scene.astype(str) + '\n' + ps.crop
    pv = ps[ps.model.isin(LEARNERS)].pivot(index='model', columns='label', values='accuracy').loc[LEARNERS]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.imshow(pv.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f'{pv.values[i, j]:.2f}', ha='center', va='center', fontsize=7)
    ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, fontsize=6)
    ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index)
    ax.set_title('Accuracy on each held-out scene (leave-one-scene-out, all features)')
    fig.tight_layout(); fig.savefig(f'{FIGS}/fig_per_scene.png'); plt.close(fig)


def fig_ablation():
    if not exists('cls_ablation_noise_raw.csv'):
        return
    r = pd.read_csv(f'{RESULTS}/cls_ablation_noise_raw.csv')
    r = r[r.noise_k == 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, prot, title in zip(axes, ['A', 'B'], ['A: random 80/20 split', 'B: leave-one-scene-out']):
        d = r[r.protocol == prot].groupby(['feature_set', 'model']).macro_f1.mean().unstack()[LEARNERS]
        order = list(d.index)
        x = np.arange(len(order)); w = 0.27
        for k, m in enumerate(LEARNERS):
            ax.bar(x + (k - 1) * w, d[m], w, color=COL[m], label=m)
        ax.set_xticks(x); ax.set_xticklabels([o.replace(' + ', '\n+ ').replace(' only', '\nonly') for o in order], fontsize=7)
        ax.set_title(title); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel('Macro-F1'); axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f'{FIGS}/fig_ablation.png'); plt.close(fig)


def fig_noise():
    if not exists('cls_ablation_noise_raw.csv'):
        return
    r = pd.read_csv(f'{RESULTS}/cls_ablation_noise_raw.csv')
    sets = ['All features', 'Optical + SAR + thermal/rain']
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey=True)
    for i, prot in enumerate(['A', 'B']):
        for j, fs in enumerate(sets):
            ax = axes[i, j]
            for m in LEARNERS:
                d = r[(r.protocol == prot) & (r.feature_set == fs) & (r.model == m)].groupby('noise_k').macro_f1.agg(['mean', 'std'])
                ax.errorbar(d.index, d['mean'], yerr=d['std'].fillna(0), marker='o', ms=3, capsize=2, color=COL[m], label=m)
            ax.set_title(f'Protocol {prot} — {fs}', fontsize=9); ax.set_ylim(0, 1.05)
    for ax in axes[1]:
        ax.set_xlabel('Noise level k  (σ = k × feature SD)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Macro-F1')
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f'{FIGS}/fig_noise.png'); plt.close(fig)


def fig_regression():
    if not exists('reg_A_runs.csv', 'reg_B_pooled.csv'):
        return
    ra = pd.read_csv(f'{RESULTS}/reg_A_runs.csv'); rb = pd.read_csv(f'{RESULTS}/reg_B_pooled.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, fs in zip(axes, ['All features', 'NDVI-free']):
        names = [m for m in LEARNERS + ['Mean predictor', 'Linear regression on NDVI'] if m in set(ra[ra.feature_set == fs].model)]
        x = np.arange(len(names)); w = 0.38
        a = [ra[(ra.feature_set == fs) & (ra.model == m)].R2.mean() for m in names]
        s = [ra[(ra.feature_set == fs) & (ra.model == m)].R2.std() for m in names]
        b = [rb[(rb.feature_set == fs) & (rb.model == m)].R2.iloc[0] for m in names]
        ax.bar(x - w / 2, a, w, yerr=s, capsize=3, color='#2a7f62', label='A: random split')
        ax.bar(x + w / 2, b, w, color='#3d5a80', label='B: leave-one-scene-out')
        ax.axhline(0, color='k', lw=.6)
        ax.set_xticks(x); ax.set_xticklabels([n.replace('Linear regression on NDVI', 'Linear\n(NDVI)').replace(' ', '\n', 1) for n in names], fontsize=8)
        ax.set_title(f'CWSI regression — {fs}', fontsize=9)
    axes[0].set_ylabel('R²'); axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f'{FIGS}/fig_regression.png'); plt.close(fig)


if __name__ == '__main__':
    for f in (fig_main, fig_confusion, fig_per_scene, fig_ablation, fig_noise, fig_regression):
        f(); print('ok', f.__name__)
