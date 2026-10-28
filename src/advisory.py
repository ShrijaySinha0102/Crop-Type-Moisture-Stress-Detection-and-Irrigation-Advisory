"""Advisory logic shared by the Streamlit app and the tests (no UI code here).

Crop classifier  : one of the three learners trained on the weather-free feature set.
Water-demand rule: FAO-56 style  ETc = Kc * ET0,  NIR = max(ETc - rain, 0)
                   with ET0 from a Hargreaves-Samani form that uses surface solar radiation.
This is a research prototype; see the report for its limits (LOSO macro-F1 0.12-0.16).
"""
import numpy as np
import pandas as pd

from common import (CLASSES, FEATURE_SETS, HONEST_SET, LEARNERS, REFERENCE, fit_predict_clf, load_data)

# Single mid-season Kc per harmonised class (values of the corresponding raw labels in build_dataset.py)
KC = {'Paddy': 1.20, 'Sugarcane': 1.25, 'Soybean': 1.05, 'Pulses': 1.05, 'Oilseeds': 1.00, 'Maize': 1.15}
FEATURES = FEATURE_SETS[HONEST_SET]
LOW_CONFIDENCE = 0.60


def reference_et0(t2m, tmax, tmin, rs):
    """mm/day. Same formula as src/build_dataset.py (Hargreaves-Samani form, Rs in place of Ra)."""
    trange = max(float(tmax) - float(tmin), 0.1)
    return max(0.0023 * (0.408 * float(rs)) * (float(t2m) + 17.8) * np.sqrt(trange), 0.0)


def water_demand(crop, t2m, tmax, tmin, rs, rain):
    et0 = reference_et0(t2m, tmax, tmin, rs)
    kc = KC[crop]
    etc = et0 * kc
    nir = max(etc - float(rain), 0.0)
    return dict(et0=et0, kc=kc, etc=etc, nir=nir, m3_per_ha=nir * 10.0)   # 1 mm over 1 ha = 10 m3


def advice_text(crop, confidence, demand):
    lines = []
    if confidence < LOW_CONFIDENCE:
        lines.append(f'Low confidence ({confidence:.0%}). Verify the crop on the ground before acting; '
                     'the estimate below assumes the predicted crop.')
    if demand['nir'] <= 0.05:
        lines.append('Rainfall covers the estimated crop demand today; no extra irrigation is indicated.')
    else:
        lines.append(f"Estimated net irrigation need for {crop}: {demand['nir']:.1f} mm/day "
                     f"(about {demand['m3_per_ha']:.0f} m3 per hectare per day).")
    lines.append('Research prototype: crop prediction does not generalise reliably to new scenes, and the water '
                 'rule uses one crop coefficient for all growth stages. Use as a suggestion to a human expert only.')
    return lines


_cache = {}


def train_model(learner, exclude_scene=None, seed=0):
    """Fit `learner` on the weather-free features; optionally leave one scene out. Returns a predict function."""
    key = (learner, exclude_scene, seed)
    if key in _cache:
        return _cache[key]
    df = load_data()
    train = df if exclude_scene is None else df[df['ISRO_ProductID'] != exclude_scene]
    ytr = train['Crop'].values

    # Fit once and keep the fitted model (re-fitting per call would be slow)
    _, _, _, model = fit_predict_clf(learner, REFERENCE[learner], train[FEATURES], ytr, df[FEATURES].iloc[:2], seed)
    present = np.array(sorted(set(ytr)))

    def predict(X):
        p = model.predict_proba(X[FEATURES])
        full = np.zeros((len(X), len(CLASSES)))
        for i, c in enumerate(present):
            full[:, list(CLASSES).index(c)] = p[:, i]
        return full

    _cache[key] = predict
    return predict


def scene_check(learner, scene):
    """Accuracy on one scene when it is IN the training data versus held out (the leakage demo)."""
    df = load_data()
    sub = df[df['ISRO_ProductID'] == scene]
    out = {}
    for label, excl in (('scene included in training', None), ('scene held out', scene)):
        proba = train_model(learner, excl)(sub)
        pred = CLASSES[proba.argmax(1)]
        out[label] = dict(accuracy=float((pred == sub['Crop'].values).mean()),
                          top=pd.Series(pred).value_counts(normalize=True).head(3).to_dict())
    return sub['Crop'].iloc[0], out


SUS_ITEMS = [
    'I think that I would like to use this system frequently.',
    'I found the system unnecessarily complex.',
    'I thought the system was easy to use.',
    'I think that I would need the support of a technical person to be able to use this system.',
    'I found the various functions in this system were well integrated.',
    'I thought there was too much inconsistency in this system.',
    'I would imagine that most people would learn to use this system very quickly.',
    'I found the system very cumbersome to use.',
    'I felt very confident using the system.',
    'I needed to learn a lot of things before I could get going with this system.',
]


def sus_score(answers):
    """Standard System Usability Scale scoring. answers: ten integers 1-5. Returns 0-100."""
    if len(answers) != 10 or any(a not in (1, 2, 3, 4, 5) for a in answers):
        raise ValueError('SUS needs ten answers between 1 and 5')
    total = sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))
    return total * 2.5
