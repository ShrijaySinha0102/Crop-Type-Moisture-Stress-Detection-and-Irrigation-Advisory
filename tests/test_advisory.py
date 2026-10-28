"""Run with:  python tests/test_advisory.py   (or: python -m pytest tests)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import advisory as A                      # noqa: E402
from common import CLASSES, load_data      # noqa: E402


def test_et0_matches_dataset():
    df = load_data().sample(200, random_state=0)
    calc = np.array([A.reference_et0(r.T2M, r.T2M_MAX, r.T2M_MIN, r.ALLSKY_SFC_SW_DWN) for r in df.itertuples()])
    assert np.allclose(calc, df['ET0_mm_day'].values, atol=1e-9), 'ET0 differs from build_dataset.py'


def test_water_demand_formula():
    d = A.water_demand('Paddy', 27.0, 33.0, 22.0, 18.0, 1.0)
    assert abs(d['etc'] - d['et0'] * 1.20) < 1e-12
    assert abs(d['nir'] - max(d['etc'] - 1.0, 0)) < 1e-12
    assert A.water_demand('Paddy', 27.0, 33.0, 22.0, 18.0, 99.0)['nir'] == 0.0     # rain exceeds demand
    assert A.water_demand('Paddy', 27.0, 30.0, 30.0, 18.0, 0.0)['et0'] > 0          # zero range is clamped


def test_kc_covers_all_classes():
    assert set(A.KC) == set(CLASSES)


def test_model_predicts_valid_probabilities():
    df = load_data()
    for learner in A.LEARNERS:
        p = A.train_model(learner)(df.iloc[:5])
        assert p.shape == (5, len(CLASSES)) and np.allclose(p.sum(1), 1.0, atol=1e-6)


def test_held_out_scene_behaves_like_loso():
    df = load_data()
    scene = df[df['Crop'] == 'Soybean']['ISRO_ProductID'].iloc[0]       # single-scene class
    truth, out = A.scene_check('Random Forest', scene)
    assert truth == 'Soybean'
    assert out['scene held out']['accuracy'] == 0.0                        # class unseen -> cannot be predicted
    assert out['scene included in training']['accuracy'] > out['scene held out']['accuracy']


def test_sus_scoring():
    assert A.sus_score([5, 1] * 5) == 100.0
    assert A.sus_score([1, 5] * 5) == 0.0
    assert A.sus_score([3] * 10) == 50.0
    try:
        A.sus_score([3] * 9)
    except ValueError:
        return
    raise AssertionError('should reject 9 answers')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('PASS', name)
