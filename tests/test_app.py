"""Headless UI smoke test: python tests/test_app.py"""
import os
import sys
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app', 'streamlit_app.py')


def run_app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_default_view_and_learners():
    at = run_app()
    assert any('Predicted crop' in s.value for s in at.subheader)
    for learner in ['XGBoost', 'AdaBoost', 'Random Forest']:
        at.sidebar.selectbox[0].set_value(learner).run()
        assert not at.exception, (learner, [e.value for e in at.exception])


def test_manual_mode():
    at = run_app()
    at.radio[0].set_value('Enter values manually').run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.number_input) >= 12 + 5


def test_leakage_demo_button():
    at = run_app()
    btn = [b for b in at.button if b.label == 'Run test'][0]
    btn.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any('held out' in m.label for m in at.metric)


def test_sus_validation_and_score():
    at = run_app()
    sub = [b for b in at.button if b.label == 'Submit survey'][0]
    sub.click().run()
    assert any('consent' in e.value.lower() for e in at.error)
    at.checkbox[0].check().run()
    for i in range(10):
        at.radio(key=f'sus{i + 1}').set_value(5 if i % 2 == 0 else 1)
    [b for b in at.button if b.label == 'Submit survey'][0].click().run()
    assert any('100.0' in s.value for s in at.success), [s.value for s in at.success]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('PASS', name)
