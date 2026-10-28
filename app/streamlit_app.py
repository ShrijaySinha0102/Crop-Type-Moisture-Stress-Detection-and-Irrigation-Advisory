"""Crop and irrigation advisory: research prototype.

Run:  streamlit run app/streamlit_app.py
Tabs: 1 Advisory (crop prediction + FAO-56 water demand)  2 Leakage demo  3 Model comparison  4 Usability survey (SUS)
"""
import csv
import datetime
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import advisory as A                              # noqa: E402
from common import CLASSES, FIGS, LEARNERS, RESULTS, load_data   # noqa: E402

st.set_page_config(page_title='Crop & irrigation advisory (research prototype)', layout='wide')


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model(learner, exclude_scene=None):
    return A.train_model(learner, exclude_scene)


@st.cache_data
def run_scene_check(learner, scene):
    return A.scene_check(learner, scene)


df = get_data()
WEATHER = [('T2M', 'Mean air temperature (°C)'), ('T2M_MAX', 'Max air temperature (°C)'), ('T2M_MIN', 'Min air temperature (°C)'),
           ('ALLSKY_SFC_SW_DWN', 'Solar radiation (MJ/m²/day)'), ('PRECTOTCORR', 'Rainfall today (mm)')]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header('Settings')
    learner = st.selectbox('Crop classifier', LEARNERS, help='All three were compared in the report.')
    st.warning('**Research prototype.** On unseen scenes the classifiers reach only about 0.12–0.16 macro-F1 '
               '(see the *Leakage demo* tab). Treat every output as a suggestion for a human expert, not as advice to act on.')
    st.caption('Data: 14 ISRO scenes (AWiFS / LISS-4), NASA POWER weather. The classifier uses optical, SAR and thermal/rain '
               'features only; weather is used for the water-demand rule.')

st.title('Crop-type and irrigation advisory')
tab1, tab2, tab3, tab4 = st.tabs(['Advisory', 'Leakage demo', 'Model comparison', 'Usability survey'])

# ------------------------------------------------------------------ tab 1
with tab1:
    mode = st.radio('Input', ['Sample point from the dataset', 'Enter values manually'], horizontal=True)
    if mode.startswith('Sample'):
        scenes = df.groupby('ISRO_ProductID').agg(crop=('Crop', 'first'), sensor=('Sensor', 'first')).reset_index()
        labels = {f'{r.ISRO_ProductID}  ·  {r.sensor}': r.ISRO_ProductID for r in scenes.itertuples()}
        c1, c2 = st.columns([3, 1])
        scene = labels[c1.selectbox('Scene (product ID · sensor)', list(labels))]
        sub = df[df['ISRO_ProductID'] == scene].reset_index(drop=True)
        idx = c2.number_input('Point (1–150)', 1, len(sub), 1) - 1
        row = sub.iloc[int(idx)]
        st.caption(f"Recorded label for this scene (region-level dominant crop): **{row['Crop']}**   ·   "
                   f"acquisition date {row['Acquisition_Date']}")
        feats = row[A.FEATURES].to_frame().T.astype(float)
        wdef = {k: float(row[k]) for k, _ in WEATHER}
    else:
        cols = st.columns(4)
        vals = {}
        for i, f in enumerate(A.FEATURES):
            vals[f] = cols[i % 4].number_input(f, value=float(df[f].median()), format='%.4f')
        feats = pd.DataFrame([vals])
        wdef = {k: float(df[k].median()) for k, _ in WEATHER}

    st.subheader('Weather for the water-demand rule')
    wc = st.columns(5)
    w = {k: wc[i].number_input(lab, value=round(wdef[k], 2), format='%.2f') for i, (k, lab) in enumerate(WEATHER)}

    proba = get_model(learner)(feats)[0]
    order = np.argsort(proba)[::-1]
    pred, conf = CLASSES[order[0]], float(proba[order[0]])
    left, right = st.columns([1, 1])
    with left:
        st.subheader('Predicted crop')
        st.metric(label=f'{learner}', value=pred, delta=f'confidence {conf:.0%}', delta_color='off')
        st.bar_chart(pd.Series(proba, index=CLASSES, name='probability'))
    with right:
        st.subheader('Water demand (FAO-56 rule)')
        crop = st.selectbox('Crop used for the calculation', list(CLASSES), index=list(CLASSES).index(pred),
                            help='Defaults to the prediction; override it if you know the crop.')
        d = A.water_demand(crop, w['T2M'], w['T2M_MAX'], w['T2M_MIN'], w['ALLSKY_SFC_SW_DWN'], w['PRECTOTCORR'])
        m = st.columns(4)
        m[0].metric('ET₀ (mm/day)', f"{d['et0']:.2f}")
        m[1].metric(f"Kc ({crop})", f"{d['kc']:.2f}")
        m[2].metric('ETc (mm/day)', f"{d['etc']:.2f}")
        m[3].metric('Net irrigation need (mm/day)', f"{d['nir']:.2f}")
        for line in A.advice_text(crop, conf if crop == pred else 1.0, d):
            (st.info if line.startswith('Estimated') or line.startswith('Rainfall') else st.warning)(line)
    with st.expander('Method and limits'):
        st.markdown(
            '* **ET₀** uses a Hargreaves–Samani form with surface solar radiation in place of extraterrestrial radiation (an approximation).\n'
            '* **Kc** is one mid-season value per crop; growth stages are not modelled.\n'
            '* **Net irrigation need** = max(ETc − rainfall, 0); rainfall is treated as fully effective; no soil-water balance.\n'
            '* **Crop prediction** was trained on 14 scenes with region-level labels; see the *Leakage demo* tab.')

# ------------------------------------------------------------------ tab 2
with tab2:
    st.markdown('The same model is tested on one scene **twice**: once with that scene in the training data (like a random split) '
                'and once with the scene **held out** (like a new field site). The gap is the leakage described in the report.')
    scenes2 = df.groupby('ISRO_ProductID').agg(crop=('Crop', 'first'), sensor=('Sensor', 'first')).reset_index()
    lab2 = {f'{r.ISRO_ProductID}  ·  {r.crop}  ·  {r.sensor}': r.ISRO_ProductID for r in scenes2.itertuples()}
    pick = lab2[st.selectbox('Scene to test', list(lab2), key='leak_scene')]
    if st.button('Run test', type='primary'):
        with st.spinner('Training two models...'):
            truth, res = run_scene_check(learner, pick)
        c = st.columns(2)
        for col, (name, r) in zip(c, res.items()):
            col.metric(name, f"{r['accuracy']:.0%} correct")
            col.write('Most frequent predictions: ' + ', '.join(f'{k} {v:.0%}' for k, v in r['top'].items()))
        st.caption(f'True label of this scene: {truth}. When the scene is held out, crops that appear in no other scene '
                   '(Maize, Oilseeds, Pulses, Soybean) cannot be predicted at all.')

# ------------------------------------------------------------------ tab 3
with tab3:
    sa, sb = os.path.join(RESULTS, 'summary_A.csv'), os.path.join(RESULTS, 'summary_B.csv')
    if os.path.exists(sa) and os.path.exists(sb):
        fs = st.radio('Feature set', ['All features', 'Optical + SAR + thermal/rain'], horizontal=True)
        st.markdown('**Protocol A: random 80/20 split (10 seeds)**')
        st.dataframe(pd.read_csv(sa).query('feature_set == @fs')[['model', 'accuracy', 'macro_f1', 'macro_f1_ci95', 'kappa']], hide_index=True)
        st.markdown('**Protocol B: leave-one-scene-out**')
        st.dataframe(pd.read_csv(sb).query('feature_set == @fs')[['model', 'accuracy', 'macro_f1', 'macro_f1_ci95', 'kappa']], hide_index=True)
        for f in ('fig_main_comparison.png', 'fig_ablation.png'):
            if os.path.exists(os.path.join(FIGS, f)):
                st.image(os.path.join(FIGS, f))
    else:
        st.info('Run the experiments first (see README) to populate results/.')

# ------------------------------------------------------------------ tab 4
with tab4:
    st.markdown('**System Usability Scale (SUS).** Ten short statements; about two minutes. Answers are anonymous and are stored '
                'only in `results/sus_responses.csv` on this computer. Please answer after trying the *Advisory* and *Leakage demo* tabs.')
    consent = st.checkbox('I agree that my anonymous answers may be used in the project report.')
    role = st.selectbox('I am a', ['Student', 'Faculty / evaluator', 'Farmer / agronomy professional', 'Other'])
    answers = []
    for i, item in enumerate(A.SUS_ITEMS, 1):
        answers.append(st.radio(f'{i}. {item}', [1, 2, 3, 4, 5], index=None, horizontal=True,
                                help='1 = strongly disagree, 5 = strongly agree', key=f'sus{i}'))
    if st.button('Submit survey'):
        if not consent:
            st.error('Please tick the consent box first.')
        elif any(a is None for a in answers):
            st.error('Please answer all ten statements.')
        else:
            score = A.sus_score(answers)
            st.success(f'Thank you. SUS score: {score:.1f} / 100 (a score around 68 is considered average).')
            path = os.path.join(RESULTS, 'sus_responses.csv')
            try:
                new = not os.path.exists(path)
                with open(path, 'a', newline='', encoding='utf-8') as fh:
                    wr = csv.writer(fh)
                    if new:
                        wr.writerow(['timestamp', 'role'] + [f'q{i}' for i in range(1, 11)] + ['sus_score'])
                    wr.writerow([datetime.datetime.now().isoformat(timespec='seconds'), role] + answers + [score])
            except OSError as e:
                st.warning(f'Could not save the response ({e}). Please note the score manually.')
