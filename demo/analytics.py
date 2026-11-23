"""Interactive analytics dashboard (tab "Analytics" of the Streamlit demo).

Reads only result files written by the scripts, never recomputes or invents numbers:
    reports/final/results.json            HC3 test results, cross-domain (scripts/run_experiments.py)
    reports/final/modern_results.json     modern LLMs, attacks, mixed texts (scripts/evaluate_modern.py)
    reports/final/augmented_results.json  retraining with Claude Haiku texts (scripts/train_augmented.py)
    reports/final/modern/{texts,scores}.parquet   per-text scores for the distribution plots and error explorer
"""
import json

import altair as alt
import pandas as pd
import streamlit as st

from detector import modern
from detector.config import CHATGPT_COLOR, DOMAIN_NAMES, FINAL, HUMAN_COLOR
from detector.inference import DETECTORS, NAMES

INK_2, MUTED, VIOLET = '#52514e', '#898781', '#4a3aa7'
BLUES = ['#f6f5f1', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
SHORT = {'human': 'Human (HC3)', 'chatgpt_hc3': 'ChatGPT 2023 (HC3)', 'claude-haiku-5-5': 'Claude Haiku 5.5',
         'claude-sonnet-5-5': 'Claude Sonnet 5.5', 'claude-opus-5-5': 'Claude Opus 5.5',
         'qwen2.5-1.5b': 'Qwen2.5-1.5B (open)', 'smollm2-1.7b': 'SmolLM2-1.7B (open)',
         'paraphrase': 'Paraphrased', 'humanize': '"Humanized"', 'typos': '5% typos', 'mix': 'Mixed'}
# results.json uses long names, the modern results use detector keys
RESULT_KEY = {'DistilRoBERTa + LoRA': 'roberta_lora', 'TF-IDF + LogReg': 'tfidf',
              'PyTorch bag-of-embeddings': 'nn', 'Stylometry + Random Forest': 'style_rf',
              'Stylometry + LogReg': 'style_lr', 'GPT-2 perplexity (zero-shot)': 'ppl',
              'Fast-DetectGPT on GPT-2 (zero-shot)': 'fdg', 'Length baseline': 'length'}


def _json(name):
    path = FINAL / name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


@st.cache_data(show_spinner=False)
def _scored(_version=None):
    """`_version` = file modification time, so new results are picked up without a restart."""
    texts, scores = modern.TEXTS_FILE, modern.MODERN / 'scores.parquet'
    if not (texts.exists() and scores.exists()):
        return None
    return pd.read_parquet(texts).merge(pd.read_parquet(scores), on='text_id', how='left')


def _kpi(col, label, value, note):
    col.markdown(f'<div class="card" style="padding:16px 20px"><p class="verdict-label">{label}</p>'
                 f'<p class="verdict-title" style="font-size:28px !important;margin:2px 0">{value}</p>'
                 f'<p class="verdict-meta">{note}</p></div>', unsafe_allow_html=True)


def _heatmap(frame, x, y, value, fmt, title_x=None, reverse=False, height=None, x_sort=None, y_sort=None):
    scale = alt.Scale(domain=[0, 1], range=BLUES[::-1] if reverse else BLUES)
    base = alt.Chart(frame).encode(
        x=alt.X(f'{x}:N', sort=x_sort, title=title_x, axis=alt.Axis(labelAngle=-30, labelLimit=160)),
        y=alt.Y(f'{y}:N', sort=y_sort, title=None, axis=alt.Axis(labelLimit=220)))
    cells = base.mark_rect(cornerRadius=3).encode(
        color=alt.Color(f'{value}:Q', scale=scale, legend=None),
        tooltip=[y, x, alt.Tooltip(f'{value}:Q', format=fmt)])
    text = base.mark_text(fontSize=12).encode(
        text=alt.Text(f'{value}:Q', format=fmt),
        color=alt.condition(f'datum["{value}"] {"<" if reverse else ">"} 0.6', alt.value('white'), alt.value('#0b0b0b')))
    return (cells + text).properties(height=height or 36 * frame[y].nunique() + 20)


def section_hc3(results):
    st.subheader('1 · In-domain: the HC3 test set')
    st.caption('4,380 test texts (questions never used for training). Bars = value, whiskers = 95% bootstrap CI.')
    metric = st.segmented_control('Metric', ['F1', 'ROC-AUC', 'Accuracy', 'Humans flagged (FPR)', 'TPR @ 1% FPR'],
                                  default='F1', key='hc3_metric')
    column = {'F1': 'f1', 'ROC-AUC': 'roc_auc', 'Accuracy': 'accuracy', 'Humans flagged (FPR)': 'false_positive_rate',
              'TPR @ 1% FPR': 'tpr_at_1fpr'}[metric or 'F1']
    ci = {'f1': 'f1_ci', 'roc_auc': 'roc_auc_ci', 'accuracy': 'accuracy_ci'}.get(column)
    rows = []
    for name, r in results['test_results'].items():
        row = {'Detector': name, 'value': r[column], 'family': 'raw threshold 0.5' if 'raw' in name else 'calibrated'}
        if ci and ci in r:
            row['low'], row['high'] = r[ci]
        rows.append(row)
    frame = pd.DataFrame(rows)
    ascending = column == 'false_positive_rate'
    order = frame.sort_values('value', ascending=ascending).Detector.tolist()
    lo = 0 if column in ('false_positive_rate', 'tpr_at_1fpr') else max(0.0, frame.value.min() - 0.05)
    hi = min(1.0, frame.value.max() * 1.15) if column == 'false_positive_rate' else 1.0
    x = alt.X('value:Q', scale=alt.Scale(domain=[lo, hi], clamp=True), title=metric)
    bars = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, size=16).encode(
        x=x, y=alt.Y('Detector:N', sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color('family:N', scale=alt.Scale(domain=['calibrated', 'raw threshold 0.5'],
                                                    range=[VIOLET, '#c3c2b7']), legend=alt.Legend(orient='bottom', title=None)),
        tooltip=['Detector', alt.Tooltip('value:Q', title=metric, format='.4f')])
    layers = [bars]
    if 'low' in frame:
        layers.append(alt.Chart(frame.dropna(subset=['low'])).mark_rule(color=INK_2).encode(
            x='low:Q', x2='high:Q', y=alt.Y('Detector:N', sort=order)))
    layers.append(alt.Chart(frame).mark_text(align='left', dx=6, fontSize=11).encode(
        x=alt.X('high:Q' if 'low' in frame else 'value:Q'), y=alt.Y('Detector:N', sort=order),
        text=alt.Text('value:Q', format='.3f')))
    st.altair_chart(alt.layer(*layers).properties(height=30 * len(frame) + 40), width='stretch')

    if results.get('lodo'):
        st.markdown('**Cross-domain: train on 4 topics, test on the 5th.** Darker = better F1 on the unseen topic.')
        lodo = pd.DataFrame(results['lodo'])
        name_col = 'model_name' if 'model_name' in lodo else 'model'
        lodo['Topic'] = lodo.domain.map(DOMAIN_NAMES).fillna(lodo.domain)
        show = st.segmented_control('Show', ['F1 on the unseen topic', 'Drop vs. topic seen in training'],
                                    default='F1 on the unseen topic', key='lodo_show')
        if show == 'Drop vs. topic seen in training':
            lodo['v'] = lodo.f1_drop.clip(lower=0) / max(lodo.f1_drop.max(), 0.15)
            lodo['label'] = -lodo.f1_drop
            chart = alt.Chart(lodo).mark_rect(cornerRadius=3).encode(
                x=alt.X('Topic:N', title=None, axis=alt.Axis(labelAngle=-30)),
                y=alt.Y(f'{name_col}:N', title=None, axis=alt.Axis(labelLimit=220)),
                color=alt.Color('v:Q', scale=alt.Scale(domain=[0, 1], range=['#fcfcfb', '#f6c3a8', CHATGPT_COLOR]),
                                legend=None),
                tooltip=[name_col, 'Topic', alt.Tooltip('label:Q', title='F1 change', format='+.3f')]
            ) + alt.Chart(lodo).mark_text(fontSize=12).encode(
                x='Topic:N', y=f'{name_col}:N', text=alt.Text('label:Q', format='+.3f'))
            st.altair_chart(chart.properties(height=36 * lodo[name_col].nunique() + 20), width='stretch')
        else:
            st.altair_chart(_heatmap(lodo, 'Topic', name_col, 'f1_unseen', '.3f'), width='stretch')


def section_generators(res):
    st.subheader('2 · New generators: Claude 5.5 family and open models')
    st.caption('Same test questions; the detectors were trained only on HC3 (ChatGPT, 2023) and are NOT retrained. '
               'Thresholds come from HC3. ROC-AUC compares the machine texts with the human answers to the same questions.')
    metric = st.segmented_control('Metric ', ['Detection rate (P ≥ 0.5)', 'ROC-AUC'], default='Detection rate (P ≥ 0.5)',
                                  key='gen_metric')
    sources = [s for s in modern.GENERATORS if s in res['detection_rate']]
    rows = []
    for d in DETECTORS:
        for s in sources:
            v = res['roc_auc'][s][d] if metric == 'ROC-AUC' else res['detection_rate'][s][d]['value']
            rows.append({'Detector': NAMES[d], 'Generator': SHORT[s], 'v': v})
        rows.append({'Detector': NAMES[d], 'Generator': 'Humans flagged (FPR)',
                     'v': res['false_positive_rate'][d]['value']})
    frame = pd.DataFrame(rows)
    gens = frame[frame.Generator != 'Humans flagged (FPR)']
    fpr = frame[frame.Generator == 'Humans flagged (FPR)']
    y_sort = [NAMES[d] for d in DETECTORS]
    left, right = st.columns([5, 1.2])
    with left:
        st.altair_chart(_heatmap(gens, 'Generator', 'Detector', 'v', '.0%' if metric != 'ROC-AUC' else '.2f',
                                 x_sort=[SHORT[s] for s in sources], y_sort=y_sort), width='stretch')
    with right:
        st.altair_chart(_heatmap(fpr, 'Generator', 'Detector', 'v', '.1%', reverse=True, y_sort=y_sort)
                        .encode(y=alt.Y('Detector:N', sort=y_sort, axis=None)), width='stretch')
    words = pd.DataFrame([{'Source': SHORT[s], 'Mean words': res['mean_words'][s], 'Texts': res['n_texts'][s]}
                          for s in ['human'] + sources])
    st.dataframe(words.style.format({'Mean words': '{:.0f}'}), hide_index=True, width='content')


def section_distributions(scored):
    st.subheader('3 · How the scores are distributed')
    st.caption('Each line: how the P(machine) scores of one text source are spread. Good separation = the human line '
               'peaks on the left, machine lines on the right of the dashed 0.5 threshold.')
    c1, c2 = st.columns([1, 2])
    detector = c1.selectbox('Detector', DETECTORS, format_func=NAMES.get, key='dist_detector')
    present = [s for s in SHORT if s in set(scored.source) and s != 'mix']
    chosen = c2.multiselect('Sources', present, default=[s for s in present if s in
                            ('human', 'chatgpt_hc3', 'claude-opus-5-5', 'humanize')] or present[:3],
                            format_func=SHORT.get, key='dist_sources')
    # histogram as share of texts per 5% bin (a density estimate is misleading: most scores sit at 0 or 1)
    frame = scored[scored.source.isin(chosen)]
    bins = (frame[detector].clip(0, 0.9999) // 0.05 * 0.05).round(2)
    hist = (frame.assign(bin=bins).groupby(['source', 'bin']).size()
            / frame.groupby('source').size()).rename('share').reset_index()
    hist['Source'] = hist.source.map(SHORT)
    colors = {SHORT[s]: (HUMAN_COLOR if s == 'human' else SERIES[(i + 1) % len(SERIES)]) for i, s in enumerate(chosen)}
    chart = alt.Chart(hist).mark_line(interpolate='step-after', strokeWidth=2).encode(
        x=alt.X('bin:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%'), title='P(machine), 5% bins'),
        y=alt.Y('share:Q', axis=alt.Axis(format='%'), title='Share of texts of this source'),
        color=alt.Color('Source:N', scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                        legend=alt.Legend(orient='bottom', title=None)),
        tooltip=['Source', alt.Tooltip('bin:Q', title='bin from', format='.0%'), alt.Tooltip('share:Q', format='.1%')])
    rule = alt.Chart(pd.DataFrame({'x': [0.5]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(x='x:Q')
    st.altair_chart((chart + rule).properties(height=300), width='stretch')


def section_attacks(res):
    st.subheader('4 · Robustness: attacks and mixed texts')
    attacks = [s for s in modern.ATTACKS if s in res['detection_rate']]
    label = {**SHORT, 'chatgpt_hc3': 'No attack'}
    rows = [{'Detector': NAMES[d], 'Attack': label[s], 'Detected': res['detection_rate'][s][d]['value'],
             'low': res['detection_rate'][s][d]['ci'][0], 'high': res['detection_rate'][s][d]['ci'][1]}
            for d in DETECTORS for s in attacks]
    frame = pd.DataFrame(rows)
    left, right = st.columns(2, gap='large')
    with left:
        st.markdown('**Share of ChatGPT texts still detected after an attack**')
        chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X('Detected:Q', axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1]), title=None),
            y=alt.Y('Attack:N', sort=[label[s] for s in attacks], title=None),
            color=alt.Color('Attack:N', scale=alt.Scale(domain=[label[s] for s in attacks],
                                                        range=['#c3c2b7'] + SERIES[1:len(attacks)]), legend=None),
            tooltip=['Detector', 'Attack', alt.Tooltip('Detected:Q', format='.1%'),
                     alt.Tooltip('low:Q', title='95% CI low', format='.1%'),
                     alt.Tooltip('high:Q', title='95% CI high', format='.1%')]
        ).properties(height=24 * len(attacks), width=170).facet(
            facet=alt.Facet('Detector:N', sort=[NAMES[d] for d in DETECTORS], title=None), columns=2)
        st.altair_chart(chart, width='content')
    with right:
        st.markdown(f'**Mixed texts: human start + ChatGPT continuation** ({res["mix_n_questions"]} questions)')
        keys = st.multiselect('Detectors', DETECTORS, default=['roberta_lora', 'tfidf', 'ppl', 'style_rf'],
                              format_func=NAMES.get, key='mix_detectors')
        mix = pd.DataFrame([{'Detector': NAMES[d], 'Machine share': s, 'Mean P(machine)': p}
                            for d in keys for s, p in zip(res['mix'][d]['share'], res['mix'][d]['mean_prob'])])
        line = alt.Chart(mix).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X('Machine share:Q', axis=alt.Axis(format='%'), title='Share of the text written by ChatGPT'),
            y=alt.Y('Mean P(machine):Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
            color=alt.Color('Detector:N', scale=alt.Scale(range=SERIES), legend=alt.Legend(orient='bottom', title=None)),
            tooltip=['Detector', alt.Tooltip('Machine share:Q', format='.0%'), alt.Tooltip('Mean P(machine):Q', format='.1%')])
        rule = alt.Chart(pd.DataFrame({'y': [0.5]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y='y:Q')
        st.altair_chart((line + rule).properties(height=320), width='stretch')
        st.caption('A detector that only says "human" or "ChatGPT" for the whole text cannot tell which part was '
                   'machine-written; the curve shows at what share it starts to flag the text.')


def section_augmented(aug):
    st.subheader('5 · Fix: retrain with texts of one modern generator')
    st.caption(f"Training = HC3 train + {aug['n_augmentation'].get('claude-haiku-train', 0):,} Claude Haiku 5.5 texts "
               f"(train questions only) + {aug['n_augmentation'].get('human-extra', 0):,} extra human answers. "
               'Sonnet, Opus, Qwen and SmolLM are never used for training: they test transfer to unseen generators.')
    options = [k for k in ('roberta_lora', 'tfidf', 'nn', 'style_rf', 'style_lr') if k in aug['modern']]
    key = st.segmented_control('Detector', options, default=options[0], format_func=NAMES.get,
                               key='aug_key') or options[0]
    data = aug['modern'][key]
    rows = []
    for s, v in data.items():
        if s == 'mix':
            continue
        for when in ('before', 'after'):
            rows.append({'Source': 'Humans flagged (FPR)' if s == 'human' else SHORT.get(s, s),
                         'Model': 'HC3 only' if when == 'before' else 'HC3 + Claude Haiku', 'Rate': v[when]})
    frame = pd.DataFrame(rows)
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X('Rate:Q', axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1]), title='Share flagged as machine'),
        y=alt.Y('Model:N', title=None, axis=None, sort=['HC3 only', 'HC3 + Claude Haiku']),
        color=alt.Color('Model:N', scale=alt.Scale(domain=['HC3 only', 'HC3 + Claude Haiku'], range=['#c3c2b7', '#256abf']),
                        legend=alt.Legend(orient='bottom', title=None)),
        row=alt.Row('Source:N', title=None, header=alt.Header(labelAngle=0, labelAlign='left', labelLimit=220),
                    sort=list(dict.fromkeys(frame.Source))),
        tooltip=['Source', 'Model', alt.Tooltip('Rate:Q', format='.1%')]).properties(height=36, width=520)
    st.altair_chart(chart, width='content')
    hc3 = aug['hc3_test'][key]
    c = st.columns(3)
    c[0].metric('HC3 test F1', f"{hc3['after']['f1']:.3f}", f"{hc3['after']['f1'] - hc3['before']['f1']:+.3f}")
    c[1].metric('HC3 ROC-AUC', f"{hc3['after']['roc_auc']:.4f}", f"{hc3['after']['roc_auc'] - hc3['before']['roc_auc']:+.4f}")
    c[2].metric('HC3 humans flagged', f"{hc3['after']['fpr']:.1%}", f"{hc3['after']['fpr'] - hc3['before']['fpr']:+.1%}",
                delta_color='inverse')


def section_errors(scored):
    st.subheader('6 · Error explorer')
    st.caption('Read the actual texts a detector gets wrong. Sorted by how confident the wrong answer was.')
    c1, c2, c3 = st.columns(3)
    detector = c1.selectbox('Detector ', DETECTORS, format_func=NAMES.get, key='err_detector')
    present = [s for s in SHORT if s in set(scored.source)]
    source = c2.selectbox('Source', present, format_func=SHORT.get, index=present.index('claude-opus-5-5')
                          if 'claude-opus-5-5' in present else 0, key='err_source')
    domain = c3.selectbox('Topic', ['all'] + sorted(scored.domain.unique()),
                          format_func=lambda d: 'All topics' if d == 'all' else DOMAIN_NAMES.get(d, d), key='err_domain')
    frame = scored[scored.source == source]
    if domain != 'all':
        frame = frame[frame.domain == domain]
    wrong = frame[(frame[detector] >= 0.5) != (frame.y == 1)]
    wrong = wrong.assign(confidence=(wrong[detector] - 0.5).abs()).sort_values('confidence', ascending=False)
    st.markdown(f'**{len(wrong)} of {len(frame)} texts are misclassified** ({len(wrong) / max(len(frame), 1):.1%}).')
    show = wrong.head(30).assign(**{'P(machine)': wrong[detector], 'Topic': wrong.domain.map(DOMAIN_NAMES),
                                   'Truth': wrong.y.map({0: 'human', 1: 'machine'})})
    st.dataframe(show[['P(machine)', 'Truth', 'Topic', 'n_words', 'text']].rename(columns={'n_words': 'Words', 'text': 'Text'}),
                 hide_index=True, width='stretch', height=360,
                 column_config={'P(machine)': st.column_config.ProgressColumn(format='percent', min_value=0, max_value=1),
                                'Text': st.column_config.TextColumn(width='large')})


def render():
    results, res, aug = _json('results.json'), _json('modern_results.json'), _json('augmented_results.json')
    stamp = modern.MODERN / 'scores.parquet'
    scored = _scored(stamp.stat().st_mtime if stamp.exists() else None)
    if not results:
        st.warning('reports/final/results.json not found. Run: python scripts/run_experiments.py')
        return
    best = max(results['test_results'].items(), key=lambda x: x[1]['f1'])
    c = st.columns(4)
    _kpi(c[0], 'Best F1 on HC3 test', f"{best[1]['f1']:.3f}", best[0])
    if res:
        unseen = [s for s in ('claude-sonnet-5-5', 'claude-opus-5-5', 'qwen2.5-1.5b', 'smollm2-1.7b')
                  if s in res['detection_rate']] or [s for s in res['detection_rate'] if s != 'chatgpt_hc3']
        # the length baseline is excluded: new models write longer answers, so it "detects" length, not style
        mean_rate = {d: sum(res['detection_rate'][s][d]['value'] for s in unseen) / len(unseen)
                     for d in DETECTORS if d != 'length'}
        top = max(mean_rate, key=mean_rate.get)
        _kpi(c[1], 'Best on new generators', f'{mean_rate[top]:.0%}', f'{NAMES[top]}: mean detection rate')
        worst_attack = min((s for s in ('paraphrase', 'humanize') if s in res['detection_rate']),
                           key=lambda s: res['detection_rate'][s]['roberta_lora']['value'], default=None)
        if worst_attack:
            v = res['detection_rate'][worst_attack]['roberta_lora']['value']
            _kpi(c[2], 'Strongest attack', f'{v:.0%}', f'{SHORT[worst_attack]}: still caught by DistilRoBERTa')
        else:
            v = res['detection_rate']['typos']['roberta_lora']['value']
            _kpi(c[2], 'Typos attack', f'{v:.0%}', 'still caught by DistilRoBERTa')
        _kpi(c[3], 'Texts evaluated', f"{sum(res['n_texts'].values()) + results['meta']['n_test']:,}",
             'HC3 test + new sets')
    section_hc3(results)
    if res:
        section_generators(res)
        if scored is not None:
            section_distributions(scored)
        section_attacks(res)
    else:
        st.info('New-generator results not found. Run: python scripts/evaluate_modern.py')
    if aug:
        section_augmented(aug)
    if scored is not None:
        section_errors(scored)
