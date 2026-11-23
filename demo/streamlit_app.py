"""Streamlit demo: paste an English text and see how our detectors judge it.

Start:   double-click "Запустить демо.bat"   or   python -m streamlit run demo/streamlit_app.py
Needs:   the models in models/ (created by scripts/run_experiments.py and scripts/train_transformer.py).
         GPT-2 and DistilRoBERTa are loaded from the local Hugging Face cache only; if they are missing,
         those detectors are shown as unavailable instead of giving a made-up result.
"""
import html
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import altair as alt                       # noqa: E402
import joblib                              # noqa: E402
import pandas as pd                        # noqa: E402
import streamlit as st                     # noqa: E402

from detector import data as dataset       # noqa: E402
from detector import features as style     # noqa: E402
from detector.config import CHATGPT_COLOR, FIGURES, FINAL, HUMAN_COLOR, MODELS, TRANSFORMER_NAME   # noqa: E402

SHORT_NAMES = {'connectives': 'Connectives (however, overall…)', 'first person I': '"I", "my", "me"',
               'contractions': "Contractions (don't, it's)", 'brackets ( [': 'Brackets',
               'hedges': 'Hedges (I think, maybe)', 'MATTR (vocabulary richness)': 'Vocabulary richness',
               'mean_sentence_len': 'Words per sentence', 'long_words_% (7+ letters)': 'Long words (7+ letters)'}
UNCERTAIN = (0.35, 0.65)          # P(ChatGPT) inside this band -> "uncertain"
SHORT_TEXT = 50                   # short ChatGPT texts are rare in HC3 and often missed (errors_by_length.png)
MODEL_ORDER = ['roberta_lora', 'tfidf', 'nn', 'style_rf', 'style_lr', 'ppl', 'fdg', 'length']
DESCRIPTIONS = {
    'roberta_lora': 'Transformer (DistilRoBERTa) fine-tuned with LoRA, calibrated on validation; reads the first 256 tokens.',
    'tfidf': 'Classical ML: TF-IDF of words, word pairs and punctuation + logistic regression.',
    'nn': 'Small neural network in PyTorch: averaged word embeddings + 2 dense layers.',
    'style_rf': 'Random forest on hand-crafted style features (sentence length, vocabulary, "I", connectives...).',
    'style_lr': 'Logistic regression on the same style features (fully interpretable).',
    'ppl': 'Zero-shot: GPT-2 perplexity of the first 128 tokens (predictable text = more likely ChatGPT).',
    'fdg': 'Zero-shot: Fast-DetectGPT criterion computed with GPT-2.',
    'length': 'Sanity-check baseline that only knows the number of words.',
}

st.set_page_config(page_title='Human or ChatGPT? Text detector', layout='wide')

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"], button, input, textarea,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ font-family: 'Manrope', system-ui, sans-serif; }}
.block-container {{ padding-top: 2rem; max-width: 1240px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; text-wrap: balance; }}
p, li {{ text-wrap: pretty; }}
.hero-title {{ font-size: 36px !important; line-height: 40px !important; font-weight: 700; margin: 0 0 8px 0; color: #0b0b0b; }}
.hero-sub {{ font-size: 18px !important; line-height: 28px !important; color: #52514e; max-width: 680px; margin: 0 0 16px 0; }}
.card {{ border: 1px solid rgba(11,11,11,0.10); border-radius: 16px; padding: 24px; background: #ffffff; margin-bottom: 16px; }}
.verdict-human {{ border-color: {HUMAN_COLOR}; background: #f2f7fd; }}
.verdict-chatgpt {{ border-color: {CHATGPT_COLOR}; background: #fdf4ef; }}
.verdict-uncertain {{ border-color: #c3c2b7; background: #f6f5f1; }}
.verdict-label {{ font-size: 14px; line-height: 20px; color: #52514e; margin: 0; }}
.verdict-title {{ font-size: 30px !important; line-height: 36px !important; font-weight: 700; margin: 4px 0 8px 0; color: #0b0b0b; }}
.verdict-meta {{ font-size: 14px; line-height: 20px; color: #52514e; margin: 0; }}
.prob-bar {{ display: flex; height: 32px; border-radius: 8px; overflow: hidden; margin: 16px 0 8px 0; }}
.prob-bar div {{ display: flex; align-items: center; padding: 0 12px; color: #ffffff; font-size: 14px; font-weight: 600; white-space: nowrap; }}
.prob-legend {{ display: flex; justify-content: space-between; font-size: 14px; color: #52514e; }}
.highlight {{ font-size: 16px; line-height: 28px; color: #0b0b0b; }}
.highlight .h {{ background: #d7e6f8; border-radius: 4px; padding: 0 2px; }}
.highlight .c {{ background: #fadccd; border-radius: 4px; padding: 0 2px; }}
.note {{ font-size: 14px; line-height: 20px; color: #52514e; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loading (cached, never fake)
@st.cache_resource(show_spinner=False)
def load_local_models():
    """Small models stored in models/. Returns ({key: model}, {key: reason it is missing})."""
    loaded, missing = {}, {}
    files = {'length': 'length_baseline.joblib', 'tfidf': 'tfidf_logreg.joblib',
             'style_rf': 'stylometry_rf.joblib', 'style_lr': 'stylometry_logreg.joblib'}
    for key, name in files.items():
        path = MODELS / name
        if path.exists():
            loaded[key] = joblib.load(path)
        else:
            missing[key] = f'{name} not found. Run: python scripts/run_experiments.py'
    if (MODELS / 'bag_of_embeddings' / 'weights.pt').exists():
        from detector.models import NeuralDetector
        loaded['nn'] = NeuralDetector.load(MODELS / 'bag_of_embeddings')
    else:
        missing['nn'] = 'models/bag_of_embeddings not found. Run: python scripts/run_experiments.py'
    return loaded, missing


@st.cache_resource(show_spinner=False)
def load_gpt2():
    calibration_path = MODELS / 'zero_shot_calibration.json'
    if not calibration_path.exists():
        return None, 'Calibration file not found. Run: python scripts/run_experiments.py'
    try:
        from detector.lm_scores import GPT2Scorer
        scorer = GPT2Scorer(local_files_only=True)
    except Exception as error:          # weights not downloaded or broken cache: report, do not guess
        return None, (f'GPT-2 is not in the local Hugging Face cache ({type(error).__name__}). '
                      'Run once with internet: python scripts/compute_lm_scores.py')
    return (scorer, json.loads(calibration_path.read_text())), None


@st.cache_resource(show_spinner=False)
def load_transformer():
    folder = MODELS / 'distilroberta_lora'
    if not (folder / 'adapter_config.json').exists():
        return None, 'LoRA adapter not found. Run: python scripts/train_transformer.py --mode lora'
    try:
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        base = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_NAME, num_labels=2,
                                                                  local_files_only=True)
        model = PeftModel.from_pretrained(base, folder).eval()
        tokenizer = AutoTokenizer.from_pretrained(folder)
    except Exception as error:
        return None, (f'DistilRoBERTa could not be loaded from the local cache ({type(error).__name__}). '
                      'Run once with internet: python scripts/train_transformer.py --mode lora')
    calibration_path = MODELS / 'transformer_calibration.json'
    if not calibration_path.exists():
        return None, 'Calibration of the transformer not found. Run: python scripts/run_experiments.py --reuse'
    return (model, tokenizer, json.loads(calibration_path.read_text())), None


@st.cache_data(show_spinner=False)
def load_json(path_str):
    path = Path(path_str)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


# ------------------------------------------------------------------ prediction
def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def predict_all(text, local, gpt2, transformer):
    """P(ChatGPT) from every available detector + extra numbers for the explanation panels."""
    probs, extra = {}, {}
    for key in ('length', 'tfidf'):
        if key in local:
            probs[key] = float(local[key].predict_proba([text])[0, 1])
    feats = style.text_features(text)
    extra['features'] = feats
    for key in ('style_rf', 'style_lr'):
        if key in local:
            bundle = local[key]
            row = pd.DataFrame([{n: feats[n] for n in bundle['features']}])
            probs[key] = float(bundle['model'].predict_proba(row)[0, 1])
    if 'nn' in local:
        probs['nn'] = float(local['nn'].predict_proba([text])[0])
    if gpt2 is not None:
        scorer, calibration = gpt2
        scores = scorer.score_batch([text]).iloc[0]
        extra['perplexity'] = float(scores.perplexity)
        extra['fast_detectgpt'] = float(scores.fast_detectgpt)
        extra['ppl_threshold'] = calibration['ppl']['perplexity_threshold']
        for key in ('ppl', 'fdg'):
            cal = calibration[key]
            probs[key] = sigmoid(cal['coef'] * cal['sign'] * float(scores[cal['column']]) + cal['intercept'])
    if transformer is not None:
        import torch
        model, tokenizer, calibration = transformer
        with torch.no_grad():
            enc = tokenizer([text], truncation=True, max_length=256, return_tensors='pt')
            raw = min(max(float(torch.softmax(model(**enc).logits, dim=-1)[0, 1]), 1e-7), 1 - 1e-7)
        # Same Platt calibration as in the evaluation: raw probabilities after 1 epoch are shifted towards ChatGPT
        probs['roberta_lora'] = sigmoid(calibration['coef'] * math.log(raw / (1 - raw)) + calibration['intercept'])
    return probs, extra


def word_contributions(pipeline, text):
    """TF-IDF value x logistic-regression weight for every single word of the text."""
    vectorizer, classifier = pipeline[0], pipeline[-1]
    vector = vectorizer.transform([text])
    weights = classifier.coef_[0]
    scores = {}
    for word in set(re.findall(r'\b\w+\b', text.lower())):
        index = vectorizer.vocabulary_.get(word)
        if index is not None and vector[0, index] != 0:
            scores[word] = float(vector[0, index] * weights[index])
    return scores


def highlighted_html(text, scores, k=10):
    to_chatgpt = {w for w, s in sorted(scores.items(), key=lambda x: -x[1])[:k] if s > 0}
    to_human = {w for w, s in sorted(scores.items(), key=lambda x: x[1])[:k] if s < 0}
    parts = []
    for piece in re.split(r'(\b\w+\b)', text):
        safe = html.escape(piece)
        low = piece.lower()
        if low in to_chatgpt:
            parts.append(f'<span class="c">{safe}</span>')
        elif low in to_human:
            parts.append(f'<span class="h">{safe}</span>')
        else:
            parts.append(safe)
    return '<div class="highlight">' + ''.join(parts) + '</div>', to_chatgpt, to_human


def share(p):
    """Probability as text; never prints a misleading 100% or 0%."""
    if p > 0.999:
        return '> 99.9%'
    if p < 0.001:
        return '< 0.1%'
    return f'{p:.1%}'


def verdict(p):
    if p >= UNCERTAIN[1]:
        return 'chatgpt', 'Likely AI-generated (ChatGPT style)'
    if p <= UNCERTAIN[0]:
        return 'human', 'Likely human-written'
    return 'uncertain', 'Uncertain: the signals are mixed'


# ------------------------------------------------------------------ page
local_models, local_missing = load_local_models()
model_card = load_json(str(MODELS / 'model_card.json')) or {}
examples = [e for e in ((load_json(str(ROOT / 'demo' / 'examples.json')) or [])
                         + (load_json(str(ROOT / 'demo' / 'extra_examples.json')) or []))   # texts from outside HC3
            if isinstance(e, dict) and e.get('text')]
for e in examples:          # tolerate older files that have no "title" field
    e.setdefault('title', f"{e.get('label', 'Example')} · {e.get('domain', e.get('topic', ''))}")
results = load_json(str(FINAL / 'results.json'))

st.markdown('<div class="hero-title">Human or ChatGPT?</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Course project, AI Essentials (AITU, 2026). We compare classical machine learning, '
            'zero-shot language-model scores and a fine-tuned transformer for detecting ChatGPT answers, '
            'all trained and tested on the cleaned HC3 dataset.</div>', unsafe_allow_html=True)

tab_analyze, tab_live, tab_dashboard, tab_compare, tab_about = st.tabs(
    ['Analyze a text', 'Live: fool the detector', 'Analytics dashboard', 'HC3 figures', 'About the project'])

with tab_analyze:
    left, right = st.columns([5, 6], gap='large')
    with left:
        if examples:
            titles = ['Write or paste your own text'] + [e['title'] for e in examples]
            choice = st.selectbox('Example text (HC3 test set, never seen in training, or outside HC3)', titles,
                                  key='example_choice')
            if choice != st.session_state.get('last_example'):
                st.session_state['last_example'] = choice
                st.session_state['text'] = '' if choice == titles[0] else examples[titles.index(choice) - 1]['text']
        st.text_area('English text to analyze', key='text', height=300,
                     placeholder='Paste an answer, an essay paragraph or a comment (best: 50 words or more).')

        with st.expander('Detectors and options'):
            use_gpt2 = st.toggle('Use GPT-2 zero-shot detectors (slower, needs the local GPT-2 model)', value=True)
            use_transformer = st.toggle('Use the fine-tuned DistilRoBERTa + LoRA', value=True)
        gpt2, gpt2_error = (load_gpt2() if use_gpt2 else (None, 'switched off in the options'))
        transformer, transformer_error = (load_transformer() if use_transformer else (None, 'switched off in the options'))
        available = [k for k in MODEL_ORDER if k in local_models or (k in ('ppl', 'fdg') and gpt2 is not None)
                     or (k == 'roberta_lora' and transformer is not None)]
        if not available:
            st.error('No trained detector was found in models/. Run: python scripts/run_experiments.py')
            st.stop()
        best_default = max(available, key=lambda k: (model_card.get(k, {}).get('f1') or 0))
        main_key = st.selectbox('Main detector for the verdict', available, index=available.index(best_default),
                                format_func=lambda k: f"{model_card.get(k, {}).get('name', k)}"
                                                      + (f"  (test F1 {model_card[k]['f1']:.3f})" if k in model_card else ''))
        st.caption(DESCRIPTIONS.get(main_key, ''))
        analyze = st.button('Analyze', type='primary', width='stretch')

        status = {k: 'ready' for k in available}
        status.update({k: v for k, v in local_missing.items()})
        if gpt2 is None:
            status.update({'ppl': gpt2_error, 'fdg': gpt2_error})
        if transformer is None:
            status['roberta_lora'] = transformer_error
        with st.expander('Model status'):
            for key in MODEL_ORDER:
                name = model_card.get(key, {}).get('name', key)
                state = status.get(key, 'not available')
                st.markdown(f"**{name}**: {'ready' if state == 'ready' else 'not available. ' + state}")

    with right:
        text = dataset.normalise(st.session_state.get('text', '') or '')
        if analyze:
            n_words = len(text.split())
            if n_words < 5:
                st.session_state.pop('analysis', None)
                st.warning('Please paste at least 5 words. Very short texts cannot be judged reliably.')
            else:
                with st.spinner('Running the detectors...'):
                    probs, extra = predict_all(text, local_models, gpt2, transformer)
                st.session_state['analysis'] = {'text': text, 'probs': probs, 'extra': extra, 'main': main_key}

        analysis = st.session_state.get('analysis')
        if not analysis:
            st.markdown('<div class="card"><p class="verdict-label">Getting started</p>'
                        '<p class="verdict-title">Choose an example or paste a text</p>'
                        '<p class="verdict-meta">Then press Analyze. You will see the verdict of the main detector, '
                        'the opinion of every other detector, text statistics and the words that pushed '
                        'the decision.</p></div>', unsafe_allow_html=True)
        else:
            probs, extra, key = analysis['probs'], analysis['extra'], analysis['main']
            if key not in probs:
                key = max(probs, key=lambda k: (model_card.get(k, {}).get('f1') or 0))
            p = probs[key]
            kind, title = verdict(p)
            confidence = max(p, 1 - p)
            card = model_card.get(key, {})
            st.markdown(f"""
<div class="card verdict-{kind}">
  <p class="verdict-label">Analysis summary · main detector: {html.escape(card.get('name', key))}</p>
  <p class="verdict-title">{title}</p>
  <div class="prob-bar">
    <div style="width:{max(1 - p, 0.001) * 100:.1f}%; background:{HUMAN_COLOR}">{'Human ' + share(1 - p) if 1 - p >= 0.12 else ''}</div>
    <div style="width:{max(p, 0.001) * 100:.1f}%; background:{CHATGPT_COLOR}">{'ChatGPT ' + share(p) if p >= 0.12 else ''}</div>
  </div>
  <div class="prob-legend"><span>P(human) = {share(1 - p)}</span><span>P(ChatGPT) = {share(p)}</span></div>
  <p class="verdict-meta" style="margin-top:12px">Model confidence: {share(confidence)} (how sure the model is, not a guarantee).
  {'Test F1 of this detector: ' + format(card['f1'], '.3f') + '; humans wrongly flagged on our test set: ' + format(card['false_positive_rate'], '.1%') + '.' if card else ''}</p>
</div>""", unsafe_allow_html=True)

            n_words = int(extra['features']['n_words'])
            if n_words < SHORT_TEXT:
                st.warning(f'Short text ({n_words} words): few signals to judge by. In HC3 almost all answers under '
                           '50 words are human, and the few short ChatGPT answers were often missed by our detectors, '
                           'so treat this result with extra caution.')
            letters = re.findall(r'[^\W\d_]', analysis['text'])
            if letters and sum(ch.isascii() for ch in letters) / len(letters) < 0.7:
                st.warning('The text does not look like English. All detectors were trained on English HC3 answers only.')
            st.info('A detector gives a statistical estimate, not proof of authorship. It can be wrong, especially on '
                    'short, edited, translated or mixed human + AI texts, and on texts from newer models than '
                    'ChatGPT (2023). Never use it as the only evidence against a person.')

            st.subheader('Model comparison')
            rows = [{'Detector': model_card.get(k, {}).get('name', k), 'P(ChatGPT)': probs[k],
                     'Verdict': 'ChatGPT' if probs[k] >= 0.5 else 'Human',
                     'Test F1': model_card.get(k, {}).get('f1')} for k in MODEL_ORDER if k in probs]
            frame = pd.DataFrame(rows)
            votes = int((frame['P(ChatGPT)'] >= 0.5).sum())
            st.caption(f'{votes} of {len(frame)} detectors say ChatGPT. Detectors are sorted from the strongest '
                       'to the weakest on our test set.')
            bars = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, size=18).encode(
                x=alt.X('P(ChatGPT):Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%', title='P(ChatGPT)')),
                y=alt.Y('Detector:N', sort=None, title=None, axis=alt.Axis(labelLimit=260)),
                color=alt.Color('Verdict:N', scale=alt.Scale(domain=['Human', 'ChatGPT'],
                                                              range=[HUMAN_COLOR, CHATGPT_COLOR]),
                                legend=alt.Legend(orient='bottom', title=None)),
                tooltip=['Detector', alt.Tooltip('P(ChatGPT):Q', format='.1%'), 'Verdict',
                         alt.Tooltip('Test F1:Q', format='.3f')])
            rule = alt.Chart(pd.DataFrame({'x': [0.5]})).mark_rule(color='#898781', strokeDash=[4, 4]).encode(x='x:Q')
            st.altair_chart((bars + rule).properties(height=36 * len(frame) + 40), width='stretch')

            st.subheader('Text statistics')
            feats = extra['features']
            cols = st.columns(3)
            cols[0].metric('Words', f'{n_words}')
            cols[1].metric('Words per sentence', f"{feats['mean_sentence_len']:.1f}")
            cols[2].metric('Vocabulary (MATTR)', f"{feats['MATTR (vocabulary richness)']:.2f}",
                           help='Share of different words in a 50-word window (0-1).')
            cols = st.columns(3)
            if 'perplexity' in extra:
                cols[0].metric('GPT-2 perplexity', f"{extra['perplexity']:.1f}",
                               help=f"Below {extra['ppl_threshold']:.1f} looks like ChatGPT on our data. "
                                    'Lower = more predictable text.')
                cols[1].metric('Fast-DetectGPT', f"{extra['fast_detectgpt']:.2f}",
                               help='Higher = closer to what GPT-2 itself would write.')
            else:
                cols[0].metric('GPT-2 perplexity', 'not available')
            cols[2].metric('Sentence variation', f"{feats['burstiness (sentence-len std)']:.1f}",
                           help='Standard deviation of sentence length (humans are usually more "bursty").')

            reference = load_json(str(MODELS / 'style_reference.json'))
            if reference:
                st.subheader('Stylistic indicators')
                shown = ['connectives', 'first person I', 'contractions', 'brackets ( [', 'hedges',
                         'MATTR (vocabulary richness)', 'mean_sentence_len', 'long_words_% (7+ letters)']
                table = []
                for name in shown:
                    value, h, c = feats[name], reference[name]['human_median'], reference[name]['chatgpt_median']
                    closer = '—' if abs(h - c) < 1e-9 else ('ChatGPT' if abs(value - c) < abs(value - h) else 'Human')
                    table.append({'Indicator': SHORT_NAMES[name], 'This text': round(value, 2),
                                  'Human (typical)': round(h, 2), 'ChatGPT (typical)': round(c, 2), 'Closer to': closer})
                st.dataframe(pd.DataFrame(table), hide_index=True, width='stretch')
                st.caption('Counts are per 100 words. Typical = median on the training set. A single indicator '
                           'proves nothing; the models combine many of them.')

            if 'tfidf' in local_models:
                st.subheader('Words that pushed the TF-IDF model')
                block, to_c, to_h = highlighted_html(analysis['text'], word_contributions(local_models['tfidf'],
                                                                                          analysis['text']))
                st.markdown('<p class="note"><span style="background:#fadccd;border-radius:4px;padding:0 4px">orange'
                            '</span> pushes towards ChatGPT, <span style="background:#d7e6f8;border-radius:4px;'
                            'padding:0 4px">blue</span> towards human (top 10 words each way)</p>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="card">{block}</div>', unsafe_allow_html=True)

sys.path.insert(0, str(ROOT / 'demo'))

with tab_live:
    import live                            # noqa: E402
    live.render()

with tab_dashboard:
    import analytics                       # noqa: E402
    analytics.render()

with tab_compare:
    if not results:
        st.warning('reports/final/results.json not found. Run: python scripts/run_experiments.py')
    else:
        st.markdown(f"All detectors are evaluated on the **same test set of {results['meta']['n_test']:,} texts** "
                    '(questions never used for training or tuning). 95% confidence intervals come from a bootstrap '
                    'over questions.')
        rows = []
        for name, r in sorted(results['test_results'].items(), key=lambda x: -x[1]['f1']):
            rows.append({'Detector': name, 'F1': r['f1'], 'F1 95% CI': f"{r['f1_ci'][0]:.3f}–{r['f1_ci'][1]:.3f}",
                         'Accuracy': r['accuracy'], 'Precision': r['precision'], 'Recall': r['recall'],
                         'ROC-AUC': r['roc_auc'], 'Humans flagged (FPR)': r['false_positive_rate']})
        formats = {c: '{:.3f}' for c in ['F1', 'Accuracy', 'Precision', 'Recall', 'Humans flagged (FPR)']}
        formats['ROC-AUC'] = '{:.4f}'
        st.dataframe(pd.DataFrame(rows).style.format(formats),
                     hide_index=True, width='stretch')
        for file, caption in (('f1_comparison.png', 'F1 with 95% confidence intervals'),
                              ('cross_domain.png', 'Generalisation to a topic never seen in training'),
                              ('roc_curves.png', 'ROC curves'), ('confusion_matrices.png', 'Confusion matrices')):
            if (FIGURES / file).exists():
                st.image(str(FIGURES / file), caption=caption, width='stretch')

with tab_about:
    st.markdown("""
### What we study
Can we tell whether an answer was written by a **human** or by **ChatGPT**, and does the detector learn real
writing style rather than accidents of how the dataset was collected?

**Research questions**
1. Which approach works best: classical ML, style features, zero-shot language-model scores or a fine-tuned transformer?
2. How well do detectors work on a topic that was not in the training data?
3. Do hidden artifacts in the dataset inflate the scores?

### Data
HC3 (Guo et al., 2023): 24,322 questions with human and ChatGPT answers from 5 domains (Reddit ELI5, finance,
medicine, open QA, Wikipedia CS/AI). After cleaning we keep one human and one ChatGPT answer per question
(21,903 questions, 43,806 texts) and split **by question** into train / validation / test (80 / 10 / 10).

### The key finding: a shortcut in the data
Human answers in HC3 were stored tokenized (`word , word`, `do n't`), ChatGPT answers were not.
A one-line rule "space before a comma means human" is right for 91.8% of all HC3 answers without any AI.
After our cleaning the same rule falls to chance. A model trained on the raw text can learn this shortcut
instead of the writing style.
""")
    for file in ('bias_rule_accuracy.png', 'bias_shortcut_stress_test.png'):
        if (FIGURES / file).exists():
            st.image(str(FIGURES / file), width='stretch')
    st.markdown("""
### Limitations
* One generator (ChatGPT, early 2023) and English only; newer models write differently.
* Paraphrased, edited or mixed human + AI texts were not tested.
* Trained detectors lose accuracy on unseen topics; short ChatGPT answers are rare in HC3 and often missed.
* A false accusation harms a real person, so the output is a hint for a human decision, not a verdict.

Team project · MSc Applied AI, Astana IT University · code and results: `reports/final/`, `FINAL_PROJECT_GUIDE.md`.
""")
