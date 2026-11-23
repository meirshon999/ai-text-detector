"""Live tab: ask a 2026 model a question, watch the old and the retrained detector judge the answer,
see which sentences look machine-written, then "humanize" the text and try again.

Live mode calls the Claude API (key in .env, never shown). Without a key or internet the tab switches to
saved texts from reports/final/modern/ so the demo still works at the defense.
"""
import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import joblib                                    # noqa: E402
import numpy as np                               # noqa: E402
import pandas as pd                              # noqa: E402
import streamlit as st                           # noqa: E402

from detector import modern                      # noqa: E402
from detector.config import CHATGPT_COLOR, HUMAN_COLOR, MODELS, TRANSFORMER_NAME   # noqa: E402
from detector.segments import split_boundary, split_sentences, split_texts       # noqa: E402

MODELS_LIVE = {'claude-sonnet-5-5': 'Claude Sonnet 5.5', 'claude-opus-5-5': 'Claude Opus 5.5',
               'claude-haiku-5-5': 'Claude Haiku 5.5'}
REWRITE_MODEL = 'claude-sonnet-5-5'          # the same rewriter as in the attack experiment
CONTINUE_PROMPT = ('Here is the beginning of an answer to the question below. Continue it for about 120 more '
                   'words in plain prose, without repeating the beginning and without Markdown. Output only the '
                   'continuation.\n\n<question>\n{question}\n</question>\n\n<beginning>\n{start}\n</beginning>')
DETECTOR_LABELS = {'old': 'Trained on ChatGPT 2023 (HC3)', 'new': 'Retrained with Claude Haiku texts'}
SAMPLE_QUESTIONS = ['Why do we get goosebumps when we are cold or scared?',
                    'Should I pay off my student loans early or invest the money?',
                    'What is the difference between a virus and a bacterial infection?',
                    'How does a neural network learn from data?']


# ------------------------------------------------------------------ models (cached)
@st.cache_resource(show_spinner=False)
def _lora(folder_name, calibration_file):
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    folder = MODELS / folder_name
    base = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_NAME, num_labels=2, local_files_only=True)
    model = PeftModel.from_pretrained(base, folder).eval()
    tokenizer = AutoTokenizer.from_pretrained(folder)
    return model, tokenizer, json.loads((MODELS / calibration_file).read_text())


@st.cache_resource(show_spinner=False)
def _tfidf(path):
    return joblib.load(MODELS / path)


def detectors_ready():
    need = [MODELS / 'distilroberta_lora' / 'adapter_config.json',
            MODELS / 'distilroberta_lora_augmented' / 'adapter_config.json',
            MODELS / 'distilroberta_lora_augmented' / 'calibration.json']
    return all(p.exists() for p in need)


def lora_probs(which, texts, max_length=256):
    import torch
    model, tokenizer, cal = (_lora('distilroberta_lora', 'transformer_calibration.json') if which == 'old' else
                             _lora('distilroberta_lora_augmented', 'distilroberta_lora_augmented/calibration.json'))
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), 16):
            enc = tokenizer(texts[start:start + 16], truncation=True, max_length=max_length, padding=True,
                            return_tensors='pt')
            out += torch.softmax(model(**enc).logits, dim=-1)[:, 1].tolist()
    raw = np.clip(np.array(out), 1e-7, 1 - 1e-7)
    return 1 / (1 + np.exp(-(cal['coef'] * np.log(raw / (1 - raw)) + cal['intercept'])))


def judge(text):
    """P(AI) from the old and the retrained detectors (LoRA as the main one, TF-IDF for reference)."""
    res = {'old': float(lora_probs('old', [text])[0]), 'new': float(lora_probs('new', [text])[0])}
    res['tfidf_old'] = float(_tfidf('tfidf_logreg.joblib').predict_proba([text])[0, 1])
    res['tfidf_new'] = float(_tfidf('augmented/tfidf_logreg.joblib').predict_proba([text])[0, 1])
    return res


def sentence_map(text, whole, which='new', min_jump=0.3):
    """Split method (best in scripts/evaluate_segments.py): for every possible boundary the retrained detector
    judges the whole part before and the whole part after it; the boundary is the largest jump in P(AI).
    Returns sentences, one P(AI) per sentence (the score of the part it belongs to) and the boundary or None."""
    sents = split_sentences(text)
    if len(sents) < 3:
        return sents, np.full(len(sents), whole), None
    before, after = split_texts(sents)
    p_before, p_after = lora_probs(which, before), lora_probs(which, after)
    k, jump = split_boundary(p_before, p_after)
    if jump < min_jump:            # no clear human -> AI change: colour the whole text by its own score
        return sents, np.full(len(sents), whole), None
    return sents, np.array([p_before[k - 1]] * k + [p_after[k - 1]] * (len(sents) - k)), k


# ------------------------------------------------------------------ Claude
def api_ready():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / '.env')
    except ImportError:
        pass
    return bool(os.environ.get('ANTHROPIC_API_KEY'))


def claude_stream(model, content, system=None, max_tokens=1500):
    import anthropic
    api_ready()                    # loads .env (the key is never printed)
    params = dict(model=model, max_tokens=max_tokens, output_config={'effort': 'low'},
                  messages=[{'role': 'user', 'content': content}])
    if system:
        params['system'] = system
    with anthropic.Anthropic().messages.stream(**params) as stream:
        for piece in stream.text_stream:
            yield piece


def api_error(error):
    if 'usage limit' in str(error).lower():
        return ('The API key has reached its spending limit (raise it in the Anthropic Console under Limits). '
                'Switch to Offline mode to use the saved texts.')
    return f'Claude API call failed ({type(error).__name__}). Switch to Offline mode to use the saved texts.'


def run_claude(model, content, system=None):
    """Streams the answer into the page and returns the cleaned text (same cleaning as the experiments)."""
    box = st.empty()
    text = ''
    for piece in claude_stream(model, content, system):
        text += piece
        box.markdown(f'<div class="card"><div class="highlight">{html.escape(text)}▌</div></div>',
                     unsafe_allow_html=True)
    box.empty()
    return modern.clean_generated(text)


@st.cache_data(show_spinner=False)
def saved_texts():
    """Offline mode: the texts generated for the experiments (same test questions)."""
    q = pd.read_parquet(modern.QUESTIONS_FILE)
    raw = pd.read_json(modern.MODERN / 'raw' / 'claude.jsonl', lines=True)
    raw = raw[~raw.source.str.startswith('train-')].assign(text=lambda d: d.raw.map(modern.clean_generated))
    wide = raw.pivot_table(index='qid', columns='source', values='text', aggfunc='first')
    q = q.merge(wide, left_on='qid', right_index=True, how='inner')
    q = q[q.question.str.len().between(25, 160)]
    return q.reset_index(drop=True)


# ------------------------------------------------------------------ rendering helpers
def _mix(a, b, t):
    a, b = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(x + (y - x) * t):02x}' for x, y in zip(a, b))


def verdict_card(title, p, note=''):
    ai = p >= 0.5
    color = CHATGPT_COLOR if ai else HUMAN_COLOR
    word = 'AI-generated' if ai else 'Human-written'
    return (f'<div class="card" style="border:2px solid {color}; text-align:left">'
            f'<p class="verdict-label">{html.escape(title)}</p>'
            f'<p style="font-size:44px;line-height:52px;font-weight:700;margin:4px 0;color:{color}">{p:.0%}</p>'
            f'<p class="verdict-title" style="font-size:22px !important;line-height:28px !important">{word}</p>'
            f'<p class="verdict-meta">{note}</p></div>')


def highlighted(sents, probs, boundary, true_boundary=None):
    parts = []
    for i, (s, p) in enumerate(zip(sents, probs)):
        if boundary is not None and i == boundary:
            parts.append('<span style="display:inline-block;background:#0b0b0b;color:#fff;border-radius:6px;'
                         'padding:0 8px;margin:0 4px;font-size:13px;font-weight:600">AI starts here?</span>')
        if true_boundary is not None and i == true_boundary:
            parts.append('<span style="display:inline-block;border:1.5px dashed #0b0b0b;border-radius:6px;'
                         'padding:0 8px;margin:0 4px;font-size:13px">human part ends</span>')
        bg = _mix('#d7e6f8', '#f7b89a', float(p))
        parts.append(f'<span title="P(AI) = {p:.0%}" style="background:{bg};border-radius:4px;padding:1px 3px;'
                     f'box-decoration-break:clone">{html.escape(s)}</span> ')
    legend = ('<p class="note">Where does the AI part start? The retrained detector scores every possible '
              'split of the text; each part is coloured by its score: '
              '<span style="background:#d7e6f8;border-radius:4px;padding:0 4px">human-like</span> → '
              '<span style="background:#f7b89a;border-radius:4px;padding:0 4px">AI-like</span>. '
              'Hover a sentence to see its score.</p>')
    return legend + '<div class="card"><div class="highlight" style="line-height:32px">' + ''.join(parts) + '</div></div>'


def show_result(text, true_boundary=None, key='main'):
    res = judge(text)
    c1, c2 = st.columns(2)
    c1.markdown(verdict_card(DETECTOR_LABELS['old'], res['old'],
                             f"DistilRoBERTa + LoRA, F1 0.990 on HC3 · TF-IDF: {res['tfidf_old']:.0%}"),
                unsafe_allow_html=True)
    c2.markdown(verdict_card(DETECTOR_LABELS['new'], res['new'],
                             f"Same model + 4,499 Claude Haiku texts · TF-IDF: {res['tfidf_new']:.0%}"),
                unsafe_allow_html=True)
    sents, probs, boundary = sentence_map(text, res['new'])
    if len(sents) >= 2:
        st.markdown(highlighted(sents, probs, boundary, true_boundary), unsafe_allow_html=True)
        if true_boundary is not None:
            if boundary is None:
                st.caption('The detector did not find a clear human → AI boundary in this text.')
            else:
                off = abs(boundary - true_boundary)
                st.caption(f'The human part ends before sentence {true_boundary + 1}; the detector puts the boundary before '
                           f'sentence {boundary + 1} ' + ('(exact).' if off == 0 else f'({off} sentence(s) off).'))
    return res


# ------------------------------------------------------------------ page
def render():
    st.markdown('### Live: can a 2026 model fool our detector?')
    st.markdown('<p class="note">Ask any question. A current Claude model answers it, our detector trained on '
                'ChatGPT 2023 and the same detector retrained with one modern model judge the answer, and the text '
                'is coloured sentence by sentence. Then one click asks Claude to make it sound human.</p>',
                unsafe_allow_html=True)
    if not detectors_ready():
        st.error('The retrained LoRA adapter is missing. Run: python scripts/train_transformer.py --mode lora '
                 '--augment 1500, then python scripts/train_augmented.py')
        return
    live_ok = api_ready()
    mode = st.radio('Mode', ['Live (Claude API)', 'Offline (saved texts)'], index=0 if live_ok else 1,
                    horizontal=True, key='live_mode',
                    help='Offline mode uses the texts generated for the experiments, no internet needed.')
    if mode.startswith('Live') and not live_ok:
        st.warning('No ANTHROPIC_API_KEY found in .env, switching to saved texts.')
        mode = 'Offline'

    state = st.session_state
    if mode.startswith('Live'):
        c1, c2 = st.columns([3, 1])
        question = c1.text_input('Question', value=state.get('live_q', SAMPLE_QUESTIONS[0]), key='live_q_input')
        model = c2.selectbox('Model', list(MODELS_LIVE), format_func=MODELS_LIVE.get, key='live_model')
        start = st.text_area('Optional: write the first sentences yourself, Claude will continue (mixed text)',
                             key='live_start', height=90,
                             placeholder='e.g. Honestly I never understood this myself until my biology teacher...')
        if st.button('Ask Claude', type='primary'):
            state['live_q'] = question
            try:
                with st.spinner(f'{MODELS_LIVE[model]} is writing...'):
                    if start.strip():
                        own = modern.clean_generated(start.strip())
                        cont = run_claude(model, CONTINUE_PROMPT.format(question=question, start=own))
                        state['live'] = {'text': f'{own} {cont}', 'boundary': len(split_sentences(own)),
                                         'who': f'You + {MODELS_LIVE[model]}'}
                    else:
                        ans = run_claude(model, question, modern.ANSWER_SYSTEM)
                        state['live'] = {'text': ans, 'boundary': None, 'who': MODELS_LIVE[model]}
                    state.pop('live_human', None)
            except Exception as error:          # no internet, quota, etc.: say so, never fake
                st.error(api_error(error))
    else:
        saved = saved_texts()
        idx = st.selectbox('Question (HC3 test set, never used for training)', range(len(saved)),
                           format_func=lambda i: saved.question.iloc[i][:120], key='off_q')
        row = saved.iloc[idx]
        sources = {'claude-sonnet-5-5': 'Claude Sonnet 5.5', 'claude-opus-5-5': 'Claude Opus 5.5',
                   'claude-haiku-5-5': 'Claude Haiku 5.5', 'chatgpt_text': 'ChatGPT 2023 (HC3)',
                   'human_text': 'Human answer (HC3)', 'mix': 'Mixed: human start + Claude Sonnet 5.5'}
        src = st.selectbox('Author', list(sources), format_func=sources.get, key='off_src')
        if st.button('Show answer', type='primary'):
            if src == 'mix':      # first ~60 words of the human answer, then Claude's answer to the same question
                own, words = [], 0
                for s in split_sentences(row.human_text):
                    if words >= 60:
                        break
                    own.append(s)
                    words += len(s.split())
                cont = split_sentences(row['claude-sonnet-5-5'])[:5]
                state['live'] = {'text': ' '.join(own + cont), 'boundary': len(own), 'who': sources[src],
                                 'qid': row.qid, 'src': src}
            else:
                state['live'] = {'text': row[src], 'boundary': None, 'who': sources[src], 'qid': row.qid,
                                 'src': src}
            state.pop('live_human', None)

    live = state.get('live')
    if not live:
        return
    st.markdown(f'#### Answer by {html.escape(live["who"])} · {len(live["text"].split())} words')
    before = show_result(live['text'], live.get('boundary'))

    st.markdown('#### Attack: make it sound human')
    can_offline = live.get('src') == 'chatgpt_text'
    if mode.startswith('Live'):
        go = st.button('Humanize with Claude Sonnet 5.5')
    else:
        go = st.button('Show the saved "humanized" version', disabled=not can_offline,
                       help='Offline, a saved humanized version exists for the ChatGPT 2023 answers.')
    if go:
        try:
            if mode.startswith('Live'):
                with st.spinner('Rewriting...'):
                    state['live_human'] = run_claude(REWRITE_MODEL, modern.HUMANIZE_PROMPT.format(text=live['text']))
            else:
                saved = saved_texts()
                state['live_human'] = saved.set_index('qid').loc[live['qid'], 'humanize']
        except Exception as error:
            st.error(api_error(error))
    if state.get('live_human'):
        after = show_result(state['live_human'], key='human')
        rows = pd.DataFrame({'Detector': [DETECTOR_LABELS['old'], DETECTOR_LABELS['new']],
                             'Original': [before['old'], before['new']],
                             'Humanized': [after['old'], after['new']]})
        st.dataframe(rows.style.format({'Original': '{:.0%}', 'Humanized': '{:.0%}'}), hide_index=True,
                     width='stretch')
        st.caption('On 555 test questions the HC3 detector caught 7% of humanized texts, the retrained one 94% '
                   '(threshold 50%). One live example can differ from these averages.')
