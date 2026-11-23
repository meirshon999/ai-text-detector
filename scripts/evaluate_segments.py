"""Can a detector show WHERE the AI part of a text starts? (sentence-level localisation)

    python scripts/evaluate_segments.py

Texts: the same test questions as in evaluate_modern.py. Each text = the human HC3 answer cut to
25 / 50 / 75% of ~150 words, continued by a machine answer to the same question (ChatGPT from HC3 or
Claude Sonnet 5.5). Every sentence keeps its true label, so we can measure:

    sentence ROC-AUC     does the per-sentence score separate human and machine sentences?
    boundary exact / ±1  is the predicted human -> AI boundary on the right sentence (or one off)?

Two ways to use a whole-text detector:
    window   score every sentence with its neighbours (radius 0 or 1), then fit a step human -> AI
    split    for every possible boundary score the whole part before and the whole part after it,
             and take the boundary with the largest jump P(after) - P(before) (longer texts = surer scores)
Baseline for the boundary: always guess the middle sentence.

Outputs: reports/final/segments_results.json, reports/final/figures/segments_boundary.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import joblib                                                     # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402

from detector import modern, plots                                # noqa: E402
from detector.config import FIGURES, FINAL, MODELS                # noqa: E402
from detector.inference import Detectors                          # noqa: E402
from detector.segments import (change_point, sentence_windows, split_boundary, split_sentences,   # noqa: E402
                               split_texts)

RESULTS = FINAL / 'segments_results.json'
SHARES = [0.25, 0.5, 0.75]
N_WORDS = 150
RADII = [0, 1]
DETECTORS = {'tfidf': 'TF-IDF (HC3)', 'tfidf_aug': 'TF-IDF (+ Haiku)',
             'roberta_lora': 'LoRA (HC3)', 'roberta_lora_aug': 'LoRA (+ Haiku)'}
CONTINUATIONS = {'chatgpt_hc3': 'ChatGPT 2023', 'claude-sonnet-5-5': 'Claude Sonnet 5.5'}


def build_texts():
    q = pd.read_parquet(modern.QUESTIONS_FILE)
    raw = pd.read_json(modern.MODERN / 'raw' / 'claude.jsonl', lines=True)
    sonnet = raw[raw.source == 'claude-sonnet-5-5'].assign(text=lambda d: d.raw.map(modern.clean_generated))
    sonnet = sonnet.set_index('qid').text
    rows = []
    for r in q.itertuples():
        human = split_sentences(r.human_text)
        if sum(len(s.split()) for s in human) < N_WORDS * 0.75:
            continue
        for cont, machine_text in (('chatgpt_hc3', r.chatgpt_text), ('claude-sonnet-5-5', sonnet.get(r.qid))):
            machine = split_sentences(machine_text or '')
            if len(machine) < 3:
                continue
            for share in SHARES:
                sents, labels, words = [], [], 0
                for s in human:
                    if words >= N_WORDS * (1 - share):
                        break
                    sents.append(s); labels.append(0); words += len(s.split())
                for s in machine:
                    if words >= N_WORDS:
                        break
                    sents.append(s); labels.append(1); words += len(s.split())
                if 0 < sum(labels) < len(labels):
                    rows.append({'qid': r.qid, 'continuation': cont, 'share': share, 'sentences': sents,
                                 'labels': labels})
    return pd.DataFrame(rows)


def score_windows(texts):
    """{(detector, radius): list of per-sentence probability arrays, one per text}."""
    flat = {r: [w for s in texts.sentences for w in sentence_windows(list(s), r)] for r in RADII}
    sizes = texts.sentences.map(len).tolist()
    tf = {'tfidf': joblib.load(MODELS / 'tfidf_logreg.joblib'),
          'tfidf_aug': joblib.load(MODELS / 'augmented' / 'tfidf_logreg.joblib')}
    det = Detectors(use_lm=False, use_transformer=False)
    aug_folder = MODELS / 'distilroberta_lora_augmented'
    out = {}
    for r in RADII:
        probs = {k: m.predict_proba(flat[r])[:, 1] for k, m in tf.items()}
        print(f'radius {r}: {len(flat[r])} windows, scoring LoRA (HC3) ...', flush=True)
        probs['roberta_lora'] = det.transformer_probs(flat[r], progress=False, max_length=128)
        print(f'radius {r}: scoring LoRA (+ Haiku) ...', flush=True)
        probs['roberta_lora_aug'] = det.transformer_probs(
            flat[r], progress=False, max_length=128, folder=aug_folder,
            calibration=json.loads((aug_folder / 'calibration.json').read_text()))
        for k, p in probs.items():
            out[(k, r)] = np.split(np.asarray(p), np.cumsum(sizes)[:-1])
    return out


def score_splits(texts):
    """{detector: list of (p_before, p_after) per text} for the split method."""
    befores, afters = zip(*[split_texts(list(s)) for s in texts.sentences])
    sizes = [len(b) for b in befores]
    flat_b, flat_a = [t for b in befores for t in b], [t for a in afters for t in a]
    tf = {'tfidf': joblib.load(MODELS / 'tfidf_logreg.joblib'),
          'tfidf_aug': joblib.load(MODELS / 'augmented' / 'tfidf_logreg.joblib')}
    det = Detectors(use_lm=False, use_transformer=False)
    aug_folder = MODELS / 'distilroberta_lora_augmented'
    aug_cal = json.loads((aug_folder / 'calibration.json').read_text())
    out = {}
    for k in DETECTORS:
        print(f'split method: scoring {len(flat_b) * 2} parts with {k} ...', flush=True)
        if k in tf:
            pb, pa = tf[k].predict_proba(flat_b)[:, 1], tf[k].predict_proba(flat_a)[:, 1]
        else:
            kw = {} if k == 'roberta_lora' else {'folder': aug_folder, 'calibration': aug_cal}
            pb = det.transformer_probs(flat_b, progress=False, **kw)
            pa = det.transformer_probs(flat_a, progress=False, **kw)
        cuts = np.cumsum(sizes)[:-1]
        out[k] = list(zip(np.split(np.asarray(pb), cuts), np.split(np.asarray(pa), cuts)))
    return out


def metrics(texts, scores, splits):
    res = {'n_texts': int(len(texts)), 'n_questions': int(texts.qid.nunique()),
           'mean_sentences': float(texts.sentences.map(len).mean()), 'results': {}}
    truth = texts.labels.map(lambda l: l.index(1)).to_numpy()
    n = texts.sentences.map(len).to_numpy()
    middle = n // 2
    for cont in CONTINUATIONS:
        mask = (texts.continuation == cont).to_numpy()
        base = np.abs(middle[mask] - truth[mask])
        block = {'baseline_middle': {'exact': float((base == 0).mean()), 'within1': float((base <= 1).mean())}}
        for (k, r), per_text in scores.items():
            idx = np.where(mask)[0]
            y = np.concatenate([texts.labels.iloc[i] for i in idx])
            p = np.concatenate([per_text[i] for i in idx])
            words = [[len(s.split()) for s in texts.sentences.iloc[i]] for i in idx]
            pred = np.array([change_point(per_text[i], w) for i, w in zip(idx, words)])
            err = np.abs(pred - truth[idx])
            block[f'{k}_r{r}'] = {'sentence_auc': float(roc_auc_score(y, p)),
                                  'sentence_acc': float(((p >= 0.5) == y).mean()),
                                  'exact': float((err == 0).mean()), 'within1': float((err <= 1).mean()),
                                  'mean_error': float(err.mean())}
        for k, per_text in splits.items():
            idx = np.where(mask)[0]
            pred = np.array([split_boundary(*per_text[i])[0] for i in idx])
            err = np.abs(pred - truth[idx])
            block[f'{k}_split'] = {'exact': float((err == 0).mean()), 'within1': float((err <= 1).mean()),
                                   'mean_error': float(err.mean())}
        res['results'][cont] = block
    return res


def main():
    texts = build_texts()
    print(f'{len(texts)} mixed texts, {texts.qid.nunique()} questions, '
          f'{texts.sentences.map(len).mean():.1f} sentences on average')
    res = metrics(texts, score_windows(texts), score_splits(texts))
    RESULTS.write_text(json.dumps(res, indent=2))
    table = pd.DataFrame({f'{c} / {k}': v for c, b in res['results'].items() for k, v in b.items()}).T
    print(table.round(3).to_string())

    methods = [f'r{r}' for r in RADII] + ['split']
    best_r = {k: max(methods, key=lambda m: np.mean([res['results'][c][f'{k}_{m}']['within1'] for c in CONTINUATIONS]))
              for k in DETECTORS}
    plots.setup_style()
    series = {CONTINUATIONS[c]: [100 * res['results'][c]['baseline_middle']['within1']]
              + [100 * res['results'][c][f'{k}_{best_r[k]}']['within1'] for k in DETECTORS] for c in CONTINUATIONS}
    plots.grouped_bars(['Guess the middle'] + list(DETECTORS.values()), series, FIGURES / 'segments_boundary.png',
                       'Human -> AI boundary found within one sentence', '% of mixed texts',
                       ['#9ec5f4', '#256abf'])
    print('best method per detector:', best_r)


if __name__ == '__main__':
    main()
