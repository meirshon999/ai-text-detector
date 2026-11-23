"""RQ4 + RQ5: how the SAVED HC3 detectors behave on modern LLMs, attacks and mixed texts.

    python scripts/evaluate_modern.py           # assemble texts, score (cached), metrics, figures

Inputs  reports/final/modern/questions.parquet and raw/*.jsonl (from generate_claude.py / generate_open_models.py);
        sources that are not generated yet are skipped, so the script can run at any stage.
Outputs reports/final/modern/texts.parquet       every evaluation text with its source
        reports/final/modern/scores.parquet      P(machine) of every detector for every text (cache)
        reports/final/modern_results.json        all numbers
        reports/final/figures/modern_*.png       figures

Metrics (thresholds are the HC3 ones, nothing is re-tuned on the new texts):
    detection rate   share of machine texts flagged (P >= 0.5)  = recall / TPR
    false-positive   share of the human HC3 answers to the same questions that are flagged
    ROC-AUC          machine texts of one source vs the human answers to the same questions
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
from sklearn.metrics import roc_auc_score                           # noqa: E402

from detector import modern, plots                                  # noqa: E402
from detector.config import DOMAIN_NAMES, FIGURES, FINAL            # noqa: E402
from detector.inference import DETECTORS, NAMES, Detectors          # noqa: E402

SCORES = modern.MODERN / 'scores.parquet'
RESULTS = FINAL / 'modern_results.json'


def text_id(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def assemble():
    q = pd.read_parquet(modern.QUESTIONS_FILE)
    base = q[['qid', 'domain']]
    parts = [base.assign(source='human', y=0, text=q.human_text),
             base.assign(source='chatgpt_hc3', y=1, text=q.chatgpt_text),
             base.assign(source='typos', y=1, text=q.chatgpt_text.map(modern.add_typos))]
    for path in sorted((modern.MODERN / 'raw').glob('*.jsonl')):
        raw = pd.read_json(path, lines=True)
        raw = raw[~raw.source.str.startswith('train-')]
        if raw.empty:
            continue
        raw = raw.merge(base, on='qid')
        parts.append(raw.assign(y=1, text=raw.raw.map(modern.clean_generated))[['qid', 'domain', 'source', 'y', 'text']])
    texts = pd.concat(parts, ignore_index=True)
    mix = modern.mixed_texts(q)
    mix = mix.assign(source='mix', y=(mix.share > 0).astype(int))
    texts = pd.concat([texts, mix[['qid', 'domain', 'source', 'y', 'text', 'share', 'machine_share_real']]],
                      ignore_index=True)
    texts['n_words'] = texts.text.str.split().str.len()
    texts = texts[texts.n_words >= 5].reset_index(drop=True)
    texts['text_id'] = texts.text.map(text_id)
    texts.to_parquet(modern.TEXTS_FILE, index=False)
    return texts


def score(texts):
    cached = pd.read_parquet(SCORES) if SCORES.exists() else pd.DataFrame(columns=['text_id'])
    todo = texts[~texts.text_id.isin(cached.text_id)].drop_duplicates('text_id')
    if len(todo):
        print(f'Scoring {len(todo)} new texts with all detectors...', flush=True)
        new = Detectors().score(todo.text.tolist())
        new.insert(0, 'text_id', todo.text_id.to_numpy())
        cached = pd.concat([cached, new], ignore_index=True)
        cached.to_parquet(SCORES, index=False)
    return texts.merge(cached, on='text_id', how='left')


def rate(flags):
    value, ci = modern.bootstrap_rate(flags)
    return {'value': value, 'ci': ci}


def metrics(scored):
    human = scored[scored.source == 'human']
    sources = [s for s in modern.SOURCE_NAMES if s != 'human' and s in set(scored.source)]
    out = {'n_texts': {s: int((scored.source == s).sum()) for s in ['human'] + sources + ['mix']},
           'mean_words': {s: float(scored.loc[scored.source == s, 'n_words'].mean()) for s in ['human'] + sources},
           'false_positive_rate': {d: rate(human[d] >= 0.5) for d in DETECTORS},
           'detection_rate': {}, 'roc_auc': {}, 'by_domain': {}, 'mix': {}}
    for s in sources:
        machine = scored[scored.source == s]
        same_human = human[human.qid.isin(machine.qid)]
        out['detection_rate'][s] = {d: rate(machine[d] >= 0.5) for d in DETECTORS}
        out['roc_auc'][s] = {d: float(roc_auc_score(np.r_[np.zeros(len(same_human)), np.ones(len(machine))],
                                                     np.r_[same_human[d], machine[d]])) for d in DETECTORS}
        out['by_domain'][s] = {d: {dom: float((g[d] >= 0.5).mean()) for dom, g in machine.groupby('domain')}
                               for d in DETECTORS}
    mix = scored[scored.source == 'mix']
    for d in DETECTORS:
        g = mix.groupby('share')[d]
        out['mix'][d] = {'share': [float(x) for x in g.mean().index], 'mean_prob': g.mean().round(4).tolist(),
                         'flagged': g.apply(lambda v: float((v >= 0.5).mean())).round(4).tolist()}
    out['mix_n_questions'] = int(mix.qid.nunique())
    return out


def figures(res):
    plots.setup_style()
    rows = [NAMES[d] for d in DETECTORS]
    gens = [s for s in modern.GENERATORS if s in res['detection_rate']]
    attacks = [s for s in modern.ATTACKS if s in res['detection_rate']]
    short = {'chatgpt_hc3': 'ChatGPT\n(HC3, seen)', 'claude-haiku-5-5': 'Claude\nHaiku 5.5',
             'claude-sonnet-5-5': 'Claude\nSonnet 5.5', 'claude-opus-5-5': 'Claude\nOpus 5.5',
             'qwen2.5-1.5b': 'Qwen2.5\n1.5B', 'smollm2-1.7b': 'SmolLM2\n1.7B', 'paraphrase': 'Paraphrased',
             'humanize': '"Humanized"', 'typos': '5% typos'}
    matrix = [[res['false_positive_rate'][d]['value']] + [res['detection_rate'][s][d]['value'] for s in gens]
              for d in DETECTORS]
    fpr_col = np.array(matrix)[:, :1]
    # the human column is "bad when high": show 1 - FPR colour but print the FPR itself
    plots.rate_heatmap(np.array(matrix)[:, 1:], rows, [short[s] for s in gens], FIGURES / 'modern_generators.png',
                       'Share of machine texts detected, by generator (threshold from HC3)',
                       f'Same test questions for every column. False-positive rate on the human answers: '
                       + ', '.join(f'{NAMES[d]} {v:.0%}' for d, v in zip(DETECTORS, fpr_col[:, 0])))
    plots.rate_heatmap([[res['roc_auc'][s][d] for s in gens] for d in DETECTORS], rows, [short[s] for s in gens],
                       FIGURES / 'modern_generators_auc.png',
                       'ROC-AUC by generator: machine texts vs the human answers to the same questions',
                       'ROC-AUC does not depend on the threshold: 0.5 = chance, 1.0 = perfect ranking.',
                       fmt='{:.2f}')
    plots.rate_heatmap([[res['detection_rate'][s][d]['value'] for s in attacks] for d in DETECTORS], rows,
                       [short[s] for s in attacks], FIGURES / 'modern_attacks.png',
                       'Robustness: share of ChatGPT texts still detected after an attack',
                       'Same 555 ChatGPT answers in every column; only the attack changes.')
    keys = ['roberta_lora', 'tfidf', 'nn', 'style_rf', 'ppl', 'fdg']
    plots.share_curves(res['mix']['tfidf']['share'], {NAMES[d]: res['mix'][d]['mean_prob'] for d in keys},
                       FIGURES / 'modern_mixed.png',
                       f'Mixed texts: human beginning + ChatGPT continuation ({res["mix_n_questions"]} questions)',
                       'Share of the text written by ChatGPT, %', 'Mean P(machine)')


def main():
    texts = assemble()
    print(texts.groupby('source').size().to_string())
    scored = score(texts)
    res = metrics(scored)
    RESULTS.write_text(json.dumps(res, indent=2))
    figures(res)
    print('\nDetection rate (P >= 0.5) by source:')
    table = pd.DataFrame({s: {NAMES[d]: v['value'] for d, v in r.items()} for s, r in res['detection_rate'].items()})
    table.insert(0, 'human FPR', {NAMES[d]: v['value'] for d, v in res['false_positive_rate'].items()})
    print(table.round(3).to_string())
    print(f'\nSaved {RESULTS} and figures modern_*.png')


if __name__ == '__main__':
    main()
