"""Fix attempt: retrain the classical detectors on HC3 + texts of ONE modern generator (Claude Haiku 5.5).

    python scripts/train_augmented.py              # retrain + evaluate (~15 min)
    python scripts/train_augmented.py --figures    # only redraw the figures

Training data = HC3 train (unchanged) + up to 4,500 Claude Haiku 5.5 texts written for TRAIN questions only
(answers, paraphrases and "humanized" rewrites of the HC3 ChatGPT answers) + the same number of EXTRA human
answers to train questions (HC3 has several human answers per question; the pairs used only one), so the
classes stay balanced and the human texts come from the same questions and domains.

Evaluation: HC3 test (must not get worse) and the modern sets. Claude Sonnet 5.5, Claude Opus 5.5,
Qwen2.5 and SmolLM2 are never used for training, so they measure transfer to UNSEEN generators.

Outputs: models/augmented/*, reports/final/augmented_results.json, reports/final/figures/augmented_*.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import joblib                                                       # noqa: E402
import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score                 # noqa: E402

from detector import data, features, models, modern, plots          # noqa: E402
from detector.config import CACHE, FIGURES, FINAL, MODELS, SEED            # noqa: E402
from detector.inference import NAMES                                # noqa: E402

OUT_MODELS = MODELS / 'augmented'
RESULTS = FINAL / 'augmented_results.json'
KEYS = ['tfidf', 'nn', 'style_rf', 'style_lr']        # + roberta_lora if the augmented adapter exists


def augmentation_texts(train):
    raw = pd.read_json(modern.MODERN / 'raw' / 'claude.jsonl', lines=True)
    raw = raw[raw.source.str.startswith('train-')]
    raw = raw.assign(text_clean=raw.raw.map(modern.clean_generated))
    raw = raw[raw.text_clean.str.split().str.len() >= 5]
    machine = raw.merge(train[['qid', 'domain']].drop_duplicates('qid'), on='qid')
    # extra human answers to the same TRAIN questions, never used in the pairs
    long_clean, _ = data.clean(data.load_raw())
    used = set(data.load_pairs().text_clean)
    pool = long_clean[(long_clean.label == 'human') & long_clean.qid.isin(set(train.qid))
                      & ~long_clean.text_clean.isin(used)]
    human = pool.sample(min(len(machine), len(pool)), random_state=SEED)
    print(f'augmentation: {len(machine)} Claude Haiku texts ({machine.source.value_counts().to_dict()}), '
          f'{len(human)} extra human answers')
    extra = pd.concat([machine.assign(y=1)[['qid', 'domain', 'text_clean', 'y']],
                       human.assign(y=0)[['qid', 'domain', 'text_clean', 'y']]], ignore_index=True)
    return extra.assign(split='train', source=np.where(extra.y == 1, 'claude-haiku-train', 'human-extra'))


def fit(train, val):
    chosen = json.loads((FINAL / 'results.json').read_text())['tuning']
    tf = chosen['tfidf']['chosen']
    style_names = chosen['style_rf']['features']
    fitted = {'tfidf': models.tfidf_logreg(C=tf['C'], ngram_range=tuple(tf['ngram_range'])).fit(train.text_clean,
                                                                                                    train.y)}
    s_train = features.feature_table(train.text_clean)
    rf = models.stylometry_forest(chosen['style_rf']['chosen']['min_samples_leaf']).fit(s_train[style_names], train.y)
    lr = models.stylometry_logreg().fit(s_train[style_names], train.y)
    fitted['style_rf'] = {'features': style_names, 'model': rf}
    fitted['style_lr'] = {'features': style_names, 'model': lr}
    fitted['nn'] = models.NeuralDetector().fit(train.text_clean.tolist(), train.y.values, val.text_clean.tolist(),
                                               val.y.values, verbose=False)
    return fitted


def predict(fitted, texts):
    texts = list(texts)
    out = {'tfidf': fitted['tfidf'].predict_proba(texts)[:, 1], 'nn': fitted['nn'].predict_proba(texts)}
    feats = features.feature_table(texts)
    for key in ('style_rf', 'style_lr'):
        out[key] = fitted[key]['model'].predict_proba(feats[fitted[key]['features']])[:, 1]
    return pd.DataFrame(out)


def save(fitted):
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted['tfidf'], OUT_MODELS / 'tfidf_logreg.joblib', compress=3)
    joblib.dump(fitted['style_rf'], OUT_MODELS / 'stylometry_rf.joblib', compress=3)
    joblib.dump(fitted['style_lr'], OUT_MODELS / 'stylometry_logreg.joblib')
    fitted['nn'].save(OUT_MODELS / 'bag_of_embeddings')


def before_hc3_scores(test):
    """Calibrated test probabilities of the ORIGINAL LoRA model (as in results.json)."""
    cached = pd.read_parquet(CACHE / 'transformer_lora_scores.parquet').query('split == "test"')
    cal = json.loads((MODELS / 'transformer_calibration.json').read_text())
    p = cached.set_index(['qid', 'label']).p_chatgpt.clip(1e-7, 1 - 1e-7)
    p = 1 / (1 + np.exp(-(cal['coef'] * np.log(p / (1 - p)) + cal['intercept'])))
    return test.set_index(['qid', 'label']).index.map(p).to_numpy()


def lora_augmented(res, test, before_test, texts, before):
    """DistilRoBERTa + LoRA retrained with `train_transformer.py --mode lora --augment N` (if it exists)."""
    folder = MODELS / 'distilroberta_lora_augmented'
    scores_path = CACHE / 'transformer_lora_augmented_scores.parquet'
    if not (folder / 'adapter_config.json').exists() or not scores_path.exists():
        print('LoRA with augmentation not trained yet: python scripts/train_transformer.py --mode lora --augment 1500')
        return
    from sklearn.linear_model import LogisticRegression
    from detector.inference import Detectors
    scores = pd.read_parquet(scores_path)
    val = scores[scores.split == 'val']
    logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    platt = LogisticRegression().fit(logit(val.p_chatgpt.to_numpy()).reshape(-1, 1), val.y)
    cal = {'coef': float(platt.coef_[0, 0]), 'intercept': float(platt.intercept_[0])}
    (folder / 'calibration.json').write_text(json.dumps(cal, indent=2))
    test_scores = scores[scores.split == 'test'].set_index(['qid', 'label']).p_chatgpt
    after_test = 1 / (1 + np.exp(-(cal['coef'] * logit(test.set_index(['qid', 'label']).index.map(test_scores)
                                                       .to_numpy()) + cal['intercept'])))
    metric = lambda p: {'f1': float(f1_score(test.y, p >= 0.5)), 'roc_auc': float(roc_auc_score(test.y, p)),
                        'fpr': float((p[test.y.values == 0] >= 0.5).mean())}
    res['hc3_test']['roberta_lora'] = {'before': metric(before_test), 'after': metric(after_test)}
    after = Detectors(use_lm=False, use_transformer=False).transformer_probs(texts.text.tolist(), folder=folder,
                                                                              calibration=cal)
    after = texts.assign(roberta_lora=after)
    res['modern']['roberta_lora'] = {}
    for source in after.source.unique():
        b, a = before[before.source == source], after[after.source == source]
        if source == 'mix':
            res['modern']['roberta_lora']['mix'] = {'share': sorted(a.share.unique().tolist()),
                                                    'before': b.groupby('share').roberta_lora.mean().round(4).tolist(),
                                                    'after': a.groupby('share').roberta_lora.mean().round(4).tolist()}
        else:
            res['modern']['roberta_lora'][source] = {'before': float((b.roberta_lora >= 0.5).mean()),
                                                     'after': float((a.roberta_lora >= 0.5).mean())}


def main():
    train, val, test = data.split_frames(data.load_pairs())
    extra = augmentation_texts(train)
    augmented = pd.concat([train, extra], ignore_index=True)
    print(f'training texts: HC3 {len(train)} -> augmented {len(augmented)}')
    fitted = fit(augmented, val)
    save(fitted)

    # HC3 test: before (saved original models) vs after
    from detector.inference import Detectors
    before_hc3 = Detectors(use_lm=False, use_transformer=False).score(test.text_clean)
    after_hc3 = predict(fitted, test.text_clean)
    res = {'n_augmentation': extra.source.value_counts().to_dict(), 'hc3_test': {}, 'modern': {}}
    for key in KEYS:
        res['hc3_test'][key] = {
            'before': {'f1': float(f1_score(test.y, before_hc3[key] >= 0.5)),
                       'roc_auc': float(roc_auc_score(test.y, before_hc3[key])),
                       'fpr': float((before_hc3[key][test.y.values == 0] >= 0.5).mean())},
            'after': {'f1': float(f1_score(test.y, after_hc3[key] >= 0.5)),
                      'roc_auc': float(roc_auc_score(test.y, after_hc3[key])),
                      'fpr': float((after_hc3[key][test.y.values == 0] >= 0.5).mean())}}

    # modern sets: before = cached scores of the original models, after = augmented models
    texts = pd.read_parquet(modern.TEXTS_FILE)
    before = texts.merge(pd.read_parquet(modern.MODERN / 'scores.parquet'), on='text_id', how='left')
    after = pd.concat([texts.reset_index(drop=True), predict(fitted, texts.text)], axis=1)
    sources = ['human'] + [s for s in modern.SOURCE_NAMES if s != 'human' and s in set(texts.source)] + ['mix']
    for key in KEYS:
        res['modern'][key] = {}
        for source in sources:
            b = before[before.source == source]
            a = after[after.source == source]
            if source == 'mix':
                res['modern'][key]['mix'] = {
                    'share': sorted(a.share.unique().tolist()),
                    'before': b.groupby('share')[key].mean().round(4).tolist(),
                    'after': a.groupby('share')[key].mean().round(4).tolist()}
                continue
            res['modern'][key][source] = {'before': float((b[key] >= 0.5).mean()),
                                          'after': float((a[key] >= 0.5).mean())}
    lora_augmented(res, test, before_hc3_scores(test), texts, before)
    RESULTS.write_text(json.dumps(res, indent=2))
    figures(res)
    print(json.dumps(res['hc3_test'], indent=1))


def figures(res):
    plots.setup_style()
    unseen = [s for s in ['claude-sonnet-5-5', 'claude-opus-5-5', 'qwen2.5-1.5b', 'smollm2-1.7b', 'humanize',
                          'paraphrase'] if s in res['modern']['tfidf']]
    labels = ['Human FPR'] + [modern.SOURCE_NAMES[s].replace('Attack: ', '') for s in ['claude-haiku-5-5'] + unseen
                              if s in res['modern']['tfidf']]
    cols = ['human'] + [s for s in ['claude-haiku-5-5'] + unseen if s in res['modern']['tfidf']]
    for key in [k for k in ('roberta_lora', 'tfidf', 'nn') if k in res['modern']]:
        series = {'Trained on HC3 only': [100 * res['modern'][key][c]['before'] for c in cols],
                  'HC3 + Claude Haiku texts': [100 * res['modern'][key][c]['after'] for c in cols]}
        plots.grouped_bars([l.replace(' (open)', '').replace('-Instruct', '') for l in labels], series,
                           FIGURES / f'augmented_{key}.png',
                           f'{NAMES[key]}: detection rate before / after adding one modern generator',
                           '% of texts flagged as machine', ['#9ec5f4', '#256abf'])
    print(pd.DataFrame({s: {k: f"{v[s]['before']:.2f}->{v[s]['after']:.2f}" for k, v in res['modern'].items()}
                        for s in cols}).to_string())


if __name__ == '__main__':
    if '--figures' in sys.argv:          # only redraw the figures from augmented_results.json
        figures(json.loads(RESULTS.read_text()))
    else:
        main()
