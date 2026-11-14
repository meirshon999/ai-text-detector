"""Single entry point for all experiments of the final project.

Usage (from any folder):
    python scripts/run_experiments.py            # full run, about 20-30 minutes on a laptop CPU
    python scripts/run_experiments.py --reuse    # only re-evaluate + redraw figures from cached predictions (1 min)
    python scripts/run_experiments.py --quick    # small smoke test (2-3 minutes), writes to reports/final/quick/

Before the full run:
    python scripts/download_data.py              # HC3 -> data/raw/              (once)
    python scripts/compute_lm_scores.py          # GPT-2 scores -> cache          (once, ~20 min)
    python scripts/train_transformer.py          # DistilRoBERTa full + LoRA      (optional, ~1 h)

Steps:
    1. data      rebuild the cleaned, balanced dataset (identical split to the midterm) + leakage audit
    2. bias      dataset-bias experiments: shortcut rule, train-raw -> test-clean stress test, URL placeholders
    3. models    train every detector on TRAIN, tune on VALIDATION, evaluate once on TEST
    4. lodo      leave-one-domain-out generalisation, same test texts for every model
    5. report    metrics with 95% bootstrap CIs, tables and figures

Outputs: reports/final/results.json, reports/final/*.md tables, reports/final/figures/*.png, models/*
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import joblib                                                                      # noqa: E402
import numpy as np                                                                 # noqa: E402
import pandas as pd                                                                # noqa: E402
import sklearn                                                                     # noqa: E402
import torch                                                                       # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer                        # noqa: E402
from sklearn.inspection import permutation_importance                              # noqa: E402
from sklearn.linear_model import LogisticRegression                                # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score                                # noqa: E402

from detector import data, features, models, plots                                 # noqa: E402
from detector.config import (CACHE, CHATGPT_COLOR, DOMAIN_NAMES, FINAL, HUMAN_COLOR, MODELS, ROOT, SEED,  # noqa: E402
                             ensure_dirs)
from detector.evaluation import bootstrap_ci, compute_metrics                      # noqa: E402

NAMES = {
    'length': 'Length baseline',
    'tfidf': 'TF-IDF + LogReg',
    'style_lr': 'Stylometry + LogReg',
    'style_rf': 'Stylometry + Random Forest',
    'nn': 'PyTorch bag-of-embeddings',
    'ppl': 'GPT-2 perplexity (zero-shot)',
    'fdg': 'Fast-DetectGPT on GPT-2 (zero-shot)',
    'roberta_full': 'DistilRoBERTa full fine-tuning',
    'roberta_lora': 'DistilRoBERTa + LoRA',
    'roberta_full_raw': 'DistilRoBERTa full FT, raw 0.5',
    'roberta_lora_raw': 'DistilRoBERTa + LoRA, raw 0.5',
}
ZERO_SHOT = {'ppl': ('log_ppl', -1.0), 'fdg': ('fast_detectgpt', 1.0)}     # score column, sign (+ = ChatGPT)
LENGTH_BUCKETS = [(5, 49, '5–49'), (50, 99, '50–99'), (100, 199, '100–199'), (200, 399, '200–399'),
                  (400, 10 ** 9, '400+')]
START = time.time()


def log(message):
    print(f'[{(time.time() - START) / 60:5.1f} min] {message}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quick', action='store_true', help='small subset, results go to reports/final/quick/')
    parser.add_argument('--reuse', action='store_true', help='reuse cached predictions, only rebuild the report')
    parser.add_argument('--n-boot', type=int, default=1000, help='bootstrap resamples for confidence intervals')
    return parser.parse_args()


# ============================================================================ 1. data
def subsample(pairs, n_train, n_eval):
    rng = np.random.default_rng(SEED)
    parts = []
    for split, n in (('train', n_train), ('val', n_eval), ('test', n_eval)):
        part = pairs[pairs.split == split]
        chosen = rng.choice(part.qid.unique(), size=min(n, part.qid.nunique()), replace=False)
        parts.append(part[part.qid.isin(chosen)])
    return pd.concat(parts).reset_index(drop=True)


def audit(pairs, tr, te):
    """Checks for data leakage between the splits."""
    by_split = {s: pairs[pairs.split == s] for s in ('train', 'val', 'test')}
    overlap = lambda col, a, b: len(set(by_split[a][col]) & set(by_split[b][col]))
    vectorizer = TfidfVectorizer(min_df=2, sublinear_tf=True).fit(tr.text_clean)
    test_matrix, train_matrix = vectorizer.transform(te.text_clean), vectorizer.transform(tr.text_clean)
    nearest = np.zeros(test_matrix.shape[0])
    for i in range(0, test_matrix.shape[0], 500):
        nearest[i:i + 500] = (test_matrix[i:i + 500] @ train_matrix.T).toarray().max(axis=1)
    return {
        'class_balance_chatgpt_share': pairs.groupby('split').y.mean().round(4).to_dict(),
        'split_sizes': pairs.split.value_counts().to_dict(),
        'questions_per_split': pairs.groupby('split').qid.nunique().to_dict(),
        'texts_per_domain_and_split': pd.crosstab(pairs.domain, pairs.split).to_dict(),
        'same_question_id_in_train_and_test': overlap('qid', 'train', 'test'),
        'same_question_id_in_train_and_val': overlap('qid', 'train', 'val'),
        'same_question_text_in_train_and_test': overlap('question', 'train', 'test'),
        'identical_texts_in_train_and_test': overlap('text_clean', 'train', 'test'),
        'identical_texts_in_train_and_val': overlap('text_clean', 'train', 'val'),
        'test_texts_with_near_duplicate_in_train_cos_ge_0.9': int((nearest >= 0.9).sum()),
        'test_texts_with_near_duplicate_in_train_cos_ge_0.8': int((nearest >= 0.8).sum()),
        'n_test_texts': int(len(te)),
    }


# ============================================================================ 2. dataset bias
def stage_bias(long_clean, tr, te):
    result = {}
    rows = []
    for name, pattern in data.ARTIFACT_PATTERNS.items():
        for label in ('human', 'chatgpt'):
            part = long_clean[long_clean.label == label]
            rows.append({'artifact': name, 'label': label,
                         'raw_percent': float(part.text.str.contains(pattern, regex=True).mean() * 100),
                         'clean_percent': float(part.text_clean.str.contains(pattern, regex=True).mean() * 100)})
    result['artifact_prevalence'] = rows

    y_long = (long_clean.label == 'chatgpt').astype(int).values
    result['shortcut_rule_accuracy'] = {
        'all_answers_raw': float((data.shortcut_rule_predict(long_clean.text) == y_long).mean()),
        'all_answers_clean': float((data.shortcut_rule_predict(long_clean.text_clean) == y_long).mean()),
        'balanced_test_raw': float((data.shortcut_rule_predict(te.text) == te.y).mean()),
        'balanced_test_clean': float((data.shortcut_rule_predict(te.text_clean) == te.y).mean()),
        'balanced_test_raw_metrics': compute_metrics(te.y, data.shortcut_rule_predict(te.text)),
    }
    log('bias: shortcut rule done')

    # Stress test: a model trained on RAW text is applied to CLEAN text (a human text without the artifact).
    stress = []
    vectorizers = {
        # 1-3 characters are enough to "see" the artifact " ," / " ." / " n'"
        'Character n-grams (sees spaces)': lambda: TfidfVectorizer(analyzer='char', ngram_range=(1, 3), min_df=5,
                                                                   max_features=100_000, sublinear_tf=True),
        'Word n-grams (our TF-IDF)': lambda: models.tfidf_vectorizer((1, 2)),
    }
    for model_name, make_vectorizer in vectorizers.items():
        fitted = {}
        for train_col in ('text', 'text_clean'):
            vectorizer = make_vectorizer().fit(tr[train_col])
            classifier = LogisticRegression(C=4, max_iter=3000).fit(vectorizer.transform(tr[train_col]), tr.y)
            fitted[train_col] = (vectorizer, classifier)
        for train_col, test_col, setting in (('text', 'text', 'Train raw → test raw'),
                                             ('text', 'text_clean', 'Train raw → test clean'),
                                             ('text_clean', 'text_clean', 'Train clean → test clean')):
            vectorizer, classifier = fitted[train_col]
            prob = classifier.predict_proba(vectorizer.transform(te[test_col]))[:, 1]
            stress.append({'model': model_name, 'setting': setting, **compute_metrics(te.y, prob)})
        log(f'bias: stress test done for {model_name}')
    result['stress_test'] = stress

    # Residual artifact: "URL_0" placeholders (links in human answers were replaced by the dataset authors).
    has_url = lambda frame: frame.text_clean.str.contains(data.URL_PLACEHOLDER, regex=True)
    result['url_placeholder_share'] = {'human_percent': float(has_url(tr[tr.y == 0]).mean() * 100),
                                       'chatgpt_percent': float(has_url(tr[tr.y == 1]).mean() * 100)}
    ablation = {}
    for setting, transform in (('with URL_0 placeholders', lambda s: s),
                               ('URL_0 placeholders removed', lambda s: s.map(data.remove_url_placeholders))):
        pipe = models.tfidf_logreg(C=4).fit(transform(tr.text_clean), tr.y)
        prob = pipe.predict_proba(transform(te.text_clean))[:, 1]
        reddit = (te.domain == 'reddit_eli5').values
        ablation[setting] = {'all': compute_metrics(te.y, prob),
                             'reddit_eli5': compute_metrics(te.y[reddit], prob[reddit])}
    result['url_placeholder_ablation'] = ablation
    log('bias: URL placeholder ablation done')
    return result


# ============================================================================ 3. models
def style_tables(pairs):
    path = CACHE / 'style_features.parquet'
    if path.exists():
        table = pd.read_parquet(path)
        if len(table) == len(pairs) and (table.qid.values == pairs.qid.values).all() \
                and (table.label.values == pairs.label.values).all():
            return table
    log('computing stylometric features for all texts...')
    table = features.feature_table(pairs.text_clean)
    table[['qid', 'label']] = pairs[['qid', 'label']].values
    table.to_parquet(path, index=False)
    return table


def zero_shot_frames(va, te):
    path = CACHE / 'gpt2_scores.parquet'
    if not path.exists():
        log(f'GPT-2 scores not found ({path}); zero-shot detectors are skipped. '
            'Run: python scripts/compute_lm_scores.py')
        return None, None
    scores = pd.read_parquet(path)[['qid', 'label', 'log_ppl', 'perplexity', 'fast_detectgpt', 'n_tokens']]
    merge = lambda frame: frame[['qid', 'label']].merge(scores, on=['qid', 'label'], how='left')
    val_scores, test_scores = merge(va), merge(te)
    if val_scores.log_ppl.isna().any() or test_scores.log_ppl.isna().any():
        log('GPT-2 scores do not cover every validation/test text; zero-shot detectors are skipped.')
        return None, None
    return val_scores, test_scores


def platt(val_score, val_y):
    """1-D logistic regression on the validation set: turns a score into P(ChatGPT)."""
    return LogisticRegression().fit(np.asarray(val_score).reshape(-1, 1), val_y)


def stage_models(pairs, tr, va, te, style, quick):
    pred_val, pred_test, tuning, extra = {}, {}, {}, {}
    model_dir = (FINAL / 'quick' / 'models') if quick else MODELS
    model_dir.mkdir(parents=True, exist_ok=True)

    # Model 0 — length only
    length_model = models.length_baseline().fit(tr.text_clean, tr.y)
    pred_val['length'] = length_model.predict_proba(va.text_clean)[:, 1]
    pred_test['length'] = length_model.predict_proba(te.text_clean)[:, 1]
    joblib.dump(length_model, model_dir / 'length_baseline.joblib')
    log('length baseline done')

    # Model 1 — TF-IDF + logistic regression, tuned on validation
    grid = []
    for ngram in ((1, 1), (1, 2)):
        vectorizer = models.tfidf_vectorizer(ngram).fit(tr.text_clean)
        x_train, x_val = vectorizer.transform(tr.text_clean), vectorizer.transform(va.text_clean)
        for C in (1.0, 4.0, 16.0):
            prob = LogisticRegression(C=C, max_iter=3000).fit(x_train, tr.y).predict_proba(x_val)[:, 1]
            grid.append({'ngram_range': list(ngram), 'C': C, 'val_f1': float(f1_score(va.y, prob >= 0.5)),
                         'val_roc_auc': float(roc_auc_score(va.y, prob))})
            log(f'TF-IDF ngram={ngram} C={C}: val F1 {grid[-1]["val_f1"]:.4f}')
    best = max(grid, key=lambda g: (round(g['val_f1'], 4), g['val_roc_auc']))
    tuning['tfidf'] = {'grid': grid, 'chosen': best}
    tfidf = models.tfidf_logreg(C=best['C'], ngram_range=tuple(best['ngram_range'])).fit(tr.text_clean, tr.y)
    pred_val['tfidf'] = tfidf.predict_proba(va.text_clean)[:, 1]
    pred_test['tfidf'] = tfidf.predict_proba(te.text_clean)[:, 1]
    joblib.dump(tfidf, model_dir / 'tfidf_logreg.joblib', compress=3)
    vocabulary = np.array(tfidf[0].get_feature_names_out())
    weights = tfidf[-1].coef_[0]
    extra['tfidf_top_features'] = {
        'chatgpt': [(str(vocabulary[i]), float(weights[i])) for i in np.argsort(weights)[-20:][::-1]],
        'human': [(str(vocabulary[i]), float(weights[i])) for i in np.argsort(weights)[:20]],
    }

    # Model 2 — stylometric features (feature set and leaf size chosen on validation)
    s_train, s_val, s_test = (style.loc[frame.index] for frame in (tr, va, te))
    grid = []
    for feature_set in ('18 base features', '18 + 2 extra features'):
        names = features.STYLE_FEATURES + ([] if feature_set == '18 base features' else features.EXTRA_FEATURES)
        for leaf in (1, 3, 5):
            forest = models.stylometry_forest(leaf).fit(s_train[names], tr.y)
            prob = forest.predict_proba(s_val[names])[:, 1]
            grid.append({'feature_set': feature_set, 'min_samples_leaf': leaf,
                         'val_f1': float(f1_score(va.y, prob >= 0.5)), 'val_roc_auc': float(roc_auc_score(va.y, prob))})
    best = max(grid, key=lambda g: (round(g['val_f1'], 3), -len(g['feature_set'])))   # tie -> fewer features
    # Extra features are kept only if they improve validation F1 by at least 0.002
    base_best = max((g for g in grid if g['feature_set'] == '18 base features'), key=lambda g: g['val_f1'])
    if best['feature_set'] != '18 base features' and best['val_f1'] - base_best['val_f1'] < 0.002:
        best = base_best
    feature_names = features.STYLE_FEATURES + ([] if best['feature_set'] == '18 base features'
                                               else features.EXTRA_FEATURES)
    tuning['style_rf'] = {'grid': grid, 'chosen': best, 'features': feature_names}
    forest = models.stylometry_forest(best['min_samples_leaf']).fit(s_train[feature_names], tr.y)
    logreg = models.stylometry_logreg().fit(s_train[feature_names], tr.y)
    for key, model in (('style_rf', forest), ('style_lr', logreg)):
        pred_val[key] = model.predict_proba(s_val[feature_names])[:, 1]
        pred_test[key] = model.predict_proba(s_test[feature_names])[:, 1]
    joblib.dump({'features': feature_names, 'model': forest}, model_dir / 'stylometry_rf.joblib', compress=3)
    joblib.dump({'features': feature_names, 'model': logreg}, model_dir / 'stylometry_logreg.joblib')
    # Typical values of every feature for each class (training set) - shown in the demo
    reference = {name: {'human_median': float(s_train.loc[tr.y.values == 0, name].median()),
                        'chatgpt_median': float(s_train.loc[tr.y.values == 1, name].median())}
                 for name in ['n_words'] + features.STYLE_FEATURES + features.EXTRA_FEATURES}
    (model_dir / 'style_reference.json').write_text(json.dumps(reference, indent=2))
    importance = permutation_importance(forest, s_val[feature_names], va.y, scoring='roc_auc', n_repeats=5,
                                        random_state=SEED, n_jobs=1)
    extra['style_permutation_importance'] = {n: [float(m), float(s)] for n, m, s in
                                             zip(feature_names, importance.importances_mean, importance.importances_std)}
    extra['style_logreg_weights'] = dict(zip(feature_names, map(float, logreg[-1].coef_[0])))
    log(f'stylometry done (features: {best["feature_set"]}, min_samples_leaf={best["min_samples_leaf"]})')

    # Model 3 — PyTorch bag-of-embeddings with early stopping
    network = models.NeuralDetector(max_epochs=3 if quick else 10).fit(tr.text_clean.tolist(), tr.y.values,
                                                                       va.text_clean.tolist(), va.y.values)
    pred_val['nn'] = network.predict_proba(va.text_clean.tolist())
    pred_test['nn'] = network.predict_proba(te.text_clean.tolist())
    network.save(model_dir / 'bag_of_embeddings')
    extra['nn_history'] = network.history
    extra['nn_best_epoch'] = network.best_epoch
    log(f'neural network done (best epoch {network.best_epoch})')

    # Overfitting check: the same metric on the TRAINING texts (a large train-test gap = memorisation)
    train_prob = {'length': length_model.predict_proba(tr.text_clean)[:, 1],
                  'tfidf': tfidf.predict_proba(tr.text_clean)[:, 1],
                  'style_rf': forest.predict_proba(s_train[feature_names])[:, 1],
                  'style_lr': logreg.predict_proba(s_train[feature_names])[:, 1],
                  'nn': network.predict_proba(tr.text_clean.tolist())}
    extra['train_f1'] = {k: float(f1_score(tr.y, p >= 0.5)) for k, p in train_prob.items()}
    extra['val_f1'] = {k: float(f1_score(va.y, pred_val[k] >= 0.5)) for k in train_prob}

    # Models 4-5 — zero-shot GPT-2 scores, calibrated on validation
    val_scores, test_scores = (None, None) if quick else zero_shot_frames(va, te)
    calibration = {}
    if val_scores is not None:
        for key, (column, sign) in ZERO_SHOT.items():
            calibrator = platt(sign * val_scores[column], va.y)
            a, b = float(calibrator.coef_[0][0]), float(calibrator.intercept_[0])
            pred_val[key] = calibrator.predict_proba((sign * val_scores[column]).values.reshape(-1, 1))[:, 1]
            pred_test[key] = calibrator.predict_proba((sign * test_scores[column]).values.reshape(-1, 1))[:, 1]
            boundary = -b / a                                    # score where P(ChatGPT) = 0.5
            calibration[key] = {'column': column, 'sign': sign, 'coef': a, 'intercept': b,
                                'raw_threshold': float(sign * boundary)}
        calibration['ppl']['perplexity_threshold'] = float(np.exp(calibration['ppl']['raw_threshold']))
        calibration['max_tokens'] = 128
        (model_dir / 'zero_shot_calibration.json').write_text(json.dumps(calibration, indent=2))
        extra['perplexity_median'] = {'human': float(test_scores.perplexity[te.y.values == 0].median()),
                                      'chatgpt': float(test_scores.perplexity[te.y.values == 1].median())}
        log('zero-shot GPT-2 detectors done')
    extra['calibration'] = calibration
    return pred_val, pred_test, tuning, extra


# ============================================================================ 4. leave-one-domain-out
def stage_lodo(tr, va, te, style, tuning, pred_test, quick):
    rows = []
    tf_best, rf_best = tuning['tfidf']['chosen'], tuning['style_rf']['chosen']
    feature_names = tuning['style_rf']['features']
    val_scores, test_scores = (None, None) if quick else zero_shot_frames(va, te)
    for domain in sorted(te.domain.unique()):
        train_o, val_o = tr[tr.domain != domain], va[va.domain != domain]
        test_mask = (te.domain == domain).values
        test_d = te[test_mask]
        unseen = {}
        unseen['length'] = models.length_baseline().fit(train_o.text_clean, train_o.y).predict_proba(
            test_d.text_clean)[:, 1]
        tfidf = models.tfidf_logreg(C=tf_best['C'], ngram_range=tuple(tf_best['ngram_range']))
        unseen['tfidf'] = tfidf.fit(train_o.text_clean, train_o.y).predict_proba(test_d.text_clean)[:, 1]
        s_train, s_test = style.loc[train_o.index, feature_names], style.loc[test_d.index, feature_names]
        unseen['style_rf'] = models.stylometry_forest(rf_best['min_samples_leaf']).fit(s_train, train_o.y) \
            .predict_proba(s_test)[:, 1]
        unseen['style_lr'] = models.stylometry_logreg().fit(s_train, train_o.y).predict_proba(s_test)[:, 1]
        network = models.NeuralDetector(max_epochs=3 if quick else 10).fit(
            train_o.text_clean.tolist(), train_o.y.values, val_o.text_clean.tolist(), val_o.y.values, verbose=False)
        unseen['nn'] = network.predict_proba(test_d.text_clean.tolist())
        if val_scores is not None:
            val_mask = (va.domain != domain).values
            for key, (column, sign) in ZERO_SHOT.items():
                calibrator = platt(sign * val_scores.loc[val_mask, column], va.y[val_mask])
                unseen[key] = calibrator.predict_proba(
                    (sign * test_scores.loc[test_mask, column]).values.reshape(-1, 1))[:, 1]
        for key, prob in unseen.items():
            seen = compute_metrics(test_d.y, pred_test[key][test_mask])
            new = compute_metrics(test_d.y, prob)
            ci = bootstrap_ci(test_d.y, prob, test_d.qid, n_boot=200)
            rows.append({'model': NAMES[key], 'key': key, 'domain': domain, 'n_test': int(len(test_d)),
                         'f1_seen': seen['f1'], 'f1_unseen': new['f1'], 'f1_drop': seen['f1'] - new['f1'],
                         'accuracy_seen': seen['accuracy'], 'accuracy_unseen': new['accuracy'],
                         'roc_auc_seen': seen['roc_auc'], 'roc_auc_unseen': new['roc_auc'],
                         'fpr_unseen': new['false_positive_rate'], 'f1_unseen_ci': ci['f1_ci']})
        log(f'LODO: held-out domain {domain} done')
    return rows


# ============================================================================ 5. report
def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def calibrate_transformer(scores):
    """Platt scaling on the validation texts of a transformer run: P(ChatGPT) = sigmoid(a * logit(p) + b).

    After one epoch on 5,000 texts the network ranks texts almost perfectly but its raw probabilities are
    shifted towards "ChatGPT". The same 2-parameter calibration on validation is used for the zero-shot scores.
    """
    val = scores[scores.split == 'val']
    calibrator = LogisticRegression().fit(logit(val.p_chatgpt).reshape(-1, 1), val.y)
    return float(calibrator.coef_[0][0]), float(calibrator.intercept_[0])


def apply_calibration(p, coef, intercept):
    return 1 / (1 + np.exp(-(coef * logit(p) + intercept)))


def load_transformer_predictions(te):
    found = {}
    for key, mode in (('roberta_full', 'full'), ('roberta_lora', 'lora')):
        path = CACHE / f'transformer_{mode}_scores.parquet'
        if not path.exists():
            continue
        scores = pd.read_parquet(path)
        merged = te[['qid', 'label']].merge(scores[scores.split == 'test'][['qid', 'label', 'p_chatgpt']],
                                            on=['qid', 'label'], how='left')
        if merged.p_chatgpt.isna().any():
            log(f'{path.name} does not cover the whole test set - skipped')
            continue
        coef, intercept = calibrate_transformer(scores)
        found[key + '_raw'] = merged.p_chatgpt.values
        found[key] = apply_calibration(merged.p_chatgpt.values, coef, intercept)
        info = json.loads((CACHE / f'transformer_{mode}_info.json').read_text())
        found[key + '_info'] = {**{k: v for k, v in info.items() if k != 'history'},
                                'calibration': {'coef': coef, 'intercept': intercept,
                                                'fitted_on': f"{info['val_texts']} validation texts"}}
        found[key + '_history'] = info['history']
    return found


def length_bucket(n_words):
    for low, high, label in LENGTH_BUCKETS:
        if low <= n_words <= high:
            return label
    return LENGTH_BUCKETS[-1][2]


def markdown_table(frame):
    head = '| ' + ' | '.join(frame.columns) + ' |'
    sep = '|' + '|'.join(['---'] + ['---:'] * (len(frame.columns) - 1)) + '|'
    body = ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in frame.values]
    return '\n'.join([head, sep] + body)


def stage_report(out_dir, figures_dir, te, pred_test, extra, tuning, bias, lodo, audit_result, steps, n_boot):
    plots.setup_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    transformer = load_transformer_predictions(te)
    for key in ('roberta_full', 'roberta_lora', 'roberta_full_raw', 'roberta_lora_raw'):
        if key in transformer:
            pred_test[key] = transformer[key]
    if 'roberta_lora_info' in transformer and out_dir == FINAL:      # the demo applies the same calibration
        (MODELS / 'transformer_calibration.json').write_text(
            json.dumps(transformer['roberta_lora_info']['calibration'], indent=2))

    # Optional generalisation test of LoRA on a held-out domain (scripts/train_transformer.py --holdout <domain>)
    lodo = list(lodo)
    for path in sorted(CACHE.glob('transformer_lora_lodo_*_scores.parquet')):
        domain = path.name[len('transformer_lora_lodo_'):-len('_scores.parquet')]
        if 'roberta_lora' not in pred_test or domain not in set(te.domain):
            continue
        scores = pd.read_parquet(path)
        coef, intercept = calibrate_transformer(scores)
        scores = scores.assign(p_chatgpt=apply_calibration(scores.p_chatgpt.values, coef, intercept))
        test_mask = (te.domain == domain).values
        test_d = te[test_mask]
        merged = test_d[['qid', 'label']].merge(scores[scores.split == 'test'][['qid', 'label', 'p_chatgpt']],
                                                on=['qid', 'label'], how='left')
        if merged.p_chatgpt.isna().any():
            log(f'{path.name} does not cover the {domain} test texts - skipped')
            continue
        seen = compute_metrics(test_d.y, pred_test['roberta_lora'][test_mask])
        new = compute_metrics(test_d.y, merged.p_chatgpt.values)
        lodo.append({'model': NAMES['roberta_lora'], 'key': 'roberta_lora', 'domain': domain, 'n_test': int(len(test_d)),
                     'f1_seen': seen['f1'], 'f1_unseen': new['f1'], 'f1_drop': seen['f1'] - new['f1'],
                     'accuracy_seen': seen['accuracy'], 'accuracy_unseen': new['accuracy'],
                     'roc_auc_seen': seen['roc_auc'], 'roc_auc_unseen': new['roc_auc'],
                     'fpr_unseen': new['false_positive_rate'],
                     'f1_unseen_ci': bootstrap_ci(test_d.y, merged.p_chatgpt.values, test_d.qid, n_boot=200)['f1_ci']})
        log(f'added LoRA held-out-domain result for {domain}')

    results, ci = {}, {}
    for key, prob in pred_test.items():
        results[NAMES[key]] = compute_metrics(te.y, prob)
        results[NAMES[key]].update(bootstrap_ci(te.y, prob, te.qid, n_boot=n_boot))
        ci[NAMES[key]] = results[NAMES[key]]['f1_ci']
    log('test metrics and bootstrap CIs done')

    # Per-domain and per-length analysis on the test set
    per_domain = {NAMES[k]: {d: compute_metrics(te.y[te.domain == d], p[(te.domain == d).values])['f1']
                             for d in sorted(te.domain.unique())} for k, p in pred_test.items()}
    # Errors by text length, separately for each class: short ChatGPT answers are rare in HC3,
    # so a single error rate per length bucket would mostly describe human texts.
    buckets = te.text_clean.str.split().str.len().map(length_bucket).values
    y_test = te.y.values
    by_length = {}
    for k, p in pred_test.items():
        pred = (p >= 0.5).astype(int)
        by_length[NAMES[k]] = {
            label: {'error_rate': float((pred != y_test)[buckets == label].mean()),
                    'false_positive_rate': float((pred[(buckets == label) & (y_test == 0)] == 1).mean()),
                    'miss_rate': float((pred[(buckets == label) & (y_test == 1)] == 0).mean())}
            for _, _, label in LENGTH_BUCKETS if ((buckets == label) & (y_test == 1)).any()}
    bucket_sizes = {label: {'human': int(((buckets == label) & (y_test == 0)).sum()),
                            'chatgpt': int(((buckets == label) & (y_test == 1)).sum())}
                    for _, _, label in LENGTH_BUCKETS}

    best_key = max(pred_test, key=lambda k: results[NAMES[k]]['f1'])
    best_prob = pred_test[best_key]
    errors = te.assign(p=best_prob, pred=(best_prob >= 0.5).astype(int))
    errors = errors[errors.pred != errors.y]
    examples = {'model': NAMES[best_key],
                'false_positives_human_flagged': [
                    {'domain': r.domain, 'p_chatgpt': round(float(r.p), 3), 'text': r.text_clean[:300]}
                    for r in errors[errors.y == 0].sort_values('p', ascending=False).head(5).itertuples()],
                'false_negatives_chatgpt_missed': [
                    {'domain': r.domain, 'p_chatgpt': round(float(r.p), 3), 'text': r.text_clean[:300]}
                    for r in errors[errors.y == 1].sort_values('p').head(5).itertuples()]}

    # ---------------- files used by the demo: model card + example texts from the TEST set
    quick = out_dir != FINAL
    model_dir = (FINAL / 'quick' / 'models') if quick else MODELS
    lodo_mean = pd.DataFrame(lodo).groupby('key').f1_unseen.mean().to_dict() if lodo else {}
    card = {key: {'name': NAMES[key], 'f1': results[NAMES[key]]['f1'], 'f1_ci': results[NAMES[key]]['f1_ci'],
                  'accuracy': results[NAMES[key]]['accuracy'], 'roc_auc': results[NAMES[key]]['roc_auc'],
                  'false_positive_rate': results[NAMES[key]]['false_positive_rate'],
                  'mean_unseen_domain_f1': lodo_mean.get(key), 'n_test': int(len(te))} for key in pred_test}
    (model_dir / 'model_card.json').write_text(json.dumps(card, indent=2))
    if not quick:
        p = pred_test['tfidf']
        words = te.text_clean.str.split().str.len().values
        demo_examples = []
        for domain in ('reddit_eli5', 'finance', 'medicine', 'open_qa', 'wiki_csai'):
            for y_value, label in ((0, 'Human'), (1, 'ChatGPT')):
                confident = (p < 0.2) if y_value == 0 else (p > 0.8)
                mask = (te.domain.values == domain) & (te.y.values == y_value) & confident & (words >= 60) & (words <= 170)
                if mask.any():
                    i = np.flatnonzero(mask)[0]
                    demo_examples.append({'title': f'{label} · {DOMAIN_NAMES[domain]}', 'label': label,
                                          'domain': domain, 'text': te.text_clean.iloc[i]})
        for y_value, title in ((0, 'Hard case: human text that TF-IDF flags as ChatGPT'),
                               (1, 'Hard case: ChatGPT text that TF-IDF misses')):
            wrong = (te.y.values == y_value) & (((p >= 0.5) if y_value == 0 else (p < 0.5))) & (words >= 40) & (words <= 200)
            if wrong.any():
                i = np.flatnonzero(wrong)[0]
                demo_examples.append({'title': title, 'label': 'Human' if y_value == 0 else 'ChatGPT',
                                      'domain': te.domain.iloc[i], 'text': te.text_clean.iloc[i]})
        (ROOT / 'demo' / 'examples.json').write_text(
            json.dumps(demo_examples, indent=2, ensure_ascii=False), encoding='utf-8')

    # ---------------- figures
    order = [k for k in NAMES if k in pred_test]
    table = {NAMES[k]: results[NAMES[k]] for k in order}
    plots.metric_bars(table, 'f1', 'F1-score on the test set (95% bootstrap CI)', figures_dir / 'f1_comparison.png',
                      xlabel='F1 (ChatGPT = positive class)', ci=ci, xlim=(0.5, 1.03))
    plots.metric_bars(table, 'accuracy', 'Accuracy on the test set (balanced classes: 50% = chance)',
                      figures_dir / 'accuracy_comparison.png', xlabel='Accuracy',
                      ci={n: results[n]['accuracy_ci'] for n in table}, xlim=(0.5, 1.03))
    # The two best detectors, the zero-shot one and the uncalibrated transformer (shows why the threshold matters)
    shown = [NAMES[k] for k in ('roberta_lora', 'tfidf', 'ppl', 'roberta_lora_raw') if k in pred_test]
    if len(shown) < 4:
        shown += [n for n in sorted(table, key=lambda n: -table[n]['f1']) if n not in shown][:4 - len(shown)]
    plots.confusion_grid(results, shown, figures_dir / 'confusion_matrices.png')
    roc_keys = [k for k in ('length', 'tfidf', 'style_rf', 'nn', 'ppl', 'fdg', 'roberta_lora', 'roberta_full')
                if k in pred_test][:8]
    plots.roc_curves({NAMES[k]: (te.y.values, pred_test[k]) for k in roc_keys}, figures_dir / 'roc_curves.png')
    if 'style_permutation_importance' in extra:
        imp = extra['style_permutation_importance']
        plots.importance_bars(list(imp), [v[0] for v in imp.values()], [v[1] for v in imp.values()],
                              figures_dir / 'feature_importance_stylometry.png',
                              'Which style features matter? (Random Forest, permutation importance)',
                              'Drop of validation ROC-AUC when the feature is shuffled')
        weights = extra['style_logreg_weights']
        plots.diverging_bars(list(weights), list(weights.values()), figures_dir / 'stylometry_logreg_weights.png',
                             'Direction of each style feature (logistic regression, standardised)', 'Weight')
    top = extra['tfidf_top_features']
    words = [w for w, _ in top['chatgpt'][:15]] + [w for w, _ in top['human'][:15]]
    weights = [v for _, v in top['chatgpt'][:15]] + [v for _, v in top['human'][:15]]
    plots.diverging_bars([repr(w)[1:-1] if w.strip() else repr(w) for w in words], weights,
                         figures_dir / 'tfidf_top_words.png', 'Words and punctuation the TF-IDF model relies on',
                         'Logistic-regression weight')
    if extra.get('nn_history'):
        plots.training_curves(extra['nn_history'], figures_dir / 'nn_training_curves.png',
                              'PyTorch network: training vs validation loss', best_x=extra['nn_best_epoch'])
    for key in ('roberta_full', 'roberta_lora'):
        if key + '_history' in transformer:
            plots.training_curves(transformer[key + '_history'], figures_dir / f'{key}_training_curve.png',
                                  f'{NAMES[key]}: training loss (moving average of 10 steps)', x_key='step',
                                  lines=(('train_loss', 'Training loss'),))
    scores_path = CACHE / 'gpt2_scores.parquet'
    if scores_path.exists() and extra.get('calibration'):
        scores = te[['qid', 'label']].merge(pd.read_parquet(scores_path), on=['qid', 'label'], how='left')
        plots.class_histogram(scores.perplexity[te.y.values == 0].values, scores.perplexity[te.y.values == 1].values,
                              figures_dir / 'perplexity_distribution.png',
                              'GPT-2 perplexity: ChatGPT texts are more predictable (test set)',
                              'GPT-2 perplexity, first 128 tokens (log scale)',
                              threshold=extra['calibration']['ppl']['perplexity_threshold'])
        plots.class_histogram(scores.fast_detectgpt[te.y.values == 0].values,
                              scores.fast_detectgpt[te.y.values == 1].values,
                              figures_dir / 'fast_detectgpt_distribution.png',
                              'Fast-DetectGPT criterion on GPT-2 (test set)', 'Criterion (higher = more machine-like)',
                              threshold=extra['calibration']['fdg']['raw_threshold'], log=False)
    if lodo:
        lodo_models = [NAMES[k] for k in NAMES if any(r['key'] == k for r in lodo)]
        plots.lodo_heatmaps(lodo, lodo_models, figures_dir / 'cross_domain.png')
    plots.error_by_length({NAMES[k]: by_length[NAMES[k]] for k in ('tfidf', 'style_rf', 'nn', 'ppl', 'roberta_lora')
                           if k in pred_test}, bucket_sizes, figures_dir / 'errors_by_length.png')
    # Dataset-bias figures
    prevalence = pd.DataFrame(bias['artifact_prevalence'])
    names = list(dict.fromkeys(prevalence.artifact))
    short = ['Space before\npunctuation', "Split contractions\n(do n't)", 'No space after\nperiod', 'Line break']
    get = lambda label, col: [float(prevalence[(prevalence.artifact == n) & (prevalence.label == label)][col].iloc[0])
                              for n in names]
    plots.grouped_bars(short, {'Human — raw': get('human', 'raw_percent'),
                               'ChatGPT — raw': get('chatgpt', 'raw_percent'),
                               'Human — cleaned': get('human', 'clean_percent'),
                               'ChatGPT — cleaned': get('chatgpt', 'clean_percent')},
                       figures_dir / 'bias_artifacts_before_after.png',
                       'Formatting artifacts in HC3 before and after our cleaning', '% of answers with the pattern',
                       [HUMAN_COLOR, CHATGPT_COLOR, '#9ec5f4', '#f6c3a8'], ylim=(0, 100))
    rule = bias['shortcut_rule_accuracy']
    plots.grouped_bars(['All HC3 answers\n(2/3 are human)', 'Balanced test set\n(50% / 50%)'],
                       {'Raw text': [rule['all_answers_raw'] * 100, rule['balanced_test_raw'] * 100],
                        'Cleaned text': [rule['all_answers_clean'] * 100, rule['balanced_test_clean'] * 100]},
                       figures_dir / 'bias_rule_accuracy.png',
                       'Rule "space before punctuation → human": accuracy before/after cleaning', 'Accuracy, %',
                       ['#6d6c66', plots.MODEL_COLOR], ylim=(0, 100))
    stress = pd.DataFrame(bias['stress_test'])
    settings = list(dict.fromkeys(stress.setting))
    char_name, word_name = list(dict.fromkeys(stress.model))
    pick = lambda model, col: [float(stress[(stress.model == model) & (stress.setting == s)][col].iloc[0]) * 100
                               for s in settings]
    plots.grouped_bars(settings, {f'{char_name}': pick(char_name, 'accuracy'),
                                  f'{word_name}': pick(word_name, 'accuracy')},
                       figures_dir / 'bias_shortcut_stress_test.png',
                       'Stress test: train on raw HC3 text, test on cleaned text (accuracy)',
                       'Accuracy on the test set, %', ['#eda100', plots.MODEL_COLOR], ylim=(0, 105))
    plots.grouped_bars(settings, {f'{char_name}': pick(char_name, 'false_positive_rate'),
                                  f'{word_name}': pick(word_name, 'false_positive_rate')},
                       figures_dir / 'bias_shortcut_false_positives.png',
                       'Stress test: humans wrongly flagged as ChatGPT (false positive rate)', '% of human texts',
                       ['#eda100', plots.MODEL_COLOR],
                       ylim=(0, max(5.0, 1.35 * max(pick(char_name, 'false_positive_rate') + pick(word_name, 'false_positive_rate')))))
    log('figures done')

    # ---------------- tables + json
    rows = []
    for name in sorted(table, key=lambda n: -table[n]['f1']):
        r = results[name]
        rows.append([name, f"{r['accuracy']:.3f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}",
                     f"{r['f1']:.3f} [{r['f1_ci'][0]:.3f}–{r['f1_ci'][1]:.3f}]", f"{r['roc_auc']:.4f}",
                     f"{r['false_positive_rate']:.3f}", f"{r['tpr_at_1fpr']:.3f}"])
    main_table = pd.DataFrame(rows, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 [95% CI]', 'ROC-AUC',
                                             'FPR (humans flagged)', 'TPR @ 1% FPR'])
    lodo_md = ''
    if lodo:
        frame = pd.DataFrame(lodo)
        summary = frame.groupby('model', sort=False).agg(f1_seen=('f1_seen', 'mean'), f1_unseen=('f1_unseen', 'mean'),
                                                         f1_drop=('f1_drop', 'mean'), worst=('f1_unseen', 'min'),
                                                         domains=('domain', 'count'))
        summary = summary.sort_values('f1_unseen', ascending=False).round(3).reset_index()
        summary.columns = ['Model', 'Mean F1, domain seen', 'Mean F1, domain unseen', 'Mean drop',
                           'Worst unseen F1', 'Domains tested']
        lodo_md = markdown_table(summary)
        detail = frame.pivot(index='model', columns='domain', values='f1_unseen').round(3)
        detail = detail.astype(object).where(detail.notna(), '—')       # "—" = not tested
        detail.columns = [DOMAIN_NAMES.get(c, c) for c in detail.columns]
        lodo_md += '\n\nF1 on each unseen domain:\n\n' + markdown_table(detail.reset_index())
    (out_dir / 'results_table.md').write_text(
        '# Final results (test set, ' + f'{len(te)} texts)\n\n' + markdown_table(main_table) +
        '\n\n# Cross-domain generalisation (leave-one-domain-out)\n\n' + lodo_md + '\n', encoding='utf-8')
    main_table.to_csv(out_dir / 'results_table.csv', index=False)

    output = {
        'meta': {'created': time.strftime('%Y-%m-%d %H:%M'), 'seed': SEED, 'python': platform.python_version(),
                 'sklearn': sklearn.__version__, 'torch': torch.__version__, 'pandas': pd.__version__,
                 'n_test': int(len(te)), 'positive_class': 'ChatGPT', 'threshold': 0.5},
        'cleaning_protocol': steps, 'audit': audit_result, 'tuning': tuning,
        'test_results': results, 'per_domain_f1': per_domain, 'error_rate_by_length': by_length,
        'length_bucket_sizes': bucket_sizes, 'error_examples': examples, 'bias': bias, 'lodo': lodo,
        'transformer': {k: v for k, v in transformer.items() if k.endswith('_info')},
        **{k: v for k, v in extra.items()},
    }
    (out_dir / 'results.json').write_text(json.dumps(output, indent=2, ensure_ascii=False, default=float),
                                          encoding='utf-8')
    log(f'saved {out_dir / "results.json"} and results_table.md')
    print('\n' + markdown_table(main_table))
    return results


def main():
    args = parse_args()
    ensure_dirs()
    torch.set_num_threads(4)            # leave CPU for a transformer job that may run in parallel
    out_dir = FINAL / 'quick' if args.quick else FINAL
    figures_dir = out_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = CACHE / ('state_quick.joblib' if args.quick else 'state.joblib')

    raw, long_clean, pairs, steps = data.build_dataset(save=not args.quick)
    if args.quick:
        pairs = subsample(pairs, 400, 100)
    tr, va, te = (pairs[pairs.split == s] for s in ('train', 'val', 'test'))
    log(f'data: train {len(tr)}, validation {len(va)}, test {len(te)} texts')

    if args.reuse and state_path.exists():
        state = joblib.load(state_path)
        log('reusing cached predictions, LODO and bias results')
    else:
        audit_result = audit(pairs, tr, te)
        log('audit done: ' + json.dumps({k: v for k, v in audit_result.items() if 'same' in k or 'identical' in k
                                         or 'near' in k}))
        bias = stage_bias(long_clean if not args.quick else long_clean.sample(5000, random_state=SEED), tr, te)
        style = style_tables(pairs) if not args.quick else features.feature_table(pairs.text_clean).set_index(
            pairs.index)
        style.index = pairs.index
        pred_val, pred_test, tuning, extra = stage_models(pairs, tr, va, te, style, args.quick)
        lodo = stage_lodo(tr, va, te, style, tuning, pred_test, args.quick)
        state = {'audit': audit_result, 'bias': bias, 'pred_test': pred_test, 'pred_val': pred_val,
                 'tuning': tuning, 'extra': extra, 'lodo': lodo}
        joblib.dump(state, state_path)
    stage_report(out_dir, figures_dir, te, dict(state['pred_test']), state['extra'], state['tuning'], state['bias'],
                 state['lodo'], state['audit'], steps, 100 if args.quick else args.n_boot)
    log('all done')


if __name__ == '__main__':
    main()
