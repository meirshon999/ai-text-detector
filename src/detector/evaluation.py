"""Metrics used for every detector. ChatGPT = positive class (y = 1)."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from .config import SEED


def tpr_at_fpr(y_true, score, max_fpr=0.01):
    """Share of ChatGPT texts caught while at most 1% of human texts are falsely flagged."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(tpr[fpr <= max_fpr].max())


def compute_metrics(y_true, score, threshold=0.5):
    """`score` is a probability or any score where higher = more likely ChatGPT."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    if len(y_true) != len(score):
        raise ValueError(f'{len(y_true)} labels but {len(score)} scores')
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'n': int(len(y_true)),
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y_true, pred)),
        'precision': float(precision_score(y_true, pred, zero_division=0)),
        'recall': float(recall_score(y_true, pred, zero_division=0)),
        'f1': float(f1_score(y_true, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, score)),
        'tpr_at_1fpr': tpr_at_fpr(y_true, score),
        'false_positive_rate': float(fp / max(fp + tn, 1)),     # humans wrongly flagged as ChatGPT
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],   # rows = true human/ChatGPT
    }


def bootstrap_ci(y_true, score, groups, threshold=0.5, n_boot=1000, seed=SEED):
    """95% confidence intervals for F1, accuracy and ROC-AUC.

    We resample whole questions (both answers together), because the two answers
    to one question are not independent.
    """
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    _, group_index = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(group_index, kind='stable')          # rows sorted by question
    counts = np.bincount(group_index)
    starts = np.cumsum(counts) - counts
    rng = np.random.default_rng(seed)
    f1s, accs, aucs = [], [], []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(counts), len(counts))
        sizes = counts[chosen]
        offsets = np.arange(sizes.sum()) - np.repeat(np.cumsum(sizes) - sizes, sizes)
        idx = order[np.repeat(starts[chosen], sizes) + offsets]
        yt, yp, sc = y_true[idx], pred[idx], score[idx]
        tp = np.sum((yp == 1) & (yt == 1))
        f1s.append(2 * tp / max(yp.sum() + yt.sum(), 1))
        accs.append((yt == yp).mean())
        aucs.append(roc_auc_score(yt, sc))
    q = lambda values: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
    return {'f1_ci': q(f1s), 'accuracy_ci': q(accs), 'roc_auc_ci': q(aucs)}


def choose_threshold(y_val, score_val):
    """Threshold for a score-based detector (no probabilities): best validation accuracy."""
    y_val = np.asarray(y_val).astype(int)
    score_val = np.asarray(score_val, dtype=float)
    candidates = np.quantile(score_val, np.linspace(0.02, 0.98, 193))
    accuracy = [((score_val >= t).astype(int) == y_val).mean() for t in candidates]
    return float(candidates[int(np.argmax(accuracy))])
