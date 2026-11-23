"""Which PART of a text looks machine-written? Sentence-level scores and the human -> AI boundary.

A detector trained on whole answers is applied to a sliding window of sentences (the sentence itself
plus `radius` neighbours on each side), so every sentence gets its own P(machine). For a text that starts
human and continues with AI, the most likely boundary is the split that best separates low scores
(before) from high scores (after).

    from detector.segments import sentence_windows, change_point
"""
import re

import numpy as np

_SENTENCE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"(])')


def split_sentences(text: str):
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def sentence_windows(sentences, radius=1):
    """One text per sentence: the sentence with `radius` neighbours on each side."""
    return [' '.join(sentences[max(0, i - radius):i + radius + 1]) for i in range(len(sentences))]


def split_texts(sentences):
    """Texts for the split method: for every split k = 1..n-1 the part before it and the part after it."""
    n = len(sentences)
    return ([' '.join(sentences[:k]) for k in range(1, n)], [' '.join(sentences[k:]) for k in range(1, n)])


def split_boundary(p_before, p_after):
    """Split method: the boundary is where P(AI | part after) - P(AI | part before) is largest.

    Each part is a longer text than a 3-sentence window, so a whole-text detector judges it more reliably.
    Returns (index of the first machine sentence, the margin)."""
    gain = np.asarray(p_after) - np.asarray(p_before)
    k = int(np.argmax(gain))
    return k + 1, float(gain[k])


def change_point(probs, weights=None):
    """Index of the first machine sentence (0 = all machine, len = all human) for a human -> AI text.

    Picks the split with the smallest weighted squared error of a step function: low before, high after.
    `weights` (e.g. words per sentence) make long sentences count more.
    """
    p = np.asarray(probs, dtype=float)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float)
    best, best_err = len(p), np.inf
    for k in range(len(p) + 1):
        err = 0.0
        for part, wp in ((p[:k], w[:k]), (p[k:], w[k:])):
            if len(part):
                mean = np.average(part, weights=wp)
                err += float(np.sum(wp * (part - mean) ** 2))
        # a split is only a human -> AI boundary if the part after it scores higher
        if k not in (0, len(p)) and np.average(p[k:], weights=w[k:]) <= np.average(p[:k], weights=w[:k]):
            continue
        if err < best_err - 1e-12:
            best, best_err = k, err
    return best
