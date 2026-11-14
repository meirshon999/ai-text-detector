"""Hand-crafted stylometric features (notebook 03), shared by the experiments and the demo.

All counts are per 100 words, so longer texts do not automatically get larger values.
The length itself (n_words) is computed for analysis but NOT given to the stylometric
detector: the length-only baseline already shows how much length alone explains.
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOP_WORDS

CONNECTIVES = ['however', 'additionally', 'furthermore', 'moreover', 'in general', 'generally', 'overall',
               'firstly', 'secondly', 'finally', 'in addition', 'therefore', 'for example', 'on the other hand',
               'it is important', 'in summary', 'ultimately', 'typically', 'it is also']
HEDGES = ['i think', 'i guess', 'maybe', 'probably', 'i believe', 'pretty much', 'kind of', 'sort of', 'imo',
          'i mean']
INFORMAL = ['lol', 'tl;dr', 'tldr', 'yeah', 'yep', 'nope', 'gonna', 'wanna', 'kinda', 'dunno', 'btw', 'haha',
            'ok', 'okay', 'stuff', 'basically', 'literally', 'damn', 'wtf', 'omg']

SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
WORD = re.compile(r"[A-Za-z']+")
VOWEL_GROUPS = re.compile(r'[aeiouy]+')
# One alternation, longest forms first, so "I'm" is counted once (the midterm regex counted it twice).
FIRST_PERSON = re.compile(r"\b(?:i'm|i've|i'd|i'll|i|my|me)\b")
SECOND_PERSON = re.compile(r'\b(?:you|your)\b')
CONTRACTION = re.compile(r"\b\w+'(?:t|s|re|ve|ll|d|m)\b")


def _phrase_regex(phrases):
    return re.compile(r'\b(?:' + '|'.join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) + r')\b')


CONNECTIVES_RE, HEDGES_RE, INFORMAL_RE = map(_phrase_regex, (CONNECTIVES, HEDGES, INFORMAL))

# The 18 features of the stylometric detector (same names as in notebook 03)
STYLE_FEATURES = [
    'mean_sentence_len', 'burstiness (sentence-len std)', 'MATTR (vocabulary richness)', 'mean_word_len',
    'long_words_% (7+ letters)', 'exclamation !', 'question ?', 'ellipsis ...', 'brackets ( [', 'quotes "',
    'commas ,', 'connectives', 'first person I', 'second person you', 'hedges', 'informal/slang',
    'contractions', 'function words %',
]
# Candidate extra features. They are kept only if they improve the validation score (see run_experiments.py).
EXTRA_FEATURES = ['readability (Flesch)', 'repeated bigrams %']


def mattr(words, window=50):
    """Moving-average type-token ratio: vocabulary richness that does not depend on text length."""
    if len(words) <= window:
        return len(set(words)) / max(len(words), 1)
    return float(np.mean([len(set(words[i:i + window])) / window for i in range(0, len(words) - window + 1, 5)]))


def _syllables(word):
    count = len(VOWEL_GROUPS.findall(word))
    if word.endswith('e') and not word.endswith('le') and count > 1:
        count -= 1
    return max(count, 1)


def text_features(text: str) -> dict:
    low = text.lower()
    words = WORD.findall(low)
    n = max(len(words), 1)
    per100 = 100 / n
    sentences = [s for s in SENTENCE_SPLIT.split(text) if WORD.search(s)]
    sentence_lengths = np.array([len(WORD.findall(s)) for s in sentences]) if sentences else np.array([n])
    bigrams = list(zip(words, words[1:]))
    repeated_bigrams = len(bigrams) - len(set(bigrams))
    syllables_per_word = np.mean([_syllables(w) for w in words]) if words else 1.0
    return {
        'n_words': n,
        'mean_sentence_len': sentence_lengths.mean(),
        'burstiness (sentence-len std)': sentence_lengths.std(),
        'MATTR (vocabulary richness)': mattr(words),
        'mean_word_len': np.mean([len(w) for w in words]) if words else 0.0,
        'long_words_% (7+ letters)': sum(len(w) >= 7 for w in words) * per100,
        'exclamation !': text.count('!') * per100,
        'question ?': text.count('?') * per100,
        'ellipsis ...': text.count('...') * per100,
        'brackets ( [': (text.count('(') + text.count('[')) * per100,
        'quotes "': text.count('"') * per100,
        'commas ,': text.count(',') * per100,
        'connectives': len(CONNECTIVES_RE.findall(low)) * per100,
        'first person I': len(FIRST_PERSON.findall(low)) * per100,
        'second person you': len(SECOND_PERSON.findall(low)) * per100,
        'hedges': len(HEDGES_RE.findall(low)) * per100,
        'informal/slang': len(INFORMAL_RE.findall(low)) * per100,
        'contractions': len(CONTRACTION.findall(low)) * per100,
        'function words %': sum(w in STOP_WORDS for w in words) * per100,
        'readability (Flesch)': 206.835 - 1.015 * sentence_lengths.mean() - 84.6 * syllables_per_word,
        'repeated bigrams %': repeated_bigrams / max(len(bigrams), 1) * 100,
    }


def feature_table(texts) -> pd.DataFrame:
    return pd.DataFrame([text_features(t) for t in texts]).astype(float)


# Plain-language explanation of each feature, used in the demo and the guide
FEATURE_EXPLANATIONS = {
    'mean_sentence_len': 'Average sentence length in words',
    'burstiness (sentence-len std)': 'How much sentence length varies (humans are usually more "bursty")',
    'MATTR (vocabulary richness)': 'Share of different words in a 50-word window (vocabulary richness)',
    'mean_word_len': 'Average word length in letters',
    'long_words_% (7+ letters)': 'Long words per 100 words',
    'exclamation !': 'Exclamation marks per 100 words',
    'question ?': 'Question marks per 100 words',
    'ellipsis ...': 'Ellipses per 100 words',
    'brackets ( [': 'Brackets per 100 words',
    'quotes "': 'Quotation marks per 100 words',
    'commas ,': 'Commas per 100 words',
    'connectives': '"However", "additionally", "overall"... per 100 words',
    'first person I': '"I", "my", "me" per 100 words',
    'second person you': '"You", "your" per 100 words',
    'hedges': '"I think", "maybe", "probably"... per 100 words',
    'informal/slang': '"lol", "yeah", "gonna", "stuff"... per 100 words',
    'contractions': '"don\'t", "it\'s"... per 100 words',
    'function words %': 'Share of function words ("the", "of", "and"...)',
    'readability (Flesch)': 'Flesch reading ease (higher = easier to read)',
    'repeated bigrams %': 'Share of word pairs that repeat inside the text',
}
