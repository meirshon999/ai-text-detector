"""New evaluation sets added after the defense: modern LLMs, attacks and mixed texts.

Every new set is built on the SAME test questions of HC3, so the question, the domain and the
human answer stay fixed and only the author of the machine text changes (a controlled comparison).

    Generators (RQ4)   Claude Haiku 5.5 / Sonnet 5.5 / Opus 5.5 (API), Qwen2.5-1.5B and SmolLM2-1.7B (open, local)
    Attacks (RQ5)      paraphrase and "humanize" rewrites of the HC3 ChatGPT answers, typos
    Mixed texts (RQ5)  human beginning + ChatGPT continuation, 0-100% machine share

All texts are stored in reports/final/modern/ so the evaluation can be re-run without the API.
"""
import random
import re

import numpy as np
import pandas as pd

from .config import FINAL, SEED
from .data import load_pairs, normalise

MODERN = FINAL / 'modern'
QUESTIONS_FILE = MODERN / 'questions.parquet'
TRAIN_QUESTIONS_FILE = MODERN / 'train_questions.parquet'   # for the augmentation experiment (TRAIN split only)
TRAIN_PER_DOMAIN = 300
TEXTS_FILE = MODERN / 'texts.parquet'          # one row per new text: qid, source, text, y
PER_DOMAIN = 120

# Plain prompt, as close as possible to how HC3 was collected (the question alone). The one extra line
# stops Markdown: modern chat models use headings / bullet lists, which would make detection trivially easy
# for the wrong reason (the same lesson as the HC3 formatting artifacts).
ANSWER_SYSTEM = ('Answer the user\'s question. Write plain prose paragraphs: no Markdown, no headings, '
                 'no bullet or numbered lists, no bold text.')
PARAPHRASE_PROMPT = ('Paraphrase the text below. Keep the meaning, the level of detail and roughly the same '
                     'length, but change the wording and sentence structure. Output only the paraphrased '
                     'text, in plain prose without Markdown.\n\n<text>\n{text}\n</text>')
HUMANIZE_PROMPT = ('Rewrite the answer below so that it reads as if a regular person typed it on an online '
                   'forum: relaxed tone, natural rhythm, contractions, varied sentence length, no '
                   'stock phrases such as "overall" or "it is important to note". Keep the facts and roughly '
                   'the same length. Output only the rewritten answer, in plain prose without Markdown.'
                   '\n\n<answer>\n{text}\n</answer>')

SOURCE_NAMES = {
    'human': 'Human (HC3)',
    'chatgpt_hc3': 'ChatGPT 2023 (HC3)',
    'claude-haiku-5-5': 'Claude Haiku 5.5',
    'claude-sonnet-5-5': 'Claude Sonnet 5.5',
    'claude-opus-5-5': 'Claude Opus 5.5',
    'qwen2.5-1.5b': 'Qwen2.5-1.5B-Instruct (open)',
    'smollm2-1.7b': 'SmolLM2-1.7B-Instruct (open)',
    'paraphrase': 'Attack: ChatGPT paraphrased by Claude',
    'humanize': 'Attack: ChatGPT "humanized" by Claude',
    'typos': 'Attack: ChatGPT + 5% typos',
}
GENERATORS = ['chatgpt_hc3', 'claude-haiku-5-5', 'claude-sonnet-5-5', 'claude-opus-5-5',
              'qwen2.5-1.5b', 'smollm2-1.7b']
ATTACKS = ['chatgpt_hc3', 'paraphrase', 'humanize', 'typos']


def select_questions(per_domain=PER_DOMAIN, split='test'):
    """Up to `per_domain` questions per domain of one split (fixed seed), with the HC3 human and ChatGPT answers."""
    test = load_pairs().query('split == @split')
    qids = (test.drop_duplicates('qid').groupby('domain', group_keys=False)
            .apply(lambda g: g.sample(min(per_domain, len(g)), random_state=SEED)).qid)
    rows = test[test.qid.isin(qids)]
    human = rows[rows.y == 0].set_index('qid')
    machine = rows[rows.y == 1].set_index('qid')
    questions = pd.DataFrame({'qid': human.index, 'domain': human.domain.values,
                              'question': human.question.values,
                              'human_text': human.text_clean.values,
                              'chatgpt_text': machine.loc[human.index, 'text_clean'].values})
    return questions.sort_values('qid').reset_index(drop=True)


_MARKDOWN = [(re.compile(r'^\s*#{1,6}\s*', re.M), ''), (re.compile(r'\*\*(.+?)\*\*'), r'\1'),
             (re.compile(r'^\s*[-*•]\s+', re.M), ''), (re.compile(r'^\s*\d+[.)]\s+', re.M), '')]


def clean_generated(text: str) -> str:
    """Removes any Markdown left by a model, then applies exactly the same normalisation as for HC3."""
    for pattern, replacement in _MARKDOWN:
        text = pattern.sub(replacement, text)
    return normalise(text)


# ------------------------------------------------------------------ attacks built locally (no API)
def add_typos(text: str, rate=0.05, seed=SEED) -> str:
    """Character-level noise in `rate` of the words: swap, drop or double one letter."""
    rng = random.Random(f'{seed}-{text[:40]}')
    words = text.split(' ')
    for i, word in enumerate(words):
        letters = [j for j, c in enumerate(word) if c.isalpha()]
        if len(letters) < 4 or rng.random() > rate:
            continue
        j = rng.choice(letters[1:-1])
        kind = rng.choice(['swap', 'drop', 'double'])
        if kind == 'swap' and j + 1 < len(word):
            word = word[:j] + word[j + 1] + word[j] + word[j + 2:]
        elif kind == 'drop':
            word = word[:j] + word[j + 1:]
        else:
            word = word[:j] + word[j] + word[j:]
        words[i] = word
    return ' '.join(words)


_SENTENCE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"(])')


def sentences(text: str):
    return [s for s in _SENTENCE.split(text) if s.strip()]


MIX_SHARES = [0.0, 0.25, 0.5, 0.75, 1.0]


def mixed_texts(questions: pd.DataFrame, shares=MIX_SHARES, n_words=150):
    """Human beginning + ChatGPT continuation to the same question.

    Both answers are cut to the same total length (~`n_words`) so that only the machine share changes.
    """
    rows = []
    for q in questions.itertuples():
        human, machine = sentences(q.human_text), sentences(q.chatgpt_text)
        if sum(len(s.split()) for s in human) < n_words * 0.75 or len(machine) < 3:
            continue
        for share in shares:
            human_part, machine_part, words = [], [], 0
            for s in human:
                if words >= n_words * (1 - share):
                    break
                human_part.append(s)
                words += len(s.split())
            for s in machine:
                if share == 0 or words >= n_words:
                    break
                machine_part.append(s)
                words += len(s.split())
            text = ' '.join(human_part + machine_part)
            real = len(' '.join(machine_part).split()) / max(len(text.split()), 1)
            rows.append({'qid': q.qid, 'domain': q.domain, 'share': share, 'machine_share_real': real,
                         'text': text})
    return pd.DataFrame(rows)


def load_texts() -> pd.DataFrame:
    if not TEXTS_FILE.exists():
        raise FileNotFoundError(f'{TEXTS_FILE} not found. Run the generation scripts first (see README).')
    return pd.read_parquet(TEXTS_FILE)


def bootstrap_rate(values, n_boot=1000, seed=SEED):
    """Mean of a 0/1 vector with a 95% bootstrap interval (detection rate, false-positive rate)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(1)
    return float(values.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
