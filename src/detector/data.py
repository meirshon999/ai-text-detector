"""Loading, cleaning and splitting HC3.

The code reproduces notebooks 01 and 02 exactly (same rules, same seed), so the
train / validation / test split is identical to the one used at the midterm.
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import DATA_INTERIM, DATA_RAW, DOMAINS, SEED

ERROR_PATTERN = 'There was an error generating a response|Too many requests'
# Found in the final audit: some "ChatGPT answers" are a Cloudflare page saved instead of an answer.
SCRAPING_ERROR_PATTERN = 'Checking if the site connection is secure'

# Formatting normalisation. The same rules are applied to human and ChatGPT texts:
# they change how a text was stored, not who wrote it.
CLEANING_RULES = [
    (r'\\n', ' '),                                  # literal "\n" characters left from scraping
    (r'\s+', ' '),                                  # line breaks / repeated spaces -> one space
    (r'([a-z0-9\)])\.([A-Z])', r'\1. \2'),          # "end.Next"     -> "end. Next"
    (r'\s+([,.!?;:%])', r'\1'),                     # "word , x"     -> "word, x"
    (r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r'\1'),    # "do n't"       -> "don't"
    (r'\(\s+', '('), (r'\s+\)', ')'),               # "( x )"        -> "(x)"
    (r'(\w)\s+-\s+(\w)', r'\1-\2'),                 # "well - known" -> "well-known"
    (r'"\s+([^"]*?)\s+"', r'"\1"'),                 # '" quote "'    -> '"quote"'
    (r'\$\s+(\d)', r'$\1'),                         # "$ 5"          -> "$5"
]
_COMPILED_RULES = [(re.compile(pattern), replacement) for pattern, replacement in CLEANING_RULES]

# Formatting artifacts of HC3 (notebook 01). Used for the dataset-bias analysis.
ARTIFACT_PATTERNS = {
    'Space before punctuation ("word , word")': r' [,.!?;:]',
    "Split contractions (\"do n't\")": r" n't\b",
    'Missing space after period ("end.Next")': r'[a-z]\.[A-Z]',
    'Line break inside the answer': r'\n',
}
SHORTCUT_RULE = r' [,.!?;:]'          # "space before punctuation -> human"
URL_PLACEHOLDER = re.compile(r'\bURL_\d+\b')


def normalise(text: str) -> str:
    for pattern, replacement in _COMPILED_RULES:
        text = pattern.sub(replacement, text)
    return text.strip()


def remove_url_placeholders(text: str) -> str:
    """HC3 replaced links in human answers with "URL_0", "URL_1"... (a dataset convention)."""
    return re.sub(r'\s+', ' ', URL_PLACEHOLDER.sub(' ', text)).strip()


def load_raw() -> pd.DataFrame:
    """HC3 jsonl files -> one row per answer (domain, qid, question, text, label)."""
    missing = [d for d in DOMAINS if not (DATA_RAW / f'{d}.jsonl').exists()]
    if missing:
        raise FileNotFoundError(f'HC3 files are missing in {DATA_RAW}: {missing}. '
                                'Run:  python scripts/download_data.py')
    rows = []
    for domain in DOMAINS:
        with open(DATA_RAW / f'{domain}.jsonl', encoding='utf-8') as f:
            for qid, line in enumerate(f):
                record = json.loads(line)
                for answer in record['human_answers']:
                    rows.append((domain, f'{domain}_{qid}', record['question'], answer or '', 'human'))
                for answer in record['chatgpt_answers']:
                    rows.append((domain, f'{domain}_{qid}', record['question'], answer or '', 'chatgpt'))
    df = pd.DataFrame(rows, columns=['domain', 'qid', 'question', 'text', 'label'])
    df['n_words'] = df.text.str.split().str.len()
    df['is_error'] = df.text.str.contains(ERROR_PATTERN, regex=True)
    return df


def clean(df: pd.DataFrame):
    """Cleaning protocol of notebook 02. Returns the cleaned table and the log of every step."""
    steps = [('Original HC3 (English)', len(df))]
    df = df[df.text.str.strip() != '']
    steps.append(('1. Remove empty answers', len(df)))
    df = df[~df.is_error].copy()
    steps.append(('2. Remove ChatGPT server error messages', len(df)))
    df['text_clean'] = df.text.map(normalise)
    df['n_words'] = df.text_clean.str.split().str.len()
    steps.append(('3. Normalise formatting (no rows removed)', len(df)))
    df = df[df.n_words >= 5]
    steps.append(('4. Remove answers shorter than 5 words', len(df)))
    df = df.drop_duplicates('text_clean')
    steps.append(('5. Remove exact duplicates (after normalisation)', len(df)))
    return df, steps


def make_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """One human + one ChatGPT answer per question, then an 80/10/10 split by question."""
    pairs = df.groupby(['qid', 'label']).sample(n=1, random_state=SEED)
    keep = pairs.groupby('qid').label.transform('nunique') == 2
    pairs = pairs[keep].reset_index(drop=True)
    pairs['y'] = (pairs.label == 'chatgpt').astype(int)            # 1 = ChatGPT, 0 = human

    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, rest_idx = next(first.split(pairs, groups=pairs.qid))
    rest = pairs.iloc[rest_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_idx, test_idx = next(second.split(rest, groups=rest.qid))
    pairs['split'] = 'train'
    pairs.loc[rest.index[val_idx], 'split'] = 'val'
    pairs.loc[rest.index[test_idx], 'split'] = 'test'
    return pairs


def remove_scraping_errors(pairs: pd.DataFrame) -> pd.DataFrame:
    """Drops whole questions (both answers) whose answer is a saved Cloudflare page.

    Done after pairing, so the pairs and the split of all other questions stay exactly as at the midterm.
    """
    bad_questions = pairs.loc[pairs.text_clean.str.contains(SCRAPING_ERROR_PATTERN, regex=False), 'qid']
    return pairs[~pairs.qid.isin(bad_questions)].reset_index(drop=True)


def build_dataset(save: bool = True):
    """Raw HC3 -> cleaned long table -> balanced pairs with split. Returns (raw, long_clean, pairs, steps)."""
    raw = load_raw()
    long_clean, steps = clean(raw)
    pairs = make_pairs(long_clean)
    steps.append(('6. Keep 1 human + 1 ChatGPT answer per question (balanced pairs)', len(pairs)))
    pairs = remove_scraping_errors(pairs)
    steps.append(('7. Remove pairs with a saved Cloudflare page instead of an answer (final audit)', len(pairs)))
    if save:
        DATA_INTERIM.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(DATA_INTERIM / 'hc3_long.parquet', index=False)
        pairs.to_parquet(DATA_INTERIM / 'hc3_pairs.parquet', index=False)
        (DATA_INTERIM / 'cleaning_protocol.json').write_text(json.dumps(steps, indent=2))
    return raw, long_clean, pairs, steps


def load_pairs() -> pd.DataFrame:
    path = DATA_INTERIM / 'hc3_pairs.parquet'
    if not path.exists():
        build_dataset(save=True)
    # The filter is idempotent, so a pairs file saved at the midterm gives the same final dataset.
    return remove_scraping_errors(pd.read_parquet(path))


def split_frames(pairs: pd.DataFrame):
    return tuple(pairs[pairs.split == s].reset_index(drop=True) for s in ('train', 'val', 'test'))


def shortcut_rule_predict(texts: pd.Series) -> np.ndarray:
    """The one-line 'detector' with no AI: space before punctuation -> human (0), else ChatGPT (1)."""
    return np.where(texts.str.contains(SHORTCUT_RULE, regex=True), 0, 1)
