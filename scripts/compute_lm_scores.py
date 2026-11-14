"""Computes GPT-2 perplexity and the Fast-DetectGPT criterion for the validation and test texts.

Usage:   python scripts/compute_lm_scores.py
Output:  reports/final/cache/gpt2_scores.parquet   (about 20 minutes on a laptop CPU)

GPT-2 (~0.5 GB) is downloaded from Hugging Face on the first run.
The scores are cached, so run_experiments.py does not recompute them.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from detector import data                                   # noqa: E402
from detector.config import CACHE, ensure_dirs              # noqa: E402
from detector.lm_scores import GPT2Scorer                   # noqa: E402

CACHE_FILE = CACHE / 'gpt2_scores.parquet'


def compute(force=False):
    if CACHE_FILE.exists() and not force:
        print(f'GPT-2 scores already cached: {CACHE_FILE}')
        return
    ensure_dirs()
    pairs = data.load_pairs()
    subset = pairs[pairs.split.isin(['val', 'test'])].reset_index(drop=True)
    print(f'Scoring {len(subset)} validation + test texts with GPT-2...')
    start = time.time()
    scores = GPT2Scorer().score(subset.text_clean.tolist())
    result = subset[['qid', 'label', 'split', 'domain', 'y']].join(scores)
    result.to_parquet(CACHE_FILE, index=False)
    print(f'Saved {CACHE_FILE} ({time.time() - start:.0f}s)')


if __name__ == '__main__':
    compute(force='--force' in sys.argv)
