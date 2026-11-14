"""Project-wide settings: paths, random seed, domains and colours.

Every path is built from the location of this file, so scripts, notebooks and the demo
work no matter which folder they are started from (no absolute paths).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../ai-text-detector
DATA_RAW = ROOT / 'data' / 'raw'
DATA_INTERIM = ROOT / 'data' / 'interim'
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'
FINAL = REPORTS / 'final'
FIGURES = FINAL / 'figures'
CACHE = FINAL / 'cache'

SEED = 42
DOMAINS = ['finance', 'medicine', 'open_qa', 'reddit_eli5', 'wiki_csai']
DOMAIN_NAMES = {'finance': 'Finance', 'medicine': 'Medicine', 'open_qa': 'Open QA',
                'reddit_eli5': 'Reddit ELI5', 'wiki_csai': 'Wikipedia CS/AI'}

# Colours used in every figure and in the demo (colour-blind-safe pair)
HUMAN_COLOR = '#2a78d6'
CHATGPT_COLOR = '#eb6834'

GPT2_NAME = 'openai-community/gpt2'
TRANSFORMER_NAME = 'distilbert/distilroberta-base'


def ensure_dirs():
    for folder in (DATA_INTERIM, MODELS, FINAL, FIGURES, CACHE):
        folder.mkdir(parents=True, exist_ok=True)
