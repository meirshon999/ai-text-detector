"""Downloads the HC3 dataset (Guo et al., 2023) from Hugging Face into data/raw/.

Usage:  python scripts/download_data.py
"""
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

RAW = Path(__file__).resolve().parents[1] / 'data' / 'raw'
DOMAINS = ['finance', 'medicine', 'open_qa', 'reddit_eli5', 'wiki_csai']

RAW.mkdir(parents=True, exist_ok=True)
for name in [f'{d}.jsonl' for d in DOMAINS]:
    path = hf_hub_download('Hello-SimpleAI/HC3', name, repo_type='dataset')
    shutil.copy(path, RAW / name)
    print('downloaded', name)
