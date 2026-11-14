"""Downloads the two pretrained models used by the demo and the final notebook (once, about 0.8 GB):
GPT-2 (perplexity and Fast-DetectGPT) and DistilRoBERTa-base (the base of the LoRA adapter in models/).

Usage:  python scripts/download_models.py

Everything else (TF-IDF, stylometry, PyTorch network, the LoRA adapter, calibrations) is already in models/.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from detector.config import GPT2_NAME, TRANSFORMER_NAME  # noqa: E402

for name, loader in ((GPT2_NAME, AutoModelForCausalLM), (TRANSFORMER_NAME, AutoModelForSequenceClassification)):
    print(f'downloading {name} ...', flush=True)
    AutoTokenizer.from_pretrained(name)
    loader.from_pretrained(name)
print('Done. The demo and notebooks/04_Final_Project.ipynb can now use GPT-2 and DistilRoBERTa offline.')
