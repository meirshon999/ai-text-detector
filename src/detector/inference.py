"""Batch scoring of new texts with every SAVED detector (the exact models used in the demo).

    from detector.inference import Detectors
    probs = Detectors().score(texts)      # DataFrame: one column of P(machine) per detector

No retraining happens here: thresholds and calibrations are the ones fitted on HC3 validation.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

from . import features as style
from .config import MODELS, TRANSFORMER_NAME

os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

DETECTORS = ['roberta_lora', 'tfidf', 'nn', 'style_rf', 'style_lr', 'ppl', 'fdg', 'length']
NAMES = {'roberta_lora': 'DistilRoBERTa + LoRA', 'tfidf': 'TF-IDF + LogReg', 'nn': 'PyTorch bag-of-embeddings',
         'style_rf': 'Stylometry + RF', 'style_lr': 'Stylometry + LogReg', 'ppl': 'GPT-2 perplexity',
         'fdg': 'Fast-DetectGPT (GPT-2)', 'length': 'Length baseline'}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Detectors:
    def __init__(self, use_lm=True, use_transformer=True, model_dir=MODELS):
        self.dir = model_dir
        self.local = {k: joblib.load(model_dir / f) for k, f in
                      {'length': 'length_baseline.joblib', 'tfidf': 'tfidf_logreg.joblib',
                       'style_rf': 'stylometry_rf.joblib', 'style_lr': 'stylometry_logreg.joblib'}.items()}
        from .models import NeuralDetector
        self.nn = NeuralDetector.load(model_dir / 'bag_of_embeddings')
        self.use_lm, self.use_transformer = use_lm, use_transformer

    def score(self, texts, gpt2_scores=None, progress=True):
        """`gpt2_scores` (optional): precomputed GPT-2 table with log_ppl / fast_detectgpt, same order."""
        texts = list(texts)
        out = pd.DataFrame(index=range(len(texts)))
        out['length'] = self.local['length'].predict_proba(texts)[:, 1]
        out['tfidf'] = self.local['tfidf'].predict_proba(texts)[:, 1]
        feats = style.feature_table(texts)
        for key in ('style_rf', 'style_lr'):
            bundle = self.local[key]
            out[key] = bundle['model'].predict_proba(feats[bundle['features']])[:, 1]
        out['nn'] = self.nn.predict_proba(texts)
        if self.use_lm:
            lm = gpt2_scores if gpt2_scores is not None else self.gpt2_scores(texts, progress)
            calibration = json.loads((self.dir / 'zero_shot_calibration.json').read_text())
            for key in ('ppl', 'fdg'):
                cal = calibration[key]
                out[key] = sigmoid(cal['coef'] * cal['sign'] * lm[cal['column']].to_numpy() + cal['intercept'])
            out['log_ppl'] = lm['log_ppl'].to_numpy()
            out['fast_detectgpt'] = lm['fast_detectgpt'].to_numpy()
        if self.use_transformer:
            out['roberta_lora'] = self.transformer_probs(texts, progress)
        return out

    @staticmethod
    def gpt2_scores(texts, progress=True):
        from .lm_scores import GPT2Scorer
        return GPT2Scorer(local_files_only=True).score(texts, progress_every=1000 if progress else 0)

    def transformer_probs(self, texts, progress=True, batch_size=32, max_length=256, folder=None, calibration=None):
        """P(machine) of a LoRA adapter (default: the demo one) after Platt calibration {coef, intercept}."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        folder = folder or self.dir / 'distilroberta_lora'
        base = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_NAME, num_labels=2,
                                                                  local_files_only=True)
        model = PeftModel.from_pretrained(base, folder).eval()
        tokenizer = AutoTokenizer.from_pretrained(folder)
        cal = calibration or json.loads((self.dir / 'transformer_calibration.json').read_text())
        order = np.argsort([len(t) for t in texts])
        raw = np.zeros(len(texts))
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                idx = order[start:start + batch_size]
                enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_length, padding=True,
                                return_tensors='pt')
                raw[idx] = torch.softmax(model(**enc).logits, dim=-1)[:, 1].numpy()
                if progress and (start // batch_size) % 40 == 0:
                    print(f'    DistilRoBERTa: {start + len(idx)}/{len(texts)}', flush=True)
        raw = np.clip(raw, 1e-7, 1 - 1e-7)
        return sigmoid(cal['coef'] * np.log(raw / (1 - raw)) + cal['intercept'])
