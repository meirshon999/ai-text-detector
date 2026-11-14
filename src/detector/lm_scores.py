"""Zero-shot scores from GPT-2 (no training on HC3).

For every text we run GPT-2 once and compute two scores from the same next-token probabilities:

* Perplexity (PPL) = exp(average negative log-probability of the real tokens).
  Low perplexity = the text is predictable for GPT-2, which is typical of LLM output.
* Fast-DetectGPT criterion (Bao et al., ICLR 2024), analytical version with GPT-2 as both the
  sampling and the scoring model:  (log p(x) - E[log p(x~)]) / sqrt(Var[log p(x~)]),
  where x~ are alternative tokens the model itself would sample. Machine text is usually
  close to the most probable choices, so its criterion is higher.
"""
import os

import numpy as np
import pandas as pd
import torch

from .config import GPT2_NAME

os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
MAX_TOKENS = 128          # same as the midterm: first 128 GPT-2 tokens of each text


class GPT2Scorer:
    def __init__(self, model_name=GPT2_NAME, max_tokens=MAX_TOKENS, local_files_only=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.tokenizer.pad_token = self.tokenizer.eos_token        # GPT-2 has no padding token
        self.tokenizer.padding_side = 'right'
        self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only).eval()
        self.max_tokens = max_tokens

    @torch.no_grad()
    def score_batch(self, texts):
        enc = self.tokenizer(list(texts), return_tensors='pt', truncation=True, max_length=self.max_tokens,
                             padding=True)
        logits = self.model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask']).logits
        log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)      # prediction for token t+1
        targets = enc['input_ids'][:, 1:]
        mask = enc['attention_mask'][:, 1:].float()                           # ignore padding
        token_ll = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        probs = log_probs.exp()
        expected_ll = (probs * log_probs).sum(-1)                              # E[log p] = -entropy
        variance_ll = (probs * log_probs.pow(2)).sum(-1) - expected_ll.pow(2)
        n_tokens = mask.sum(1).clamp(min=1)
        log_ppl = -(token_ll * mask).sum(1) / n_tokens
        fast_detectgpt = ((token_ll - expected_ll) * mask).sum(1) / (variance_ll * mask).sum(1).clamp(min=1e-6).sqrt()
        return pd.DataFrame({'log_ppl': log_ppl.numpy(), 'perplexity': log_ppl.exp().numpy(),
                             'fast_detectgpt': fast_detectgpt.numpy(), 'n_tokens': n_tokens.numpy().astype(int)})

    def score(self, texts, batch_size=8, progress_every=500):
        texts = list(texts)
        # Sorting by length keeps padding (wasted computation) small; the order is restored at the end.
        order = np.argsort([len(t) for t in texts])
        parts = []
        for start in range(0, len(texts), batch_size):
            batch_idx = order[start:start + batch_size]
            part = self.score_batch([texts[i] for i in batch_idx])
            part.index = batch_idx
            parts.append(part)
            if progress_every and (start // batch_size) % max(progress_every // batch_size, 1) == 0:
                print(f'    GPT-2 scores: {start + len(batch_idx)}/{len(texts)}', flush=True)
        return pd.concat(parts).sort_index()
