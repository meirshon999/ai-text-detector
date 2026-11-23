"""Answers of small OPEN models to the same HC3 test questions (runs locally on CPU, no API key).

    python scripts/generate_open_models.py                       # both models, 50 questions per domain (~2-3 h)
    python scripts/generate_open_models.py --model qwen2.5-1.5b --limit 20

Resumable: answers are appended to reports/final/modern/raw/<model>.jsonl and skipped on the next run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch                                                       # noqa: E402

from detector import modern                                        # noqa: E402
from detector.config import SEED                                   # noqa: E402

OPEN_MODELS = {'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
               'smollm2-1.7b': 'HuggingFaceTB/SmolLM2-1.7B-Instruct'}
RAW = modern.MODERN / 'raw'


def questions_file():
    modern.MODERN.mkdir(parents=True, exist_ok=True)
    if not modern.QUESTIONS_FILE.exists():
        modern.select_questions().to_parquet(modern.QUESTIONS_FILE, index=False)
    import pandas as pd
    return pd.read_parquet(modern.QUESTIONS_FILE)


def done_ids(path):
    if not path.exists():
        return set()
    return {json.loads(line)['qid'] for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}


def generate(key, questions, batch_size, max_new_tokens, limit, per_domain):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = OPEN_MODELS[key]
    out = RAW / f'{key}.jsonl'
    RAW.mkdir(parents=True, exist_ok=True)
    if per_domain:      # CPU generation is slow: a fixed, balanced subset of the selected questions
        questions = questions.groupby('domain', group_keys=False).head(per_domain)
    todo = questions[~questions.qid.isin(done_ids(out))]
    if limit:
        todo = todo.head(limit)
    if todo.empty:
        print(f'{key}: nothing to do')
        return
    print(f'{key}: {len(todo)} questions with {name}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    torch.manual_seed(SEED)
    start = time.time()
    for i in range(0, len(todo), batch_size):
        batch = todo.iloc[i:i + batch_size]
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'system', 'content': modern.ANSWER_SYSTEM}, {'role': 'user', 'content': q}],
            tokenize=False, add_generation_prompt=True) for q in batch.question]
        enc = tokenizer(prompts, return_tensors='pt', padding=True)
        with torch.no_grad():
            generated = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7,
                                       top_p=0.9, repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id)
        answers = tokenizer.batch_decode(generated[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        with open(out, 'a', encoding='utf-8') as f:
            for qid, answer in zip(batch.qid, answers):
                f.write(json.dumps({'qid': qid, 'source': key, 'model': name, 'raw': answer}) + '\n')
        done = i + len(batch)
        print(f'  {done}/{len(todo)}  {(time.time() - start) / done:.1f} s per answer', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(OPEN_MODELS), help='default: all')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--per-domain', type=int, default=50, help='questions per domain (0 = all 555)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=280)
    args = parser.parse_args()
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    questions = questions_file()
    for key in [args.model] if args.model else OPEN_MODELS:
        generate(key, questions, args.batch_size, args.max_new_tokens, args.limit, args.per_domain)


if __name__ == '__main__':
    main()
