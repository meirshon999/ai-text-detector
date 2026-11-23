"""Fine-tunes DistilRoBERTa-base as a human / ChatGPT classifier: full fine-tuning vs LoRA.

Usage:
    python scripts/train_transformer.py                 # both modes (about 1 hour on a laptop CPU)
    python scripts/train_transformer.py --mode lora     # only LoRA
    python scripts/train_transformer.py --quick         # smoke test on a few texts (1-2 minutes)

Why DistilRoBERTa and a subset: we have no GPU. DistilRoBERTa (82M parameters, 6 layers) is a
distilled version of RoBERTa-base (125M, 12 layers) and is about 2x faster. We train on a random
subset of the training questions (default 2,500 questions = 5,000 texts, 1 epoch) and evaluate on
the FULL test set, the same one used for every other detector.

Outputs:
    reports/final/cache/transformer_<mode>_scores.parquet   P(ChatGPT) for validation-subset and test texts
    reports/final/cache/transformer_<mode>_info.json        settings, training curve, time
    models/distilroberta_lora/                               LoRA adapter + classifier head (small, used by the demo)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score                     # noqa: E402

from detector import data                                               # noqa: E402
from detector.config import CACHE, MODELS, SEED, TRANSFORMER_NAME, ensure_dirs   # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['full', 'lora', 'both'], default='both')
    parser.add_argument('--train-questions', type=int, default=2500)
    parser.add_argument('--val-questions', type=int, default=500)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--threads', type=int, default=8, help='CPU threads for PyTorch')
    parser.add_argument('--holdout', choices=['finance', 'medicine', 'open_qa', 'reddit_eli5', 'wiki_csai'],
                        help='generalisation test: train without this domain, test only on its test texts')
    parser.add_argument('--augment', type=int, default=0, metavar='N',
                        help='add N Claude Haiku 5.5 texts of TRAIN questions + N extra human answers '
                             '(see scripts/train_augmented.py); saves models/distilroberta_lora_augmented/')
    parser.add_argument('--quick', action='store_true', help='tiny run to check that everything works')
    return parser.parse_args()


def sample_questions(frame, n_questions, seed):
    rng = np.random.default_rng(seed)
    questions = frame.qid.unique()
    chosen = rng.choice(questions, size=min(n_questions, len(questions)), replace=False)
    return frame[frame.qid.isin(chosen)].reset_index(drop=True)


def length_grouped_batches(lengths, batch_size, rng):
    """Random batches of texts with similar length -> less padding -> faster training on CPU."""
    order = rng.permutation(len(lengths))
    chunk = batch_size * 50
    batches = []
    for start in range(0, len(order), chunk):
        part = order[start:start + chunk]
        part = part[np.argsort(lengths[part])]
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def predict(model, tokenizer, texts, max_length, batch_size=32):
    model.eval()
    order = np.argsort([len(t) for t in texts])
    probs = np.zeros(len(texts))
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_length, padding=True,
                        return_tensors='pt')
        probs[idx] = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].numpy()
    return probs


def build_model(mode):
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_NAME, num_labels=2)
    if mode == 'lora':
        from peft import LoraConfig, get_peft_model
        config = LoraConfig(task_type='SEQ_CLS', r=8, lora_alpha=16, lora_dropout=0.1,
                            target_modules=['query', 'value'])
        model = get_peft_model(model, config)       # base weights frozen; LoRA matrices + classifier trainable
    return model


def train_one(mode, train, val, test, args, tokenizer):
    from transformers import get_linear_schedule_with_warmup
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = build_model(mode)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    lr = 5e-4 if mode == 'lora' else 3e-5
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)

    texts = train.text_clean.tolist()
    labels = torch.tensor(train.y.values, dtype=torch.long)
    lengths = np.array([len(t) for t in texts])
    steps_per_epoch = int(np.ceil(len(texts) / args.batch_size))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps_per_epoch * args.epochs),
                                                steps_per_epoch * args.epochs)
    print(f'[{mode}] trainable parameters: {trainable:,} of {total:,} ({trainable / total:.1%}), lr={lr}')

    history, start, step = [], time.time(), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for batch in length_grouped_batches(lengths, args.batch_size, rng):
            enc = tokenizer([texts[i] for i in batch], truncation=True, max_length=args.max_length, padding=True,
                            return_tensors='pt')
            loss = model(**enc, labels=labels[batch]).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            running.append(loss.item())
            if step % 10 == 0:
                history.append({'step': step, 'epoch': epoch, 'train_loss': float(np.mean(running[-10:]))})
            if step % 25 == 0:
                elapsed = time.time() - start
                print(f'[{mode}] step {step}/{steps_per_epoch * args.epochs}  loss {np.mean(running[-25:]):.4f}  '
                      f'{elapsed / 60:.1f} min', flush=True)
    train_minutes = (time.time() - start) / 60

    start = time.time()
    val_prob = predict(model, tokenizer, val.text_clean.tolist(), args.max_length)
    test_prob = predict(model, tokenizer, test.text_clean.tolist(), args.max_length)
    predict_minutes = (time.time() - start) / 60
    print(f'[{mode}] val F1 {f1_score(val.y, val_prob >= 0.5):.4f} AUC {roc_auc_score(val.y, val_prob):.4f} | '
          f'test F1 {f1_score(test.y, test_prob >= 0.5):.4f} AUC {roc_auc_score(test.y, test_prob):.4f}')

    scores = (val.assign(p_chatgpt=val_prob), test.assign(p_chatgpt=test_prob))
    scores = __import__('pandas').concat(scores)[['qid', 'label', 'split', 'domain', 'y', 'p_chatgpt']]
    suffix = '_quick' if args.quick else (f'_lodo_{args.holdout}' if args.holdout else
                                          ('_augmented' if args.augment else ''))
    scores.to_parquet(CACHE / f'transformer_{mode}{suffix}_scores.parquet', index=False)
    info = {'mode': mode, 'base_model': TRANSFORMER_NAME, 'trainable_params': trainable, 'total_params': total,
            'learning_rate': lr, 'epochs': args.epochs, 'batch_size': args.batch_size,
            'max_length': args.max_length, 'train_texts': len(train), 'val_texts': len(val),
            'test_texts': len(test), 'train_minutes': train_minutes, 'predict_minutes': predict_minutes,
            'history': history}
    if mode == 'lora':
        info['lora'] = {'r': 8, 'alpha': 16, 'dropout': 0.1, 'target_modules': ['query', 'value']}
        if not args.quick and not args.holdout:
            folder = MODELS / ('distilroberta_lora_augmented' if args.augment else 'distilroberta_lora')
            model.save_pretrained(folder)
            tokenizer.save_pretrained(folder)
    (CACHE / f'transformer_{mode}{suffix}_info.json').write_text(json.dumps(info, indent=2))
    return info


def main():
    args = parse_args()
    ensure_dirs()
    torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
    train, val, test = data.split_frames(data.load_pairs())
    if args.quick:
        args.train_questions, args.val_questions = 40, 20
        test = sample_questions(test, 20, SEED)
    if args.holdout:      # the held-out domain is never seen in training or validation
        train, val = train[train.domain != args.holdout], val[val.domain != args.holdout]
        test = test[test.domain == args.holdout].reset_index(drop=True)
    train = sample_questions(train, args.train_questions, SEED)
    val = sample_questions(val, args.val_questions, SEED)
    if args.augment:
        import pandas as pd
        from train_augmented import augmentation_texts
        extra = augmentation_texts(data.split_frames(data.load_pairs())[0])
        extra = pd.concat([g.sample(min(args.augment, len(g)), random_state=SEED) for _, g in extra.groupby('y')])
        train = pd.concat([train, extra], ignore_index=True)
    print(f'train {len(train)} texts, validation {len(val)}, test {len(test)}; '
          f'torch threads: {torch.get_num_threads()}')
    for mode in (['lora', 'full'] if args.mode == 'both' else [args.mode]):   # LoRA first: the demo uses it
        train_one(mode, train, val, test, args, tokenizer)


if __name__ == '__main__':
    main()
