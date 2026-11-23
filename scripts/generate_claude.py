"""Modern-LLM texts from the Claude API, sent as one Message Batch (50% cheaper than normal requests).

    python scripts/generate_claude.py submit      # sends the batch, saves its id
    python scripts/generate_claude.py status      # shows progress
    python scripts/generate_claude.py collect     # downloads the results -> reports/final/modern/raw/claude.jsonl

Needs ANTHROPIC_API_KEY in a file .env in the project folder (never committed) or in the environment.

Tasks, all on the SAME HC3 test questions (reports/final/modern/questions.parquet):
    claude-haiku-5-5 / claude-sonnet-5-5 / claude-opus-5-5   answer the question
    paraphrase   Claude Sonnet 5.5 paraphrases the HC3 ChatGPT answer   (attack)
    humanize     Claude Sonnet 5.5 rewrites it to sound like a person   (attack)
and, on TRAIN questions only (reports/final/modern/train_questions.parquet), for the augmentation experiment:
    train-haiku / train-paraphrase / train-humanize   the same three kinds of text, all from Claude Haiku 5.5
    (the test generators Sonnet, Opus, Qwen and SmolLM are never used for training)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import anthropic                                                                      # noqa: E402
import pandas as pd                                                                   # noqa: E402
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming     # noqa: E402
from anthropic.types.messages.batch_create_params import Request                      # noqa: E402
from dotenv import load_dotenv                                                        # noqa: E402

from detector import modern                                                           # noqa: E402

RAW = modern.MODERN / 'raw'
STATE = RAW / 'claude_batch.json'
OUT = RAW / 'claude.jsonl'
ANSWER_MODELS = ['claude-haiku-5-5', 'claude-sonnet-5-5', 'claude-opus-5-5']
REWRITE_MODEL = 'claude-sonnet-5-5'
TRAIN_MODEL = 'claude-haiku-5-5'
MAX_TOKENS = 4000          # room for adaptive thinking + an answer of a few hundred words


def client():
    load_dotenv(ROOT / '.env')
    return anthropic.Anthropic()


def questions():
    modern.MODERN.mkdir(parents=True, exist_ok=True)
    if not modern.QUESTIONS_FILE.exists():
        modern.select_questions().to_parquet(modern.QUESTIONS_FILE, index=False)
    if not modern.TRAIN_QUESTIONS_FILE.exists():
        modern.select_questions(modern.TRAIN_PER_DOMAIN, split='train').to_parquet(modern.TRAIN_QUESTIONS_FILE,
                                                                                  index=False)
    return pd.read_parquet(modern.QUESTIONS_FILE), pd.read_parquet(modern.TRAIN_QUESTIONS_FILE)


def request(custom_id, model, content, system=None):
    params = dict(model=model, max_tokens=MAX_TOKENS, output_config={'effort': 'low'},
                  messages=[{'role': 'user', 'content': content}])
    if system:
        params['system'] = system
    return Request(custom_id=custom_id, params=MessageCreateParamsNonStreaming(**params))


def build_requests(q, train_q):
    requests = []
    for model in ANSWER_MODELS:
        requests += [request(f'{model}__{r.qid}', model, r.question, modern.ANSWER_SYSTEM) for r in q.itertuples()]
    for task, prompt in (('paraphrase', modern.PARAPHRASE_PROMPT), ('humanize', modern.HUMANIZE_PROMPT)):
        requests += [request(f'{task}__{r.qid}', REWRITE_MODEL, prompt.format(text=r.chatgpt_text))
                     for r in q.itertuples()]
    requests += [request(f'train-haiku__{r.qid}', TRAIN_MODEL, r.question, modern.ANSWER_SYSTEM)
                 for r in train_q.itertuples()]
    for task, prompt in (('train-paraphrase', modern.PARAPHRASE_PROMPT), ('train-humanize', modern.HUMANIZE_PROMPT)):
        requests += [request(f'{task}__{r.qid}', TRAIN_MODEL, prompt.format(text=r.chatgpt_text))
                     for r in train_q.itertuples()]
    return requests


PRICE = {'claude-haiku-5-5': (0.10, 0.50), 'claude-sonnet-5-5': (2.0, 10.0), 'claude-opus-5-5': (4.0, 20.0)}


def estimate():
    """Rough cost before sending: ~1.3 tokens per word in, ~600 tokens out (answer + short thinking), batch = 50%."""
    total, counts = 0.0, {}
    for r in build_requests(*questions()):
        p = r['params']
        tokens_in = 1.3 * len(str(p['messages'][0]['content']).split()) + 60
        price_in, price_out = PRICE[p['model']]
        total += 0.5 * (tokens_in * price_in + 600 * price_out) / 1e6
        counts[p['model']] = counts.get(p['model'], 0) + 1
    print(f'{sum(counts.values())} requests {counts}; estimated cost about ${total:.2f}')


def submit():
    if STATE.exists():
        sys.exit(f'A batch was already submitted ({STATE}). Use "status" / "collect", or delete the file to resend.')
    requests = build_requests(*questions())
    batch = client().messages.batches.create(requests=requests)
    RAW.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({'batch_id': batch.id, 'n_requests': len(requests)}, indent=2))
    print(f'Submitted {len(requests)} requests, batch {batch.id}')


def status():
    batch = client().messages.batches.retrieve(json.loads(STATE.read_text())['batch_id'])
    print(batch.processing_status, batch.request_counts)
    return batch


def collect():
    batch = status()
    if batch.processing_status != 'ended':
        sys.exit('The batch is still running; try again later.')
    rows, problems, usage = [], {}, {}
    api = client()          # keep a reference: a temporary client is closed while the stream is read
    for result in api.messages.batches.results(batch.id):
        task, qid = result.custom_id.split('__', 1)
        if result.result.type != 'succeeded':
            problems[result.custom_id] = result.result.type
            continue
        message = result.result.message
        text = ''.join(b.text for b in message.content if b.type == 'text')
        if message.stop_reason in ('refusal', 'max_tokens') or not text.strip():
            problems[result.custom_id] = message.stop_reason
            continue
        u = usage.setdefault(message.model, {'input_tokens': 0, 'output_tokens': 0})
        u['input_tokens'] += message.usage.input_tokens
        u['output_tokens'] += message.usage.output_tokens
        rows.append({'qid': qid, 'source': task, 'model': message.model, 'raw': text})
    with open(OUT, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    state = json.loads(STATE.read_text())
    state.update({'collected': len(rows), 'problems': problems, 'usage': usage})
    STATE.write_text(json.dumps(state, indent=2))
    print(f'Saved {len(rows)} texts to {OUT}; {len(problems)} not usable; usage {usage}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['estimate', 'submit', 'status', 'collect'])
    {'estimate': estimate, 'submit': submit, 'status': status, 'collect': collect}[parser.parse_args().action]()
