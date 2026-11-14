"""Demo: paste a text -> probability that it was written by ChatGPT.

Start:  python demo/app.py   (or double-click "Запустить демо.bat")
Uses the TF-IDF + logistic regression detector (demo/model.joblib) and GPT-2 perplexity.
"""
import os
import re
import json
from pathlib import Path

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import numpy as np
import joblib
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
saved = joblib.load(HERE / 'model.joblib')
vectorizer, model = saved['vectorizer'], saved['model']
examples = json.loads((HERE / 'examples.json').read_text(encoding='utf-8'))

gpt2_tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
gpt2 = AutoModelForCausalLM.from_pretrained('openai-community/gpt2').eval()
PPL_THRESHOLD = 16.1          # chosen on the validation set in the notebook

# The same cleaning rules as in the notebooks
CLEANING_RULES = [
    (r'\\n', ' '), (r'\s+', ' '), (r'([a-z0-9\)])\.([A-Z])', r'\1. \2'), (r'\s+([,.!?;:%])', r'\1'),
    (r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r'\1'), (r'\(\s+', '('), (r'\s+\)', ')'),
    (r'(\w)\s+-\s+(\w)', r'\1-\2'), (r'"\s+([^"]*?)\s+"', r'"\1"'), (r'\$\s+(\d)', r'$\1'),
]


def clean_text(text):
    for pattern, replacement in CLEANING_RULES:
        text = re.sub(pattern, replacement, text)
    return text.strip()


def perplexity(text):
    tokens = gpt2_tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        return float(torch.exp(gpt2(**tokens, labels=tokens['input_ids']).loss))


def word_contributions(text):
    """How much each single word pushes the decision: TF-IDF weight x model weight."""
    vector = vectorizer.transform([text])
    weights = model.coef_[0]
    vocabulary = vectorizer.vocabulary_
    scores = {}
    for word in set(re.findall(r'\b\w+\b', text.lower())):
        index = vocabulary.get(word)
        if index is not None:
            scores[word] = vector[0, index] * weights[index]
    return scores


def analyse(text):
    text = clean_text(text or '')
    if len(text.split()) < 5:
        return {}, 'Please paste a longer text (at least a few sentences).', [], ''

    p_chatgpt = float(model.predict_proba(vectorizer.transform([text]))[0, 1])
    ppl = perplexity(text)
    ppl_verdict = 'typical of ChatGPT (predictable text)' if ppl < PPL_THRESHOLD else 'typical of a human (less predictable text)'
    summary = (f"### TF-IDF detector: **{p_chatgpt:.0%}** probability of ChatGPT\n"
               f"### GPT-2 perplexity: **{ppl:.1f}** → {ppl_verdict}\n"
               f"(threshold {PPL_THRESHOLD}: lower = more predictable)")

    scores = word_contributions(text)
    top_chatgpt = {w for w, s in sorted(scores.items(), key=lambda x: -x[1])[:8] if s > 0}
    top_human = {w for w, s in sorted(scores.items(), key=lambda x: x[1])[:8] if s < 0}
    highlighted = []
    for piece in re.split(r'(\b\w+\b)', text):
        word = piece.lower()
        if word in top_chatgpt:
            highlighted.append((piece, 'ChatGPT'))
        elif word in top_human:
            highlighted.append((piece, 'human'))
        elif piece:
            highlighted.append((piece, None))

    explanation = ('**Words pushing towards ChatGPT:** ' + ', '.join(sorted(top_chatgpt)) +
                   '\n\n**Words pushing towards human:** ' + ', '.join(sorted(top_human)))
    return {'ChatGPT': p_chatgpt, 'Human': 1 - p_chatgpt}, summary, highlighted, explanation


with gr.Blocks(title='ChatGPT text detector — demo') as demo:
    gr.Markdown('# Is this text written by ChatGPT?\n'
                'Early prototype of our course project. Detector: TF-IDF + logistic regression trained on the HC3 dataset; '
                'second opinion: GPT-2 perplexity. *Trained on ChatGPT answers from 2023 — on very different texts it can be wrong.*')
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(label='Paste an English text', lines=10)
            button = gr.Button('Analyse', variant='primary')
            gr.Examples([[e['text']] for e in examples], inputs=text_input,
                        example_labels=[f"{e['label']} · {e['topic']}" for e in examples], label='Examples from the test set')
        with gr.Column(scale=1):
            probability_output = gr.Label(label='Prediction', num_top_classes=2)
            summary_output = gr.Markdown()
            highlight_output = gr.HighlightedText(label='Words that influenced the decision (orange → ChatGPT, blue → human)',
                                                  color_map={'ChatGPT': '#e8833a', 'human': '#2a78d6'},
                                                  show_legend=True, show_inline_category=False)
            explanation_output = gr.Markdown()
    button.click(analyse, inputs=text_input, outputs=[probability_output, summary_output, highlight_output, explanation_output])

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=7860, inbrowser=os.environ.get('DEMO_NO_BROWSER') != '1')
