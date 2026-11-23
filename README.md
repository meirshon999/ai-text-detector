# Detecting ChatGPT-Generated Text

**A comparison of classical machine learning, stylometry, zero-shot language-model scores and fine-tuned
transformers on HC3, with a critical study of dataset bias**

Course project · *AI Essentials: Theory and Applications* · MSc Applied Artificial Intelligence, Astana IT University, 2026

---

## Problem

Given an answer to a question, decide whether it was written by a **human** or by **ChatGPT**,
check that the detector learns writing style rather than artifacts of the dataset,
and measure how well it works on topics it has never seen.

**Research questions**

- **RQ1.** Which approach detects ChatGPT text best: classical ML, style features, zero-shot LLM scores or a fine-tuned transformer?
- **RQ2.** How well do detectors generalise to a domain not seen during training?
- **RQ3.** Do hidden artifacts of the dataset inflate detection scores?
- **RQ4.** Do detectors trained on ChatGPT (2023) still work on 2026 models, on rewriting attacks and on mixed human/AI texts?
- **RQ5.** Can adding texts of *one* modern model to training fix the detectors for *other, unseen* models?
- **RQ6.** Can a detector show *which part* of a mixed human + AI text was machine-written?

## Key results

All detectors are evaluated on the **same test set of 4,380 texts** (balanced, split by question), threshold 0.5,
95% bootstrap confidence intervals. Zero-shot scores and the transformers are first calibrated on validation
(rows marked "raw 0.5" show the transformers without calibration). ChatGPT = positive class; FPR = share of human texts wrongly flagged.

| Model | Accuracy | Precision | Recall | F1 [95% CI] | ROC-AUC | FPR (humans flagged) | TPR @ 1% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DistilRoBERTa + LoRA | 0.990 | 0.990 | 0.990 | 0.990 [0.987–0.993] | 0.9996 | 0.011 | 0.990 |
| TF-IDF + LogReg | 0.989 | 0.991 | 0.987 | 0.989 [0.986–0.992] | 0.9993 | 0.009 | 0.988 |
| DistilRoBERTa full fine-tuning | 0.986 | 0.975 | 0.997 | 0.986 [0.982–0.989] | 0.9991 | 0.026 | 0.990 |
| DistilRoBERTa full FT, raw 0.5 | 0.963 | 0.932 | 0.999 | 0.964 [0.959–0.969] | 0.9991 | 0.073 | 0.990 |
| GPT-2 perplexity (zero-shot) | 0.958 | 0.953 | 0.963 | 0.958 [0.953–0.964] | 0.9897 | 0.047 | 0.791 |
| Fast-DetectGPT on GPT-2 (zero-shot) | 0.957 | 0.956 | 0.957 | 0.957 [0.951–0.962] | 0.9912 | 0.044 | 0.877 |
| PyTorch bag-of-embeddings | 0.953 | 0.957 | 0.947 | 0.952 [0.946–0.958] | 0.9898 | 0.042 | 0.779 |
| DistilRoBERTa + LoRA, raw 0.5 | 0.944 | 0.900 | 0.999 | 0.947 [0.941–0.953] | 0.9996 | 0.111 | 0.990 |
| Stylometry + Random Forest | 0.935 | 0.944 | 0.925 | 0.934 [0.927–0.941] | 0.9826 | 0.055 | 0.718 |
| Stylometry + LogReg | 0.890 | 0.895 | 0.884 | 0.889 [0.880–0.898] | 0.9525 | 0.104 | 0.431 |
| Length baseline | 0.752 | 0.716 | 0.835 | 0.771 [0.761–0.782] | 0.7664 | 0.331 | 0.000 |

**Main findings**

1. **In-domain, HC3 is easy:** DistilRoBERTa + LoRA reaches F1 0.990. A length-only baseline gets 0.771.
2. **Dataset bias (RQ3):** a one-line rule "space before punctuation → human" is right for **91.8%** of raw HC3 answers
   (tokenization artifact) and drops to chance after our cleaning. A character model trained on raw text wrongly flags
   **15.0%** of clean human texts (vs 2.0% when trained on clean text): a shortcut that the benchmark score hides.
3. **Generalisation (RQ2):** trained models lose F1 on an unseen domain (TF-IDF: mean −0.069, Reddit 0.992 → 0.866),
   while zero-shot GPT-2 detectors lose almost nothing. On unseen domains GPT-2 perplexity (mean F1 0.947) beats TF-IDF (0.911).
4. **Transformer:** DistilRoBERTa + LoRA (0.9% of the weights trained, 5,000 texts, laptop CPU) reaches F1 0.990 after calibration on validation, tied with TF-IDF, and the best ROC-AUC (0.9996). With the raw threshold 0.5 it flags 11.1% of humans. Full fine-tuning reached F1 0.986 after calibration, LoRA 0.990; their confidence intervals overlap, so LoRA, training 0.9% of the weights, is as good as full fine-tuning. For the transformer we could afford one such test: trained without Reddit, LoRA goes from 0.995 to 0.980 on Reddit, while TF-IDF goes from 0.992 to 0.866.

### Extension: modern models, attacks, mixed texts (RQ4–RQ5)

We took **555 test questions** (never used for training; 77–120 per domain) and collected new answers to them:
Claude Haiku / Sonnet / Opus 5.5 (Anthropic Message Batches API, total cost **$9.51**), two small open models run
locally (Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct; 50 questions per domain), three attacks on the HC3 ChatGPT
answers (paraphrase and "humanize" rewrites by Claude Sonnet 5.5, 5% character typos) and **905 mixed texts**
(first part human, rest ChatGPT, machine share 0 / 25 / 50 / 75 / 100%). All detectors are used **exactly as saved**
(same threshold 0.5 and calibration as on HC3). Detection rate = share of machine texts flagged; the human row is the
false positive rate on the human answers to the same questions.

| Detector | Humans flagged | ChatGPT (HC3) | Claude Haiku 5.5 | Claude Sonnet 5.5 | Claude Opus 5.5 | Qwen2.5-1.5B | SmolLM2-1.7B | Paraphrased | "Humanized" | 5% typos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DistilRoBERTa + LoRA | 2.3% | 97.8% | 21.1% | 9.7% | 11.9% | 73.6% | 96.8% | 74.6% | 7.0% | 83.2% |
| TF-IDF + LogReg | 0.9% | 96.9% | 39.1% | 33.9% | 32.3% | 63.6% | 87.2% | 49.7% | 1.8% | 96.4% |
| PyTorch bag-of-embeddings | 4.9% | 91.2% | 17.1% | 6.1% | 7.2% | 47.6% | 58.8% | 47.2% | 0.7% | 80.7% |
| GPT-2 perplexity | 6.1% | 96.6% | 11.9% | 10.1% | 11.5% | 32.4% | 66.8% | 33% | 18% | 49.7% |
| Fast-DetectGPT (GPT-2) | 3.6% | 93.2% | 11.0% | 10.1% | 6.8% | 18.0% | 46.4% | 20% | 18% | 48.5% |
| Length baseline | 29.5% | 76.9% | 87.7% | 95.9% | 95.9% | 78.8% | 84.4% | — | — | — |

5. **Detectors trained on 2023 ChatGPT miss most 2026 texts.** Small open models (Qwen, SmolLM) write closer to
   2023 ChatGPT and are caught much better (LoRA 74% / 97%), the large Claude models are not. The best HC3 model (DistilRoBERTa + LoRA) catches
   only 10–21% of Claude 5.5 answers; one "humanize" rewrite hides 93–99% of ChatGPT texts from every trained model.
   ROC-AUC on Claude stays at 0.77–0.92 for the trained models, so the scores still carry signal: a large part of the
   failure is the **threshold/calibration**, not a total loss of signal.
6. **The length baseline "wins" on Claude only by accident:** Claude answers are 2–3.5× longer than human ones
   (259–411 vs 116 words), and it still flags 29.5% of humans. It is a confound, not a detector.
7. **Mixed texts:** the GPT-2 detectors read only the first 128 tokens, so they barely react until half of the text is
   machine-written and flag under 50% of texts even at 75% machine share; trained models rise roughly linearly and cross 0.5 between 50% and 75% machine share.
8. **Fix (RQ5):** retraining with 4,499 Claude **Haiku** texts (train questions only, balanced with 4,499 extra human
   answers) makes the detectors work on models they never saw: DistilRoBERTa + LoRA Sonnet 9.7% → 93%, Opus 11.9% → 94%,
   "humanized" 7.0% → 94%, Qwen 74% → 89%; TF-IDF Sonnet 34% → 98%, Opus 32% → 98%, Qwen 64% → 80%. The price is paid on humans and on HC3:
   FPR 2.3% → 4.5% (LoRA) and 0.9% → 4.0% (TF-IDF); HC3 test F1 0.990 → 0.980 (LoRA) and 0.989 → 0.984 (TF-IDF).
   Caveats: the *rewrite types* (paraphrase, humanize) were seen in training through Haiku, only the rewriting model
   (Sonnet) was new; and the retrained LoRA became worse on typos (83% → 70%).

9. **Where does the AI part start? (RQ6)** On 1,082 mixed texts (human start, machine end) every sentence split is
   scored by a whole-text detector. With a ChatGPT continuation the boundary is found within one sentence in 82–89% of
   texts (guessing the middle: 59%). With a Claude Sonnet 5.5 continuation the HC3 detectors are at guessing level
   (52–54% vs 53%), the retrained ones reach 78% (TF-IDF) and 69% (LoRA). The exact sentence is found in at most 42%.
   The demo tab *Live: fool the detector* shows this split on any text.

Figures: `reports/final/figures/modern_*.png`, `augmented_*.png`, `segments_boundary.png` · numbers: `reports/final/modern_results.json`,
`reports/final/augmented_results.json`, `reports/final/segments_results.json` · all of it interactive in the demo tab **Analytics dashboard**.

Full tables: [`reports/final/results_table.md`](reports/final/results_table.md) · all numbers: `reports/final/results.json` ·
figures: [`reports/final/figures/`](reports/final/figures/).

## Quick start

**Requirements:** Python 3.10+ (tested on 3.14), Windows / macOS / Linux, no GPU needed.

```bash
git clone https://github.com/meirshon999/ai-text-detector.git
cd ai-text-detector
pip install -r requirements.txt
python scripts/download_models.py    # GPT-2 + DistilRoBERTa from Hugging Face, ~0.8 GB, once
```

**Run the demo** (uses the trained models in `models/`). Tabs: *Analyze a text*; *Live: fool the detector*
(a Claude 5.5 model answers any question, the HC3 detector and the retrained detector judge it, the text is split into
its human and AI parts, one click "humanizes" it; needs `ANTHROPIC_API_KEY` in `.env`, otherwise it uses the saved
texts offline); *Analytics dashboard* (all results, interactive); *HC3 figures*; *About*.

- Windows: double-click **`Запустить демо.bat`** — the browser opens `http://localhost:8501` by itself.
- Any OS:

```bash
python -m streamlit run demo/streamlit_app.py
```

GPT-2 (~0.5 GB) and DistilRoBERTa (~0.3 GB) are loaded **only from the local Hugging Face cache**.
If they are missing, the demo shows those detectors as *not available* (it never invents a result);
the TF-IDF, stylometric, PyTorch and length models always work offline.

**Open the final code:** [`notebooks/04_Final_Project.ipynb`](notebooks/04_Final_Project.ipynb) — the whole final
pipeline, already executed (a browser copy: `reports/final/Final_Project.html`). To run it again, download the data
first (`python scripts/download_data.py`). The slow parts — GPT-2 scores and the transformer predictions — are cached
in `reports/final/cache/` in this repository, so the notebook runs in about 5–7 minutes and reproduces the final
numbers exactly (it checks this itself).

## Reproduce all experiments

Every script finds the project folder by itself, so it can be started from any directory.

| Step | Command | Time (laptop CPU) | Output |
|---|---|---|---|
| 1. Download HC3 | `python scripts/download_data.py` | 1 min | `data/raw/*.jsonl` |
| 2. GPT-2 scores (perplexity + Fast-DetectGPT) | `python scripts/compute_lm_scores.py` | ~20 min | `reports/final/cache/gpt2_scores.parquet` |
| 3. DistilRoBERTa: full fine-tuning + LoRA (optional) | `python scripts/train_transformer.py` | ~1 h | `reports/final/cache/transformer_*`, `models/distilroberta_lora/` |
| 3b. LoRA generalisation test: train without Reddit, test on Reddit (optional) | `python scripts/train_transformer.py --mode lora --holdout reddit_eli5` | ~30 min | `reports/final/cache/transformer_lora_lodo_*` |
| 4. All other models, evaluation, figures | `python scripts/run_experiments.py` | ~30 min | `reports/final/`, `models/` |
| 4b. Only rebuild report + figures from cache | `python scripts/run_experiments.py --reuse` | 1 min | same |
| 5. Claude 5.5 texts (needs `ANTHROPIC_API_KEY` in `.env`, ~$10) | `python scripts/generate_claude.py estimate` → `submit` → `status` → `collect` | ~1 h (batch) | `reports/final/modern/raw/claude.jsonl` |
| 6. Open-model texts (local, CPU) | `python scripts/generate_open_models.py` | several hours | `reports/final/modern/raw/*.jsonl` |
| 7. Score all detectors on modern texts, attacks, mixed texts | `python scripts/evaluate_modern.py` | ~30 min | `reports/final/modern_results.json`, `figures/modern_*.png` |
| 8. LoRA retrained with Claude Haiku texts (optional) | `python scripts/train_transformer.py --mode lora --augment 1500` | ~1.5 h | `models/distilroberta_lora_augmented/` |
| 9. Retrain with Claude Haiku texts, before/after | `python scripts/train_augmented.py` | ~15 min | `reports/final/augmented_results.json`, `models/augmented/` |
| 10. Where does the AI part start? (sentence-level boundary) | `python scripts/evaluate_segments.py` | ~45 min | `reports/final/segments_results.json`, `figures/segments_boundary.png` |
| Smoke test of the whole pipeline | `python scripts/run_experiments.py --quick` | 2 min | `reports/final/quick/` |

Random seed 42 everywhere. The rebuilt split is **identical** to the one used at the midterm
(checked row by row), minus 10 questions whose "ChatGPT answer" is a saved Cloudflare page (found in the final audit).

## Repository structure

```
ai-text-detector/
├── README.md                     ← you are here
├── FINAL_PROJECT_GUIDE.md        ← full explanation of the project (for the team and the committee)
├── DEFENSE_PREPARATION.md        ← talk for 3 speakers: English text + Russian translation
├── DEFENSE_QA.md                 ← 40+ likely questions with answers
├── FINAL_STATUS.md               ← what is done, what is not, final metrics, limitations
├── Запустить демо.bat             ← starts the demo on Windows
├── requirements.txt
├── src/detector/                 ← shared code (used by scripts AND the demo)
│   ├── config.py                 paths (relative to the project), seed, colours
│   ├── data.py                   loading, cleaning rules, pairing, split by question
│   ├── features.py               20 stylometric features (18 from the midterm + 2 validated new ones)
│   ├── models.py                 length baseline, TF-IDF+LogReg, stylometry, PyTorch network
│   ├── lm_scores.py              GPT-2 perplexity and Fast-DetectGPT
│   ├── evaluation.py             metrics, bootstrap confidence intervals
│   ├── inference.py              batch scoring with every saved detector
│   ├── modern.py                 modern-model experiment: questions, prompts, attacks, mixed texts
│   ├── segments.py               which part of a text is AI: sentence splits and the human -> AI boundary
│   └── plots.py                  all figures in one style
├── scripts/
│   ├── download_data.py
│   ├── compute_lm_scores.py
│   ├── train_transformer.py
│   ├── run_experiments.py        ← single entry point for the HC3 experiments
│   ├── generate_claude.py        Claude 5.5 answers + rewrites (Message Batches API)
│   ├── generate_open_models.py   Qwen2.5 / SmolLM2 answers, local
│   ├── evaluate_modern.py        RQ4: modern models, attacks, mixed texts
│   ├── train_augmented.py        RQ5: retraining with one modern generator
│   └── evaluate_segments.py      RQ6: locating the AI part inside a mixed text
├── demo/
│   ├── streamlit_app.py          ← the final demo
│   ├── analytics.py              interactive Analytics dashboard tab
│   ├── live.py                   "Live: fool the detector" tab (Claude API or saved texts offline)
│   └── examples.json             example texts from the TEST set
├── models/                       trained detectors used by the demo
├── reports/
│   ├── final/                    final results, tables, figures (created by the scripts)
│   ├── figures/                  midterm figures
│   ├── Project_Proposal.docx     midterm report
│   └── Midterm_Presentation.pptx midterm slides
├── notebooks/04_Final_Project.ipynb  ← the FINAL code end to end, already executed (HTML copy: reports/final/Final_Project.html)
├── notebooks/                    midterm notebooks 01–03 + Full_Project_Explained (step-by-step code);
│                                 their numbers are the MIDTERM ones, final numbers come from run_experiments.py
├── archive/                      old versions (midterm Gradio demo, long drafts)
├── backup_midterm/               untouched copies of midterm results, split and demo
└── data/                         raw and interim data (created by the scripts, not in git)
```

## Data

[HC3 — Human ChatGPT Comparison Corpus](https://huggingface.co/datasets/Hello-SimpleAI/HC3) (Guo et al., 2023,
[arXiv:2301.07597](https://arxiv.org/abs/2301.07597)), English part: 24,322 questions, 85,449 answers,
5 domains (Reddit ELI5, finance, medicine, open QA, Wikipedia CS/AI). Licence CC-BY-SA.

## Documentation

- [`FINAL_PROJECT_GUIDE.md`](FINAL_PROJECT_GUIDE.md) — dataset, every model, metrics, experiments, demo, limitations, conclusions.
- [`DEFENSE_PREPARATION.md`](DEFENSE_PREPARATION.md) — who says what at the defense, with slides and figures.
- [`DEFENSE_QA.md`](DEFENSE_QA.md) — questions the committee may ask, with short answers.
- [`FINAL_STATUS.md`](FINAL_STATUS.md) — honest status: done / not done / limitations.
