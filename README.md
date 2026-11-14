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

## Key results

All detectors are evaluated on the **same test set of 4,380 texts** (balanced, split by question), threshold 0.5,
95% bootstrap confidence intervals. Zero-shot scores and the transformers are first calibrated on validation
(rows marked "raw 0.5" show the transformers without calibration). ChatGPT = positive class; FPR = share of human texts wrongly flagged.

| Model | Accuracy | Precision | Recall | F1 [95% CI] | ROC-AUC | FPR (humans flagged) | TPR @ 1% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
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

**Run the demo** (uses the trained models in `models/`):

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
│   └── plots.py                  all figures in one style
├── scripts/
│   ├── download_data.py
│   ├── compute_lm_scores.py
│   ├── train_transformer.py
│   └── run_experiments.py        ← single entry point for the experiments
├── demo/
│   ├── streamlit_app.py          ← the final demo
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
