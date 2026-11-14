# Final results (test set, 4380 texts)

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

# Cross-domain generalisation (leave-one-domain-out)

| Model | Mean F1, domain seen | Mean F1, domain unseen | Mean drop | Worst unseen F1 | Domains tested |
|---|---:|---:|---:|---:|---:|
| DistilRoBERTa + LoRA | 0.995 | 0.98 | 0.015 | 0.98 | 1 |
| GPT-2 perplexity (zero-shot) | 0.947 | 0.947 | -0.001 | 0.9 | 5 |
| Fast-DetectGPT on GPT-2 (zero-shot) | 0.94 | 0.938 | 0.002 | 0.902 | 5 |
| TF-IDF + LogReg | 0.98 | 0.911 | 0.069 | 0.84 | 5 |
| Stylometry + Random Forest | 0.91 | 0.875 | 0.035 | 0.769 | 5 |
| PyTorch bag-of-embeddings | 0.933 | 0.857 | 0.076 | 0.726 | 5 |
| Stylometry + LogReg | 0.852 | 0.839 | 0.013 | 0.675 | 5 |
| Length baseline | 0.733 | 0.726 | 0.007 | 0.538 | 5 |

F1 on each unseen domain:

| model | Finance | Medicine | Open QA | Reddit ELI5 | Wikipedia CS/AI |
|---|---:|---:|---:|---:|---:|
| DistilRoBERTa + LoRA | — | — | — | 0.98 | — |
| Fast-DetectGPT on GPT-2 (zero-shot) | 0.946 | 0.902 | 0.924 | 0.963 | 0.955 |
| GPT-2 perplexity (zero-shot) | 0.957 | 0.991 | 0.9 | 0.962 | 0.925 |
| Length baseline | 0.741 | 0.867 | 0.538 | 0.77 | 0.712 |
| PyTorch bag-of-embeddings | 0.95 | 0.992 | 0.726 | 0.751 | 0.865 |
| Stylometry + LogReg | 0.919 | 0.919 | 0.675 | 0.863 | 0.821 |
| Stylometry + Random Forest | 0.932 | 0.969 | 0.769 | 0.86 | 0.844 |
| TF-IDF + LogReg | 0.988 | 0.983 | 0.84 | 0.866 | 0.877 |
