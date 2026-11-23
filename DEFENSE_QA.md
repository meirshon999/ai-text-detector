# DEFENSE Q&A — вопросы комиссии и ответы

> Ответ на английском — то, что говорим комиссии (коротко, 20–40 секунд).
> **Простыми словами** — пояснение для себя на русском.
> Все числа взяты из `reports/final/results.json` (test = 4,380 текстов). Если чего-то не делали, так и говорим: «we did not test this».

**Главные числа (выучить)**

| Что | Число |
|---|---|
| Лучший детектор на test | DistilRoBERTa + LoRA, F1 0.990 (вровень с TF-IDF: интервалы пересекаются) |
| TF-IDF + LogReg | F1 0.989, ROC-AUC 0.9993, люди ложно помечены: 0.9% |
| GPT-2 perplexity без обучения | F1 0.958; медиана perplexity: ChatGPT 10.5, люди 38.9 |
| DistilRoBERTa + LoRA (5 000 текстов, CPU) | F1 0.990, обучено 0.9% весов |
| Baseline только по длине | F1 0.771 |
| Правило «пробел перед запятой → человек» | 91.8% на сырых данных → 33.1% после очистки |
| Новая тема (Reddit не видели в обучении), TF-IDF | F1 0.866 вместо 0.992 |

---

## 1. General Project Questions

**Q1. What is your project about, in one sentence?**
We compare different ways to detect whether an answer was written by a human or by ChatGPT. We check that the detectors learn writing style and not artifacts of the dataset, and we test them on topics they have never seen.

**Q2. Why is this problem important?**
LLMs produce fluent text almost for free, so universities and platforms want to know what is generated. But a detector error is not free: a false positive means accusing a real person. That is why we report the false positive rate and confidence intervals, not only accuracy.
> **Простыми словами:** важна не только точность, но и цена ошибки для человека.

**Q3. What is new in your work compared with the HC3 paper?**
The HC3 authors report high in-domain accuracy. We (1) found and removed a formatting shortcut that alone gives 91.8% accuracy, (2) showed with a stress test that a model trained on raw text breaks on clean human text, (3) compared nine detectors under identical conditions with confidence intervals, and (4) measured generalisation to unseen domains for every model.

**Q4. Why did you choose HC3?** *(каверзный)*
It is open (CC-BY-SA) and widely used. It has human and ChatGPT answers **to the same questions**, so topic is controlled. It covers five domains, which lets us test generalisation to an unseen domain. Its weaknesses are a single generator (ChatGPT, early 2023) and English only, and we state them as limitations.
> **Простыми словами:** одинаковые вопросы для обоих классов означают честное сравнение, а 5 тем позволяют проверить перенос.

**Q5. What is the difference between classification and detection?** *(каверзный)*
Technically our task is binary classification on a closed dataset: two known classes, one known generator. Real-world detection is open-world: unknown generators, edited or mixed texts, adversarial users, and a requirement of a very low false positive rate. That is why we also report TPR at 1% FPR and do not claim that our detector works "in the wild".
> **Простыми словами:** мы решаем классификацию на HC3. Настоящая детекция в реальном мире сложнее, и мы этого не обещаем.

**Q6. What did you do yourselves and what did you take ready-made?**
Ready-made: the HC3 dataset and the pretrained GPT-2 and DistilRoBERTa weights. Everything else is ours: the data audit and cleaning, the shortcut experiments, all training and tuning, the PyTorch network, the LoRA fine-tuning, the evaluation protocol and the demo.

---

## 2. Dataset and Data Preprocessing

**Q7. How was the data cleaned?**
Seven logged steps: remove empty answers, ChatGPT server error messages, answers under 5 words, exact duplicates (6,012), and 10 saved Cloudflare pages that we found in the final audit. Formatting is normalised with the **same** rules for both classes. Then we keep one human and one ChatGPT answer per question: 43,806 texts, perfectly balanced.

**Q8. Why one human and one ChatGPT answer per question?**
Reddit has about three human answers per ChatGPT answer. Without pairing, the classes would be imbalanced and the topic distribution would differ between classes. With pairing, both classes answer exactly the same questions, so the topic cannot reveal the label.

**Q9. How did you split the data?**
80/10/10 into train, validation and test, **grouped by question**, so both answers to a question are always in the same part. Train is for fitting, validation for tuning and early stopping, and test is used once at the end.

**Q10. Why is class balance important, and did you use class weights?**
With 50/50 classes, accuracy has a clear meaning (50% = chance) and the model has no reason to prefer one class. Class weights were not needed because the data is exactly balanced.

**Q11. Why do you normalise formatting? Doesn't it remove real information?** *(каверзный)*
It removes information about **how the dataset was stored**, not about the author. Human answers were tokenized by older NLP pipelines ("do n't", "word , word"), and ChatGPT answers lost their line breaks during scraping. A real human never writes "do n't". If we kept this, the model would learn the storage format.
> **Простыми словами:** пробелы перед запятыми поставил скрипт подготовки данных, а не человек.

---

## 3. Machine Learning Algorithms

**Q12. Why do you need a length-only baseline?**
ChatGPT answers are longer, so length alone gives F1 0.771. Every real detector must clearly beat this baseline. Otherwise it may simply be measuring length.

**Q13. How does TF-IDF + logistic regression work?**
TF-IDF turns a text into a vector of 100,000 weights for words, word pairs and punctuation marks: frequent in this text and rare elsewhere means a high weight. Logistic regression learns one weight per feature, sums them and applies a sigmoid to get P(ChatGPT). We tuned the n-gram range and the regularisation C on validation and chose unigrams + bigrams, C = 16.

**Q14. Why can TF-IDF achieve such high performance (F1 0.989)?** *(каверзный)*
Inside HC3 the two classes are very consistent. ChatGPT uses the same register: "it is important to", ", which", generic structure. The human answers come from a few sources with their own habits: first person, Reddit markdown, link placeholders. A linear model over 100,000 n-grams captures this easily. But part of it is topic- and source-specific, which is why TF-IDF drops on unseen domains (Reddit: 0.866).
> **Простыми словами:** внутри одного датасета это легко; трудно перенести на новые тексты.

**Q15. What are the stylometric features and why Random Forest?**
Eighteen hand-crafted features plus two that we added and validated: sentence length and its variation, vocabulary richness (MATTR), punctuation, discourse connectives, first person, hedges, slang, contractions, function words, readability and repeated word pairs. Length itself is excluded. Random forest captures non-linear interactions between them; logistic regression on the same features gives fully interpretable weights. The stylometric RF reaches F1 0.934: lower than word models, but every decision is explainable.

**Q16. Which style features matter most?**
By permutation importance: the share of repeated word pairs (by far the strongest: ChatGPT repeats formulaic phrases more), mean sentence length, sentence-length variation, brackets, commas, first-person words and connectives. ChatGPT uses more connectives and longer sentences; humans use more "I", brackets and contractions and have a richer vocabulary.

**Q17. Did you add new features? Why not paragraph structure?**
We tested two candidates, Flesch readability and the share of repeated word pairs, and kept them only if validation F1 improved by at least 0.002: both were kept, because validation F1 rose from 0.921 to 0.931. The share of repeated word pairs even became the most important feature, and we checked that it is not hidden text length: on texts of 100 to 300 words the effect is almost the same (Cliff's delta +0.74 versus +0.80 on all texts). We deliberately did **not** add paragraph features, because ChatGPT answers in HC3 lost their line breaks during scraping, so paragraphs would re-introduce an artifact.

**Q18. Why not use only the best model?**
Because the research question is a comparison. Different families fail differently: word models drop on new topics, zero-shot scores depend on the scoring model, and style features are weaker but more transparent. The comparison is the result.

---

## 4. Neural Networks and Transformers

**Q19. Describe your PyTorch network.**
A fastText-style network: every word gets a learned 64-dimensional embedding, the vectors of a text are averaged, then Linear(64→64), ReLU, Dropout 0.2 and Linear(64→1). Loss: binary cross-entropy. Optimizer: Adam, learning rate 0.003, batch 64. Early stopping on validation loss with patience 2. The best epoch was 2. Test F1 0.952.

**Q20. How do you know the network really learns and does not just memorise?**
Training loss goes down every epoch, while validation loss reaches a minimum at epoch 2 and then rises. This is overfitting, and early stopping keeps the best epoch. Train F1 is 0.986 and test F1 is 0.952: a gap of a few points, as expected for a neural network, and validation and test agree, so the model generalises within the dataset.

**Q21. Why is your neural network worse than TF-IDF?**
It averages word vectors, so it loses word order and punctuation patterns, and it is small. TF-IDF has 100,000 features including word pairs and punctuation, which are exactly the strongest cues in HC3.

**Q22. What is a transformer and self-attention?**
A transformer represents each token in the context of all other tokens. Self-attention computes, for every pair of tokens, how much one should look at the other, so the model understands word order and context. RoBERTa is a BERT-type transformer pretrained on a large corpus; we fine-tune it for classification.

**Q23. Why DistilRoBERTa and not RoBERTa-base or a larger model?**
We have no GPU. DistilRoBERTa has 6 layers and 82M parameters instead of 12 layers and 125M, and runs about twice as fast. Even so, we fine-tuned on 5,000 training texts for one epoch. This is a compute limit, and we say so.

**Q24. What is LoRA and why use it?**
LoRA freezes the pretrained weights and adds small trainable low-rank matrices to the attention layers: W' = W + B·A with rank 8. We train only 0.74M parameters, which is 0.9% of the model, plus the classifier head. It needs less memory, the saved adapter is only a few megabytes, and it keeps the pretrained knowledge. Result: LoRA F1 0.990 vs full fine-tuning F1 0.986.
> **Простыми словами:** не переучиваем всю модель, а добавляем маленькую «насадку».

**Q24b. You calibrated the transformer on validation. Isn't that tuning on the test set?** *(каверзный)*
No. Calibration uses only the 1,000 **validation** texts and fits two numbers: P = sigmoid(a·logit(p) + b). The test set is untouched. We do the same for the zero-shot GPT-2 scores. It does not change the ranking, so ROC-AUC stays 0.9996. It only moves the decision point. We report both versions: raw at threshold 0.5, F1 0.947 with **11.1% of humans flagged**, and calibrated, F1 0.990 with 1.1%. After one epoch on 5,000 texts the raw probabilities are shifted towards "ChatGPT", and this is a practical lesson: the threshold matters as much as the model.
> **Простыми словами:** модель отлично сортирует тексты, но порог 0.5 у неё смещён. Мы сдвинули порог по validation, тест при этом не трогали.

**Q25. Why is perplexity lower for ChatGPT texts?** *(каверзный)*
Perplexity measures how surprised a language model is by a text. ChatGPT generates by choosing high-probability tokens, so its text is smooth and typical for another language model. Humans make less predictable choices: rare words, slang, typos, unusual structure. On our test set the median GPT-2 perplexity is 10.5 for ChatGPT and 38.9 for humans.

**Q26. Is perplexity a universal detector?**
No. It depends on topic and length. Simple or formulaic human writing is also predictable, which causes false positives. Non-native writers are flagged more often (Liang et al., 2023). Paraphrasing or a different generator changes it. And GPT-2 is a small model from 2019.

**Q27. What is Fast-DetectGPT?**
A zero-shot method by Bao et al. (ICLR 2024). It compares the log-probability of the actual text with the expected log-probability of the alternative tokens the model itself would sample, normalised by their spread. Machine text sits near the top of the model's distribution. We compute it from the same GPT-2 pass as perplexity. F1 0.957.

---

## 5. Evaluation Metrics

**Q28. Why is accuracy not enough?** *(каверзный)*
First, with imbalanced data a trivial model can have high accuracy. Second, accuracy does not show **which** errors the model makes, and wrongly flagging a human is worse than missing an AI text. Third, it depends on the threshold. So we also report precision, recall, F1, ROC-AUC, the false positive rate and TPR at 1% FPR.

**Q29. What is ROC-AUC?**
The probability that a randomly chosen ChatGPT text gets a higher score than a randomly chosen human text. It measures ranking quality independently of the threshold. 0.5 is random and 1.0 is perfect.

**Q30. Why do you report TPR at 1% FPR?**
In a university, we can accept very few false accusations. TPR at 1% FPR answers: if we allow at most 1% of humans to be flagged, how many ChatGPT texts do we still catch? For TF-IDF it is 98.8%; for GPT-2 perplexity it is 79.1%.

**Q31. Does a high F1-score mean your detector is reliable?** *(каверзный)*
No. F1 0.989 is measured in-distribution: the same dataset, generator, sources and time. On an unseen domain it drops, a shortcut can inflate it, and at a strict 1% false-positive budget recall is lower. High F1 on HC3 means the classes are separable **in HC3**, not that the detector is reliable in the real world.

**Q32. What do the confidence intervals mean?**
We resampled the test questions 1,000 times with replacement and recomputed the metric. The 95% interval shows how much the score could change with a different test sample. If the intervals of two models overlap, we do not claim one is better.

---

## 6. Data Leakage and Dataset Bias

**Q33. How do you prevent data leakage?** *(каверзный)*
(1) We remove duplicates **before** splitting. (2) We split **by question**, so both answers to a question are in the same part. (3) All fitting, including the TF-IDF vocabulary, scalers, the vocabulary of the neural network and the style-feature medians, is done on train only. (4) Hyperparameters, thresholds and early stopping use validation only. (5) The test set is used once. We audited it: 0 shared questions, 0 identical texts, and 1 near-duplicate out of 4,380 (a doctor's template answer).

**Q34. What is shortcut learning?** *(каверзный)*
When a model solves the task through an accidental feature that correlates with the label in the training data, instead of the intended concept (Geirhos et al., 2020). In HC3, "space before a comma" correlates with "human" because of tokenization. A rule using only this gives 91.8% accuracy without any AI.

**Q35. How did you prove that the shortcut is dangerous, not just present?**
Stress test: we trained a character n-gram model, which sees spaces, on raw text and tested it on cleaned text, where humans no longer have the artifact. Accuracy changes from 98.8% to 91.9%, and the share of humans flagged as ChatGPT rises to 15.0%. The same model trained on clean text gets 97.7%. So a model can look excellent on the benchmark and fail on real human text.
> **Простыми словами:** модель, выучившая пробелы, начинает обвинять людей, которые пишут без лишних пробелов, то есть всех нормальных людей.

**Q36. Are there other artifacts left?**
Yes, and we measured them. Links in human answers were replaced by "URL_0" (9.1% of human vs 0.1% of ChatGPT training texts). Removing the placeholders changes TF-IDF F1 from 0.986 to 0.987, so the model does not depend on it much. In medicine, about half of the human answers contain doctor-platform phrases like "Thanks for the query", which explains near-perfect scores there. This is a source effect, not a general human style.

---

## 7. Generalization

**Q37. How did you test generalisation?**
Leave-one-domain-out: train on four domains and test on the fifth. We compare with the model trained on all domains **on exactly the same test texts**, so the difference is caused only by not having seen the domain. Zero-shot detectors only calibrate their threshold on the four domains.

**Q38. Why does performance decrease on unseen domains?** *(каверзный)*
Because of domain shift. Word models learn topic words and source habits, and the human answers come from different communities with different styles: Reddit users, doctors, finance forums. Answer length and structure also differ. On average, F1 drops by 0.069 for TF-IDF; the worst unseen domain for TF-IDF is 0.840 (Open QA).

**Q39. Which model generalises best?**
Of the models tested on all five domains, the zero-shot GPT-2 detectors. They do not learn topics, so they lose practically nothing on an unseen domain, and among the models tested on all five domains GPT-2 perplexity has the best mean F1 on unseen domains, 0.947, while TF-IDF falls to 0.911 and our PyTorch network drops the most (0.076 on average). Among trained models, style features transfer better than words. For the transformer we could afford one such test: trained without Reddit, LoRA goes from 0.995 to 0.980 on Reddit, while TF-IDF goes from 0.992 to 0.866. So the transformer may transfer even better, but we checked only one domain.
> **Простыми словами:** лучше всех переносятся модели, которые не учат слова темы (GPT-2). Трансформер на Reddit тоже переносится отлично, но мы проверили только одну тему.

**Q40. Can your model detect GPT-4 or newer LLMs?** *(каверзный)*
We did not test it, so we cannot claim it. HC3 contains only ChatGPT from early 2023. Training-based detectors are known to generalise poorly to new generators (Xu et al., 2024). Zero-shot methods depend on the scoring model. To claim this, we would need test data from GPT-4, Claude, Llama and others.

**Q41. What happens if ChatGPT paraphrases its answer?** *(каверзный)*
We did not test paraphrasing. The literature shows it is the main weakness: Krishna et al. (2023) report that paraphrasing with their DIPPER model reduces DetectGPT's detection from 70.3% to 4.6% at 1% FPR. We expect our detectors to drop too, especially TF-IDF, which relies on specific phrases.

---

## 8. Practical Implementation

**Q42. How does your demo work?**
Streamlit app. The text is normalised with the same rules as in training, then every available detector gives P(ChatGPT). We show the verdict of the main detector with an uncertainty zone (0.35–0.65 means "uncertain"), the opinions of all detectors, text statistics, GPT-2 perplexity, style indicators compared with typical human and ChatGPT values, and the words that pushed the TF-IDF decision.

**Q43. What happens if a model file is missing?**
The demo marks that detector as "not available" with the command to create it. It never shows a made-up result. GPT-2 and DistilRoBERTa are loaded only from the local cache, so the demo works offline.

**Q44. Can human-written text be classified as AI-generated?** *(каверзный)*
Yes. On our test set, the best detector flags 1.1% of human texts, GPT-2 perplexity 4.7%, and the stylometric logistic regression 10.4%. Long human answers are flagged a bit more often than short ones (TF-IDF: 2.4% for 200–399 words vs 0.3% under 50 words), because long, structured human answers look more like ChatGPT. Formal, simple or non-native writing is at higher risk (Liang et al., 2023). That is why the demo says "statistical estimate, not proof".
> **Простыми словами:** да, и мы это измерили. Ни один детектор не даёт ноль ложных обвинений.

**Q44b. Are short texts harder?**
Yes for AI texts, but carefully: in HC3 ChatGPT almost never writes short answers. Only 19 of the 777 test texts under 50 words are ChatGPT answers, and the detectors miss many of them: TF-IDF 2 of 19 (11%), GPT-2 perplexity 12 of 19 (63%). With so few examples this is a warning sign, not a precise number.
> **Простыми словами:** коротких ответов ChatGPT в данных почти нет. Те немногие, что есть, детекторы часто пропускают.

**Q45. How is the project reproducible?**
One command per stage: download data, compute GPT-2 scores, train the transformer, run all experiments. Seed 42, relative paths only, the split is rebuilt from raw data identically to the midterm, and every number comes from `results.json`.

---

## 9. Limitations and Future Work

**Q46. What are the main limitations?**
One generator and English only; the question-answer format; no paraphrased or mixed texts; the transformer was trained on a subset on CPU without a domain-generalisation test; GPT-2 sees only 128 tokens; small domains have wide confidence intervals.

**Q47. Would you deploy this detector in a university?**
Not as evidence against a student. At most as a hint for a human reviewer, with the false positive rate known, an uncertainty zone, and never as the only evidence.

**Q48. How can the project be improved in the future?** *(каверзный)*
(1) Test data from newer generators (GPT-4, Claude, Llama) and other genres (essays). (2) Paraphrased and mixed human + AI texts. (3) Full RoBERTa fine-tuning on a GPU with the whole training set, plus a domain-generalisation test for it. (4) Stronger zero-shot scoring models (Binoculars with 7B models). (5) Calibrated ensembles and evaluation at a fixed low false positive rate. (6) Multilingual data, including Russian and Kazakh.

**Q49. What did you learn from this project?**
That a benchmark score can be misleading. The most valuable part of our work was not the 0.99 F1 but finding why it is so high, and showing where the detectors fail.

---

## 10. After the defense: modern models (RQ4–RQ5)

**Q50. Does your detector work on today's models, not only on ChatGPT 2023?**
Mostly no, and we measured it. On 555 unseen test questions, DistilRoBERTa + LoRA catches 97.8% of HC3 ChatGPT answers but only 21% of Claude Haiku 5.5, 10% of Sonnet 5.5 and 12% of Opus 5.5 answers. ROC-AUC is still 0.80–0.88, so the score keeps signal; the HC3 threshold is what fails. Small open models (Qwen2.5-1.5B, SmolLM2-1.7B) are much easier: 74% and 97%.
*Простыми словами:* модель «узнаёт» ChatGPT 2023 года, а новый стиль Claude — нет.

**Q51. What about paraphrasing and "humanizing"?**
One "humanize" rewrite by Claude Sonnet hides 93–99% of ChatGPT texts from every trained detector. Plain paraphrasing is weaker: LoRA still catches 75%. 5% typos hurt GPT-2 perplexity most (97% → 50%), TF-IDF almost not at all.

**Q52. Can it find AI text inside a human text?**
Only when the machine part is large. Trained models cross 50% flagged between 50% and 75% machine share. GPT-2 detectors read only the first 128 tokens, so they barely react until the machine part reaches the beginning of the text.

**Q53. Why does the length baseline look best on Claude?**
Claude answers are 259–411 words, humans 116. Length "detects" Claude by accident and still flags 29.5% of humans. It is a confound, which is why we exclude it when we name the best detector.

**Q54. Can you fix it?**
Partly. We added 4,499 texts of one model (Claude Haiku) for training questions only, balanced with extra human answers. On models never seen in training, LoRA goes from 10% to 93% on Sonnet and from 12% to 94% on Opus. The cost: humans wrongly flagged 2.3% → 4.5%, HC3 F1 0.990 → 0.980. Caveat: the rewrite types were already seen through Haiku, only the rewriting model was new.

**Q55. How much did the new data cost and is it reproducible?**
$9.51 for 7,274 Claude texts via the Message Batches API (50% cheaper); open models run locally. Prompts and all scripts are in the repository (`scripts/generate_claude.py`, `generate_open_models.py`, `evaluate_modern.py`, `train_augmented.py`), raw outputs are saved in `reports/final/modern/raw/`.

**Q56. Can your tool show which part of a text was written by AI?**
Partly. We built 1,082 mixed texts where a human answer is continued by a machine and the boundary is known. For every possible boundary the detector scores the part before and the part after it. With ChatGPT continuations the boundary is found within one sentence in 82–89% of texts; with Claude Sonnet continuations the HC3 detectors are at guessing level (52–54%), the retrained ones reach 78%. The exact sentence is found in at most 42% of texts, so it is a hint, not a proof.
*Простыми словами:* после дообучения детектор неплохо показывает, где «примерно» начинается ИИ, но точное предложение угадывает реже чем в половине случаев.

**Q57. What happens in the live demo if the model output is different from your averages?**
One example can differ from the averages; that is why the slides show results on hundreds of texts with confidence intervals. The live demo shows the mechanism, the numbers come from the evaluation. Without internet the same tab uses the saved texts from the experiments.
