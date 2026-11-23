# DEFENSE PREPARATION — защита втроём

> **Формат:** около 13–14 минут доклада (по ~4.5 минуты на спикера, у спикера 3 внутри демо) и вопросы.
> **Как читать:** левая колонка — что говорить по-английски, правая — перевод, чтобы понимать каждое слово.
> Учить наизусть не нужно: перескажите своими словами, числа держите перед глазами.
> Все числа взяты из `reports/final/results.json`. Графики лежат в `reports/final/figures/`.

## ⏱ Минимум на вечер (если времени мало — читайте только это)

Весь проект — это **три истории**. Если вы можете рассказать их своими словами, вы можете защитить проект.

**История 1. Подсказка в данных (главная находка).**
У людей в HC3 перед каждой запятой стоит пробел («word , word»), потому что их ответы когда-то прогнали через старый токенизатор. У ChatGPT такого нет. Правило «есть пробел → человек» без всякого ИИ угадывает **91.8%**. Мы почистили данные. И показали, что модель, обученная на «грязных» данных, на нормальном тексте людей ошибается в 12 раз чаще: ложно обвиняет **15.0%** людей вместо 1.2%.

**История 2. Сравнение моделей на одном тесте (4,380 текстов).**
Трансформер DistilRoBERTa + LoRA (0.990) и TF-IDF (0.989) — лучшие, и они вровень. GPT-2 **без обучения** — 0.958. Одна длина текста — 0.771. Без калибровки порога трансформер ложно обвинял 11.1% людей, поэтому важна не только точность.

**История 3. Новые темы.**
Если тема не встречалась в обучении, обучаемые модели теряют качество (TF-IDF на Reddit: 0.992 → 0.866). А GPT-2 почти не теряет: на новых темах она даже лучше TF-IDF.

**Пять терминов, которые надо уметь объяснить одной фразой:**
| Термин | Одна фраза |
|---|---|
| TF-IDF + logistic regression | считаем, какие слова и знаки чаще у ChatGPT, а какие у людей, и складываем их «голоса» |
| Perplexity | насколько GPT-2 «удивлена» текстом; текст ChatGPT предсказуем, perplexity низкая |
| LoRA | дообучаем не всю модель, а маленькую «насадку» — 0.9% параметров |
| F1 и FPR | F1 — общая точность по классу ChatGPT; FPR — доля людей, которых ложно обвинили |
| Shortcut learning | модель учит случайную подсказку (пробелы) вместо настоящего признака (стиль) |

**Живой пример на защите:** «Красная Шапочка» (сказка 1889 года, есть в списке примеров демо). Сильные детекторы говорят «человек» (TF-IDF, GPT-2, трансформер — около 0% ChatGPT). Слабые стилистические модели ошибаются, потому что сказка полна повторов («what great eyes… what great ears…»), а частые повторы фраз — признак ChatGPT в наших данных. Это наглядно показывает, почему нельзя доверять одному признаку.

**Если спросят, кто писал код:** отвечайте честно. Если для кода или текстов вы пользовались ИИ-ассистентом, так и скажите, если это разрешено правилами курса. Комиссии важнее, понимаете ли вы, что сделано и почему. Для этого и написаны этот файл, `DEFENSE_QA.md` и `FINAL_PROJECT_GUIDE.md`.

## Распределение ролей

| Спикер | Блок | Слайды | Время | Отвечает на вопросы про |
|---|---|:---:|:---:|---|
| **Speaker 1** | Introduction, Problem, Research Questions, Dataset, Preprocessing | 1–6 | ~4 мин | тему, литературу, HC3, очистку, разбиение, утечки |
| **Speaker 2** | ML approaches, architecture, training, evaluation, results | 7–12 | ~4.5 мин | модели, обучение, метрики, результаты, трансформер |
| **Speaker 3** | Dataset bias, generalisation, demo, limitations, conclusion | 13–19 | ~4.5 мин + демо | shortcut learning, новые темы, демо, ограничения, этику |

## Структура слайдов (материалы для презентации)

| # | Слайд | Что на слайде | Файл графика |
|:---:|---|---|---|
| 1 | Title | Тема, команда, курс AITU 2026 | — |
| 2 | Why detect AI text? | Актуальность + цена ложного обвинения | — |
| 3 | Research questions & hypotheses | RQ1–RQ3, H1–H3 | — |
| 4 | Related work | Zero-shot vs supervised; известные проблемы | — |
| 5 | Data: HC3 | 5 тем, пары ответов на один вопрос, разбиение по вопросам | `reports/figures/fig1_class_balance.png` |
| 6 | Cleaning & leakage audit | Протокол очистки (7 шагов), аудит: 0 утечек | таблица из `FINAL_PROJECT_GUIDE.md` Part 2 |
| 7 | Nine detectors, four families | Baseline / classical / neural / zero-shot / transformer | таблица |
| 8 | Classical & neural models | TF-IDF+LogReg, stylometry RF, PyTorch | `tfidf_top_words.png`, `feature_importance_stylometry.png` |
| 9 | Zero-shot & transformer | GPT-2 perplexity, Fast-DetectGPT, DistilRoBERTa full vs LoRA | `perplexity_distribution.png` |
| 10 | Training & evaluation protocol | train/val/test, тюнинг, early stopping, метрики, CI | `nn_training_curves.png` |
| 11 | Results | F1 всех моделей с 95% CI + таблица | `f1_comparison.png` |
| 12 | Where models fail | Матрицы ошибок, ROC при низком FPR, ошибки по длине текста (отдельно люди и ChatGPT) | `confusion_matrices.png`, `roc_curves.png`, `errors_by_length.png` |
| 13 | Dataset bias: a shortcut | Правило «пробел → человек»: 91.8% → 33.1% | `bias_artifacts_before_after.png`, `bias_rule_accuracy.png` |
| 14 | Shortcut stress test | Обучили на сырых, проверили на чистых; остаточные артефакты | `bias_shortcut_stress_test.png`, `bias_shortcut_false_positives.png` |
| 15 | Generalisation | Обучение на 4 темах, тест на 5-й | `cross_domain.png` |
| 16 | Live demo | Streamlit-приложение | живое демо или скриншот |
| 17 | Limitations & ethics | 6 ограничений, детектор ≠ доказательство | — |
| 18 | Conclusions | Ответы на RQ1–3, статус H1–H3 | — |
| 19 | References · Thank you | 8–10 ключевых источников | — |
| B1–B6 | Backup | все метрики, аудит, тюнинг, Fast-DetectGPT, обучение трансформера, примеры ошибок | `results_table.md`, `fast_detectgpt_distribution.png`, `roberta_*_training_curve.png`, `stylometry_logreg_weights.png` |

---

# 🎤 SPEAKER 1 — Introduction, Problem, Research Questions, Dataset, Preprocessing (слайды 1–6, ~4 мин)

### Slide 1. Title (~15 s)
| English (say this) | Русский перевод |
|---|---|
| Good morning. Our project is "Detecting ChatGPT-Generated Text". | Доброе утро. Наш проект — «Распознавание текстов, сгенерированных ChatGPT». |
| We compare classical machine learning, zero-shot language-model scores and fine-tuned transformers. We also study how a dataset can make a detector look better than it really is. | Мы сравниваем классическое машинное обучение, оценки языковых моделей без обучения и дообученные трансформеры. А ещё исследуем, как датасет может сделать детектор лучше, чем он есть на самом деле. |

### Slide 2. Why detect AI-generated text? (~35 s)
| English | Русский |
|---|---|
| Large language models write fluent text in seconds and almost for free. | Большие языковые модели пишут связный текст за секунды и почти бесплатно. |
| Universities want to know if an essay was written by a student; platforms want to find mass-generated content. | Университетам нужно понимать, написал ли эссе студент; платформам — находить массово сгенерированный контент. |
| But a detector error has a price. If it flags a human as AI, a real person is accused. | Но у ошибки детектора есть цена. Если он пометит человека как ИИ, обвинят реального человека. |
| So in our project we care not only about accuracy, but also about false positives and about how much we can trust the scores. | Поэтому нам важна не только точность, но и ложные срабатывания, и то, насколько можно доверять цифрам. |

### Slide 3. Research questions and hypotheses (~45 s)
| English | Русский |
|---|---|
| We ask three questions. First: which approach detects ChatGPT text best — classical ML, style features, zero-shot language-model scores, or a fine-tuned transformer? | Мы задаём три вопроса. Первый: какой подход лучше распознаёт текст ChatGPT — классическое ML, стилистические признаки, оценки языковой модели без обучения или дообученный трансформер? |
| Second: does a detector still work on a topic it has never seen? | Второй: работает ли детектор на теме, которую он никогда не видел? |
| Third: are there hidden artifacts in the dataset that inflate the scores? | Третий: есть ли в датасете скрытые артефакты, которые завышают результат? |
| At the midterm we stated three hypotheses: the transformer will be best; it will lose less on a new topic; and a simple formatting rule will work on raw data but not after cleaning. We will come back to them at the end. | На мидтерме мы сформулировали три гипотезы: трансформер будет лучшим; он меньше потеряет на новой теме; и простое правило по форматированию сработает на сырых данных, но не после очистки. В конце мы к ним вернёмся. |

### Slide 4. Related work (~35 s)
| English | Русский |
|---|---|
| There are two main families of detectors. | Есть два основных семейства детекторов. |
| Zero-shot methods use a language model without training: AI text is more predictable. Examples are perplexity, DetectGPT and Fast-DetectGPT. | Методы без обучения используют языковую модель как есть: текст ИИ более предсказуем. Примеры — perplexity, DetectGPT и Fast-DetectGPT. |
| Supervised methods are trained on labelled examples, for example a fine-tuned RoBERTa, as in the HC3 paper. | Методы с обучением учатся на размеченных примерах, например дообученный RoBERTa, как в статье HC3. |
| Known problems: detectors generalise poorly to new domains and generators, they can be fooled by paraphrasing, and they are biased against non-native English writers. | Известные проблемы: детекторы плохо переносятся на новые темы и генераторы, их обманывает перефразирование, и они предвзяты к людям, для которых английский не родной. |

### Slide 5. Data: HC3 (~40 s) — показать `fig1_class_balance.png`
| English | Русский |
|---|---|
| We use HC3, the Human ChatGPT Comparison Corpus: about 24 thousand questions and 85 thousand answers in five domains — Reddit, finance, medicine, open questions and Wikipedia. | Мы используем HC3 — корпус сравнения человека и ChatGPT: около 24 тысяч вопросов и 85 тысяч ответов в пяти темах — Reddit, финансы, медицина, общие вопросы и Википедия. |
| The key property: for every question there are human answers and ChatGPT answers. So both classes talk about the same topics. | Главное свойство: на каждый вопрос есть ответы людей и ответы ChatGPT. Значит, оба класса говорят на одни и те же темы. |
| We keep one human and one ChatGPT answer per question. The data is perfectly balanced, so 50 percent accuracy means random guessing. | Мы оставляем один ответ человека и один ответ ChatGPT на вопрос. Данные идеально сбалансированы, поэтому 50% accuracy — это случайное угадывание. |
| We split by question: 80 percent train, 10 validation, 10 test. Both answers to a question always stay in the same part, so the model cannot memorise a topic. | Мы делим данные по вопросам: 80% train, 10% validation, 10% test. Оба ответа на вопрос всегда в одной части, поэтому модель не может запомнить тему. |

### Slide 6. Cleaning and leakage audit (~45 s)
| English | Русский |
|---|---|
| Cleaning has seven logged steps. We removed empty answers, server error messages, very short answers and six thousand duplicates. In the final audit we also found ten "ChatGPT answers" that were actually a saved Cloudflare page, and removed them. | Очистка — семь записанных шагов. Мы удалили пустые ответы, сообщения об ошибках сервера, очень короткие ответы и шесть тысяч дубликатов. В финальном аудите мы нашли ещё десять «ответов ChatGPT», которые на самом деле оказались сохранённой страницей Cloudflare, и удалили их. |
| We also normalised formatting with the same rules for both classes. Why this matters, my colleague will show later — it is our main finding. | Ещё мы нормализовали форматирование, одинаковыми правилами для обоих классов. Почему это важно, покажет мой коллега позже — это наша главная находка. |
| Then we checked for data leakage: zero shared questions between train and test, zero identical texts, and only one near-duplicate out of 4,380 test texts. | Потом мы проверили утечку данных: ноль общих вопросов между train и test, ноль одинаковых текстов и всего один почти-дубликат из 4,380 текстов теста. |
| Final dataset: 43,806 texts. The test set has 4,380 texts and is used only once, at the very end. | Итоговый датасет: 43 806 текстов. В тесте 4,380 текстов, и он используется только один раз, в самом конце. |

### ➡️ Transition to Speaker 2
| English | Русский |
|---|---|
| Now [Speaker 2] will present the detectors we built and how they performed. | Теперь [Спикер 2] расскажет, какие детекторы мы построили и как они сработали. |

### Термины для Speaker 1
| Термин | Простыми словами |
|---|---|
| Data leakage | в обучение попала информация из теста, и оценка завышена |
| Balanced dataset | классов поровну, 50/50 |
| Split by question | оба ответа на вопрос в одной части выборки |
| Near-duplicate | почти одинаковые тексты (косинусная близость TF-IDF ≥ 0.9) |
| Zero-shot | без обучения на наших данных |

### Вопросы, которые могут задать Speaker 1
- *Why HC3?* → DEFENSE_QA Q4. *Why one answer per question?* → Q8. *How did you split?* → Q9. *How do you prevent leakage?* → Q33. *Why normalise formatting?* → Q11.

---

# 🎤 SPEAKER 2 — ML approaches, architecture, training, evaluation, results (слайды 7–12, ~4.5 мин)

### Slide 7. Nine detectors, four families (~30 s)
| English | Русский |
|---|---|
| We compare nine detectors under exactly the same conditions: the same training data, the same test set and the same threshold. | Мы сравниваем девять детекторов в абсолютно одинаковых условиях: одни и те же обучающие данные, один тест, один порог. |
| A length-only baseline; classical machine learning with TF-IDF; style features with logistic regression and random forest; our own PyTorch neural network; two zero-shot scores from GPT-2; and the DistilRoBERTa transformer, fine-tuned fully and with LoRA. | Baseline только по длине; классическое ML на TF-IDF; стилистические признаки с логистической регрессией и случайным лесом; наша нейросеть на PyTorch; две оценки GPT-2 без обучения; и трансформер DistilRoBERTa, дообученный полностью и через LoRA. |
| The baseline is important: ChatGPT writes longer answers, so length alone already gives F1 0.771. Every real detector must beat it. | Baseline важен: ChatGPT пишет длиннее, поэтому одна длина уже даёт F1 0.771. Любой настоящий детектор должен его обогнать. |

### Slide 8. Classical and neural models (~50 s) — `tfidf_top_words.png`
| English | Русский |
|---|---|
| TF-IDF turns a text into 100 thousand numbers — weights of words, word pairs and punctuation marks. Logistic regression learns which of them point to ChatGPT and which to a human. | TF-IDF превращает текст в 100 тысяч чисел — веса слов, пар слов и знаков препинания. Логистическая регрессия учит, какие из них указывают на ChatGPT, а какие на человека. |
| For example, "important to", ", which", ", including" and sentence starts like ". it" or ". this" point to ChatGPT, while "I", brackets, slashes, asterisks, "etc", "basically" and the link placeholder "URL_0" point to a human. | Например, «important to», «, which», «, including» и начала предложений «. it», «. this» указывают на ChatGPT, а «I», скобки, слэши, звёздочки, «etc», «basically» и заглушка ссылки «URL_0» — на человека. |
| The stylometric model uses 20 interpretable features: sentence length and its variation, vocabulary richness, punctuation, connectives like "additionally", the word "I", contractions. | Стилометрическая модель использует 20 понятных признаков: длину предложений и её разброс, богатство словаря, пунктуацию, связки вроде «additionally», слово «I», сокращения. |
| Our PyTorch network learns a 64-dimensional vector for every word, averages them and passes the result through two dense layers. | Наша нейросеть на PyTorch учит вектор из 64 чисел для каждого слова, усредняет их и пропускает через два полносвязных слоя. |

### Slide 9. Zero-shot detectors and the transformer (~55 s) — `perplexity_distribution.png`
| English | Русский |
|---|---|
| Zero-shot detectors need no training. GPT-2 reads the text and measures perplexity — how surprised it is. ChatGPT chooses highly probable words, so its text is predictable: median perplexity 10.5 for ChatGPT versus 38.9 for humans. | Детекторы без обучения не учатся на наших данных. GPT-2 читает текст и измеряет perplexity — насколько она «удивлена». ChatGPT выбирает самые вероятные слова, поэтому его текст предсказуем: медиана perplexity 10.5 у ChatGPT против 38.9 у людей. |
| Fast-DetectGPT goes one step further: it compares the real text with what the model itself would likely write at each position. | Fast-DetectGPT идёт дальше: сравнивает реальный текст с тем, что сама модель скорее всего написала бы на каждой позиции. |
| Our transformer is DistilRoBERTa, a smaller version of RoBERTa. We fine-tuned it in two ways: full fine-tuning, which changes all 82 million weights, and LoRA, which freezes the model and trains small add-on matrices — only 0.9 percent of the parameters. | Наш трансформер — DistilRoBERTa, уменьшенная версия RoBERTa. Мы дообучили его двумя способами: полностью, меняя все 82 миллиона весов, и через LoRA, где модель заморожена и учатся маленькие добавочные матрицы — всего 0.9% параметров. |
| Honestly: we have no GPU, so the transformer was trained on 5,000 texts for one epoch on a laptop CPU. | Честно: у нас нет GPU, поэтому трансформер обучен на 5 000 текстов за одну эпоху на CPU ноутбука. |

### Slide 10. Training and evaluation protocol (~45 s) — `nn_training_curves.png`
| English | Русский |
|---|---|
| All models are trained on the training set. Hyperparameters, thresholds and early stopping use only the validation set. The test set is used once. | Все модели обучаются на train. Гиперпараметры, пороги и ранняя остановка — только на validation. Test используется один раз. |
| Here is the training of our neural network: training loss keeps going down, but validation loss starts to rise after epoch 2 — that is overfitting. Early stopping keeps the best epoch. | Вот обучение нашей нейросети: ошибка на train продолжает падать, а на validation начинает расти после эпохи 2 — это переобучение. Early stopping сохраняет лучшую эпоху. |
| We report accuracy, precision, recall, F1 and ROC-AUC, and also the false positive rate — the share of humans wrongly flagged — because that is the error that hurts people. | Мы считаем accuracy, precision, recall, F1 и ROC-AUC, а ещё false positive rate — долю людей, ложно помеченных как ИИ, — потому что именно эта ошибка вредит людям. |
| Every score has a 95 percent confidence interval from a bootstrap over test questions. | У каждой метрики есть 95% доверительный интервал, полученный бутстрепом по вопросам теста. |

### Slide 11. Results (~50 s) — `f1_comparison.png`
| English | Русский |
|---|---|
| Here are the results on the same test set. The two best detectors are DistilRoBERTa + LoRA, F1 0.990, and TF-IDF + LogReg, F1 0.989. Their confidence intervals overlap, so we treat them as equally good. | Вот результаты на одном и том же тесте. Два лучших детектора — DistilRoBERTa + LoRA, F1 0.990, и TF-IDF + LogReg, F1 0.989. Их доверительные интервалы пересекаются, поэтому мы считаем их одинаково хорошими. |
| TF-IDF reaches F1 0.989; GPT-2 perplexity without any training reaches 0.958; the style-feature forest 0.934; our PyTorch network 0.952. | TF-IDF достигает F1 0.989; perplexity GPT-2 без всякого обучения — 0.958; случайный лес на стилистических признаках — 0.934; наша нейросеть на PyTorch — 0.952. |
| DistilRoBERTa with LoRA reaches 0.990 and with full fine-tuning 0.986, trained on only 5,000 texts. One lesson here: with the default threshold 0.5, LoRA flagged 11.1% of humans; after calibrating on the validation set only 1.1%. The model ranks texts almost perfectly, ROC-AUC 0.9996, but the threshold must be chosen carefully. | DistilRoBERTa с LoRA достигает 0.990, при полном дообучении — 0.986, обучившись всего на 5 000 текстов. Здесь есть урок: при стандартном пороге 0.5 LoRA ложно помечала 11.1% людей, а после калибровки на validation — только 1.1%. Модель почти идеально сортирует тексты, ROC-AUC 0.9996, но порог нужно выбирать аккуратно. |
| All of them are far above the length baseline, so they learn more than length. | Все они намного выше baseline по длине, значит, учат больше, чем длину. |

### Slide 12. Where the models fail (~40 s) — `errors_by_length.png`, `roc_curves.png`
| English | Русский |
|---|---|
| Errors are not random. ChatGPT almost never writes short answers in HC3: only 19 of 777 short test texts. But those few are often missed: GPT-2 perplexity misses 12 of 19 (63%) of them, TF-IDF 2 of 19 (11%). | Ошибки не случайны. В HC3 ChatGPT почти не пишет коротких ответов: всего 19 из 777 коротких текстов теста. Но эти немногие часто пропускаются: perplexity GPT-2 пропускает 12 of 19 (63%) из них, TF-IDF — 2 of 19 (11%). |
| If we allow only 1 percent of humans to be falsely flagged, TF-IDF still catches 98.8% of ChatGPT texts, while GPT-2 perplexity catches 79.1%. | Если допустить ложные обвинения только у 1% людей, TF-IDF всё ещё ловит 98.8% текстов ChatGPT, а perplexity GPT-2 — 79.1%. |
| But these are in-domain numbers. [Speaker 3] will show why they should not be trusted blindly. | Но это числа внутри одного датасета. [Спикер 3] покажет, почему им нельзя слепо доверять. |

### ➡️ Transition to Speaker 3
| English | Русский |
|---|---|
| Now [Speaker 3] will show what these numbers hide. | Теперь [Спикер 3] покажет, что скрывают эти цифры. |

### Термины для Speaker 2
| Термин | Простыми словами |
|---|---|
| TF-IDF | вес слова: часто в этом тексте и редко в остальных |
| Logistic regression | взвешенная сумма признаков → вероятность |
| Random forest | много деревьев решений голосуют |
| Embedding | вектор чисел, представляющий слово |
| Perplexity | «удивление» модели текстом; низкая = предсказуемый текст |
| LoRA | дообучение маленькой «насадки» вместо всей модели |
| Overfitting / early stopping | модель запоминает train; останавливаемся, когда validation loss растёт |
| ROC-AUC | вероятность, что текст ChatGPT получит score выше, чем текст человека |
| False positive rate | доля людей, ложно помеченных как ИИ |
| Bootstrap CI | интервал, в котором метрика лежит с вероятностью 95% |

### Вопросы, которые могут задать Speaker 2
- *Why is TF-IDF so good?* → Q14. *Why is accuracy not enough?* → Q28. *Why is perplexity lower for ChatGPT?* → Q25. *What is LoRA?* → Q24. *Why DistilRoBERTa?* → Q23. *Does high F1 mean reliable?* → Q31. *Why is the NN worse than TF-IDF?* → Q21.

---

# 🎤 SPEAKER 3 — Dataset bias, generalisation, demo, limitations, conclusion (слайды 13–19, ~4.5 мин + демо)

### Slide 13. Dataset bias: a shortcut in the data (~50 s) — `bias_artifacts_before_after.png`, `bias_rule_accuracy.png`
| English | Русский |
|---|---|
| This is our main finding. Human answers in HC3 were stored tokenized: "do n't" with a space, a space before every comma. ChatGPT answers were scraped differently and lost their line breaks. | Это наша главная находка. Ответы людей в HC3 сохранены токенизированными: «do n't» с пробелом, пробел перед каждой запятой. Ответы ChatGPT собраны иначе и потеряли переносы строк. |
| A one-line rule with no AI — "if there is a space before a comma, it is human" — is correct for 91.8% of all HC3 answers. | Правило в одну строку без всякого ИИ — «если перед запятой пробел, значит человек» — верно для 91.8% всех ответов HC3. |
| After our cleaning, the same rule drops to 33.1%; on our balanced test set it is exactly 50.0%, a coin flip. | После нашей очистки то же правило падает до 33.1%; на нашем сбалансированном тесте — ровно 50.0%, подбрасывание монетки. |
| This is called shortcut learning: a model can learn how the data was stored instead of who wrote the text. | Это называется shortcut learning: модель может выучить, как хранились данные, вместо того, кто написал текст. |

### Slide 14. Is the shortcut dangerous? A stress test (~45 s) — `bias_shortcut_stress_test.png`
| English | Русский |
|---|---|
| To prove it is dangerous, we trained a character-level model, which sees spaces, on the raw text, and tested it on clean text — like real human writing without the artifact. | Чтобы доказать опасность, мы обучили посимвольную модель, которая видит пробелы, на сырых текстах и проверили её на чистых — как настоящий человеческий текст без артефакта. |
| On raw test text it looks great: 98.8% accuracy. On clean text, the share of humans wrongly flagged as ChatGPT jumps from 1.2% to 15.0% — more than twelve times more false accusations. | На сыром тесте она выглядит отлично: 98.8% accuracy. На чистом тексте доля людей, ложно помеченных как ChatGPT, подскакивает с 1.2% до 15.0% — в двенадцать раз больше ложных обвинений. |
| The same model trained on clean text gets 97.7% with only 2.0% false positives. Our word-level TF-IDF does not see spaces, so it was safe — but transformer tokenizers do see them. That is why we cleaned the data before training anything. | Та же модель, обученная на чистых текстах, получает 97.7% и всего 2.0% ложных срабатываний. Наш словесный TF-IDF пробелов не видит, поэтому не пострадал, — а вот токенизаторы трансформеров их видят. Поэтому мы очистили данные до обучения любых моделей. |
| We also measured residual artifacts: link placeholders "URL_0" appear in 9.1% of human answers; removing them changes TF-IDF F1 only from 0.986 to 0.987. | Мы измерили и остаточные артефакты: заглушки ссылок «URL_0» есть в 9.1% ответов людей; их удаление меняет F1 TF-IDF всего с 0.986 до 0.987. |

### Slide 15. Generalisation to unseen topics (~45 s) — `cross_domain.png`
| English | Русский |
|---|---|
| Next, we trained every model on four domains and tested it on the fifth, comparing with the same test texts when the domain was seen. | Дальше мы обучили каждую модель на четырёх темах и проверили на пятой, сравнивая на тех же тестовых текстах, когда тема была в обучении. |
| Every trained model loses quality on a new topic. The biggest drop is on Reddit: TF-IDF falls from 0.992 to 0.866, our neural network from 0.955 to 0.751. | Каждая обучаемая модель теряет качество на новой теме. Сильнее всего падение на Reddit: TF-IDF падает с 0.992 до 0.866, наша нейросеть — с 0.955 до 0.751. |
| Interestingly, the zero-shot GPT-2 detectors lose almost nothing, because they never learned topics. Among the models tested on all five domains, GPT-2 perplexity has the best average F1 on unseen domains, 0.947, while TF-IDF falls to 0.911. The ranking flips. Our transformer, trained without Reddit, goes from 0.995 to 0.980 on Reddit, so it transfers much better than TF-IDF. | Интересно, что zero-shot детекторы GPT-2 почти ничего не теряют, потому что не учили темы. Среди моделей, проверенных на всех пяти темах, у perplexity GPT-2 лучший средний F1 на незнакомых темах — 0.947, а TF-IDF падает до 0.911. Рейтинг переворачивается. Наш трансформер, обученный без Reddit, на Reddit даёт 0.980 вместо 0.995 — переносится заметно лучше TF-IDF. |
| So a detector partly learns the topic and the source of the human answers, not only AI style. | Значит, детектор частично учит тему и источник ответов людей, а не только стиль ИИ. |

### NEW after the defense: slides 11–15 of `reports/Final_Presentation.pptx` (~3 min + live demo)
> Говорит Speaker 3 сразу после слайда «Where the detectors struggle». Числа: `reports/final/modern_results.json`,
> `augmented_results.json`, `segments_results.json`.

**Slide 11. 2026 models slip past a 2023 detector** — `modern_generators.png`
| English | Русский |
|---|---|
| A score of 0.99 raised a question: does our detector still work on today's models? We took 555 test questions and generated new answers with Claude Haiku, Sonnet and Opus 5.5 and with two small open models. | F1 0.99 поставил вопрос: работает ли детектор на современных моделях? Мы взяли 555 тестовых вопросов и получили новые ответы от Claude Haiku, Sonnet и Opus 5.5 и двух малых открытых моделей. |
| The same detector that catches 98% of ChatGPT 2023 answers catches only 10 to 21% of Claude answers. Small open models are caught much better. ROC-AUC is still 0.8–0.9, so the signal is there; the threshold learned on HC3 is what fails. | Тот же детектор, что ловит 98% ответов ChatGPT 2023, ловит только 10–21% ответов Claude. Малые открытые модели ловятся гораздо лучше. ROC-AUC всё ещё 0.8–0.9: сигнал есть, не работает порог, выученный на HC3. |

**Slide 12. One rewrite hides AI text** — `modern_attacks.png`
| English | Русский |
|---|---|
| Then we attacked. One request to "make it sound like a person on a forum" hides 93 to 99% of ChatGPT texts from every trained detector. Typos mainly hurt the GPT-2 detectors. | Потом мы атаковали. Одна просьба «перепиши как человек на форуме» прячет 93–99% текстов ChatGPT от всех обучаемых детекторов. Опечатки сильнее всего бьют по детекторам на GPT-2. |

**Slide 13. One modern model in training helps on unseen ones** — `augmented_roberta_lora.png`
| English | Русский |
|---|---|
| Can we fix it? We added texts of only one model, Claude Haiku, written for training questions only. On Sonnet and Opus, never seen in training, detection goes from 10 to 93 and from 12 to 94 percent; humanized texts from 7 to 94. | Можно ли это исправить? Мы добавили тексты только одной модели, Claude Haiku, и только к обучающим вопросам. На Sonnet и Opus, которых не было в обучении, детекция выросла с 10 до 93% и с 12 до 94%, «очеловеченные» тексты — с 7 до 94%. |
| The price is honest: humans wrongly flagged rise from 2.3 to 4.5 percent, HC3 F1 drops from 0.990 to 0.980. | Цена честная: ложных обвинений людей стало 4.5% вместо 2.3%, F1 на HC3 — 0.980 вместо 0.990. |

**Slide 14. Retraining also reveals where the AI part starts** — `segments_boundary.png`
| English | Русский |
|---|---|
| Real texts are often mixed: a student writes the start and lets AI finish. On 1,082 mixed texts with a known boundary, detectors trained on HC3 find where Claude takes over no better than guessing, 52–54%. After retraining, 78% of boundaries are found within one sentence. The exact sentence is still hard: at most 42%. | Реальные тексты часто смешанные: студент пишет начало, ИИ дописывает. На 1 082 смешанных текстах с известной границей детекторы, обученные на HC3, находят начало части Claude не лучше угадывания — 52–54%. После дообучения 78% границ найдены с точностью до одного предложения. Точно в предложение пока трудно: максимум 42%. |

**Slide 15. Which part did the AI write? → живое демо**
| English | Русский |
|---|---|
| Here two sentences are from a real person, the rest is Claude. The old detector says 0% AI for the whole text, the retrained one 94%, and it puts the boundary exactly where the human part ends. Let me show it live. | Здесь два предложения написал человек, остальное — Claude. Старый детектор говорит 0% ИИ на весь текст, дообученный — 94%, и ставит границу ровно там, где кончается часть человека. Покажу вживую. |

**Живое демо (вкладка «Live: fool the detector»):**
1. Режим **Live (Claude API)**. Попросить комиссию задать любой вопрос → вписать → **Ask Claude**. Ответ печатается на глазах.
2. Показать две карточки: «Trained on ChatGPT 2023» обычно говорит *Human*, «Retrained» — *AI*.
3. Нажать **Humanize with Claude Sonnet 5.5** → ещё раз две карточки и таблица «Original → Humanized».
4. Для смешанного текста: в поле «write the first sentences yourself» член комиссии пишет 1–2 предложения → Claude продолжает → цветная подсветка и метка «AI starts here?».
5. **Если нет интернета или API не отвечает:** переключить на **Offline (saved texts)** → Author = «Mixed: human start + Claude Sonnet 5.5» → Show answer. Всё то же самое на сохранённых текстах.
6. Одно живое срабатывание может отличаться от средних цифр — так и сказать: «one example can differ, the averages are on the slides».

### Slide 16. Live demo (~60–90 s)
**Подготовка до защиты:** двойной щелчок по `Запустить демо.bat`. Браузер откроет `http://localhost:8501` через 10–20 секунд. Окно не закрывать. Один раз нажать любой пример → **Analyze**, чтобы всё «прогрелось»: первая загрузка GPT-2 и трансформера медленнее.

| English | Русский |
|---|---|
| This is our demo. I paste a text and press Analyze. | Это наше демо. Я вставляю текст и нажимаю Analyze. |
| *(выбрать «ChatGPT · Finance» → Analyze)* The main detector says "likely AI-generated" with its probability, and below we see that all eight detectors agree. | *(выбрать «ChatGPT · Finance» → Analyze)* Главный детектор говорит «вероятно, сгенерировано ИИ» и показывает вероятность, а ниже видно, что все восемь детекторов согласны. |
| Here are the text statistics, GPT-2 perplexity and style indicators compared with typical human and ChatGPT values, and the words that pushed the decision. | Здесь статистика текста, perplexity GPT-2 и стилистические индикаторы рядом с типичными значениями у людей и у ChatGPT, а также слова, которые повлияли на решение. |
| *(выбрать «Hard case: human text that TF-IDF flags as ChatGPT» → Analyze)* And this is a real human answer from our test set. TF-IDF flags it as ChatGPT, and the detectors disagree with each other: look at the votes here. A calibrated model puts it into the "uncertain" zone instead of accusing the author. That is why the app says: a statistical estimate, not proof of authorship. | *(выбрать «Hard case: human text…» → Analyze)* А это настоящий ответ человека из нашего теста. TF-IDF считает его текстом ChatGPT, и детекторы расходятся во мнениях: посмотрите на голоса. Откалиброванная модель относит его к зоне «неуверенно», а не обвиняет автора. Поэтому приложение пишет: статистическая оценка, а не доказательство авторства. |

**Если демо не открылось:** не тратить время, показать скриншот на слайде и сказать: «The demo runs locally; here is a screenshot of the same analysis».

### Slide 17. Limitations and ethics (~35 s)
| English | Русский |
|---|---|
| Our limitations: one generator, ChatGPT from 2023, and English only. We did not test paraphrased or mixed human and AI texts, or newer models like GPT-4. | Наши ограничения: один генератор, ChatGPT 2023 года, и только английский. Мы не проверяли перефразированные и смешанные тексты, а также новые модели вроде GPT-4. |
| The transformer was trained on a subset on a CPU. GPT-2 reads only the first 128 tokens. | Трансформер обучен на подвыборке на CPU. GPT-2 читает только первые 128 токенов. |
| And ethically: false accusations are worse than missed AI texts. A detector should support a human decision, never replace it. | И этика: ложное обвинение хуже, чем пропущенный ИИ-текст. Детектор должен помогать человеку принимать решение, но никогда не заменять его. |

### Slide 18. Conclusions (~45 s)
| English | Русский |
|---|---|
| RQ1: in-domain, the fine-tuned transformer and TF-IDF are equally the best, about 0.99 F1, and zero-shot GPT-2 reaches 0.96 without any training. | RQ1: на знакомых темах дообученный трансформер и TF-IDF одинаково лучшие, около 0.99 F1, а GPT-2 без всякого обучения даёт 0.96. |
| RQ2: trained models lose quality on an unseen topic, up to 0.205 F1, while zero-shot GPT-2 detectors lose almost nothing; among the models tested on all five topics they are the best on new topics, and our transformer, tested on Reddit, also transfers well. | RQ2: обучаемые модели теряют качество на незнакомой теме, до 0.205 F1, а zero-shot детекторы GPT-2 почти ничего не теряют; среди моделей, проверенных на всех пяти темах, они лучшие на новых темах, а наш трансформер, проверенный на Reddit, тоже хорошо переносится. |
| RQ3: yes. A formatting shortcut alone gives 91.8%, and a model trained on it fails on clean human text. | RQ3: да. Одно лишь форматирование даёт 91.8%, а модель, обученная на нём, ломается на чистом человеческом тексте. |
| Hypotheses: H1 is partly confirmed: the transformer reaches 0.99 but is tied with TF-IDF. H2 is confirmed on Reddit: the transformer lost 0.015 F1 versus 0.126 for TF-IDF. H3 is confirmed. | Гипотезы: H1 подтверждена частично: трансформер достигает 0.99, но вровень с TF-IDF. H2 подтверждена на Reddit: трансформер потерял 0.015 F1 против 0.126 у TF-IDF. H3 подтверждена. |
| Our main lesson: a 99 percent benchmark score is not the end of the analysis. It is where the analysis starts. | Наш главный вывод: 99% на бенчмарке — не конец анализа, а его начало. |

### Slide 19. References · Thank you (~10 s)
| English | Русский |
|---|---|
| Here are our main references. Thank you for your attention — we are happy to answer your questions. | Здесь наши основные источники. Спасибо за внимание, мы готовы ответить на вопросы. |

### Термины для Speaker 3
| Термин | Простыми словами |
|---|---|
| Shortcut learning | модель решает задачу «короткой дорогой» по случайному признаку |
| Dataset bias / artifact | отличие классов, созданное сбором данных, а не задачей |
| Tokenized text | текст, где знаки отделены пробелами: «do n't», «word ,» |
| Stress test | проверка в условиях, где «подсказки» нет |
| Leave-one-domain-out | обучить на 4 темах, проверить на 5-й |
| Domain shift | новые тексты отличаются от обучающих |

### Вопросы, которые могут задать Speaker 3
- *What is shortcut learning?* → Q34. *How did you prove the shortcut is dangerous?* → Q35. *Why does performance drop on new topics?* → Q38. *Can it detect GPT-4?* → Q40. *What about paraphrasing?* → Q41. *Can a human be flagged as AI?* → Q44. *Would you use it at a university?* → Q47.

---

## Чек-лист перед защитой

1. Каждый прочитал свой блок и таблицу «Главные числа» в `DEFENSE_QA.md`.
2. Один полный прогон втроём вслух с таймером (цель — 13–14 минут).
3. Демо запущено заранее и «прогрето» (первый анализ уже сделан).
4. На ноутбуке открыты: презентация, браузер с демо, `reports/final/results_table.md` (на случай вопроса «покажите все метрики»).
5. Если спросят код: `src/detector/` (модели и метрики), `scripts/run_experiments.py` (весь эксперимент), `notebooks/Full_Project_Explained.ipynb` (пошаговое объяснение).
6. Если не знаете ответа: *"That is a good question. We did not test it, but our expectation is … and it is a direction for future work."*
