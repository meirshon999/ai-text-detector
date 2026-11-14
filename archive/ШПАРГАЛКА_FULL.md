# Шпаргалка к защите мидтерма

> Тема: **Распознавание текста, сгенерированного ChatGPT: сравнение классического машинного обучения, детекторов на основе perplexity и дообученных трансформеров**
> *Detecting ChatGPT-Generated Text: A Comparison of Classical Machine Learning, Perplexity-Based and Fine-Tuned Transformer Detectors*

**Читать в таком порядке:** раздел 1 (рассказ по слайдам) → раздел 3 (цифры) → раздел 4 (литература и данные) → раздел 6 (вопросы и ответы).

**Файлы:** презентация `reports/Midterm_Presentation.pptx` (17 слайдов и 10 слайдов приложения) · письменный отчёт `reports/Project_Proposal.docx` · код `notebooks/01–03` · скриншоты-доказательства `reports/evidence/`.

---

## 1. Рассказ по слайдам (около 7 минут)

| Слайд | Что сказать (по-английски, своими словами) | Смысл по-русски |
|:---:|---|---|
| 1 | *My project is about detecting text written by ChatGPT.* | Тема |
| 2 | *Universities and platforms need detectors, but mistakes lead to unfair accusations. Task: given an answer, decide if a human or ChatGPT wrote it.* | Зачем и какая задача |
| 3 | *Four research questions: which detector family is best, which language features differ, does it work on new topics, and do dataset artifacts inflate scores.* | 4 вопроса, 4 гипотезы, цели |
| 4 | *The literature has four families: watermarking, zero-shot statistics like perplexity and DetectGPT, trained classifiers like RoBERTa, and stylometry. Their known weakness is shortcut learning: they fail on new domains and attacks.* | **4 семейства методов** |
| 5 | *Key studies and the gap: nobody audited HC3 for artifacts and compared all families on the same cleaned splits.* | Ключевые статьи и пробел |
| 6 | *HC3 combines five public QA datasets — Reddit, FiQA, WikiQA, MedDialog, Wikipedia. ChatGPT answers were collected through the ChatGPT web preview; the paper appeared in January 2023.* | **Откуда данные** (скриншот Hugging Face) |
| 7 | *Here is the evidence: the dataset page with the five files we downloaded, and the original paper on arXiv.* | **Доказательство источника** |
| 8 | *ChatGPT writes answers of a standard length; length alone gives F1 0.77.* | Длина |
| 9 | *Key finding: human texts have spaces before commas. A one-line rule gets 92% without any AI.* | **Ловушка в данных** |
| 10 | *My cleaning protocol: five steps, every removed row is logged, the same rules for both classes. After it the rule drops to 33%.* | **Как чистили** |
| 11 | *I tested the literature statistically: ChatGPT uses more connectives, humans have richer vocabulary — like Guo et al. on HC3, but opposite to Herbold et al. on essays. Burstiness works only when length is controlled.* | **Лингвистика** |
| 12 | *Six detectors: length, TF-IDF, stylometry, a PyTorch network, GPT-2 perplexity; RoBERTa with LoRA in week 7.* | Что обучали |
| 13 | *All are accurate in-domain; TF-IDF is best with F1 0.987; GPT-2 works without training.* | Результаты |
| 14 | *On an unseen domain F1 drops up to 13 points — the main problem for the transformer stage.* | Новые темы |
| 15 | *Limitations: one generator, English. Ethics: false accusations, bias, explainability.* | Ограничения, этика |
| 16 | *Next: RoBERTa full vs LoRA, Binoculars and Fast-DetectGPT, robustness tests, a demo.* | План |
| 17 | *References.* | Литература |
| A1–A9 | Показывать, только если попросят доказать: *"Here are screenshots of the executed notebook cells — code and real output."* | **Приложение: скриншоты кода** |

### Как отвечать на «Это сделал ИИ?» / «Покажите, что вы это реализовали»
1. Под каждым графиком и таблицей в презентации есть синяя строка **Source:** с точным ноутбуком и номером ячейки, например `notebooks/02_preprocessing_and_baselines.ipynb, cell [16]`.
2. В приложении (слайды A1–A9) скриншоты этих ячеек: код и его реальный вывод, номера ячеек видны.
3. Лучшее доказательство — открыть ноутбук на ноутбуке и показать ту же ячейку вживую. Подготовьте к защите открытый Jupyter.
4. Скриншоты источников: страница датасета на Hugging Face и статья на arXiv (слайды 6–7).

---

## 2. Термины простыми словами

| Термин | Что это | Аналогия |
|---|---|---|
| **Детектор** | Программа, которая решает: текст написал человек или ИИ | Как антиплагиат, только ищет не копии, а «почерк» ИИ |
| **HC3** | Набор данных: вопросы, на каждый есть ответы людей и ответы ChatGPT | Сборник «вопрос → два ответа» для тренировки |
| **Домен** | Тематическая область: финансы, медицина, Reddit и т. д. | Разные «предметы» в школе |
| **Классификация** | Задача выбрать один ответ из вариантов (здесь: человек / ChatGPT) | Сортировка писем на «спам» и «не спам» |
| **TF-IDF** | Способ превратить текст в числа: важные для этого текста слова получают больший вес | Подсчёт характерных слов |
| **Логистическая регрессия** | Простая модель: каждому слову даёт «голос» за человека или за ChatGPT и складывает голоса | Весы с гирьками |
| **Embedding (эмбеддинг)** | Слово превращается в набор чисел, похожие слова получают похожие числа | Координаты слова на карте смыслов |
| **Нейросеть bag-of-embeddings** | Усредняем эмбеддинги всех слов текста и отдаём нейросети. Сделано на **PyTorch** | «Средний вкус» текста |
| **Perplexity (перплексия)** | Насколько языковая модель «удивлена» текстом. Текст ИИ предсказуем → низкая перплексия | Читаешь книгу и угадываешь следующее слово: в тексте ИИ угадываешь чаще |
| **GPT-2** | Небольшая открытая языковая модель OpenAI (2019), используем её для подсчёта перплексии | — |
| **Zero-shot** | Метод без обучения на наших данных | Сдать экзамен без подготовки, на общих знаниях |
| **RoBERTa** | Модель-трансформер, улучшенная версия BERT, хорошо понимает текст | Опытный читатель, которого мы доучим |
| **Fine-tuning (дообучение)** | Берём готовую модель и доучиваем на нашей задаче | Опытного врача доучиваем узкой специальности |
| **LoRA** | Дообучение, при котором меняется лишь маленькая добавка к модели, а не вся модель. Быстрее и дешевле | Вместо переписывания учебника вклеиваем закладки с пометками |
| **Артефакт данных** | Случайная особенность данных, не связанная с задачей, по которой модель может «жульничать» | Учитель ставит пятёрки тем, кто пишет синей ручкой, а не тем, кто знает |
| **Train / validation / test** | Данные для обучения / для настройки / для финальной проверки | Учебник / пробный тест / экзамен |
| **Разбиение по вопросу** | Оба ответа на один вопрос всегда в одной части, чтобы модель не запомнила тему | Не даём на экзамене вопросы из учебника |
| **Leave-one-domain-out** | Учим на 4 темах, проверяем на 5-й, которую модель не видела | Учили физику, химию, биологию, историю — экзамен по географии |

| **Стилометрия** | Распознавание автора по стилю: длина предложений, словарь, связки, пунктуация | Почерк, только в тексте |
| **MATTR** | Богатство словаря: доля разных слов в «окне» из 50 слов, усреднённая по тексту | Сколько разных слов человек использует |
| **Burstiness («ритм»)** | Насколько сильно различается длина предложений. У людей «рвано», у ИИ ровно | Музыка с паузами или метроном |
| **Связки (connectives)** | Слова-переходы: Additionally, Overall, Firstly, In general | «Во-первых… в-итоге» |
| **Тест Манна–Уитни** | Статистическая проверка, что две группы действительно различаются, а не случайно | — |
| **Cliff's δ (дельта Клиффа)** | Насколько сильно различаются группы: от −1 до +1. Больше 0,47 по модулю — сильная разница | «Сила» различия |
| **Shortcut learning** | Модель учится на случайных подсказках в данных, а не на сути | Ученик запомнил, что правильный ответ всегда «В» |
| **Случайный лес (random forest)** | Много деревьев решений голосуют вместе | Консилиум врачей |

### Метрики

| Метрика | Что показывает |
|---|---|
| **Accuracy** | Доля правильных ответов |
| **Precision** | Из всех текстов, которые модель назвала «ChatGPT», какая доля правда от ChatGPT |
| **Recall** | Из всех текстов ChatGPT какую долю модель нашла |
| **F1** | Среднее Precision и Recall в одном числе (от 0 до 1) |
| **ROC-AUC** | Насколько хорошо модель ранжирует: 1,0 — идеально, 0,5 — как монетка |
| **TPR@1%FPR** | Какую долю текстов ChatGPT детектор ловит, если ложно обвиняет не больше 1% людей. Главная метрика «честности» |

---

## 3. Главные цифры (выучить)

### Данные
| | |
|---|---|
| Вопросов / ответов в HC3 | **24 322 / 85 449** |
| Тем (доменов) | **5**: финансы, медицина, общие вопросы, Reddit, Википедия |
| Дубликатов / ошибок сервера / пустых | **6 117 / 100 / 18** |
| Сбалансированный набор | **21 913 вопросов × (1 ответ человека + 1 ответ ChatGPT)** |
| Разбиение | **80 / 10 / 10** (обучение / настройка / проверка), по вопросам |

### Ловушка в данных
| | |
|---|---|
| Ответов людей с пробелом перед запятой | **88,8%** (у ChatGPT 0,3%) |
| Правило «пробел перед запятой → человек» на сырых данных | **91,8%** |
| То же правило после очистки | **33,1%**, хуже монетки |

**Почему 33%, а не 50%?** После очистки пробелов нет ни у кого, и правило на всё отвечает «ChatGPT». В полном наборе ответов людей примерно 2/3, поэтому оно угадывает только треть.

### Протокол очистки
| Шаг | Осталось | Удалено |
|---|:---:|:---:|
| Исходный HC3 | 85 449 | — |
| 1. Пустые ответы | 85 431 | 18 |
| 2. Ошибки сервера ChatGPT | 85 331 | 100 |
| 3. Нормализация форматирования | 85 331 | 0 |
| 4. Короче 5 слов | 85 316 | 15 |
| 5. Дубликаты | 79 304 | 6 012 |
| 6. Пары «1 человек + 1 ChatGPT» | 43 826 (21 913 пар) | — |

### Результаты на проверочных данных
| Детектор | F1 | TPR@1%FPR | Простыми словами |
|---|:---:|:---:|---|
| Только длина текста | **0,77** | 0,00 | Нижняя планка |
| TF-IDF + логистическая регрессия | **0,987** | 0,98 | Лучший сейчас |
| Стилометрия + лог. регрессия | **0,86** | 0,07 | Объяснимо, но при 1% ложных обвинений почти ничего не ловит |
| Стилометрия + случайный лес | **0,92** | 0,57 | Объяснимо и неплохо |
| Нейросеть на PyTorch | **0,953** | 0,79 | Своя нейросеть с нуля, 6 эпох |
| Perplexity GPT-2 (без обучения) | **0,953** | 0,79 | Работает вообще без обучения |

### Лингвистика (Cliff's δ: плюс — больше у ChatGPT, минус — больше у людей)
| Признак | Все тексты | Тексты 100–300 слов | Вывод |
|---|:---:|:---:|---|
| Связки (Additionally, Overall…) | **+0,49** | +0,46 | ChatGPT гораздо чаще |
| Богатство словаря (MATTR) | **−0,52** | −0,43 | У людей богаче |
| Длина предложения | +0,44 | +0,33 | У ChatGPT длиннее |
| «I», «my» | −0,28 | −0,39 | Люди чаще пишут от себя |
| Скобки | −0,26 | −0,47 | Люди чаще |
| **Burstiness («ритм»)** | **+0,02** | **−0,43** | **Видно только при одинаковой длине!** |

**Главная лингвистическая находка:** «ритм» текста (на нём построен GPTZero) по всем текстам не отличается. Но если сравнивать тексты одинаковой длины, у людей ритм заметно неровнее. Короткие ответы людей (2–3 предложения) прятали эффект. Значит, длину нужно контролировать.

- Медианная perplexity: **ChatGPT около 10, люди около 39.** Текст ChatGPT гораздо предсказуемее.

### Проверка на новой теме (обучили на 4 темах, проверили на 5-й)
| Тема | F1 на знакомой теме | F1 на новой теме | Падение |
|---|:---:|:---:|:---:|
| Финансы | 0,994 | 0,977 | 1,7 |
| Медицина | 1,000 | 0,993 | 0,7 |
| Общие вопросы | 0,939 | 0,849 | 9,0 |
| **Reddit** | 0,991 | 0,860 | **13,1** |
| Википедия | 0,955 | 0,875 | 8,0 |

**Главная мысль:** на знакомых темах задача почти решена, но на новых качество падает до 13 пунктов. Это и решаем трансформером на 7-й неделе.

Стилометрия на новых темах теряет меньше на 3 из 5 тем, но её F1 всё равно ниже.

---

## 4. Литература и данные: что сказать, если спросят

### Четыре семейства методов (по обзору Wu et al., 2025, журнал Computational Linguistics, Q1)
| Семейство | Идея | Ключевые работы | У нас |
|---|---|---|---|
| Водяные знаки | Генератор прячет в текст статистическую метку | Kirchenbauer et al., ICML 2023 | Нельзя: нет доступа к ChatGPT |
| Статистика без обучения | Текст ИИ предсказуемее для языковой модели | GLTR 2019, DetectGPT 2023, Fast-DetectGPT 2024, Binoculars 2024 | GPT-2 perplexity; Binoculars на 9-й неделе |
| Обучаемые классификаторы | Учим модель на примерах «человек / ИИ» | RoBERTa в HC3, Ghostbuster 2024 | TF-IDF, PyTorch; RoBERTa на 7-й неделе |
| Стилометрия | Признаки стиля | Guo et al. 2023, Herbold et al. 2023 (Scientific Reports, Q1) | 18 признаков |

**Известная слабость всех методов** — shortcut learning:
- **M4 (EACL 2024):** детекторы ломаются на новых темах.
- **RAID (ACL 2024):** их легко обмануть перефразированием.
- **Xu et al. (EMNLP 2024):** цепляются за длину и тему.
- **Park et al. (2024):** в HC3 есть «ярлыки» от промптов.
- **Liang et al. (Patterns, Q1):** детекторы чаще ошибочно обвиняют тех, для кого английский не родной.

**Пробел, который закрывает проект:** HC3 никто не проверял на ловушки форматирования, и все семейства методов не сравнивали на одних и тех же очищенных данных с проверкой на новых темах и метрикой ложных обвинений.

### Откуда данные HC3
| Часть | Откуда вопросы и ответы людей |
|---|---|
| reddit_eli5 (80%) | Reddit «Explain Like I'm Five», датасет ELI5 (Fan et al., 2019) |
| finance | FiQA: посты StackExchange об инвестициях (Maia et al., 2018) |
| open_qa | WikiQA: поисковые запросы с ответами из Википедии (Yang et al., 2015) |
| medicine | MedDialog: диалоги врач–пациент (Zeng et al., 2020) |
| wiki_csai | Описания понятий информатики из Википедии |

**Как собирали ответы ChatGPT:** через веб-превью ChatGPT, по новому диалогу на каждый вопрос. К вопросам Reddit добавляли «Explain like I'm five». Обычно брали один ответ на вопрос. Статья вышла на arXiv 18 января 2023. Точных дат сбора в статье нет, поэтому называть их не нужно.

**Важный нюанс про Herbold et al. (2023):** они изучали эссе и нашли у ChatGPT *меньше* связок и *более богатый* словарь, то есть противоположное нашему результату на HC3. Это нормально: стиль ИИ зависит от жанра (эссе или ответы на вопросы) и версии модели. Если спросят: *"Our results agree with Guo et al. on HC3, but differ from Herbold et al. on essays — stylistic signals depend on genre and model version."*

### Что именно обучали
| Модель | Что учится |
|---|---|
| TF-IDF + логистическая регрессия | По одному весу на каждое слово или пару слов (100 тысяч признаков): «голос» за человека или за ChatGPT |
| Стилометрия | Веса 18 признаков стиля (лог. регрессия) или 300 деревьев решений (случайный лес) |
| Нейросеть на PyTorch | Векторы слов (64 числа на слово, 30 тысяч слов) и два слоя сверху. 6 эпох, оптимизатор Adam |
| GPT-2 | **Ничего не обучали**, подбирали только порог perplexity (около 16) |

---

## 5. Что сделано и что дальше

| Неделя | Что | Статус |
|---|---|:---:|
| 3 | Тема, датасет, анализ данных | ✅ |
| 4 | Очистка, сбалансированный набор, простые модели | ✅ |
| 5 | Мидтерм: обзор литературы, протокол очистки, 6 детекторов, стилометрия, проверка на новых темах | ✅ |
| 7 | Дообучение RoBERTa: полное и LoRA (в Google Colab с видеокартой) | ⏳ |
| 9 | Полная проверка, разбор ошибок, проверка без «меток» датасета, ускорение модели | ⏳ |
| 10 | Отчёт в LaTeX, презентация, демо-страница | ⏳ |

---

## 6. Вероятные вопросы и ответы

### Новые вопросы (литература, данные, стилометрия)

**Which literature did you use?**
> I followed the taxonomy of the survey by Wu et al. in Computational Linguistics, 2025. I covered watermarking, zero-shot methods like DetectGPT and Binoculars, trained classifiers like the RoBERTa detector from HC3 and Ghostbuster, and stylometry. I also used papers on robustness — M4, RAID, Xu et al. — and on bias, Liang et al.

*Какую литературу использовали?* Обзор Wu 2025 и четыре семейства методов плюс работы про устойчивость и предвзятость.

**Where does HC3 come from?**
> Human answers come from five public datasets — ELI5 from Reddit, FiQA, WikiQA, MedDialog and Wikipedia. The authors collected ChatGPT answers to the same questions through the web interface in early 2023.

*Откуда HC3?* Пять открытых датасетов плюс ответы ChatGPT на те же вопросы.

**How did you clean the data?**
> Five logged steps: remove empty answers, server error messages, very short answers and duplicates, and normalise formatting with the same rules for both classes. Then one human and one ChatGPT answer per question.

*Как чистили?* Пять шагов, каждый записан. Одинаковые правила для обоих классов.

**Why do you use TPR at 1% FPR?**
> Because a false accusation is worse than a missed AI text. This metric shows how many AI texts we catch if only 1% of humans are wrongly flagged. Binoculars and RAID report it.

*Зачем TPR@1%FPR?* Ложное обвинение хуже пропуска.

**What did the stylometric analysis show?**
> ChatGPT uses many more connectives and longer sentences; humans have a richer vocabulary and write more in the first person. Burstiness shows no difference overall, but a clear one when I compare texts of the same length.

*Что показала стилометрия?* Связки, богатство словаря, «I», а «ритм» виден только при одинаковой длине.

**Why is the stylometric detector weaker if it is explainable?**
> Eighteen features capture only part of the style, while TF-IDF uses one hundred thousand. But every decision of the stylometric model can be explained, which matters ethically. It also loses less on three of five unseen domains.

*Почему стилометрия слабее?* 18 признаков против 100 тысяч, зато решения объяснимы.

**What is the difference between your work and HC3 paper?**
> HC3 reported RoBERTa with 99% F1 in-domain. I audited the data, found formatting artifacts, cleaned them, added stylometric statistics with length control, a low false-positive metric and leave-one-domain-out tests.

*Чем отличается от статьи HC3?* Проверка данных, очистка, контроль длины, метрика ложных обвинений, новые темы.

### Базовые вопросы

### Про проект

**Why did you choose this topic?**
> Detecting AI text is an important problem for universities and online platforms right now, and it is directly connected to the course topics: LLMs, transformers, perplexity, fine-tuning and LoRA.

*Почему эта тема?* Актуальна для университетов и платформ и прямо связана с темами курса.

**Why HC3?**
> It is a public, widely cited dataset where human and ChatGPT answers are given to the same questions, in five domains. This lets me compare detectors fairly and test generalisation across domains.

*Почему HC3?* Открытый, известный, ответы людей и ChatGPT на одни и те же вопросы, 5 тем.

**What is your main finding so far?**
> The dataset has formatting artifacts: human answers have spaces before punctuation. A simple rule without any AI reaches about 92% accuracy. After normalisation the rule drops to about 33%. So without cleaning, high scores can be misleading.

*Главная находка?* Ловушка с пробелами: правило без ИИ даёт 92%, после очистки около 33%.

**Why did TF-IDF give the same score on raw and cleaned text?**
> TF-IDF splits text into words and punctuation, so it does not see spaces. The artifact is dangerous for simple rules and for transformer tokenizers, which do encode spaces. That is why I clean the data before fine-tuning RoBERTa.

*Почему TF-IDF не изменился после очистки?* TF-IDF не видит пробелы, а трансформер видит, поэтому очистка нужна именно для него.

**Your accuracy is almost 99%. Isn't the problem solved?**
> Only in-domain. When I train on four domains and test on the fifth, F1 drops noticeably. Real detectors meet new topics all the time, so generalisation is the real challenge, and that is my RQ2.

*Почти 99% — задача решена?* Только на знакомых темах. На новой теме качество падает, это и есть главный вопрос.

**Why balance the data?**
> Reddit had three human answers per ChatGPT answer and 80% of all data. Without balancing the model would favour the human class. I keep one human and one ChatGPT answer per question.

*Зачем балансировать?* Иначе модель привыкнет отвечать «человек».

**Why split by question?**
> Both answers to the same question are always in the same part, so the model cannot memorise topics from training and use them in testing.

*Зачем разбиение по вопросам?* Чтобы модель не запоминала темы.

**Why F1 and not only accuracy?**
> F1 combines precision and recall for the ChatGPT class. Our data is balanced, so accuracy is also meaningful, but F1 is the standard metric in detection papers and is robust if classes become imbalanced.

*Почему F1?* Стандарт в статьях, учитывает и ложные обвинения, и пропуски.

**What if your hypothesis is not confirmed?**
> That is also a valid scientific result. If RoBERTa does not generalise better, it means that pre-training alone does not solve domain shift, and I will analyse which errors remain.

*А если гипотеза не подтвердится?* Это тоже результат, разберу, какие ошибки остались.

**What are the ethical risks?**
> False positives can lead to unfair accusations of students, and detectors can be biased against non-native English writers. A detector should only support human judgement.

*Этические риски?* Ложные обвинения и предвзятость к тем, для кого английский не родной.

**Limitations?**
> Only one generator — early ChatGPT — and only English. Human texts come from older datasets, so some signals are source-specific, like the "URL_0" placeholder in Reddit answers.

*Ограничения?* Одна модель-генератор, только английский, у человеческих текстов свои «метки» источника.

### По теории курса

**What is a transformer / self-attention?**
> A transformer is a neural network architecture where every token looks at every other token and decides how much attention to pay to it. Self-attention uses three vectors for each token: Query — what I am looking for, Key — what I contain, Value — what I pass on.

*Трансформер:* каждое слово «смотрит» на все другие слова и решает, какие важны. Query — что ищу, Key — что у меня есть, Value — что передаю.

**What is tokenization?**
> Splitting text into units — tokens — that the model works with. RoBERTa and GPT-2 use byte-level BPE: frequent words are one token, rare words are split into pieces.

*Токенизация:* разбиение текста на кусочки. Частые слова — один токен, редкие режутся на части.

**What is perplexity?**
> The exponent of the average negative log-probability of the tokens. Low perplexity means the model finds the text predictable. LLM-generated text usually has lower perplexity: in my data the median is about 10 for ChatGPT and 39 for humans.

*Perplexity:* насколько модель «удивлена» текстом. У ChatGPT около 10, у людей около 39.

**What is the difference between pre-training and fine-tuning?**
> Pre-training teaches a model general language on huge text collections. Fine-tuning adapts the pre-trained model to a specific task with a small labelled dataset.

*Предобучение и дообучение:* сначала модель учит язык вообще, потом её доучивают на конкретной задаче.

**What is LoRA and why use it?**
> Low-Rank Adaptation: the original weights are frozen, and small low-rank matrices are trained instead. It needs far fewer trainable parameters and less memory, with quality close to full fine-tuning.

*LoRA:* основная модель заморожена, учатся только маленькие добавки. Быстрее и дешевле.

**What are RLHF and DPO? Do you use them?**
> They are alignment methods that teach a generative model to answer as people prefer, using human preference data. My task is classification, not generation, so they are not applicable — but they explain why ChatGPT text sounds polite and structured, which detectors pick up.

*RLHF и DPO:* настройка генерирующей модели под предпочтения людей. В проекте не нужны, это классификация, но они объясняют «вежливый» стиль ChatGPT.

**What does the PyTorch model do?**
> Each word gets a learned 64-dimensional embedding, embeddings are averaged, and two linear layers with ReLU output the probability of ChatGPT. It is trained with binary cross-entropy loss and the Adam optimiser; the best epoch is chosen by validation ROC-AUC.

*PyTorch-модель:* слова → векторы → среднее → два слоя → вероятность ChatGPT. Функция потерь — бинарная кросс-энтропия, оптимизатор Adam.

**What is an epoch, loss, optimiser?**
> An epoch is one pass over the training data. Loss measures how wrong the model is. The optimiser — Adam — changes the weights to reduce the loss using gradients computed by backpropagation.

*Эпоха* — один проход по данным; *loss* — величина ошибки; *оптимизатор* — меняет веса, чтобы ошибка уменьшалась.

---

## 7. Советы на защите

- Говорите своими словами, не читайте со слайдов.
- Если не знаете ответа: *"That's a good question, I will check it in the next stage of the project."*
- Держите открытым ноутбук `02_preprocessing_and_baselines.ipynb`, чтобы показать код, если попросят.
- Главная мысль, которую нужно донести: **высокая точность может обманывать, если данные содержат ловушки, поэтому я проверяю детекторы на очищенных данных и на новых темах.**
