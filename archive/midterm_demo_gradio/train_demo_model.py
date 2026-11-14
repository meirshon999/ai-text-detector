"""Trains the demo detector (same TF-IDF + logistic regression as in the notebooks) and saves it.

Run once:  python demo/train_demo_model.py
Creates:   demo/model.joblib   (vectorizer + classifier)
           demo/examples.json  (example texts from the TEST set for the demo buttons)
"""
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
pairs = __import__('pandas').read_parquet(HERE.parents[1] / 'data' / 'interim' / 'hc3_pairs.parquet')
train = pairs[pairs.split == 'train']
test = pairs[pairs.split == 'test']

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True,
                             token_pattern=r"(?u)\b\w+\b|[^\w\s]")
model = LogisticRegression(C=4, max_iter=2000)
model.fit(vectorizer.fit_transform(train.text_clean), train.y)

probability = model.predict_proba(vectorizer.transform(test.text_clean))[:, 1]
print(f'Test F1 (must be 0.987): {f1_score(test.y, (probability >= 0.5).astype(int)):.3f}')
joblib.dump({'vectorizer': vectorizer, 'model': model}, HERE / 'model.joblib')

# Example texts for the demo: test-set answers (never seen in training) of medium length, correctly classified.
test = test.assign(p=probability, n=test.text_clean.str.split().str.len())
good = test[(test.n.between(60, 140)) & (((test.y == 1) & (test.p > 0.9)) | ((test.y == 0) & (test.p < 0.1)))]
examples = []
for label in ['human', 'chatgpt']:
    for topic in ['reddit_eli5', 'finance', 'medicine']:
        row = good[(good.label == label) & (good.domain == topic)].head(1)
        if len(row):
            examples.append({'label': label, 'topic': topic, 'text': row.text_clean.iloc[0]})
(HERE / 'examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding='utf-8')
print('Saved model and', len(examples), 'examples')
