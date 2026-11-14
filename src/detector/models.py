"""Trainable detectors: length baseline, TF-IDF + logistic regression, stylometric models
and the PyTorch bag-of-embeddings network. All of them output P(ChatGPT)."""
import json
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import SEED


# ---------------------------------------------------------------- Model 0: length only
def log_word_count(texts):
    """Feature of the baseline: log(1 + number of words). Module-level so the model can be saved."""
    return np.log1p(np.array([len(str(t).split()) for t in texts], dtype=float)).reshape(-1, 1)


def length_baseline():
    return make_pipeline(FunctionTransformer(log_word_count), LogisticRegression())


# ---------------------------------------------------------------- Model 1: TF-IDF + logistic regression
TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"          # words AND punctuation marks are tokens


def tfidf_vectorizer(ngram_range=(1, 2)):
    return TfidfVectorizer(ngram_range=ngram_range, min_df=3, max_features=100_000, sublinear_tf=True,
                           lowercase=True, token_pattern=TOKEN_PATTERN)


def tfidf_logreg(C=4.0, ngram_range=(1, 2)):
    return make_pipeline(tfidf_vectorizer(ngram_range), LogisticRegression(C=C, max_iter=3000))


# ---------------------------------------------------------------- Model 2: stylometric features
def stylometry_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def stylometry_forest(min_samples_leaf=3):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=min_samples_leaf, n_jobs=-1,
                                  random_state=SEED)


# ---------------------------------------------------------------- Model 3: PyTorch bag-of-embeddings
NN_TOKEN = re.compile(r"[a-z0-9']+|[.,!?;:]")
MAX_TOKENS = 400


def nn_tokenize(text):
    return NN_TOKEN.findall(text.lower())[:MAX_TOKENS]


class BagOfEmbeddings(nn.Module):
    """fastText-style network: average of learned word vectors -> 2 dense layers -> 1 logit."""

    def __init__(self, n_vocab, dim=64, hidden=64, dropout=0.2):
        super().__init__()
        self.embedding = nn.EmbeddingBag(n_vocab + 1, dim, mode='mean')    # index 0 = unknown word
        self.head = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, ids, offsets):
        return self.head(self.embedding(ids, offsets)).squeeze(-1)


class NeuralDetector:
    """Vocabulary + network + training loop with early stopping on validation loss."""

    def __init__(self, vocab_size=30_000, dim=64, lr=3e-3, batch_size=64, max_epochs=10, patience=2):
        self.vocab_size, self.dim, self.lr = vocab_size, dim, lr
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.vocab, self.model, self.history = None, None, []

    def _encode(self, texts):
        ids, offsets = [], []
        for text in texts:
            offsets.append(len(ids))
            ids.extend([self.vocab.get(w, 0) for w in nn_tokenize(text)] or [0])
        return torch.tensor(ids, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)

    def _logits(self, texts, batch_size=512):
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                out.append(self.model(*self._encode(texts[i:i + batch_size])))
        return torch.cat(out)

    def fit(self, train_texts, train_y, val_texts, val_y, verbose=True):
        torch.manual_seed(SEED)
        counts = Counter(w for t in train_texts for w in nn_tokenize(t))
        self.vocab = {w: i + 1 for i, (w, c) in enumerate(counts.most_common(self.vocab_size)) if c >= 2}
        self.model = BagOfEmbeddings(len(self.vocab), self.dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        y_train = torch.tensor(np.asarray(train_y), dtype=torch.float32)
        y_val = torch.tensor(np.asarray(val_y), dtype=torch.float32)
        train_texts, val_texts = list(train_texts), list(val_texts)
        generator = torch.Generator().manual_seed(SEED)

        best_loss, best_state, bad_epochs = float('inf'), None, 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            order = torch.randperm(len(train_texts), generator=generator)
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                batch = order[i:i + self.batch_size]
                ids, offsets = self._encode([train_texts[j] for j in batch])
                loss = loss_fn(self.model(ids, offsets), y_train[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            val_logits = self._logits(val_texts)
            val_loss = loss_fn(val_logits, y_val).item()
            val_auc = roc_auc_score(val_y, torch.sigmoid(val_logits).numpy())
            self.history.append({'epoch': epoch, 'train_loss': total / len(order), 'val_loss': val_loss,
                                 'val_auc': float(val_auc)})
            if verbose:
                print(f'    epoch {epoch}: train loss {total / len(order):.4f}  val loss {val_loss:.4f}  '
                      f'val AUC {val_auc:.4f}')
            if val_loss < best_loss - 1e-4:
                best_loss, bad_epochs = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.best_epoch = int(np.argmin([h['val_loss'] for h in self.history]) + 1)
        return self

    def predict_proba(self, texts):
        return torch.sigmoid(self._logits(list(texts))).numpy()

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), folder / 'weights.pt')
        config = {'dim': self.dim, 'n_vocab': len(self.vocab), 'best_epoch': self.best_epoch,
                  'history': self.history}
        (folder / 'config.json').write_text(json.dumps(config, indent=2))
        (folder / 'vocab.json').write_text(json.dumps(self.vocab))

    @classmethod
    def load(cls, folder):
        config = json.loads((folder / 'config.json').read_text())
        detector = cls(dim=config['dim'])
        detector.vocab = json.loads((folder / 'vocab.json').read_text())
        detector.model = BagOfEmbeddings(config['n_vocab'], config['dim'])
        detector.model.load_state_dict(torch.load(folder / 'weights.pt', map_location='cpu'))
        detector.model.eval()
        detector.history, detector.best_epoch = config['history'], config['best_epoch']
        return detector
