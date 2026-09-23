"""
Shared surrogate model classes for the Hermes router proxy.

Both server.py (inference) and fit_surrogate.py (training) import from here
so that joblib can deserialize pipelines containing SentenceTransformerVectorizer.
"""

from __future__ import annotations

import time
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):
    """Wrap a SentenceTransformer model as an sklearn Transformer for Pipeline use.

    Caches embeddings so repeated calls don't recompute.
    Falls back gracefully if sentence_transformers is not installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        self._cache_max = 500  # max cached embeddings; LRU eviction

    def _load_model(self):
        if self._model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    f"Cannot load SentenceTransformer model '{self.model_name}': "
                    "sentence_transformers not installed"
                )
            t0 = time.time()
            self._model = SentenceTransformer(self.model_name, device="cpu")
            elapsed = time.time() - t0
            print(f"    Loaded {self.model_name} in {elapsed:.1f}s")

    def fit(self, X, y=None):
        self._load_model()
        return self

    def transform(self, X):
        self._load_model()
        if isinstance(X, (list, np.ndarray)):
            texts = list(X)
        else:
            texts = X.tolist() if hasattr(X, 'tolist') else list(X)

        # Check cache
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            embeddings = self._model.encode(
                uncached,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            for t, emb in zip(uncached, embeddings):
                self._cache[t] = emb

        return np.array([self._cache[t] for t in texts])