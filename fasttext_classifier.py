"""
fastText supervised classifier for the Hermes Router Proxy.

Tier-0 classifier: ultra-fast (<0.01ms) bag-of-words + n-gram classifier.
Trained on the same pretrain traces as the surrogate. Handles easy cases
instantly; uncertain cases fall through to SetFit (tier 1) or LLM (tier 3).

Usage:
    from fasttext_classifier import get_fasttext
    clf = get_fasttext(cfg)
    if clf is not None:
        label, confidence = clf.classify("hello world")

Config (router_config.yaml):
    classifier:
      fasttext:
        enabled: true
        model_path: .router/fasttext/model.ftz   # quantized model
        confidence_threshold: 0.85                 # min probability to accept
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("hermes-router")

# ── Singleton ────────────────────────────────────────────────────────────────
_model: Any = None  # fasttext model instance
_model_path: str = ""


class FastTextClassifier:
    """
    fastText supervised text classifier.

    Loads a pre-trained quantized fastText model and provides sub-millisecond
    classification. The model is trained on the same pretrain traces as the
    surrogate classifier.
    """

    def __init__(self, cfg: dict):
        ft_cfg = cfg.get("classifier", {}).get("fasttext", {})
        self.enabled = ft_cfg.get("enabled", False)
        self.confidence_threshold = ft_cfg.get("confidence_threshold", 0.85)

        model_path = ft_cfg.get("model_path", ".router/fasttext/model.ftz")
        p = Path(model_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / model_path
        self.model_path = str(p)

        self._model: Any = None
        self._labels: list[str] = []

        if self.enabled and p.exists():
            self._load()
        elif self.enabled:
            log.warning("fastText enabled but model not found at %s — disabling", p)
            self.enabled = False

    def _load(self) -> None:
        """Load the fastText model and label list."""
        try:
            import fasttext

            t0 = time.time()
            self._model = fasttext.load_model(self.model_path)

            # Load labels from the model itself
            raw_labels = self._model.get_labels()
            self._labels = [lbl.replace("__label__", "") for lbl in raw_labels]

            log.info(
                "fastText model loaded: %s (%d labels, %.1fs)",
                self.model_path,
                len(self._labels),
                time.time() - t0,
            )
        except Exception as exc:
            log.warning("Failed to load fastText model from %s: %s", self.model_path, exc)
            self.enabled = False
            self._model = None

    def classify(self, text: str) -> tuple[str, float]:
        """
        Classify text into a category.

        Returns (category_name, confidence) where confidence is a
        temperature-scaled probability (0.0 to 1.0).
        Uses temperature=5 to soften fastText's overconfident softmax.
        Returns ("", 0.0) on any error.
        """
        if not self.enabled or self._model is None:
            return "", 0.0

        try:
            # Get all labels with probabilities
            labels, probs = self._model.predict(text, k=-1)
            label = labels[0].replace("__label__", "")

            # Apply temperature scaling to soften overconfident predictions
            import numpy as np
            temp = 5.0
            log_probs = np.log(np.array(probs) + 1e-10) / temp
            scaled = np.exp(log_probs - log_probs.max())
            scaled = scaled / scaled.sum()
            confidence = float(scaled[0])
            return label, confidence
        except Exception as exc:
            log.warning("fastText predict error: %s", exc)
            return "", 0.0

    @property
    def is_available(self) -> bool:
        return self.enabled and self._model is not None


# ── Singleton instance ───────────────────────────────────────────────────────
_instance: FastTextClassifier | None = None


def get_fasttext(cfg: dict) -> FastTextClassifier | None:
    """Return the singleton FastTextClassifier, or None if disabled."""
    global _instance
    if _instance is None:
        _instance = FastTextClassifier(cfg)
    if not _instance.is_available:
        return None
    return _instance