#!/usr/bin/env python3
"""Laya classifier — self-contained Tier 4 replacement for the LLM classifier.

Uses laya's Router (auto-detects language, dispatches to English or multilingual
checkpoint) to classify user messages into routing categories via a `choice`
question with category-specific criteria.

No external model server required. The model loads from HuggingFace cache
(first download ~400MB for multilingual, ~420MB for English) and stays resident
in memory.

API:
    clf = get_laya_classifier(cfg)
    label, confidence = clf.classify(user_message)
"""
import logging
import time
from typing import Optional

log = logging.getLogger("router-proxy.laya")

# Lazy-loaded singleton
_classifier: Optional["LayaClassifier"] = None


class LayaClassifier:
    """Wraps laya.Router for routing-category classification."""

    def __init__(self, model_name: str = "convaiinnovations/laya-multilingual",
                 device: str = "cpu",
                 confidence_threshold: float = 0.3,
                 categories: dict | None = None):
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.categories = categories or {}
        self._router = None
        self._questions = None
        self._load()

    def _load(self):
        """Load the laya Router and build the choice question from categories."""
        from laya import Router

        log.info("Loading laya Router (default=multilingual, device=%s)...", self.device)
        # Router auto-detects language and dispatches to the right checkpoint.
        # For our bilingual CN/EN environment, default to multilingual so
        # short CJK messages don't fall through to English.
        self._router = Router(
            device=self.device,
            default="multilingual",
            preload=True,
        )

        # Build choice criteria from category labels
        criteria = {}
        for cat_key, cat_cfg in self.categories.items():
            label = cat_cfg.get("label", cat_key)
            criteria[cat_key] = label

        if not criteria:
            # Fallback defaults
            criteria = {
                "chat": "Chat & Trivia",
                "code": "Code & Debug",
                "devops": "DevOps & Infra",
                "research": "Research & Analysis",
                "homeassistant": "Home Assistant",
            }

        self._questions = {
            "category": {
                "type": "choice",
                "instructions": (
                    "Route this user message to the correct category for an AI assistant routing system. "
                    "Key rules: "
                    "1) Questions asking WHAT IS something (e.g. \"what is Docker\", \"what is quantum computing\") are chat, NOT devops or research. "
                    "2) Zigbee2mqtt, ESPHome, MQTT, smart home devices, sensors are homeassistant, NOT code. "
                    "3) deploy, LXC, Proxmox, nginx, docker-compose, systemd are devops. "
                    "4) Only actual code writing/debugging/refactoring is code. "
                    "5) Greetings, thanks, and casual conversation are chat."
                ),
                "criteria": criteria,
            }
        }

        # Warm up
        self._router.predict(
            {"state": "warmup message"},
            self._questions,
        )
        log.info("Laya Router loaded and warmed up (model=%s, device=%s).",
                 self.model_name, self.device)

    def classify(self, message: str) -> tuple[str, float]:
        """Classify a user message.

        Returns (category_label, confidence).
        If laya fails or returns low confidence, returns ("chat", 0.0) as
        the safe default.
        """
        try:
            t0 = time.perf_counter()
            result = self._router.predict(
                {"state": message[:2000]},  # laya handles up to 1024 tokens
                self._questions,
            )
            dt = time.perf_counter() - t0

            answer = result["answers"]["category"]
            choice = answer.get("choice", "chat")
            # Confidence is the probability assigned to the chosen option
            probabilities = answer.get("probabilities", {})
            confidence = probabilities.get(choice, 0.0)

            log.info(
                "Laya classified: '%s' -> %s (%.3f conf, %.0fms, model=%s)",
                message[:60], choice, confidence, dt * 1000,
                result.get("routing", {}).get("model", "?"),
            )

            return choice, float(confidence)

        except Exception as exc:
            log.warning("Laya classification failed: %s", exc)
            return "chat", 0.0


def get_laya_classifier(cfg: dict) -> LayaClassifier | None:
    """Return the singleton LayaClassifier, initialized from config if needed.

    Reads config from cfg["classifier"]["laya"] section:
        enabled: bool
        model: str (HuggingFace model ID, default convaiinnovations/laya-multilingual)
        device: str (cpu or cuda, default cpu)
        confidence_threshold: float (default 0.3)
    """
    global _classifier

    laya_cfg = cfg.get("classifier", {}).get("laya", {})
    if not laya_cfg.get("enabled", False):
        return None

    if _classifier is None:
        try:
            model_name = laya_cfg.get("model", "convaiinnovations/laya-multilingual")
            device = laya_cfg.get("device", "cpu")
            threshold = laya_cfg.get("confidence_threshold", 0.3)
            categories = cfg.get("categories", {})

            _classifier = LayaClassifier(
                model_name=model_name,
                device=device,
                confidence_threshold=threshold,
                categories=categories,
            )
        except Exception as exc:
            log.error("Failed to initialize laya classifier: %s", exc)
            return None

    return _classifier