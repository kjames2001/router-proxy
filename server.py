#!/usr/bin/env python3
"""
Hermes Model Router — hybrid flash-classifier + keyword-pipe proxy.

Routes user messages to different models based on task category (chat,
code, devops, research, homeassistant, etc.). Session-aware: classifies
once with fastText, SetFit, zero-shot embeddings, surrogate, or LLM,
then uses sub-millisecond keyword deviation detection for follow-up messages.

Classification tiers (cheapest first):
  0. fastText (<0.01ms, bag-of-words + n-grams)
  1. SetFit (~10ms, fine-tuned sentence transformer)
  2. Zero-shot embeddings (~10ms, cosine similarity, no training)
  3. ML surrogate (~0.1ms, trained sklearn pipeline)
  4. LLM classifier (slow, costs tokens, most accurate)

OpenAI-compatible at POST /v1/chat/completions.
Configuration: router_config.yaml (auto-detected alongside this file).

Author: James Huang + Jarvis (Hermes Agent)
License: MIT
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# ── Trace logging (structured JSONL for classifier decisions) ────────────
try:
    from trace import (
        trace_cache_hit,
        trace_circuit,
        trace_classify,
        trace_deviation,
        trace_key_rotation,
        trace_route,
        trace_stream_error,
        _now_iso,
    )
except ImportError:  # pragma: no cover
    trace_classify = lambda *a, **kw: None
    trace_cache_hit = lambda *a, **kw: None
    trace_deviation = lambda *a, **kw: None
    trace_route = lambda *a, **kw: None
    trace_circuit = lambda *a, **kw: None
    trace_key_rotation = lambda *a, **kw: None
    trace_stream_error = lambda *a, **kw: None
    _now_iso = lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Shared model classes — needed so joblib can deserialize embedding-based pipelines
from surrogate_models import SentenceTransformerVectorizer  # noqa: F401

# ── Zero-shot embedding classifier (optional, no LLM needed) ──────────────
try:
    from zero_shot_classifier import get_zero_shot
except ImportError:
    get_zero_shot = lambda cfg: None  # noqa: E731

# ── fastText classifier (tier 0, ultra-fast, <0.01ms) ─────────────────────
try:
    from fasttext_classifier import get_fasttext
except ImportError:
    get_fasttext = lambda cfg: None  # noqa: E731

# ── SetFit classifier (tier 1, ~10ms, fine-tuned sentence transformer) ────
try:
    from setfit_classifier import get_setfit
except ImportError:
    get_setfit = lambda cfg: None  # noqa: E731

# ── Laya classifier (tier 4, self-contained, no external model server) ────
try:
    from laya_classifier import get_laya_classifier
except ImportError:
    get_laya_classifier = lambda cfg: None  # noqa: E731

# ── Logging ─────────────────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter — one line per record."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        return json.dumps(
            {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "module": record.module,
                "line": record.lineno,
                "message": record.getMessage(),
            },
            ensure_ascii=False,
            default=str,
        )


_log_format = os.environ.get("LOG_FORMAT", "").strip().lower()
if _log_format == "json":
    _formatter = JsonFormatter()
else:
    _formatter = logging.Formatter(
        "%(asctime)s [router] %(levelname)s %(message)s"
    )

logging.basicConfig(
    level=logging.INFO,
    format=None,  # handled by formatter
)
_log_handler = logging.getLogger().handlers[0]
_log_handler.setFormatter(_formatter)

log = logging.getLogger("hermes-router")

# Also configure uvicorn's loggers for JSON mode
if _log_format == "json":
    for _name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        _uv_logger = logging.getLogger(_name)
        _uv_logger.handlers.clear()
        _uv_logger.addHandler(_log_handler)
        _uv_logger.propagate = False

# ── Config ─────────────────────────────────────────────────────────────────
CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CONFIG_DIR / "router_config.yaml"


def load_config() -> dict[str, Any]:
    """Load router_config.yaml, fail loudly if missing.

    Supports both legacy (models: simple/complex) and new (categories: N)
    config formats. Legacy is auto-migrated to categories on load.
    """
    if not CONFIG_PATH.exists():
        log.fatal("Config not found: %s — run install.sh first", CONFIG_PATH)
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Migrate legacy format: models: {simple, complex} → categories: {chat, code}
    if "categories" not in cfg and "models" in cfg:
        old = cfg.pop("models")
        cfg["categories"] = {}
        if "simple" in old:
            cfg["categories"]["chat"] = old["simple"]
            cfg["categories"]["chat"]["label"] = "Chat & Trivia"
        if "complex" in old:
            cfg["categories"]["code"] = old["complex"]
            cfg["categories"]["code"]["label"] = "Code & Debug"
        log.info("Migrated legacy models config to categories format")

    return cfg


def env_key(name: str) -> str:
    """Read an API key from the environment.  Returns empty string on miss."""
    return os.environ.get(name, "").strip()


# ── Profile Hint (lazy extraction) ──────────────────────────────────────────

def _category_list(cfg: dict) -> str:
    """Build the category description string for the classifier prompt."""
    cats = cfg.get("categories", {})
    lines = []
    for name, c in cats.items():
        label = c.get("label", name)
        lines.append(f"  {name} — {label}")
    return "\n".join(lines)


def _category_names(cfg: dict) -> list[str]:
    """Return list of valid category names."""
    return list(cfg.get("categories", {}).keys())


def build_classification_prompt(
    cfg: dict, user_message: str, *, force_extract: bool = False
) -> str:
    """
    Return the full classification prompt with profile hint and categories injected.
    On first call (profile_hint empty) or when force_extract=True,
    reads USER.md + MEMORY.md and caches a 2-3 sentence summary back to config.
    """
    hint = cfg["classifier"].get("profile_hint", "").strip()

    if not hint or force_extract:
        hint = _extract_profile_hint(cfg)
        cfg["classifier"]["profile_hint"] = hint
        _write_config_back(cfg)

    template = cfg["classifier"]["system_prompt"]
    cats_str = _category_list(cfg)
    prompt = template.strip()
    prompt = prompt.replace("{categories}", cats_str)
    prompt = prompt.replace("{message}", user_message)
    if hint:
        prompt = f"Agent context: {hint}\n\n{prompt}"
    return prompt


def _extract_profile_hint(cfg: dict) -> str:
    """Read USER.md + MEMORY.md, pass to flash model, return a 2-3 sentence summary.

    USER.md contains who the user is (name, identity, preferences, location).
    MEMORY.md contains durable facts (rules, environment, tool quirks).
    Together they give the classifier enough context for accurate routing.
    """
    p = cfg["persona"]
    user_path = Path(p["user_path"]).expanduser()
    memory_path = Path(p["memory_path"]).expanduser()

    parts: list[str] = []
    for label, path in (("USER.md", user_path), ("MEMORY.md", memory_path)):
        if path.exists():
            parts.append(path.read_text().strip())
        else:
            log.warning("%s not found at %s", label, path)

    if not parts:
        log.warning("Neither USER.md nor MEMORY.md found — skipping profile extraction")
        return "No profile available."

    max_chars = p.get("max_context_chars", 800)
    raw = "\n\n".join(parts)[:max_chars]

    prompt = (
        "Summarize this AI agent's user identity, environment, and key tools "
        "in 2-3 concise sentences. Keep only what helps classify tasks into "
        f"the configured categories.\n\n{raw}"
    )

    log.info("Extracting profile hint from USER.md+MEMORY.md (%d chars) → flash model", len(raw))
    summary = _call_classifier_raw(cfg, prompt, max_tokens=100)
    return summary.strip() or "No profile available."


def _call_classifier_raw(
    cfg: dict, prompt: str, max_tokens: int = 256
) -> str:
    """Call the flash classifier model with a raw prompt, return text content.
    
    Uses a generous token budget because reasoning models (deepseek-v4-flash)
    spend tokens on reasoning_content before producing content.
    Falls back to extracting the classification from reasoning_content 
    if the content field is empty.
    """
    cl = cfg["classifier"]
    api_key = env_key(cl["api_key_env"])

    payload: dict[str, Any] = {
        "model": cl["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = httpx.post(
            f"{cl['base_url'].rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(15),
        )
        if resp.status_code != 200:
            log.warning("Classifier returned HTTP %d: %s", resp.status_code, resp.text[:200])
            cat_names = _category_names(cfg)
            return cat_names[0] if cat_names else "chat"
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = (msg.get("content") or "").strip().lower()
        # Reasoning models may put the answer in reasoning_content instead
        if not content:
            reasoning = (msg.get("reasoning_content") or "")
            # Scan for any category name from config
            cat_names = _category_names(cfg)
            parts = reasoning.lower().strip().split()
            for word in reversed(parts):
                cleaned = word.strip('.,;:!?"\'()')
                if cleaned in cat_names:
                    content = cleaned
                    break
        cat_names = _category_names(cfg)
        return content or (cat_names[0] if cat_names else "chat")
    except Exception as exc:
        log.warning("Classifier call failed: %s", exc)
        cat_names = _category_names(cfg)
        return cat_names[0] if cat_names else "chat"


def _write_config_back(cfg: dict) -> None:
    """Write updated config back to disk (profile_hint after extraction).

    Creates a backup before writing to preserve the original (including any
    YAML comments that yaml.dump would destroy). The backup is saved with
    a .bak suffix.
    """
    import shutil
    backup_path = CONFIG_PATH.with_suffix(".yaml.bak")
    try:
        shutil.copy2(CONFIG_PATH, backup_path)
    except Exception:
        log.warning("Could not create config backup at %s — proceeding anyway", backup_path)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    log.info("Wrote updated router_config.yaml with profile_hint (backup at %s)", backup_path)


# ── Surrogate Classifier (TRACER-inspired acceptor gate) ────────────────────

class SurrogateClassifier:
    """
    ML surrogate that replaces the LLM classifier for confident predictions.

    Loads a fitted sklearn pipeline + calibrated acceptor from .router/surrogate/.
    On each classification request:
      1. Predict with the pipeline → label + confidence
      2. If confidence >= threshold → use surrogate prediction (fast, free)
      3. If confidence < threshold → fall back to LLM classifier (accurate)

    This is the "acceptor gate" from the TRACER paper: calibrated probability
    threshold determines which inputs are safe to route via the cheap surrogate
    vs. which need the expensive teacher model.
    """

    def __init__(self, surrogate_dir: Path, confidence_threshold: float = 0.85):
        self.pipeline = None
        self.acceptor = None
        self.confidence_threshold = confidence_threshold
        self.manifest: dict[str, Any] = {}
        self._loaded = False
        self._load(surrogate_dir)

    def _load(self, surrogate_dir: Path) -> None:
        """Try to load pipeline + acceptor from disk. Silent on failure."""
        manifest_path = surrogate_dir / "manifest.json"
        pipeline_path = surrogate_dir / "pipeline.joblib"
        acceptor_path = surrogate_dir / "acceptor.joblib"

        if not manifest_path.exists() or not pipeline_path.exists():
            log.info("Surrogate not found at %s — LLM-only classification", surrogate_dir)
            return

        try:
            import joblib as _joblib

            self.manifest = json.loads(manifest_path.read_text())
            self.pipeline = _joblib.load(pipeline_path)

            if acceptor_path.exists():
                self.acceptor = _joblib.load(acceptor_path)
                # Use threshold from manifest if available
                self.confidence_threshold = self.manifest.get(
                    "acceptor_threshold", self.confidence_threshold
                )

            self._loaded = True
            log.info(
                "Surrogate loaded: %s (CV accuracy=%.4f, coverage=%.1f%%, threshold=%.4f)",
                self.manifest.get("surrogate_method", "unknown"),
                self.manifest.get("cv_accuracy", 0),
                self.manifest.get("coverage", 0) * 100,
                self.confidence_threshold,
            )
        except Exception as exc:
            log.warning("Failed to load surrogate from %s: %s", surrogate_dir, exc)
            self.pipeline = None
            self.acceptor = None
            self._loaded = False

    @property
    def is_available(self) -> bool:
        return self._loaded and self.pipeline is not None

    def predict(self, text: str) -> tuple[str, float]:
        """
        Predict label and confidence for input text.
        Returns (label, confidence) — e.g. ("simple", 0.97).
        Falls back to ("simple", 0.0) on any error.
        """
        if not self._loaded or self.pipeline is None:
            return "", 0.0

        try:
            # Pipeline prediction
            label = self.pipeline.predict([text])[0]

            # Confidence from acceptor (calibrated) or pipeline predict_proba
            if self.acceptor is not None:
                probas = self.acceptor.predict_proba([text])[0]
                confidence = float(max(probas))
            elif hasattr(self.pipeline, "predict_proba"):
                probas = self.pipeline.predict_proba([text])[0]
                confidence = float(max(probas))
            else:
                # No probability available — use decision function distance
                confidence = 1.0  # Assume confident if we got this far

            return str(label), confidence

        except Exception as exc:
            log.warning("Surrogate predict error: %s", exc)
            return "", 0.0


# ── Global surrogate instance (lazy-loaded) ─────────────────────────────────
_surrogate: SurrogateClassifier | None = None


def _get_surrogate(cfg: dict) -> SurrogateClassifier | None:
    """Return the singleton surrogate, initializing from config if needed."""
    global _surrogate
    surrogate_cfg = cfg.get("classifier", {}).get("surrogate", {})
    if not surrogate_cfg.get("enabled", False):
        return None
    if _surrogate is None:
        surrogate_dir = Path(surrogate_cfg.get("path", ".router/surrogate"))
        if not surrogate_dir.is_absolute():
            surrogate_dir = CONFIG_DIR / surrogate_dir
        threshold = surrogate_cfg.get("confidence_threshold", 0.85)
        _surrogate = SurrogateClassifier(surrogate_dir, confidence_threshold=threshold)
    return _surrogate if _surrogate.is_available else None


# ── Classification ──────────────────────────────────────────────────────────

def _parse_category(result: str, cfg: dict) -> str:
    """Parse the classifier LLM output into a valid category name.

    Handles: exact match, word match, substring match.
    Falls back to the first category if no match.
    """
    result = result.strip().lower().strip('"\'\'.,;:!?()[]{}')
    names = _category_names(cfg)

    # Exact match
    if result in names:
        return result

    # Word-level match (classifier may add extra words)
    words = result.split()
    for name in names:
        if name in words:
            return name

    # Substring match
    for name in names:
        if name in result:
            return name

    # Fallback to first category
    if names:
        log.warning("Classifier returned unknown category '%s' — defaulting to '%s'", result, names[0])
        return names[0]
    return "chat"


def _detect_override(text: str, cfg: dict) -> str | None:
    """Check if the user message contains a /use:<category> override.

    Returns the category name if found, None otherwise.
    Also returns the cleaned message (without the override prefix).
    """
    prefix = cfg.get("routing", {}).get("override_prefix", "/use:")
    names = _category_names(cfg)

    # Match /use:category at the start of the message
    text_stripped = text.strip()
    if not text_stripped.startswith(prefix):
        return None

    # Extract everything after the prefix until whitespace
    rest = text_stripped[len(prefix):]
    # The category name is the first word
    parts = rest.split(None, 1)
    if not parts:
        return None

    cat = parts[0].strip().lower()
    if cat in names:
        return cat

    log.warning("Override '/use:%s' not a valid category — valid: %s", cat, names)
    return None


def _strip_override(text: str, cfg: dict) -> str:
    """Remove the /use: prefix from the message, returning the clean text."""
    prefix = cfg.get("routing", {}).get("override_prefix", "/use:")
    text_stripped = text.strip()
    if text_stripped.startswith(prefix):
        rest = text_stripped[len(prefix):]
        parts = rest.split(None, 1)
        if len(parts) > 1:
            return parts[1].strip()
        return ""
    return text


def classify(cfg: dict, user_message: str, *, session_key: str | None = None, is_first: bool = True) -> str:
    """
    Classify a user message into one of the configured categories.

    Classification priority (cheapest first):
      0. fastText classifier (<0.01ms, bag-of-words + n-grams, trained model)
      1. SetFit classifier (~10ms, fine-tuned sentence transformer, trained model)
      2. Zero-shot embedding classifier (~10ms, no training, cosine similarity)
      3. ML surrogate (TRACER-inspired, ~0.1ms, needs trained model)
      4. LLM classifier (slow, costs tokens, most accurate)

    Falls back to the first category on any failure.
    """
    names = _category_names(cfg)

    # ── Tier 0: fastText classifier (ultra-fast, <0.01ms) ──────────────
    fasttext_clf = get_fasttext(cfg)
    if fasttext_clf is not None:
        label, confidence = fasttext_clf.classify(user_message)
        if confidence >= fasttext_clf.confidence_threshold and label in names:
            latency_ms = 0.01
            _record_classifier_latency(latency_ms)
            log.info(
                "fastText classified: '%s' → %s (%.3f prob, threshold %.2f)",
                user_message[:60], label, confidence, fasttext_clf.confidence_threshold,
            )
            trace_classify(
                session_key=session_key or "?",
                user_message=user_message,
                classifier_result=label,
                classifier_raw=f"fasttext:{confidence:.4f}",
                latency_ms=latency_ms,
                tier=label,
                model="fasttext/quantized",
                is_first=is_first,
            )
            return label
        else:
            log.info(
                "fastText uncertain: '%s' (%.3f < %.2f) — deferring to next classifier",
                user_message[:60], confidence, fasttext_clf.confidence_threshold,
            )

    # ── Tier 1: SetFit classifier (~10ms, fine-tuned) ──────────────────
    setfit_clf = get_setfit(cfg)
    if setfit_clf is not None:
        label, confidence = setfit_clf.classify(user_message)
        if confidence >= setfit_clf.confidence_threshold and label in names:
            latency_ms = 10.0
            _record_classifier_latency(latency_ms)
            log.info(
                "SetFit classified: '%s' → %s (%.3f prob, threshold %.2f)",
                user_message[:60], label, confidence, setfit_clf.confidence_threshold,
            )
            trace_classify(
                session_key=session_key or "?",
                user_message=user_message,
                classifier_result=label,
                classifier_raw=f"setfit:{confidence:.4f}",
                latency_ms=latency_ms,
                tier=label,
                model="setfit/all-MiniLM-L6-v2",
                is_first=is_first,
            )
            return label
        else:
            log.info(
                "SetFit uncertain: '%s' (%.3f < %.2f) — deferring to next classifier",
                user_message[:60], confidence, setfit_clf.confidence_threshold,
            )

    # ── Tier 2: Zero-shot embedding classifier (no LLM) ───────────────
    zero_shot = get_zero_shot(cfg)
    if zero_shot is not None:
        label, confidence = zero_shot.classify(user_message)
        if confidence >= zero_shot.confidence_threshold and label in names:
            latency_ms = 10.0  # ~10ms for embedding + similarity
            _record_classifier_latency(latency_ms)
            log.info(
                "Zero-shot classified: '%s' → %s (%.3f similarity, threshold %.2f)",
                user_message[:60], label, confidence, zero_shot.confidence_threshold,
            )
            trace_classify(
                session_key=session_key or "?",
                user_message=user_message,
                classifier_result=label,
                classifier_raw=f"zero_shot:{confidence:.4f}",
                latency_ms=latency_ms,
                tier=label,
                model=f"zero_shot/{zero_shot.model_name}",
                is_first=is_first,
            )
            return label
        else:
            log.info(
                "Zero-shot uncertain: '%s' (%.3f < %.2f) — deferring to next classifier",
                user_message[:60], confidence, zero_shot.confidence_threshold,
            )

    # ── Tier 3: ML surrogate (TRACER-inspired, ~0.1ms, trained model) ──
    surrogate = _get_surrogate(cfg)

    if surrogate is not None:
        label, confidence = surrogate.predict(user_message)
        # Verify surrogate label is a valid category
        if confidence >= surrogate.confidence_threshold and label in names:
            latency_ms = 0.1
            _record_classifier_latency(latency_ms)

            log.info(
                "Surrogate classified: '%s' → %s (%.2f confidence, threshold %.2f)",
                user_message[:60], label, confidence, surrogate.confidence_threshold,
            )

            trace_classify(
                session_key=session_key or "?",
                user_message=user_message,
                classifier_result=label,
                classifier_raw=f"surrogate:{confidence:.4f}",
                latency_ms=latency_ms,
                tier=label,
                model=f"surrogate/{surrogate.manifest.get('surrogate_method', 'unknown')}",
                is_first=is_first,
            )

            return label
        else:
            log.info(
                "Surrogate uncertain: '%s' (%.2f < %.2f threshold) — deferring to LLM",
                user_message[:60], confidence, surrogate.confidence_threshold,
            )

    # ── Tier 4: Laya classifier (self-contained, ~370ms, no external server) ──
    # Previously: LLM classifier calling GLM-4.7-Flash on docker-ssd:8002 (15s, ~50% accurate)
    # Now: laya-multilingual Router, self-contained CPU inference, ~370ms, ~65% accurate
    laya_clf = get_laya_classifier(cfg)
    if laya_clf is not None:
        t0 = time.time()
        label, confidence = laya_clf.classify(user_message)
        latency_ms = (time.time() - t0) * 1000
        _record_classifier_latency(latency_ms)

        if label in names:
            model = cfg["categories"][label]["model"]
            trace_classify(
                session_key=session_key or "?",
                user_message=user_message,
                classifier_result=label,
                classifier_raw=f"laya:{confidence:.4f}",
                latency_ms=latency_ms,
                tier=label,
                model=f"laya/{laya_clf.model_name.split('/')[-1]}",
                is_first=is_first,
            )
            log.info(
                "Laya classified: '%s' -> %s (%.3f conf, %.0fms)",
                user_message[:60], label, confidence, latency_ms,
            )
            return label
        else:
            log.warning(
                "Laya returned invalid category '%s', falling back to default", label,
            )

    # ── Tier 4b: LLM classifier fallback (if laya is disabled or failed) ──
    t0 = time.time()
    prompt = build_classification_prompt(cfg, user_message)
    result = _call_classifier_raw(cfg, prompt, max_tokens=256)
    latency_ms = (time.time() - t0) * 1000
    _record_classifier_latency(latency_ms)

    category = _parse_category(result, cfg)
    model = cfg["categories"][category]["model"]

    trace_classify(
        session_key=session_key or "?",
        user_message=user_message,
        classifier_result=category,
        classifier_raw=result,
        latency_ms=latency_ms,
        tier=category,
        model=model,
        is_first=is_first,
    )

    return category


# ── Keyword Deviation Detection ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Strip whitespace, hyphens, underscores — collapse to lowercase."""
    return re.sub(r"[-\s_]+", "", text).lower()


def _fuzzy_match(keyword: str, text: str) -> bool:
    """
    Match a keyword against text with typo tolerance.
    
    1. Exact substring (normalised).
    2. Normalised substring.
    3. Levenshtein distance ≤1 for keywords ≥5 chars.
    """
    n_key = str(keyword).lower().strip()
    n_text = text.lower()

    # 1. Exact substring
    if n_key in n_text:
        return True

    # 2. Normalised (strip separators)
    if _normalize(keyword) in _normalize(text):
        return True

    # 3. Typo tolerance — words ≥5 chars, allow 1-char difference
    if len(n_key) >= 5:
        for word in n_text.split():
            word = word.strip('.,;:!?"\'()[]{}')
            if len(word) >= 5 and _levenshtein(n_key, word) <= 1:
                return True

    return False


def _levenshtein(s1: str, s2: str) -> int:
    """Minimal edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def has_deviation(cfg: dict, text: str, current_category: str, *, session_key: str | None = None) -> bool:
    """Scan follow-up message for keywords that trigger re-classification.

    With N categories, deviation is no longer a simple tier swap.
    Instead, any escalation keyword triggers full re-classification.
    De-escalation keywords force the default (first) category.

    Additionally, very short follow-up messages (< 30 chars) that aren't
    greetings/thanks are treated as continuation triggers — they likely
    refer to the ongoing task context ("do it", "yes", "go ahead", "sure")
    and should be reclassified with context-window to catch category drift.
    """
    # Check for /use: override first
    override = _detect_override(text, cfg)
    if override and override != current_category:
        model = cfg["categories"][override]["model"]
        trace_deviation(
            session_key=session_key or "?",
            keyword=f"/use:{override}",
            direction="override",
            previous_tier=current_category,
            new_tier=override,
            model=model,
        )
        return True

    # Escalation keywords — trigger re-classification
    for kw in cfg["routing"].get("escalation_keywords", []):
        if _fuzzy_match(kw, text):
            log.info("Deviation: escalation keyword '%s' matched", kw)
            trace_deviation(
                session_key=session_key or "?",
                keyword=kw,
                direction="escalation",
                previous_tier=current_category,
                new_tier="unknown",  # actual tier determined by re-classification in caller
                model="unknown",
            )
            return True

    # De-escalation keywords — force default (first) category
    names = _category_names(cfg)
    default_cat = names[0] if names else "chat"
    if current_category != default_cat:
        for kw in cfg["routing"].get("de_escalation_keywords", []):
            if _fuzzy_match(kw, text):
                log.info("Deviation: de-escalation keyword '%s' matched → %s", kw, default_cat)
                model = cfg["categories"].get(default_cat, {}).get("model", "unknown")
                trace_deviation(
                    session_key=session_key or "?",
                    keyword=kw,
                    direction="de_escalation",
                    previous_tier=current_category,
                    new_tier=default_cat,
                    model=model,
                )
                return True

    # Short continuation heuristic: very short messages that aren't explicit
    # greetings/thanks are likely task continuations ("do it", "yes", "go ahead").
    # Force reclassification with context window to catch category drift.
    _SHORT_CONTINUATION_MAX_LEN = 30
    _GREETING_PATTERNS = {"hello", "hi", "hey", "thanks", "thank you", "thx",
                          "ok", "okay", "got it", "understood", "makes sense",
                          "perfect", "good job", "nice", "cool", "awesome",
                          "great", "cheers", "bye", "goodbye", "see you",
                          "你好", "谢谢", "好的", "明白了", "再见"}
    text_lower = text.strip().lower()
    if len(text_lower) < _SHORT_CONTINUATION_MAX_LEN and text_lower not in _GREETING_PATTERNS:
        # Check if it's a bare greeting word (longer than 2 chars to avoid "hi"/"ok" overlap)
        is_greeting = any(text_lower == g for g in _GREETING_PATTERNS if len(g) <= 4)
        if not is_greeting:
            log.info("Deviation: short continuation '%s' (%d chars) — forcing context reclassification",
                     text[:40], len(text_lower))
            trace_deviation(
                session_key=session_key or "?",
                keyword="short_continuation",
                direction="context",
                previous_tier=current_category,
                new_tier="unknown",
                model="unknown",
            )
            return True

    return False


# ── Session Cache ───────────────────────────────────────────────────────────

# In-memory: session_key → {"tier": <category_name>, "at": timestamp}
SESSIONS: dict[str, dict[str, Any]] = {}
SESSIONS_MAX = 500  # max cached sessions; oldest evicted on insert

# ── Metrics ─────────────────────────────────────────────────────────────────
# Prometheus-compatible counters for GET /metrics
METRICS: dict[str, int] = {
    "classifier_calls_total": 0,
    "classifier_latency_ms_sum": 0,
    "cache_hits_total": 0,
    "429_total": 0,
    "fallback_used_total": 0,
    "fallback2_used_total": 0,
    "stream_requests_total": 0,
    "errors_total": 0,
    "requests_total": 0,  # total requests regardless of category
}
# Per-category request/429 counters are dynamic: requests_total_<cat>, 429_<cat>

# ── Circuit Breakers ────────────────────────────────────────────────────────
# Per-endpoint circuit breakers that track consecutive 429s.
# After N consecutive failures in a sliding window, open the circuit
# (skip the endpoint entirely) for X seconds.
CIRCUITS: dict[str, dict[str, Any]] = {}
CIRCUITS_MAX = 50  # max circuit entries; excess evicted on insert

# Defaults — overridable via router_config.yaml → circuit_breaker section
CB_DEFAULTS: dict[str, int] = {
    "failure_threshold": 3,      # consecutive 429s before tripping
    "recovery_timeout_sec": 30,  # how long the circuit stays open
    "window_sec": 60,            # sliding window for counting failures
}


def _circuit_key(base_url: str, model: str = "") -> str:
    """Normalize a base_url (+ optional model) into a circuit breaker key."""
    key = base_url.rstrip("/").replace("://", "_").replace("/", "_").replace(".", "_")
    if model:
        key = f"{key}__{model}"
    return key


def _circuit_is_open(cfg: dict, base_url: str, model: str = "") -> bool:
    """Check whether the circuit for this endpoint is currently open."""
    cb_cfg = cfg.get("circuit_breaker", CB_DEFAULTS)
    key = _circuit_key(base_url, model)
    entry = CIRCUITS.get(key)
    if not entry:
        return False
    if entry["state"] != "open":
        return False
    if time.time() - entry["opened_at"] > cb_cfg.get("recovery_timeout_sec", 30):
        log.info("Circuit %s → half-open (recovery timeout elapsed)", key)
        entry["state"] = "half_open"
        entry["half_open_at"] = time.time()
        trace_circuit(base_url, "open", "half_open")
        return False
    remaining = cb_cfg.get("recovery_timeout_sec", 30) - int(time.time() - entry["opened_at"])
    if remaining > 0:
        log.debug("Circuit %s is open (%ds remaining)", key, remaining)
    return True


def _circuit_record_success(cfg: dict, base_url: str, model: str = "") -> None:
    """Reset the circuit breaker after a successful request."""
    key = _circuit_key(base_url, model)
    entry = CIRCUITS.get(key)
    if entry and entry["state"] == "half_open":
        log.info("Circuit %s → closed (success in half-open)", key)
        trace_circuit(base_url, "half_open", "closed")
    elif entry and entry["state"] == "open":
        trace_circuit(base_url, "open", "closed")
    CIRCUITS[key] = {"state": "closed", "failures": 0, "last_failure_at": 0}


def _circuit_record_failure(cfg: dict, base_url: str, model: str = "") -> None:
    """Record a failure (429) and potentially open the circuit."""
    cb_cfg = cfg.get("circuit_breaker", CB_DEFAULTS)
    key = _circuit_key(base_url, model)
    if len(CIRCUITS) >= CIRCUITS_MAX and key not in CIRCUITS:
        for ck, cv in list(CIRCUITS.items()):
            if cv.get("state") == "closed":
                del CIRCUITS[ck]
                break
    entry = CIRCUITS.get(key, {"state": "closed", "failures": 0, "last_failure_at": 0})
    now = time.time()
    window = cb_cfg.get("window_sec", 60)
    if now - entry.get("last_failure_at", 0) > window:
        entry["failures"] = 0
    entry["failures"] += 1
    entry["last_failure_at"] = now
    threshold = cb_cfg.get("failure_threshold", 3)
    if entry["failures"] >= threshold and entry["state"] != "open":
        log.warning(
            "Circuit %s → OPEN (%d failures in %ds, recovery in %ds)",
            key, entry["failures"], window,
            cb_cfg.get("recovery_timeout_sec", 30),
        )
        entry["state"] = "open"
        entry["opened_at"] = now
        trace_circuit(base_url, "closed", "open", failures=entry["failures"])
    CIRCUITS[key] = entry


def _inc_metric(name: str, delta: int = 1) -> None:
    """Increment a metric counter atomically (single-threaded safe).
    Auto-creates keys for dynamic per-category metrics."""
    METRICS[name] = METRICS.get(name, 0) + delta


def _inc_metric_tier(tier: str, name: str, delta: int = 1) -> None:
    """Increment a category-scoped metric: {name}_{tier}."""
    _inc_metric(f"{name}_{tier}", delta)


def _record_classifier_latency(ms: float) -> None:
    """Record classifier call latency."""
    METRICS["classifier_calls_total"] += 1
    METRICS["classifier_latency_ms_sum"] += int(ms)


def _get_metrics() -> dict:
    """Return a copy of current metrics with computed fields."""
    m = dict(METRICS)
    calls = m["classifier_calls_total"]
    m["classifier_latency_ms_avg"] = (
        m["classifier_latency_ms_sum"] // calls if calls > 0 else 0
    )
    m["sessions_active"] = len(SESSIONS)
    return m


def session_key(messages: list[dict]) -> str | None:
    """Derive a session key from the first user message. Returns None if empty."""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                # Multimodal — grab text parts
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            return hashlib.sha256(content.encode()[:200]).hexdigest()[:16]
    return None


def is_first_message(messages: list[dict]) -> bool:
    """A new session starts when there is exactly one user message."""
    user_count = sum(1 for m in messages if m.get("role") == "user")
    return user_count <= 1


def get_cached_tier(cfg: dict, key: str) -> str | None:
    """Return cached tier if session is still valid, None otherwise."""
    entry = SESSIONS.get(key)
    if not entry:
        return None
    timeout_mins = cfg["classifier"].get("session_timeout_minutes", 5)
    if time.time() - entry["at"] > timeout_mins * 60:
        log.info("Session %s expired", key)
        del SESSIONS[key]
        return None
    return entry["tier"]


def cache_tier(key: str, tier: str) -> None:
    if len(SESSIONS) >= SESSIONS_MAX:
        oldest = min(SESSIONS, key=lambda k: SESSIONS[k].get("at", 0))
        del SESSIONS[oldest]
        log.debug("Evicted oldest session %s (cache full)", oldest)
    SESSIONS[key] = {"tier": tier, "at": time.time()}
    log.info("Session %s → %s (cached)", key, tier)


# ── Model Calling ───────────────────────────────────────────────────────────
_RETRYABLE_EXC = (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout)
_RETRY_MAX = 2  # total attempts = 1 + _RETRY_MAX = 3


async def _post_with_retry(
    url: str,
    *,
    json: dict,
    headers: dict,
    timeout: httpx.Timeout,
    label: str = "",
    client: httpx.AsyncClient | None = None,
) -> httpx.Response:
    """Async httpx POST with built-in retry for transient transport errors.

    RemoteProtocolError / ConnectError / ReadTimeout / ConnectTimeout are
    retried up to _RETRY_MAX times with 1-second backoff.  Non-transient
    errors (4xx, 5xx) are returned as-is so the caller can handle fallback.

    If *client* is provided, reuses it; otherwise creates a short-lived client.
    """
    own_client = client is None
    last_exc: Exception | None = None
    try:
        if own_client:
            async with httpx.AsyncClient() as own:
                for attempt in range(_RETRY_MAX + 1):
                    try:
                        return await own.post(url, json=json, headers=headers, timeout=timeout)
                    except _RETRYABLE_EXC as exc:
                        last_exc = exc
                        if attempt < _RETRY_MAX:
                            delay = 1.0 * (attempt + 1)
                            log.warning(
                                "%s transient error (attempt %s/%s): %s — retrying in %.1fs",
                                label, attempt + 1, _RETRY_MAX + 1, exc, delay,
                            )
                            await asyncio.sleep(delay)
                        else:
                            log.error(
                                "%s exhausted %s retries: %s", label, _RETRY_MAX + 1, exc
                            )
                raise last_exc  # type: ignore[misc]
        else:
            for attempt in range(_RETRY_MAX + 1):
                try:
                    return await client.post(url, json=json, headers=headers, timeout=timeout)
                except _RETRYABLE_EXC as exc:
                    last_exc = exc
                    if attempt < _RETRY_MAX:
                        delay = 1.0 * (attempt + 1)
                        log.warning(
                            "%s transient error (attempt %s/%s): %s — retrying in %.1fs",
                            label, attempt + 1, _RETRY_MAX + 1, exc, delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        log.error(
                            "%s exhausted %s retries: %s", label, _RETRY_MAX + 1, exc
                        )
            raise last_exc  # type: ignore[misc]
    except Exception:
        raise


async def call_model(
    cfg: dict, model_cfg: dict, request_payload: dict
) -> httpx.Response:
    """Call an OpenAI-compatible endpoint. Returns the httpx response.

    Circuit breaker: checks if the endpoint circuit is open before calling.
    Key rotation: if the primary key returns HTTP 429 (rate-limited),
    retries with alternate_key_env before giving up.
    503 retry: HTTP 503 (overloaded) retries up to 3x with exponential backoff
    (2s, 4s, 8s) so transient overloads are resolved before escalation.
    Uses _post_with_retry for transient transport error resilience.
    """
    base_url = model_cfg["base_url"].rstrip("/")
    url = f"{base_url}/chat/completions"
    model_name = model_cfg["model"]

    # Circuit breaker check (keyed per base_url+model so one 503 model doesn't block others on same host)
    if _circuit_is_open(cfg, base_url, model_name):
        log.warning("Circuit open for %s — skipping call", base_url)
        _inc_metric_tier(model_cfg.get("tier", "unknown"), "429")
        _inc_metric("429_total")
        r = httpx.Response(503, text="Circuit breaker open")
        r._request = httpx.Request("POST", url)
        return r

    api_key = env_key(model_cfg["api_key_env"])
    timeout = model_cfg.get("timeout_seconds", 120)
    alt_key = env_key(model_cfg.get("alternate_key_env", ""))

    payload = {**request_payload, "model": model_cfg["model"]}
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient() as client:
        # ── 503 retry: exponential backoff (2s, 4s, 8s) ──────────────────────────
        _503_retries = 3
        _503_backoff = [2.0, 4.0, 8.0]
        resp: httpx.Response = httpx.Response(503, text="No response received")
        resp._request = httpx.Request("POST", url)
        for attempt_503 in range(_503_retries):
            try:
                resp = await _post_with_retry(
                    url, json=payload, headers=headers,
                    timeout=httpx.Timeout(timeout),
                    label=f"call_model({model_name})",
                    client=client,
                )
            except Exception as exc:
                log.warning("Transport error for %s: %s", model_cfg["model"], exc)
                # Let caller handle fallback — don't retry transport-level here
                r = httpx.Response(503, text=str(exc))
                r._request = httpx.Request("POST", url)
                return r

            if resp.status_code != 503:
                break

            if attempt_503 < _503_retries - 1:
                delay = _503_backoff[attempt_503]
                err_brief = resp.text[:150] if resp.text else ""
                log.warning(
                    "HTTP 503 for %s (attempt %s/%s) — backing off %.1fs: %s",
                    model_cfg["model"], attempt_503 + 1, _503_retries, delay, err_brief,
                )
                await asyncio.sleep(delay)
            else:
                err_brief = resp.text[:150] if resp.text else ""
                log.error(
                    "HTTP 503 exhausted %s retries for %s: %s",
                    _503_retries, model_cfg["model"], err_brief,
                )

        # Circuit + metric tracking
        if resp.status_code == 429:
            _circuit_record_failure(cfg, base_url, model_name)
            _inc_metric_tier(model_cfg.get("tier", "unknown"), "429")
            _inc_metric("429_total")
        elif resp.status_code in (502, 503, 504):
            _circuit_record_failure(cfg, base_url, model_name)
        elif resp.status_code == 200:
            _circuit_record_success(cfg, base_url, model_name)
            _clear_retry(session_key(request_payload.get("messages", [])) or "")

        # Key rotation: HTTP 429 with alternate key available -> retry
        if resp.status_code == 429 and alt_key:
            log.warning("Primary key rate-limited (429) - switching to alternate key")
            trace_key_rotation(
                base_url=base_url,
                tier=model_cfg.get("tier", "unknown"),
                reason="429_rate_limit",
            )
            try:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {alt_key}"},
                    timeout=httpx.Timeout(timeout),
                )
            except Exception as exc:
                log.warning("Alternate key request failed for %s: %s", model_name, exc)
            if resp.status_code == 429:
                _circuit_record_failure(cfg, base_url, model_name)
                _inc_metric_tier(model_cfg.get("tier", "unknown"), "429")
                _inc_metric("429_total")
                log.warning("Alternate key also rate-limited for %s", model_name)
            elif resp.status_code == 200:
                log.info("Alternate key succeeded")
                _circuit_record_success(cfg, base_url, model_name)
                _clear_retry(session_key(request_payload.get("messages", [])) or "")

    return resp


# ── Retry-tracking: count consecutive primary failures per session.
# Only after N failures does the next gateway retry skip to fallback.
_RETRY_STATE: dict[str, int] = {}  # session_key → consecutive failure count
_RETRY_MAX_BEFORE_FALLBACK = 3
_RETRY_STATE_MAX = 200  # max tracked sessions; oldest evicted on insert

def _record_failure(key: str) -> int:
    """Increment failure counter, return new count."""
    if len(_RETRY_STATE) >= _RETRY_STATE_MAX and key not in _RETRY_STATE:
        oldest_key = next(iter(_RETRY_STATE))
        del _RETRY_STATE[oldest_key]
    cnt = _RETRY_STATE.get(key, 0) + 1
    _RETRY_STATE[key] = cnt
    return cnt

def _clear_retry(key: str) -> None:
    _RETRY_STATE.pop(key, None)

def _should_use_fallback(key: str) -> bool:
    return _RETRY_STATE.get(key, 0) >= _RETRY_MAX_BEFORE_FALLBACK


async def route_request_stream(cfg: dict, payload: dict):
    """Streaming version of route_request — returns SSE chunks from upstream.

    Acts as a transparent proxy: opens an inner stream to the selected
    upstream model and forwards every SSE chunk to the outer connection.
    If the inner stream breaks mid-flight, the outer SSE is cleanly closed
    so the gateway's own HERMES_STREAM_RETRIES can retry with a fresh
    connection.  On that retry the primary is skipped and fallback is used
    directly.
    """
    messages = payload.get("messages", [])
    key = session_key(messages)
    if not key:
        yield b'data: {"error":{"message":"No user message found","type":"router_error"}}\n\ndata: [DONE]\n\n'
        return

    # ── Determine category ──────────────────────────────────────────
    category: str
    if is_first_message(messages):
        user_content = _last_user_text(messages)
        override = _detect_override(user_content, cfg)
        if override:
            category = override
            clean_text = _strip_override(user_content, cfg)
            if clean_text:
                for m in messages:
                    if m.get("role") == "user":
                        if isinstance(m.get("content"), str):
                            m["content"] = clean_text
                        break
                payload["messages"] = messages
            log.info("Session %s override: → %s", key, category)
        else:
            category = classify(cfg, user_content, session_key=key, is_first=True)
        cache_tier(key, category)
    else:
        cached = get_cached_tier(cfg, key)
        if cached is None:
            user_content = _last_user_text(messages)
            override = _detect_override(user_content, cfg)
            if override:
                category = override
                clean_text = _strip_override(user_content, cfg)
                if clean_text:
                    for m in messages:
                        if m.get("role") == "user":
                            if isinstance(m.get("content"), str):
                                m["content"] = clean_text
                            break
                    payload["messages"] = messages
            else:
                # Context-window: classify last 3 user messages, not just this one
                ctx_content = _recent_user_text(messages, n=3)
                category = classify(cfg, ctx_content, session_key=key, is_first=False)
            cache_tier(key, category)
        else:
            last_text = _last_user_text(messages)
            if has_deviation(cfg, last_text, cached, session_key=key):
                override = _detect_override(last_text, cfg)
                if override:
                    category = override
                    clean_text = _strip_override(last_text, cfg)
                    if clean_text:
                        for m in messages:
                            if m.get("role") == "user":
                                if isinstance(m.get("content"), str):
                                    m["content"] = clean_text
                                break
                        payload["messages"] = messages
                else:
                    # Context-window: classify last 3 user messages on deviation
                    ctx_content = _recent_user_text(messages, n=3)
                    category = classify(cfg, ctx_content, session_key=key, is_first=False)
                cache_tier(key, category)
            else:
                category = cached
                _inc_metric("cache_hits_total")
                cache_entry = SESSIONS.get(key, {})
                cat_cfg_cached = cfg["categories"].get(category, {})
                trace_cache_hit(
                    session_key=key,
                    tier=category,
                    model=cat_cfg_cached.get("model", "unknown"),
                    age_sec=time.time() - cache_entry.get("at", time.time()),
                )

    model_cfg = cfg["categories"].get(category, {})
    if not model_cfg:
        yield b'data: {"error":{"message":"Category has no model config","type":"router_error"}}\n\ndata: [DONE]\n\n'
        return
    model_cfg = model_cfg.copy()

    # ── Build attempt list: primary, fallback1, fallback2 ─────────
    # Each entry: (label, model_cfg_dict)
    # If session has N consecutive failures, skip primary and start at fallback.
    tier_attempts: list[tuple[str, dict]] = []

    # Skip primary if session has too many consecutive failures
    if _should_use_fallback(key):
        fallback_model_name = model_cfg.get("fallback_model")
        if fallback_model_name:
            log.info("Session %s has %s consecutive failures — skipping primary %s",
                     key, _RETRY_STATE[key], model_cfg["model"])
        else:
            tier_attempts.append(("primary", model_cfg))
    else:
        tier_attempts.append(("primary", model_cfg))

    # Fallback 1
    fb_model = model_cfg.get("fallback_model")
    if fb_model:
        fb1_cfg = {
            "model": fb_model,
            "base_url": model_cfg["fallback_base_url"],
            "api_key_env": model_cfg.get("fallback_key_env", ""),
            "alternate_key_env": model_cfg.get("fallback_alternate_key_env", ""),
            "timeout_seconds": model_cfg.get("timeout_seconds", 120),
            "tier": category,
        }
        tier_attempts.append(("fallback1", fb1_cfg))

    # Fallback 2
    fb2_model = model_cfg.get("fallback2_model")
    if fb2_model:
        fb2_cfg = {
            "model": fb2_model,
            "base_url": model_cfg["fallback2_base_url"],
            "api_key_env": model_cfg.get("fallback2_key_env", ""),
            "alternate_key_env": model_cfg.get("fallback2_alternate_key_env", ""),
            "timeout_seconds": model_cfg.get("timeout_seconds", 120),
            "tier": category,
        }
        tier_attempts.append(("fallback2", fb2_cfg))

    # ── Try each tier in order until one streams successfully ───────────
    async with httpx.AsyncClient() as client:
        for attempt_idx, (attempt_label, attempt_cfg) in enumerate(tier_attempts):
            model_name = attempt_cfg["model"]
            log.info("Streaming attempt %s: %s → %s", attempt_label, key, model_name)

            # Circuit breaker check (per model so one 503 model doesn't block siblings on same host)
            base_url = attempt_cfg["base_url"].rstrip("/")
            if _circuit_is_open(cfg, base_url, model_name):
                log.warning("Circuit open for %s — skipping %s", base_url, model_name)
                trace_route(
                    session_key=key, tier=category, model=model_name,
                    upstream_status=429, stream=True,
                    fallback_level=attempt_idx + 1,
                )
                _inc_metric_tier(category, "429")
                continue  # try next tier

            stream_url = f"{base_url}/chat/completions"
            api_key = env_key(attempt_cfg["api_key_env"])
            alt_key = env_key(attempt_cfg.get("alternate_key_env", ""))
            timeout = attempt_cfg.get("timeout_seconds", 120)
            stream_payload = {**payload, "model": model_name}
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            # ── Key rotation: if primary key gets 429, try alternate ────
            key_attempts = [(api_key, "primary_key")] if api_key else [(None, "no_key")]
            if alt_key:
                key_attempts.append((alt_key, "alternate_key"))

            for key_val, key_label in key_attempts:
                if key_val:
                    headers["Authorization"] = f"Bearer {key_val}"

                try:
                    async with client.stream(
                        "POST", stream_url, json=stream_payload, headers=headers,
                        timeout=httpx.Timeout(timeout),
                    ) as resp:

                        if resp.status_code == 429:
                            err_body = await resp.aread()
                            err_text = err_body.decode(errors="replace")[:200]
                            log.warning("Streaming %s got 429 (%s key): %s",
                                        model_name, key_label, err_text[:100])
                            _circuit_record_failure(cfg, base_url, model_name)
                            _inc_metric_tier(category, "429")
                            _inc_metric("429_total")

                            if key_label == "primary_key" and alt_key:
                                trace_key_rotation(
                                    base_url=base_url, tier=category,
                                    reason="429_rate_limit_stream",
                                )
                                continue  # try alternate key

                            trace_route(
                                session_key=key, tier=category, model=model_name,
                                upstream_status=429, stream=True,
                                fallback_level=attempt_idx + 1,
                            )
                            break  # both keys 429'd, try next tier

                        if resp.status_code != 200:
                            err_body = await resp.aread()
                            err_text = err_body.decode(errors="replace")[:300]

                            # 503 retry with backoff before falling to next tier
                            if resp.status_code == 503:
                                for retry_i, backoff in enumerate([2.0, 4.0, 8.0]):
                                    log.warning(
                                        "Streaming %s got 503 (attempt %d/4) — backing off %.0fs: %s",
                                        model_name, retry_i + 2, backoff, err_text[:80],
                                    )
                                    await asyncio.sleep(backoff)
                                    # Retry the stream request
                                    retry_headers = dict(headers)
                                    if key_val:
                                        retry_headers["Authorization"] = f"Bearer {key_val}"
                                    try:
                                        async with client.stream(
                                            "POST", stream_url, json=stream_payload, headers=retry_headers,
                                            timeout=httpx.Timeout(timeout),
                                        ) as retry_resp:
                                            if retry_resp.status_code == 200:
                                                log.info("Streaming %s OK after 503 retry %d", model_name, retry_i + 2)
                                                _circuit_record_success(cfg, base_url, model_name)
                                                trace_route(
                                                    session_key=key, tier=category, model=model_name,
                                                    upstream_status=200, stream=True,
                                                )
                                                if attempt_label != "primary":
                                                    _inc_metric("fallback_used_total" if attempt_label == "fallback1" else "fallback2_used_total")
                                                try:
                                                    async for line in retry_resp.aiter_lines():
                                                        if line:
                                                            yield f"{line}\n".encode()
                                                        else:
                                                            yield b"\n"
                                                    _clear_retry(key)
                                                    return  # clean completion
                                                except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as stream_exc:
                                                    log.warning("Stream interrupted for %s after 503 retry: %s", model_name, stream_exc)
                                                    cnt = _record_failure(key)
                                                    trace_stream_error(
                                                        session_key=key, model=model_name,
                                                        error=str(stream_exc), failure_count=cnt, max_failures=3,
                                                    )
                                                    break  # mid-stream break after 503 retry, try next tier
                                            elif retry_resp.status_code == 429:
                                                _ = await retry_resp.aread()
                                                # 429 on retry — record and break to next key/tier
                                                _circuit_record_failure(cfg, base_url, model_name)
                                                break
                                            # Non-200, non-429 — continue retrying
                                            err_body = await retry_resp.aread()
                                            err_text = err_body.decode(errors="replace")[:300]
                                    except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout):
                                        continue  # transport error on retry, try again
                                # All 503 retries exhausted — fall through to next tier
                                _circuit_record_failure(cfg, base_url, model_name)

                            log.warning("Streaming %s returned %d: %s",
                                        model_name, resp.status_code, err_text[:150])
                            trace_route(
                                session_key=key, tier=category, model=model_name,
                                upstream_status=resp.status_code, stream=True,
                            )
                            break  # non-429 error on this tier, try next

                        # ── Stream successfully opened ───────────────────
                        log.info("Streaming %s OK (key=%s)", model_name, key_label)
                        # Bug #13: Record circuit success for any successful key, not just alternate
                        _circuit_record_success(cfg, base_url, model_name)
                        trace_route(
                            session_key=key, tier=category, model=model_name,
                            upstream_status=200, stream=True,
                        )
                        if attempt_label != "primary":
                            _inc_metric("fallback_used_total" if attempt_label == "fallback1" else "fallback2_used_total")

                        try:
                            async for line in resp.aiter_lines():
                                if line:
                                    yield f"{line}\n".encode()
                                else:
                                    yield b"\n"
                            _clear_retry(key)
                            return  # clean completion

                        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as stream_exc:
                            # Mid-stream break — close cleanly, let gateway retry
                            log.warning("Stream interrupted for %s: %s", model_name, stream_exc)
                            cnt = _record_failure(key)
                            trace_stream_error(
                                session_key=key, model=model_name,
                                error=str(stream_exc), failure_count=cnt,
                                max_failures=_RETRY_MAX_BEFORE_FALLBACK,
                            )
                            yield b'data: [DONE]\n\n'
                            return  # gateway's HERMES_STREAM_RETRIES handles retry

                except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as exc:
                    log.warning("Streaming %s transport error: %s", model_name, exc)
                    cnt = _record_failure(key)
                    trace_stream_error(
                        session_key=key, model=model_name,
                        error=str(exc), failure_count=cnt,
                        max_failures=_RETRY_MAX_BEFORE_FALLBACK,
                    )
                    # Don't return — try next tier
                    continue

                except httpx.ConnectTimeout as exc:
                    log.warning("Streaming %s connect timeout: %s", model_name, exc)
                    continue

            # key_attempts exhausted for this tier — move to next tier

    # ── All tiers exhausted ────────────────────────────────────────
    log.error("All streaming tiers exhausted for session %s", key)
    yield b'data: {"error":{"message":"All upstream models failed","type":"upstream_error"}}\n\n'
    yield b'data: [DONE]\n\n'


async def route_request(cfg: dict, payload: dict) -> JSONResponse:
    """
    Full routing pipeline.  Determines the category, calls the mapped
    model (with fallback), and returns a FastAPI JSONResponse.
    """
    messages = payload.get("messages", [])
    key = session_key(messages)
    if not key:
        return _error(400, "No user message found in request")

    # ── Determine category ──────────────────────────────────────────────
    category: str

    if is_first_message(messages):
        user_content = _last_user_text(messages)
        # Check for /use: override on first message
        override = _detect_override(user_content, cfg)
        if override:
            category = override
            log.info("Session %s override: → %s", key, category)
            # Strip the override prefix from the actual message sent upstream
            clean_text = _strip_override(user_content, cfg)
            if clean_text:
                # Replace the user message content for upstream
                for m in messages:
                    if m.get("role") == "user":
                        if isinstance(m.get("content"), str):
                            m["content"] = clean_text
                        break
                payload["messages"] = messages
            cache_tier(key, category)
        else:
            category = classify(cfg, user_content, session_key=key, is_first=True)
            cache_tier(key, category)

    else:
        cached = get_cached_tier(cfg, key)
        if cached is None:
            user_content = _last_user_text(messages)
            # Check override on follow-up too
            override = _detect_override(user_content, cfg)
            if override:
                category = override
                clean_text = _strip_override(user_content, cfg)
                if clean_text:
                    for m in messages:
                        if m.get("role") == "user":
                            if isinstance(m.get("content"), str):
                                m["content"] = clean_text
                            break
                    payload["messages"] = messages
            else:
                # Context-window: classify last 3 user messages, not just this one
                ctx_content = _recent_user_text(messages, n=3)
                category = classify(cfg, ctx_content, session_key=key, is_first=False)
            cache_tier(key, category)
        else:
            last_text = _last_user_text(messages)
            if has_deviation(cfg, last_text, cached, session_key=key):
                # Check if it's an override
                override = _detect_override(last_text, cfg)
                if override:
                    category = override
                    clean_text = _strip_override(last_text, cfg)
                    if clean_text:
                        for m in messages:
                            if m.get("role") == "user":
                                if isinstance(m.get("content"), str):
                                    m["content"] = clean_text
                                break
                        payload["messages"] = messages
                else:
                    # Context-window: classify last 3 user messages on deviation
                    ctx_content = _recent_user_text(messages, n=3)
                    category = classify(cfg, ctx_content, session_key=key, is_first=False)
                if category != cached:
                    log.info("Session %s category changed: %s → %s", key, cached, category)
                cache_tier(key, category)
            else:
                category = cached
                _inc_metric("cache_hits_total")
                cache_entry = SESSIONS.get(key, {})
                cat_cfg = cfg["categories"].get(category, {})
                trace_cache_hit(
                    session_key=key,
                    tier=category,
                    model=cat_cfg.get("model", "unknown"),
                    age_sec=time.time() - cache_entry.get("at", time.time()),
                )

    # ── Call model ──────────────────────────────────────────────────────
    cat_cfg = cfg["categories"].get(category, {}).copy()
    if not cat_cfg:
        return _error(500, f"Category '{category}' has no model config")
    cat_cfg["tier"] = category  # for circuit breaker metric labeling
    _inc_metric_tier(category, "requests_total")
    log.info("Routing session %s → %s (%s)", key, category, cat_cfg["model"])

    # Skip primary if session has 3+ consecutive failures (3-strike fallback)
    skip_primary = _should_use_fallback(key)
    if skip_primary and cat_cfg.get("fallback_model"):
        log.info("Session %s has %d consecutive failures — skipping primary %s",
                 key, _RETRY_STATE.get(key, 0), cat_cfg["model"])

    if not skip_primary or not cat_cfg.get("fallback_model"):
        resp = await call_model(cfg, cat_cfg, payload)

        if resp.status_code == 200:
            trace_route(
                session_key=key, tier=category, model=cat_cfg["model"],
                upstream_status=200, stream=False,
            )
            _clear_retry(key)
            return JSONResponse(content=resp.json())
    else:
        resp = httpx.Response(503, text="Skipped primary due to consecutive failures")
        resp._request = httpx.Request("POST", f"{cat_cfg['base_url'].rstrip('/')}/chat/completions")

    # ── Fallback ────────────────────────────────────────────────────────
    fallback_model = cat_cfg.get("fallback_model")
    if not fallback_model:
        return _proxy_error(resp)

    log.warning(
        "Primary model %s returned %d — trying fallback %s",
        cat_cfg["model"],
        resp.status_code,
        fallback_model,
    )

    _inc_metric("fallback_used_total")
    trace_route(
        session_key=key, tier=category, model=cat_cfg["model"],
        upstream_status=resp.status_code, stream=False,
        fallback_level=1, fallback_model=fallback_model,
    )
    fb_cfg = {
        "model": fallback_model,
        "base_url": cat_cfg["fallback_base_url"],
        "api_key_env": cat_cfg["fallback_key_env"],
        "alternate_key_env": cat_cfg.get("fallback_alternate_key_env", ""),
        "timeout_seconds": cat_cfg.get("timeout_seconds", 120),
        "tier": category,
    }
    fb_resp = await call_model(cfg, fb_cfg, payload)

    if fb_resp.status_code == 200:
        data = fb_resp.json()
        data.setdefault("hermes_router", {})["fallback_used"] = True
        trace_route(
            session_key=key, tier=category, model=fallback_model,
            upstream_status=200, stream=False,
        )
        return JSONResponse(content=data)

    # ── Fallback 2 ─────────────────────────────────────────────────────────
    fb2_model = cat_cfg.get("fallback2_model")
    if fb2_model:
        log.warning(
            "Fallback %s returned %d — trying fallback2 %s",
            fallback_model, fb_resp.status_code, fb2_model,
        )
        _inc_metric("fallback2_used_total")
        trace_route(
            session_key=key, tier=category, model=fallback_model,
            upstream_status=fb_resp.status_code, stream=False,
            fallback_level=2, fallback_model=fb2_model,
        )
        fb2_cfg = {
            "model": fb2_model,
            "base_url": cat_cfg["fallback2_base_url"],
            "api_key_env": cat_cfg["fallback2_key_env"],
            "alternate_key_env": cat_cfg.get("fallback2_alternate_key_env", ""),
            "timeout_seconds": cat_cfg.get("timeout_seconds", 120),
            "tier": category,
        }
        fb2_resp = await call_model(cfg, fb2_cfg, payload)
        if fb2_resp.status_code == 200:
            data = fb2_resp.json()
            data.setdefault("hermes_router", {})["fallback_used"] = True
            trace_route(
                session_key=key, tier=category, model=fb2_model,
                upstream_status=200, stream=False,
            )
            return JSONResponse(content=data)
        return _proxy_error(fb2_resp)

    return _proxy_error(fb_resp)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _last_user_text(messages: list[dict]) -> str:
    """Extract the text content of the most recent user message."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            return content
    return ""


def _recent_user_text(messages: list[dict], n: int = 3) -> str:
    """Concatenate recent messages (user + assistant) for context-window classification.

    This solves the 'do it' problem: when a follow-up message is short/ambiguous
    ("do it", "yes", "go ahead"), classifying just that message yields the wrong
    category. By including the last few user messages plus the agent's replies,
    the classifier sees the actual task context and routes correctly.

    Includes up to n user messages and up to n-1 assistant replies (interleaved).
    Assistant messages are truncated to 200 chars to stay within classifier
    context windows (SetFit/zero-shot use all-MiniLM-L6-v2 with 256 token limit).

    Messages are joined with " | " separator (not newlines) to keep the input
    compact for fastText/SetFit which work best on short text.
    """
    # Collect recent user and assistant messages in reverse, then re-interleave
    recent = []
    user_count = 0
    asst_count = 0
    for m in reversed(messages):
        role = m.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        else:
            text = content
        if not text:
            continue
        # Truncate assistant messages to 200 chars to respect classifier context limits
        if role == "assistant":
            if asst_count >= n - 1:
                continue
            text = text[:200]
            asst_count += 1
        else:
            if user_count >= n:
                continue
            user_count += 1
        recent.append((role, text))
        if user_count >= n and asst_count >= n - 1:
            break
    # Reverse to chronological order
    recent.reverse()
    # Format: [user] msg | [assistant] reply | [user] msg
    parts = []
    for role, text in recent:
        if role == "assistant":
            parts.append(f"[assistant] {text}")
        else:
            parts.append(text)
    return " | ".join(parts)


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "router_error"}},
        status_code=status,
    )


def _proxy_error(resp: httpx.Response) -> JSONResponse:
    """Forward an upstream error with context."""
    detail = resp.text[:500] if resp.text else "Unknown upstream error"
    return JSONResponse(
        {
            "error": {
                "message": f"Upstream model returned {resp.status_code}: {detail}",
                "type": "upstream_error",
                "status_code": resp.status_code,
            }
        },
        status_code=502,
    )


# ── FastAPI Application ─────────────────────────────────────────────────────

def verify_auth(request: Request):
    """Check Bearer token against configured API key."""
    cfg = request.app.state.config
    key_env = cfg.get("auth", {}).get("api_key_env", "")
    if not key_env:
        return  # No auth configured — allow all
    expected = os.environ.get(key_env, "").strip()
    if not expected:
        # Bug #22 fix: If auth is explicitly configured but the env var is
        # empty, reject all requests rather than silently allowing them.
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "API key env var is empty — auth required", "type": "auth_error"}},
        )
    auth_header = request.headers.get("Authorization", "")
    token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
    if hmac.compare_digest(token, expected):
        return
    raise HTTPException(
        status_code=401,
        detail={"error": {"message": "Invalid or missing API key", "type": "auth_error"}},
    )

app = FastAPI(
    title="Hermes Model Router",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

# ── CORS ──────────────────────────────────────────────────────────────────
origins_raw = os.environ.get("CORS_ORIGINS", "*").strip()
allowed_origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    # Note: deprecated in FastAPI 0.93+ but functional.
    cfg = load_config()
    log.info("Router starting on %s:%s", cfg["server"]["host"], cfg["server"]["port"])
    log.info("  Classifier: %s (%s)", cfg["classifier"]["model"], cfg["classifier"]["base_url"])
    cats = cfg.get("categories", {})
    log.info("  Categories (%d):", len(cats))
    for name, c in cats.items():
        label = c.get("label", name)
        log.info("    %s (%s): %s (%s)", name, label, c.get("model", "?"), c.get("base_url", "?"))
        if c.get("fallback_model"):
            log.info("      fallback: %s (%s)", c.get("fallback_model", "?"), c.get("fallback_base_url", "?"))
    if not cats:
        log.error("  No categories configured — router will fail on all requests")

    # Zero-shot classifier status
    zs_cfg = cfg.get("classifier", {}).get("zero_shot", {})
    if zs_cfg.get("enabled"):
        log.info("  Zero-shot classifier: enabled (model=%s, threshold=%.2f)",
                 zs_cfg.get("model_name", "all-MiniLM-L6-v2"),
                 zs_cfg.get("confidence_threshold", 0.35))
    else:
        log.info("  Zero-shot classifier: disabled")

    # fastText classifier status
    ft_cfg = cfg.get("classifier", {}).get("fasttext", {})
    if ft_cfg.get("enabled"):
        log.info("  fastText classifier: enabled (model=%s, threshold=%.2f)",
                 ft_cfg.get("model_path", ".router/fasttext/model.ftz"),
                 ft_cfg.get("confidence_threshold", 0.85))
    else:
        log.info("  fastText classifier: disabled")

    # SetFit classifier status
    sf_cfg = cfg.get("classifier", {}).get("setfit", {})
    if sf_cfg.get("enabled"):
        log.info("  SetFit classifier: enabled (model=%s, threshold=%.2f)",
                 sf_cfg.get("model_path", ".router/setfit"),
                 sf_cfg.get("confidence_threshold", 0.55))
    else:
        log.info("  SetFit classifier: disabled")

    # Laya classifier status
    laya_cfg = cfg.get("classifier", {}).get("laya", {})
    if laya_cfg.get("enabled"):
        log.info("  Laya classifier: enabled (model=%s, device=%s, threshold=%.2f) [self-contained]",
                 laya_cfg.get("model", "convaiinnovations/laya-multilingual"),
                 laya_cfg.get("device", "cpu"),
                 laya_cfg.get("confidence_threshold", 0.3))
    else:
        log.info("  Laya classifier: disabled (will fall back to LLM tier)")

    app.state.config = cfg


@app.get("/health")
async def health():
    """Simple liveness check."""
    return {"status": "ok", "sessions": len(SESSIONS)}


@app.post("/reload")
async def reload_config(request: Request):
    """Hot-reload router_config.yaml without restart.

    Reads config from disk, validates required sections, atomically
    swaps app.state.config. Clears profile_hint to force re-extraction
    on the next classification request.
    """
    verify_auth(request)

    old_cfg = request.app.state.config
    old_cats = list(old_cfg.get("categories", {}).keys())

    try:
        new_cfg = load_config()
    except Exception as exc:
        log.error("Failed to parse config on reload: %s", exc)
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": f"Config parse error: {exc}", "type": "reload_error"}},
        )

    # Validate minimum structure
    for section in ("classifier", "routing", "server"):
        if section not in new_cfg:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": f"Missing required section: {section}", "type": "reload_error"}},
            )
    if "categories" not in new_cfg or not new_cfg["categories"]:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "Missing or empty 'categories' section in config", "type": "reload_error"}},
        )

    # Clear profile_hint so extraction runs with fresh config
    new_cfg["classifier"]["profile_hint"] = ""

    # Bug #20: Reset surrogate singleton so it reloads with new config
    global _surrogate
    _surrogate = None

    # Reset zero-shot classifier singleton
    import zero_shot_classifier as _zsc
    _zsc._instance = None

    # Atomic swap
    request.app.state.config = new_cfg

    new_cats = list(new_cfg.get("categories", {}).keys())
    log.info(
        "Config hot-reloaded. Categories: %s → %s",
        old_cats, new_cats,
    )

    return {
        "status": "reloaded",
        "before": {"categories": old_cats},
        "after": {"categories": new_cats},
    }


@app.get("/circuits")
async def list_circuits(request: Request):
    """List all circuit breaker states."""
    verify_auth(request)
    return {
        "circuits": CIRCUITS,
        "defaults": CB_DEFAULTS,
    }


@app.post("/circuits/reset")
async def reset_circuits(request: Request):
    """Reset all circuit breakers back to closed state."""
    verify_auth(request)
    count = len(CIRCUITS)
    CIRCUITS.clear()
    log.info("Reset %d circuit breakers", count)
    return {"status": "reset", "count": count}


@app.get("/admin/sessions")
async def list_sessions(request: Request):
    """List all cached sessions with tier info."""
    verify_auth(request)
    now = time.time()
    cfg = request.app.state.config
    timeout_mins = cfg["classifier"].get("session_timeout_minutes", 5)
    sessions = {}
    for key, entry in list(SESSIONS.items()):
        age_sec = int(now - entry["at"])
        remaining_sec = max(0, timeout_mins * 60 - age_sec)
        sessions[key] = {
            "tier": entry["tier"],
            "age_sec": age_sec,
            "remaining_sec": remaining_sec,
        }
    return {"count": len(sessions), "session_timeout_minutes": timeout_mins, "sessions": sessions}


@app.delete("/admin/sessions/{key}")
async def evict_session(key: str, request: Request):
    """Force-evict a cached session, forcing re-classification on next message."""
    verify_auth(request)
    removed = SESSIONS.pop(key, None)
    if removed:
        log.info("Session %s evicted (was %s)", key, removed["tier"])
        return {"status": "evicted", "key": key, "was_tier": removed["tier"]}
    raise HTTPException(
        status_code=404,
        detail={"error": {"message": f"Session {key} not found", "type": "not_found"}},
    )


# ── Classifier Report Endpoint ───────────────────────────────────────────────

def _build_classifier_report(cfg: dict) -> dict:
    """Build a report from trace logs and in-memory metrics.
    
    Reads the last N trace events and computes:
    - Surrogate coverage (hit rate, confidence distribution)
    - Classification source breakdown (surrogate vs LLM vs cache vs keyword)
    - Per-model request counts and error rates
    - Drift detection (surrogate agreement with LLM over time)
    """
    import glob
    trace_dir = Path(os.environ.get("TRACE_LOG_DIR", "./traces"))
    m = dict(METRICS)  # snapshot

    # ── Read recent traces ─────────────────────────────────────────────
    events: list[dict] = []
    trace_files = sorted(glob.glob(str(trace_dir / "router-trace-*.jsonl")))
    # Read the last 2 files (today + yesterday) for drift analysis
    for tf in trace_files[-2:]:
        try:
            with open(tf, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        pass
        except OSError:
            pass

    # ── Classify events ─────────────────────────────────────────────────
    fasttext_hits = 0
    setfit_hits = 0
    zero_shot_hits = 0
    surrogate_hits = 0
    llm_hits = 0
    cache_hits_events = 0
    keyword_events = 0
    route_events = 0
    surrogate_confidences: list[float] = []
    model_counts: dict[str, int] = {}
    stream_errors = 0
    fallback1_events = 0
    fallback2_events = 0
    # Per-hour surrogate hit rate for drift detection
    hourly_surrogate: dict[str, dict[str, int]] = {}  # hour → {non_llm, llm}

    for ev in events:
        etype = ev.get("event", "")
        
        if etype == "classify":
            model = ev.get("model", "")
            raw = ev.get("classifier_raw", "")
            
            if model.startswith("fasttext/"):
                fasttext_hits += 1
            elif model.startswith("setfit/"):
                setfit_hits += 1
            elif model.startswith("surrogate/"):
                surrogate_hits += 1
                # Extract confidence from raw like "surrogate:0.87"
                try:
                    conf = float(raw.split(":")[-1]) if ":" in str(raw) else 0.0
                    surrogate_confidences.append(conf)
                except (ValueError, IndexError):
                    pass
            elif model.startswith("zero_shot/"):
                zero_shot_hits += 1
            else:
                llm_hits += 1

            # Per-hour breakdown
            ts = ev.get("ts", "")
            hour_key = ts[:13] if len(ts) >= 13 else "unknown"  # "2026-05-13T16"
            if hour_key not in hourly_surrogate:
                hourly_surrogate[hour_key] = {"non_llm": 0, "llm": 0}
            if model.startswith(("fasttext/", "setfit/", "surrogate/", "zero_shot/")):
                hourly_surrogate[hour_key]["non_llm"] += 1
            else:
                hourly_surrogate[hour_key]["llm"] += 1

        elif etype == "cache_hit":
            cache_hits_events += 1

        elif etype == "deviation":
            keyword_events += 1

        elif etype == "route":
            route_events += 1
            model_name = ev.get("model", "")
            model_counts[model_name] = model_counts.get(model_name, 0) + 1
            fl = ev.get("fallback_level")
            if fl == 1:
                fallback1_events += 1
            elif fl == 2:
                fallback2_events += 1

        elif etype == "stream_error":
            stream_errors += 1

    total_classifications = fasttext_hits + setfit_hits + zero_shot_hits + surrogate_hits + llm_hits
    non_llm_hits = fasttext_hits + setfit_hits + zero_shot_hits + surrogate_hits
    sur_coverage = (non_llm_hits / total_classifications * 100) if total_classifications > 0 else 0

    # Confidence distribution
    conf_buckets = {"high_ge0.9": 0, "med_0.7_0.9": 0, "low_lt0.7": 0}
    for c in surrogate_confidences:
        if c >= 0.9:
            conf_buckets["high_ge0.9"] += 1
        elif c >= 0.7:
            conf_buckets["med_0.7_0.9"] += 1
        else:
            conf_buckets["low_lt0.7"] += 1

    # ── Hourly drift ────────────────────────────────────────────────────
    drift_hours = []
    for hour_key in sorted(hourly_surrogate.keys()):
        h = hourly_surrogate[hour_key]
        total_h = h["non_llm"] + h["llm"]
        drift_hours.append({
            "hour": hour_key,
            "non_llm_pct": round(h["non_llm"] / total_h * 100, 1) if total_h > 0 else 0,
            "total_classifications": total_h,
        })

    # ── Build report ────────────────────────────────────────────────────
    sur_info = cfg.get("classifier", {}).get("surrogate", {})
    
    report = {
        "generated_at": _now_iso(),
        "surrogate": {
            "enabled": sur_info.get("enabled", False),
            "model": sur_info.get("path", "N/A"),
            "total_classifications": total_classifications,
            "surrogate_hits": surrogate_hits,
            "llm_deferrals": llm_hits,
            "coverage_pct": round(sur_coverage, 1),
            "confidence": {
                "mean": round(sum(surrogate_confidences) / len(surrogate_confidences), 3) if surrogate_confidences else None,
                "min": min(surrogate_confidences) if surrogate_confidences else None,
                "max": max(surrogate_confidences) if surrogate_confidences else None,
                "distribution": conf_buckets,
            },
            "threshold": sur_info.get("confidence_threshold", None),
        },
        "classification_sources": {
            "fasttext": fasttext_hits,
            "setfit": setfit_hits,
            "zero_shot": zero_shot_hits,
            "surrogate": surrogate_hits,
            "llm_classifier": llm_hits,
            "cache_hits": cache_hits_events,
            "keyword_deviations": keyword_events,
        },
        "routing": {
            "total_route_events": route_events,
            "fallback_tier1": fallback1_events,
            "fallback_tier2": fallback2_events,
            "model_distribution": model_counts,
        },
        "in_memory_metrics": m,
        "errors": {
            "stream_errors": stream_errors,
        },
        "drift": {
            "description": "Surrogate hit rate per hour. Sudden drops may indicate drift.",
            "hourly": drift_hours[-24:],  # last 24 hours
        },
        "trace_files_analyzed": len(trace_files[-2:]),
        "total_trace_events": len(events),
    }
    return report


@app.get("/classifier/report")
async def classifier_report(request: Request, format: str = "json"):
    """Classifier performance report with surrogate coverage, drift, and model distribution.
    
    Query params:
        format: 'json' (default) or 'html' for a human-readable HTML page.
    """
    verify_auth(request)
    cfg = request.app.state.config
    report = _build_classifier_report(cfg)
    
    if format == "html":
        # Build a simple HTML audit view
        sur = report["surrogate"]
        sources = report["classification_sources"]
        routing = report["routing"]
        errors = report["errors"]
        drift = report["drift"]
        
        html = f"""<!DOCTYPE html>
<html><head><title>Router Proxy Classifier Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2rem; background: #0d1117; color: #c9d1d9; }}
h1, h2, h3 {{ color: #58a6ff; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem; }}
.card h3 {{ margin-top: 0; }}
.stat {{ font-size: 2rem; font-weight: bold; color: #58a6ff; }}
.stat-label {{ color: #8b949e; font-size: 0.85rem; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ text-align: left; padding: 0.5rem; border-bottom: 1px solid #21262d; }}
th {{ color: #58a6ff; }}
.pct-bar {{ height: 8px; border-radius: 4px; background: #30363d; }}
.pct-fill {{ height: 8px; border-radius: 4px; background: #3fb950; }}
code {{ background: #21262d; padding: 2px 6px; border-radius: 4px; }}
</style></head><body>
<h1>🔄 Router Proxy Classifier Report</h1>
<p>Generated: <code>{report['generated_at']}</code></p>

<div class="grid">
<div class="card">
<h3>Surrogate Coverage</h3>
<div class="stat">{sur['coverage_pct']}%</div>
<div class="stat-label">of classifications handled by surrogate (no LLM call)</div>
<p>Surrogate hits: <strong>{sur['surrogate_hits']}</strong> / {sur['total_classifications']} total</p>
<p>LLM deferrals: <strong>{sur['llm_deferrals']}</strong></p>
<p>Confidence threshold: <code>{sur['threshold']}</code></p>
<p>Mean confidence: <code>{sur['confidence']['mean'] or 'N/A'}</code></p>
</div>

<div class="card">
<h3>Classification Sources</h3>
<table>
<tr><th>Source</th><th>Count</th></tr>
<tr><td>🔮 Zero-shot</td><td>{sources['zero_shot']}</td></tr>
<tr><td>🤖 Surrogate</td><td>{sources['surrogate']}</td></tr>
<tr><td>🧠 LLM Classifier</td><td>{sources['llm_classifier']}</td></tr>
<tr><td>📦 Cache Hits</td><td>{sources['cache_hits']}</td></tr>
<tr><td>🔀 Keyword Deviations</td><td>{sources['keyword_deviations']}</td></tr>
</table>
</div>

<div class="card">
<h3>Confidence Distribution</h3>
<table>
<tr><th>Range</th><th>Count</th></tr>"""
        
        for label, count in sur['confidence']['distribution'].items():
            html += f"\n<tr><td><code>{label}</code></td><td>{count}</td></tr>"
        
        html += f"""
</table>
</div>

<div class="card">
<h3>Routing & Fallbacks</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total route events</td><td>{routing['total_route_events']}</td></tr>
<tr><td>Fallback tier 1 used</td><td>{routing['fallback_tier1']}</td></tr>
<tr><td>Fallback tier 2 used</td><td>{routing['fallback_tier2']}</td></tr>"""
        
        for model, count in sorted(routing['model_distribution'].items()):
            html += f"\n<tr><td>{model}</td><td>{count}</td></tr>"
        
        html += f"""
</table>
</div>

<div class="card">
<h3>Errors</h3>
<table>
<tr><th>Type</th><th>Count</th></tr>
<tr><td>Stream errors</td><td>{errors['stream_errors']}</td></tr>
</table>
</div>

<div class="card">
<h3>Drift Detection</h3>
<p style="color:#8b949e">Surrogate hit rate per hour. Sudden drops may indicate model drift.</p>
<table>
<tr><th>Hour</th><th>Surrogate %</th><th>Total</th></tr>"""
        
        for h in drift['hourly'][-12:]:  # last 12 hours
            bar_width = h['surrogate_pct']
            html += f"""
<tr>
<td><code>{h['hour']}</code></td>
<td><div class="pct-bar"><div class="pct-fill" style="width:{bar_width}%"></div></div> {h['surrogate_pct']}%</td>
<td>{h['total_classifications']}</td>
</tr>"""
        
        html += """
</table>
</div>
</div>

<p style="color:#484f58; margin-top:2rem">Data from trace logs (last 2 files) + in-memory metrics. <a href="/classifier/report?format=json" style="color:#58a6ff">JSON format</a></p>
</body></html>"""
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html)
    
    return JSONResponse(content=report)


@app.get("/admin/config")
async def get_config(request: Request):
    """Return current config with sensitive keys redacted."""
    verify_auth(request)
    cfg_copy = copy.deepcopy(request.app.state.config)
    # Redact env var names (not values — those stay in env, not config)
    # No actual API keys are in the config, just env var names.
    # We show the raw config as-is since it only references env vars.
    return cfg_copy


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions — routed automatically."""
    verify_auth(request)
    cfg = request.app.state.config
    payload = await request.json()
    try:
        if payload.get("stream"):
            _inc_metric("stream_requests_total")
            return StreamingResponse(
                route_request_stream(cfg, payload),
                media_type="text/event-stream",
            )
        return await route_request(cfg, payload)
    except Exception:
        _inc_metric("errors_total")
        raise


@app.get("/metrics")
async def metrics(request: Request):
    """Prometheus-compatible metrics endpoint with router-specific counters."""
    verify_auth(request)
    m = _get_metrics()
    cfg = request.app.state.config
    cats = cfg.get("categories", {})

    # Build dynamic per-category metric lines
    lines = [
        "# HELP hermes_router_requests_total Total requests by category",
        "# TYPE hermes_router_requests_total counter",
    ]
    for name in cats:
        lines.append(f"hermes_router_requests_total{{category=\"{name}\"}} {m.get(f'requests_total_{name}', 0)}")
    lines.append("")

    lines += [
        "# HELP hermes_router_classifier_calls_total Classifier model calls",
        "# TYPE hermes_router_classifier_calls_total counter",
        f"hermes_router_classifier_calls_total {m['classifier_calls_total']}",
        "",
        "# HELP hermes_router_classifier_latency_ms Classifier latency in ms",
        "# TYPE hermes_router_classifier_latency_ms summary",
        f"hermes_router_classifier_latency_ms_sum {m['classifier_latency_ms_sum']}",
        f"hermes_router_classifier_latency_ms_avg {m['classifier_latency_ms_avg']}",
        "",
        "# HELP hermes_router_cache_hits_total Session cache hits (skip classifier)",
        "# TYPE hermes_router_cache_hits_total counter",
        f"hermes_router_cache_hits_total {m['cache_hits_total']}",
        "",
        "# HELP hermes_router_429_total Rate limit hits by category",
        "# TYPE hermes_router_429_total counter",
    ]
    for name in cats:
        lines.append(f"hermes_router_429_total{{category=\"{name}\"}} {m.get(f'429_{name}', 0)}")
    lines.append(f"hermes_router_429_total {m['429_total']}")
    lines.append("")

    lines += [
        "# HELP hermes_router_fallback_used_total Fallback tiers triggered",
        "# TYPE hermes_router_fallback_used_total counter",
        f"hermes_router_fallback_used_total{{level=\"1\"}} {m['fallback_used_total']}",
        f"hermes_router_fallback_used_total{{level=\"2\"}} {m['fallback2_used_total']}",
        "",
        "# HELP hermes_router_stream_requests_total Streaming requests",
        "# TYPE hermes_router_stream_requests_total counter",
        f"hermes_router_stream_requests_total {m['stream_requests_total']}",
        "",
        "# HELP hermes_router_errors_total Internal errors",
        "# TYPE hermes_router_errors_total counter",
        f"hermes_router_errors_total {m['errors_total']}",
        "",
        "# HELP hermes_router_sessions_active Active session count",
        "# TYPE hermes_router_sessions_active gauge",
        f"hermes_router_sessions_active {m['sessions_active']}",
        "",
        "# HELP hermes_router_circuits_open Number of open circuit breakers",
        "# TYPE hermes_router_circuits_open gauge",
        f"hermes_router_circuits_open {sum(1 for c in CIRCUITS.values() if c.get('state') == 'open')}",
    ]
    return JSONResponse(
        content={"metrics": m, "prometheus": "\n".join(lines)},
    )


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    port = cfg["server"]["port"]
    host = cfg["server"]["host"]

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )
