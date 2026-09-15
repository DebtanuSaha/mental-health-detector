"""
text_cleaner.py

Conservative, config-driven text preprocessing.

Per the project brief's caution: mental-health/crisis text carries signal
in places generic NLP pipelines often strip — punctuation, repeated
characters ("sooo tired"), capitalization, emojis. This module defaults
to doing the *minimum* (Unicode + whitespace normalization) and makes
every other transformation an explicit, off-by-default config flag, so
nothing gets removed without a deliberate decision (and, eventually, an
ablation showing it doesn't hurt Phase 2/3 metrics).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


@dataclass
class PreprocessConfig:
    lowercase: bool = False
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    replace_urls: bool = True
    replace_mentions: bool = False
    strip_emojis: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


def clean_text(text: str, config: PreprocessConfig) -> str:
    """Apply the configured cleaning steps, in a fixed, documented order."""
    if text is None:
        return ""
    out = str(text)

    if config.normalize_unicode:
        out = unicodedata.normalize("NFKC", out)

    if config.replace_urls:
        out = _URL_RE.sub(" <URL> ", out)

    if config.replace_mentions:
        out = _MENTION_RE.sub(" <MENTION> ", out)

    if config.strip_emojis:
        out = _EMOJI_RE.sub(" ", out)

    if config.lowercase:
        out = out.lower()

    if config.normalize_whitespace:
        out = _WHITESPACE_RE.sub(" ", out).strip()

    return out


def clean_series(series, config: PreprocessConfig):
    """Vectorized helper for a pandas Series of raw text."""
    return series.astype(str).apply(lambda t: clean_text(t, config))
