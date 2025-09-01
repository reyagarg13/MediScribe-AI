"""Lightweight NLP helpers with guarded heavy dependency imports.

This module avoids importing `transformers` at import-time. If `transformers` is
available it will lazily create a summarization pipeline. Otherwise a very small
heuristic summarizer is used as a fallback so the server remains usable.
"""
from typing import Optional
import os

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

_summarizer: Optional[object] = None

def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        if not TRANSFORMERS_AVAILABLE:
            return None
        model_name = os.environ.get("SUMMARIZER_MODEL", "facebook/bart-large-cnn")
        _summarizer = pipeline("summarization", model=model_name)
    return _summarizer


def summarize_text(text: str) -> str:
    """Return a summary for `text`.

    Uses the transformers pipeline when available. Otherwise falls back to a
    simple heuristic: take the first 2-3 sentences or truncate to 200 chars.
    """
    summarizer = _get_summarizer()
    if summarizer is not None:
        try:
            summary = summarizer(text, max_length=120, min_length=30, do_sample=False)
            return summary[0]["summary_text"]
        except Exception:
            # Fall back to heuristic if the heavy model call fails at runtime
            pass

    # Heuristic fallback summarizer
    # Split into sentences naively and return the first 2 sentences or truncated text.
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) >= 2:
        return " ".join(sentences[:2])
    # final fallback: truncate
    return (text.strip()[:200] + "...") if len(text.strip()) > 200 else text.strip()
