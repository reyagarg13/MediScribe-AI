"""
Simple Gemini adapter. Uses GEMINI_API_KEY and GEMINI_API_URL env vars.
This is a minimal example: depending on your chosen Gemini API contract you may
need to adjust headers, input shape, and response parsing.
"""
import os
import requests
from typing import Optional, Dict, Any

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL")


def available() -> bool:
    return bool(GEMINI_API_KEY and GEMINI_API_URL)


def analyze_text(prompt: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Send text to Gemini-like endpoint and return JSON response."""
    if not available():
        raise RuntimeError("Gemini not configured")
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"input": prompt}
    if params:
        payload.update(params)
    resp = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()
