"""
Simple Gemini adapter. Uses GEMINI_API_KEY and GEMINI_API_URL env vars.
This is a minimal example: depending on your chosen Gemini API contract you may
need to adjust headers, input shape, and response parsing.
"""
import os
import json
import requests
from typing import Optional, Dict, Any

from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleRequest

# Use GEMINI_API_URL if provided, otherwise default to the Google Generative Language endpoint
GEMINI_API_URL = os.getenv(
    "GEMINI_API_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
)

# Service account creds (we reuse FIREBASE service account if present)
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
FIREBASE_CREDENTIALS_JSON = os.getenv("FIREBASE_CREDENTIALS_JSON")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def _get_access_token() -> Optional[str]:
    """Return an OAuth2 access token using service account JSON if available.

    Falls back to returning None if no service account is configured; in that
    case if GEMINI_API_KEY is provided it may be used instead (not recommended
    for production when service accounts are available).
    """
    try:
        if FIREBASE_CREDENTIALS_PATH and os.path.exists(FIREBASE_CREDENTIALS_PATH):
            creds = service_account.Credentials.from_service_account_file(
                FIREBASE_CREDENTIALS_PATH, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        elif FIREBASE_CREDENTIALS_JSON:
            info = json.loads(FIREBASE_CREDENTIALS_JSON)
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        else:
            return None

        auth_req = GoogleRequest()
        creds.refresh(auth_req)
        return creds.token
    except Exception as e:
        # Do not fail hard here; return None and let caller decide
        print("Failed to obtain access token from service account:", e)
        return None


def available() -> bool:
    return bool(GEMINI_API_URL and (FIREBASE_CREDENTIALS_PATH or FIREBASE_CREDENTIALS_JSON or GEMINI_API_KEY))


def analyze_text(prompt: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Send text to Gemini-like endpoint and return JSON response.

    Uses service account token if available, otherwise attempts to use GEMINI_API_KEY
    as a bearer token (less ideal).
    """
    if not available():
        raise RuntimeError("Gemini not configured (set FIREBASE_CREDENTIALS_PATH or GEMINI_API_KEY and GEMINI_API_URL)")

    token = _get_access_token()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif GEMINI_API_KEY:
        headers["Authorization"] = f"Bearer {GEMINI_API_KEY}"

    # The exact request body depends on the Gemini API contract. We'll send a
    # simple input wrapper that many examples accept; adapt if you have a
    # different required schema.
    payload: Dict[str, Any] = {"input": {"text": prompt}}
    if params:
        payload.update(params)

    resp = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()
