"""
Simple Gemini adapter. Uses GEMINI_API_KEY and GEMINI_API_URL env vars.
This is a minimal example: depending on your chosen Gemini API contract you may
need to adjust headers, input shape, and response parsing.
"""
import os
import json
import requests
from typing import Optional, Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

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


def analyze_prescription_image(image_bytes: bytes) -> Dict[str, Any]:
    """Analyze prescription image using Gemini Vision API."""
    if not available():
        raise RuntimeError("Gemini not configured (set FIREBASE_CREDENTIALS_PATH or GEMINI_API_KEY and GEMINI_API_URL)")
    
    # Use Vision URL if available, otherwise use standard URL
    vision_url = os.getenv("GEMINI_VISION_URL", GEMINI_API_URL)
    
    # Convert image to base64
    import base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    
    # For Google's Generative AI API, use API key directly
    if GEMINI_API_KEY:
        # Use the key as query parameter for Google's API
        vision_url = f"{vision_url}?key={GEMINI_API_KEY}"
    else:
        token = _get_access_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            raise RuntimeError("No authentication method available (need GEMINI_API_KEY or service account)")
    
    # Prescription analysis prompt
    prompt = """Analyze this prescription image and extract medication information. For each medication found, provide:
1. Medication name
2. Dosage (strength/amount)
3. Frequency (how often to take)
4. Duration (how long to take)

Please be accurate and only extract medications that are clearly visible. Format your response as a structured list.

Example format:
* **Medication Name:** [Name]
* **Dosage:** [Strength]
* **Frequency:** [How often]
* **Duration:** [How long]

Focus on handwritten prescription text."""
    
    # Prepare payload for Google's Generative AI Vision API
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "topK": 32,
            "topP": 1,
            "maxOutputTokens": 2048,
        }
    }
    
    try:
        response = requests.post(vision_url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract text from Google's response format
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                text_parts = candidate["content"]["parts"]
                analysis_text = ""
                for part in text_parts:
                    if "text" in part:
                        analysis_text += part["text"] + "\n"
                
                return {
                    "analysis": analysis_text.strip(),
                    "raw_response": result
                }
        
        return {
            "analysis": "",
            "raw_response": result,
            "error": "No text content found in response"
        }
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Gemini Vision API request failed: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Gemini Vision API error: {str(e)}")
