"""Transcription helper.

This module lazily loads the Whisper model if the `whisper` package is installed.
If not installed, it raises a clear runtime error when transcription is attempted.
"""
from typing import Optional
import os

try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    whisper = None
    WHISPER_AVAILABLE = False

# Lazy cached model reference
_model: Optional[object] = None

def _get_model():
    """Lazily load and return the Whisper model.

    Loading the model at import time can be heavy and may cause startup failures
    in environments where the package or model files aren't available. We delay
    loading until transcription is actually requested.
    """
    global _model
    if _model is None:
        if not WHISPER_AVAILABLE:
            raise RuntimeError(
                "Whisper is not installed. Install it with `pip install openai-whisper` "
                "or ensure your Python environment has the package available."
                "\nYou can also install all backend deps with: `pip install -r backend/requirements.txt`."
            )
        # model name can be overridden via env if needed
        model_name = os.environ.get("WHISPER_MODEL", "large-v3")
        _model = whisper.load_model(model_name)
    return _model


def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file to text.

    Raises a RuntimeError with actionable instructions if Whisper isn't installed.
    """
    model = _get_model()
    result = model.transcribe(file_path)
    return result.get("text", "")
