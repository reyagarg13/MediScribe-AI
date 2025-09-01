import io
import re
from typing import List, Dict, Any, Optional

try:
    import numpy as np
    import cv2
    CV2_AVAILABLE = True
except Exception:
    np = None
    cv2 = None
    CV2_AVAILABLE = False

from PIL import Image

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    TESSERACT_AVAILABLE = False

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    process = None
    fuzz = None
    RAPIDFUZZ_AVAILABLE = False

# Small medication vocabulary to fuzzy-match against. Extend as needed.
DRUG_LIST = [
    "amoxicillin",
    "paracetamol",
    "acetaminophen",
    "ibuprofen",
    "metformin",
    "atorvastatin",
    "lisinopril",
    "amlodipine",
    "omeprazole",
    "azithromycin",
    "ciprofloxacin",
    "ceftriaxone",
    "insulin",
    "warfarin",
    "aspirin",
    "prednisone",
]


def _pil_to_cv2(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) not installed; install opencv-python-headless to enable image preprocessing")
    # Load image
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = _pil_to_cv2(pil)

    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize if very large
    h, w = gray.shape
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)

    # Adaptive threshold to handle lighting
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 15)

    # Deskew using orientation from tesseract if available
    try:
        osd = pytesseract.image_to_osd(th)
        rot = re.search(r"Rotate: (\d+)", osd)
        if rot:
            angle = float(rot.group(1))
            if angle != 0:
                # rotate to correct orientation
                (h2, w2) = th.shape[:2]
                M = cv2.getRotationMatrix2D((w2 / 2, h2 / 2), -angle, 1.0)
                th = cv2.warpAffine(th, M, (w2, h2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass

    # Morphological opening to reduce small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    return th


def run_ocr(preprocessed_image: np.ndarray) -> str:
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract not installed; install pytesseract and Tesseract engine to enable OCR")
    # Use tesseract to extract text
    custom_config = r"-l eng --oem 1 --psm 6"
    text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
    return text


def _extract_dosage(line: str) -> Optional[str]:
    m = re.search(r"(\d+(?:\.\d+)?\s*(?:mg|ml|g|mcg|iu))", line, flags=re.I)
    return m.group(1) if m else None


def _extract_duration(line: str) -> Optional[str]:
    m = re.search(r"(\d+\s*(?:days|day|weeks|week|months|month))", line, flags=re.I)
    return m.group(1) if m else None


def _extract_frequency(line: str) -> Optional[str]:
    freq_patterns = [
        r"once (?:daily|a day|daily)",
        r"twice (?:daily|a day|daily)",
        r"three times a day",
        r"\btds\b",
        r"\bbd\b",
        r"\bq(\d+)h\b",
        r"\b\d+ times? (?:a|per) day\b",
    ]
    for p in freq_patterns:
        m = re.search(p, line, flags=re.I)
        if m:
            return m.group(0)
    # Generic look for 'daily' or 'per day'
    if re.search(r"daily|per day|day", line, flags=re.I):
        return "daily"
    return None


def fuzzy_match_drug(name: str) -> Dict[str, Any]:
    if not RAPIDFUZZ_AVAILABLE:
        # Fallback: simple substring match
        name_clean = re.sub(r"[^a-zA-Z0-9 ]+", "", name).strip().lower()
        for d in DRUG_LIST:
            if name_clean and name_clean in d:
                return {"match": d, "score": 80}
        return {"match": None, "score": 0}
    # Use rapidfuzz to find closest drug
    name_clean = re.sub(r"[^a-zA-Z0-9 ]+", "", name).strip().lower()
    if not name_clean:
        return {"match": None, "score": 0}
    choice, score, _ = process.extractOne(name_clean, DRUG_LIST, scorer=fuzz.token_sort_ratio)
    return {"match": choice, "score": int(score)}


def parse_prescription_text(text: str) -> List[Dict[str, Any]]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    results: List[Dict[str, Any]] = []

    for line in lines:
        # Heuristic: ignore lines that look like addresses or signatures
        if len(line) < 3:
            continue
        # Extract dosage, frequency, duration
        dosage = _extract_dosage(line)
        duration = _extract_duration(line)
        frequency = _extract_frequency(line)

        # Remove extracted parts to isolate possible drug name
        name_candidate = line
        for part in filter(None, [dosage, duration, frequency]):
            name_candidate = name_candidate.replace(part, " ")

        # Remove common punctuation and numerics
        name_candidate = re.sub(r"[0-9\-\(\)\/]+", " ", name_candidate)
        name_candidate = re.sub(r"[^A-Za-z ]+", " ", name_candidate).strip()

        # If multiple words, take first two as possible drug name
        tokens = name_candidate.split()
        guess = " ".join(tokens[:2]) if tokens else ""

        fuzzy = fuzzy_match_drug(guess)

        entry = {
            "raw_line": line,
            "name_candidate": guess,
            "matched_name": fuzzy.get("match"),
            "match_score": fuzzy.get("score"),
            "dosage": dosage,
            "frequency": frequency,
            "duration": duration,
            "notes": None,
        }

        # Suggest corrections if low score
        if fuzzy.get("score", 0) < 70:
            entry["notes"] = f"Low confidence match for '{guess}' — possible matches: {process.extract(guess, DRUG_LIST, limit=3)}"

        results.append(entry)

    return results


def analyze_prescription_image(image_bytes: bytes) -> Dict[str, Any]:
    pre = preprocess_image(image_bytes)
    text = run_ocr(pre)
    parsed = parse_prescription_text(text)
    return {"ocr_text": text, "medications": parsed}
