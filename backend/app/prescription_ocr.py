import io
import re
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

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

# Enhanced medication vocabulary with common drug names, brand names, and generic names
DRUG_LIST = [
    # Pain relievers
    "acetaminophen", "paracetamol", "tylenol", "ibuprofen", "advil", "motrin", 
    "naproxen", "aleve", "aspirin", "tramadol", "codeine", "morphine", "oxycodone", 
    "hydrocodone", "vicodin", "percocet", "fentanyl",
    
    # Antibiotics
    "amoxicillin", "amoxil", "azithromycin", "zithromax", "ciprofloxacin", "cipro",
    "ceftriaxone", "rocephin", "doxycycline", "vibramycin", "clindamycin", "cleocin",
    "metronidazole", "flagyl", "penicillin", "erythromycin", "clarithromycin", "biaxin",
    "cephalexin", "keflex", "trimethoprim", "sulfamethoxazole", "bactrim",
    
    # Heart medications
    "atorvastatin", "lipitor", "simvastatin", "zocor", "lisinopril", "prinivil",
    "amlodipine", "norvasc", "metoprolol", "lopressor", "carvedilol", "coreg",
    "losartan", "cozaar", "valsartan", "diovan", "enalapril", "vasotec",
    "propranolol", "inderal", "diltiazem", "cardizem", "verapamil", "calan",
    
    # Diabetes medications
    "metformin", "glucophage", "insulin", "glipizide", "glucotrol", "glyburide",
    "diabeta", "sitagliptin", "januvia", "pioglitazone", "actos", "glimepiride",
    "amaryl", "empagliflozin", "jardiance", "liraglutide", "victoza",
    
    # Stomach/GI medications
    "omeprazole", "prilosec", "pantoprazole", "protonix", "lansoprazole", "prevacid",
    "esomeprazole", "nexium", "ranitidine", "zantac", "famotidine", "pepcid",
    "simethicone", "gas-x", "loperamide", "imodium", "ondansetron", "zofran",
    
    # Mental health medications
    "sertraline", "zoloft", "fluoxetine", "prozac", "escitalopram", "lexapro",
    "paroxetine", "paxil", "venlafaxine", "effexor", "bupropion", "wellbutrin",
    "trazodone", "desyrel", "alprazolam", "xanax", "lorazepam", "ativan",
    "clonazepam", "klonopin", "diazepam", "valium", "zolpidem", "ambien",
    
    # Allergy/Respiratory
    "prednisone", "deltasone", "prednisolone", "methylprednisolone", "medrol",
    "albuterol", "ventolin", "montelukast", "singulair", "fluticasone", "flonase",
    "cetirizine", "zyrtec", "loratadine", "claritin", "fexofenadine", "allegra",
    "diphenhydramine", "benadryl", "guaifenesin", "mucinex",
    
    # Blood thinners
    "warfarin", "coumadin", "rivaroxaban", "xarelto", "apixaban", "eliquis",
    "dabigatran", "pradaxa", "clopidogrel", "plavix", "enoxaparin", "lovenox",
    
    # Thyroid medications
    "levothyroxine", "synthroid", "liothyronine", "cytomel", "methimazole", "tapazole",
    
    # Common vitamins and supplements
    "vitamin", "calcium", "iron", "folic", "folate", "biotin", "thiamine", "riboflavin",
    "niacin", "pyridoxine", "cobalamin", "ascorbic", "cholecalciferol", "ergocalciferol",
    "magnesium", "potassium", "zinc", "selenium", "chromium", "multivitamin",
    
    # Eye/Ear medications
    "timolol", "latanoprost", "xalatan", "brimonidine", "alphagan", "dorzolamide",
    "trusopt", "ciprofloxacin", "cortisporin", "neomycin", "polymyxin",
    
    # Skin medications
    "hydrocortisone", "triamcinolone", "mupirocin", "bactroban", "clotrimazole",
    "lotrimin", "ketoconazole", "nizoral", "tretinoin", "retin-a", "adapalene",
    "differin", "benzoyl", "peroxide", "salicylic", "calamine"
]


def _pil_to_cv2(image: Image.Image) -> Any:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def preprocess_image(image_bytes: bytes) -> Any:
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


def run_ocr(preprocessed_image: Any) -> str:
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
    """Enhanced fuzzy matching with better error handling and logging."""
    if not name or len(name.strip()) < 2:
        return {"match": None, "score": 0, "candidates": []}
    
    if not RAPIDFUZZ_AVAILABLE:
        # Fallback: improved substring matching
        name_clean = re.sub(r"[^a-zA-Z0-9 ]+", "", name).strip().lower()
        best_match = None
        best_score = 0
        candidates = []
        
        for drug in DRUG_LIST:
            drug_lower = drug.lower()
            # Exact match
            if name_clean == drug_lower:
                return {"match": drug, "score": 100, "candidates": [drug]}
            # Substring match
            elif name_clean in drug_lower or drug_lower in name_clean:
                score = min(80, int((len(name_clean) / len(drug_lower)) * 80))
                candidates.append({"drug": drug, "score": score})
                if score > best_score:
                    best_match = drug
                    best_score = score
        
        # Sort candidates by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return {
            "match": best_match,
            "score": best_score,
            "candidates": [c["drug"] for c in candidates[:3]]
        }
    
    # Use rapidfuzz for better matching
    name_clean = re.sub(r"[^a-zA-Z0-9 ]+", "", name).strip().lower()
    if not name_clean:
        return {"match": None, "score": 0, "candidates": []}
    
    try:
        # Get top 3 matches
        matches = process.extract(name_clean, DRUG_LIST, scorer=fuzz.token_sort_ratio, limit=3)
        if matches:
            best_match = matches[0]
            return {
                "match": best_match[0],
                "score": int(best_match[1]),
                "candidates": [match[0] for match in matches]
            }
    except Exception as e:
        print(f"Fuzzy matching error: {e}")
    
    return {"match": None, "score": 0, "candidates": []}


def parse_prescription_text(text: str) -> List[Dict[str, Any]]:
    """Enhanced prescription parsing with better error handling and validation."""
    if not text or not text.strip():
        return []
    
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    results: List[Dict[str, Any]] = []

    for line_idx, line in enumerate(lines):
        # Enhanced heuristics to ignore non-medication lines
        if len(line) < 3:
            continue
            
        # Skip lines that are clearly not medications
        skip_patterns = [
            r'^dr\.?\s+',  # Doctor names
            r'clinic|hospital|pharmacy|address|phone|fax',
            r'patient\s+name|date\s+of\s+birth|dob',
            r'signature|signed|date\s*:',
            r'refill|dispensed|quantity',
            r'^\d+[\s\-\.]*\d*[\s\-\.]*\d*$',  # Phone numbers
        ]
        
        if any(re.search(pattern, line, re.I) for pattern in skip_patterns):
            continue
        
        # Extract dosage, frequency, duration with improved patterns
        dosage = _extract_dosage(line)
        duration = _extract_duration(line)
        frequency = _extract_frequency(line)

        # Enhanced name extraction
        name_candidate = line
        for part in filter(None, [dosage, duration, frequency]):
            name_candidate = name_candidate.replace(part, " ")

        # Remove common non-drug words and patterns
        name_candidate = re.sub(r'\b(take|tablet|capsule|mg|ml|g|once|twice|daily|times|day|per|with|without|food|water|before|after|meals?)\b', ' ', name_candidate, flags=re.I)
        name_candidate = re.sub(r'[0-9\-\(\)\/\#\*]+', ' ', name_candidate)
        name_candidate = re.sub(r'[^A-Za-z ]+', ' ', name_candidate).strip()
        name_candidate = re.sub(r'\s+', ' ', name_candidate)  # Normalize whitespace

        # If multiple words, try different combinations
        tokens = name_candidate.split()
        if not tokens:
            continue
            
        # Try different combinations: single word, two words, three words
        candidates_to_try = []
        if len(tokens) >= 1:
            candidates_to_try.append(tokens[0])
        if len(tokens) >= 2:
            candidates_to_try.append(" ".join(tokens[:2]))
        if len(tokens) >= 3:
            candidates_to_try.append(" ".join(tokens[:3]))
        
        best_fuzzy = {"match": None, "score": 0, "candidates": []}
        best_candidate = ""
        
        for candidate in candidates_to_try:
            if len(candidate) >= 3:  # Minimum length check
                fuzzy = fuzzy_match_drug(candidate)
                if fuzzy.get("score", 0) > best_fuzzy.get("score", 0):
                    best_fuzzy = fuzzy
                    best_candidate = candidate

        entry = {
            "raw_line": line,
            "line_number": line_idx + 1,
            "name_candidate": best_candidate,
            "matched_name": best_fuzzy.get("match"),
            "match_score": best_fuzzy.get("score"),
            "match_candidates": best_fuzzy.get("candidates", []),
            "dosage": dosage,
            "frequency": frequency,
            "duration": duration,
            "confidence": "high" if best_fuzzy.get("score", 0) >= 80 else "medium" if best_fuzzy.get("score", 0) >= 60 else "low",
            "notes": None,
            "warnings": []
        }

        # Add warnings and suggestions for low confidence matches
        if best_fuzzy.get("score", 0) < 70 and best_fuzzy.get("candidates"):
            entry["notes"] = f"Low confidence match for '{best_candidate}'. Consider: {', '.join(best_fuzzy['candidates'][:3])}"
            entry["warnings"].append("manual_review_recommended")
        
        # Flag potentially dangerous medications that need special attention
        dangerous_keywords = ["insulin", "warfarin", "digoxin", "lithium", "methotrexate", "phenytoin"]
        if best_fuzzy.get("match") and any(keyword in best_fuzzy["match"].lower() for keyword in dangerous_keywords):
            entry["warnings"].append("high_risk_medication")
        
        # Only add entries with reasonable confidence or any extracted information
        if best_fuzzy.get("score", 0) > 30 or dosage or frequency or duration:
            results.append(entry)

    return results


def analyze_prescription_image(image_bytes: bytes) -> Dict[str, Any]:
    """Main function to analyze prescription image with comprehensive error handling."""
    try:
        # Validate input
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Empty image data provided")
        
        # Preprocess image
        try:
            preprocessed = preprocess_image(image_bytes)
        except Exception as e:
            # Fall back to basic processing if advanced preprocessing fails
            try:
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                if not CV2_AVAILABLE:
                    # Use PIL image directly for OCR if cv2 not available
                    text = run_ocr_on_pil(pil_image)
                else:
                    # Convert PIL to cv2 format for basic processing
                    import numpy as np
                    cv_image = np.array(pil_image)
                    text = run_ocr(cv_image)
            except Exception as e2:
                raise RuntimeError(f"Image preprocessing failed: {str(e)}. Fallback also failed: {str(e2)}")
        else:
            # Run OCR on preprocessed image
            try:
                text = run_ocr(preprocessed)
            except Exception as e:
                raise RuntimeError(f"OCR processing failed: {str(e)}")
        
        # Parse the extracted text
        try:
            parsed = parse_prescription_text(text)
        except Exception as e:
            # Return raw text if parsing fails
            return {
                "ocr_text": text,
                "medications": [],
                "error": f"Text parsing failed: {str(e)}",
                "status": "partial_success"
            }
        
        # Calculate overall confidence metrics
        total_medications = len(parsed)
        high_confidence = sum(1 for med in parsed if med.get("confidence") == "high")
        medium_confidence = sum(1 for med in parsed if med.get("confidence") == "medium")
        
        overall_confidence = "high" if high_confidence >= total_medications * 0.8 else \
                           "medium" if (high_confidence + medium_confidence) >= total_medications * 0.6 else \
                           "low"
        
        return {
            "ocr_text": text,
            "medications": parsed,
            "status": "success",
            "metadata": {
                "total_medications_found": total_medications,
                "high_confidence_matches": high_confidence,
                "medium_confidence_matches": medium_confidence,
                "overall_confidence": overall_confidence,
                "ocr_engine": "tesseract" if TESSERACT_AVAILABLE else "unavailable",
                "preprocessing_used": CV2_AVAILABLE
            }
        }
        
    except Exception as e:
        return {
            "ocr_text": "",
            "medications": [],
            "error": str(e),
            "status": "error",
            "metadata": {
                "ocr_engine": "tesseract" if TESSERACT_AVAILABLE else "unavailable",
                "preprocessing_available": CV2_AVAILABLE
            }
        }


def run_ocr_on_pil(pil_image: Image.Image) -> str:
    """Fallback OCR function for when cv2 is not available."""
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract not installed; install pytesseract and Tesseract engine to enable OCR")
    
    # Convert to grayscale for better OCR
    gray_image = pil_image.convert("L")
    
    # Use tesseract to extract text
    custom_config = r"-l eng --oem 1 --psm 6"
    text = pytesseract.image_to_string(gray_image, config=custom_config)
    return text
