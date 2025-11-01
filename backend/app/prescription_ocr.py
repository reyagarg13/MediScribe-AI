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

# Legacy drug list - kept as fallback only (now using enhanced database)
# This hardcoded list is only used when the enhanced API-based database fails
LEGACY_DRUG_LIST = [
    # Most common drugs only - reduced from full list
    "acetaminophen", "tylenol", "ibuprofen", "advil", "aspirin", "amoxicillin", 
    "lipitor", "metformin", "lisinopril", "omeprazole", "prednisone", "albuterol",
    "warfarin", "insulin", "simvastatin", "amlodipine", "sertraline", "zoloft",
    "atorvastatin", "levothyroxine", "synthroid", "losartan", "metoprolol",
    "fluoxetine", "prozac", "hydrocodone", "vicodin", "oxycodone", "tramadol"
]

# Note: International drug name mapping is now handled by LLM-powered normalization
# in drug_database.py - no more hardcoded mappings needed!

# For backward compatibility - some old code might reference DRUG_LIST
DRUG_LIST = LEGACY_DRUG_LIST


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


def fuzzy_match_drug_legacy(name: str) -> Dict[str, Any]:
    """Enhanced fuzzy matching with better error handling and logging."""
    if not name or len(name.strip()) < 2:
        return {"match": None, "score": 0, "candidates": []}
    
    if not RAPIDFUZZ_AVAILABLE:
        # Fallback: improved substring matching
        name_clean = re.sub(r"[^a-zA-Z0-9 ]+", "", name).strip().lower()
        best_match = None
        best_score = 0
        candidates = []
        
        for drug in LEGACY_DRUG_LIST:
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
        matches = process.extract(name_clean, LEGACY_DRUG_LIST, scorer=fuzz.token_sort_ratio, limit=3)
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


# Import enhanced drug database functions
try:
    from .drug_database import fuzzy_match_drug_enhanced, search_drug_enhanced
    ENHANCED_DB_AVAILABLE = True
    print("✅ Enhanced drug database loaded successfully")
except ImportError as e:
    ENHANCED_DB_AVAILABLE = False
    print(f"❌ Enhanced drug database not available: {e}")


def fuzzy_match_drug(name: str) -> Dict[str, Any]:
    """
    Enhanced drug matching using RxNorm/OpenFDA APIs with fallback to legacy
    """
    if ENHANCED_DB_AVAILABLE:
        try:
            # Try enhanced database first
            result = fuzzy_match_drug_enhanced(name)
            if result and result.get("score", 0) > 50:
                print(f"✅ Enhanced DB match for '{name}': {result['match']} ({result['score']}%)")
                return result
        except Exception as e:
            print(f"⚠️ Enhanced drug lookup failed for '{name}': {e}")
    
    # Fallback to legacy hardcoded list
    legacy_result = fuzzy_match_drug_legacy(name)
    if legacy_result.get("score", 0) > 0:
        print(f"📋 Legacy DB match for '{name}': {legacy_result['match']} ({legacy_result['score']}%)")
    return legacy_result


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


def parse_gemini_response(analysis_text: str) -> Dict[str, Any]:
    """Parse enhanced Gemini Vision API response and extract structured prescription data."""
    result = {
        "prescription_header": {},
        "medications": []
    }
    
    if not analysis_text:
        return result
    
    # Check if Gemini found no medications
    if "NO_MEDICATIONS_FOUND" in analysis_text.upper():
        return result
    
    lines = analysis_text.split('\n')
    current_section = None
    medication_data = None
    
    # Parse the response into sections
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect sections
        if "=== PRESCRIPTION HEADER ===" in line:
            current_section = "header"
            continue
        elif "=== PRESCRIBED MEDICATIONS ===" in line:
            current_section = "medications"
            continue
        
        # Parse header information
        if current_section == "header":
            for field_mapping in [
                ("**Patient Name:**", "patient_name"),
                ("**Patient Age:**", "patient_age"),
                ("**Patient Sex:**", "patient_sex"),
                ("**Date of Birth:**", "patient_dob"),
                ("**Doctor Name:**", "doctor_name"),
                ("**Clinic Name:**", "clinic_name"),
                ("**Clinic Address:**", "clinic_address"),
                ("**Phone:**", "phone"),
                ("**Prescription Date:**", "prescription_date"),
                ("**Prescription Number:**", "prescription_number")
            ]:
                field_label, field_key = field_mapping
                if line.startswith(field_label):
                    value = line.replace(field_label, "").strip()
                    if value and value.lower() not in ["not visible", "not specified", "none"]:
                        result["prescription_header"][field_key] = value
        
        # Parse medication information
        elif current_section == "medications":
            # Look for numbered medication entries (1. **Item Name:** ...)
            if re.match(r'^\d+\.\s*\*\*Item Name:\*\*', line):
                # Save previous medication if exists
                if medication_data:
                    processed_med = _process_medication_data(medication_data)
                    if processed_med:
                        result["medications"].append(processed_med)
                
                # Start new medication
                name_match = re.search(r'\*\*Item Name:\*\*\s*(.+)', line)
                if name_match:
                    item_name = name_match.group(1).strip()
                    medication_data = {
                        "raw_line": line,
                        "name_candidate": item_name,
                        "complete_dosage": None,
                        "frequency": None,
                        "duration": None,
                        "meal_instructions": None
                    }
            
            # Look for medication details
            elif medication_data is not None:
                if '**Complete Dosage:**' in line:
                    dosage_match = re.search(r'\*\*Complete Dosage:\*\*\s*(.+)', line)
                    if dosage_match:
                        dosage_text = dosage_match.group(1).strip()
                        if dosage_text.lower() not in ["not specified", "none", ""]:
                            medication_data["complete_dosage"] = dosage_text
                        
                elif '**Frequency:**' in line:
                    freq_match = re.search(r'\*\*Frequency:\*\*\s*(.+)', line)
                    if freq_match:
                        freq_text = freq_match.group(1).strip()
                        if freq_text.lower() not in ["not specified", "none", ""]:
                            medication_data["frequency"] = freq_text
                        
                elif '**Duration:**' in line:
                    dur_match = re.search(r'\*\*Duration:\*\*\s*(.+)', line)
                    if dur_match:
                        duration_text = dur_match.group(1).strip()
                        if duration_text.lower() not in ["not specified", "none", ""]:
                            medication_data["duration"] = duration_text
                            
                elif '**Meal Instructions:**' in line:
                    meal_match = re.search(r'\*\*Meal Instructions:\*\*\s*(.+)', line)
                    if meal_match:
                        meal_text = meal_match.group(1).strip()
                        if meal_text.lower() not in ["not specified", "none", ""]:
                            medication_data["meal_instructions"] = meal_text
    
    # Collect all medications first (without individual processing)
    if medication_data:
        result["medications"].append(medication_data)
    
    # OPTIMIZATION: Skip individual processing, go straight to batch processing
    if result["medications"]:
        try:
            from .drug_database import batch_fuzzy_match_drugs_enhanced
            
            # Extract all drug names for batch processing
            drug_names = [med["name_candidate"] for med in result["medications"]]
            print(f"🚀 Starting batch processing for {len(drug_names)} drugs...")
            
            # Single batch call for all drugs
            batch_results = batch_fuzzy_match_drugs_enhanced(drug_names)
            
            # Process medications with batch results
            optimized_medications = []
            for med_data in result["medications"]:
                drug_name = med_data["name_candidate"]
                batch_fuzzy_result = batch_results.get(drug_name, {})
                
                # Process with batch result (no individual LLM calls)
                processed_med = _process_medication_data(med_data, batch_fuzzy_result)
                if processed_med:
                    optimized_medications.append(processed_med)
            
            result["medications"] = optimized_medications
            print(f"✅ Batch processed {len(drug_names)} drugs in single LLM call")
            
        except ImportError:
            print("⚠️ Batch processing not available, falling back to individual processing")
            # Only fall back to individual processing if batch completely fails
            optimized_medications = []
            for med_data in result["medications"]:
                processed_med = _process_medication_data(med_data)
                if processed_med:
                    optimized_medications.append(processed_med)
            result["medications"] = optimized_medications
    
    # Fallback to legacy parsing if no structured data found
    if not result["medications"]:
        legacy_medications = _parse_legacy_gemini_format(analysis_text)
        result["medications"] = legacy_medications
    
    return result


def _process_medication_data(medication_data: Dict, fuzzy_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process medication data and add fuzzy matching."""
    item_name = medication_data["name_candidate"]
    
    # Use provided fuzzy result from batch processing, or do individual lookup
    if fuzzy_result is None:
        fuzzy_result = fuzzy_match_drug(item_name)
    
    # Extract simple dosage and frequency for backward compatibility
    complete_dosage = medication_data.get("complete_dosage", "")
    simple_dosage = _extract_dosage(complete_dosage) if complete_dosage else None
    simple_frequency = medication_data.get("frequency")
    
    medication = {
        "raw_line": medication_data["raw_line"],
        "name_candidate": item_name,
        "matched_name": fuzzy_result.get("match"),
        "match_score": fuzzy_result.get("score", 0),
        "match_candidates": fuzzy_result.get("candidates", []),
        "dosage": simple_dosage,  # Backward compatibility
        "complete_dosage": complete_dosage,  # New detailed dosage
        "frequency": simple_frequency,
        "duration": medication_data.get("duration"),
        "meal_instructions": medication_data.get("meal_instructions"),  # Add meal timing
        "confidence": "high" if fuzzy_result.get("score", 0) >= 80 else "medium" if fuzzy_result.get("score", 0) >= 60 else "low",
        "method": "gemini_vision_enhanced",
        "notes": None,
        "warnings": [],
        "normalization": fuzzy_result.get("normalization")  # Add LLM normalization info
    }
    
    return medication


def _parse_legacy_gemini_format(analysis_text: str) -> List[Dict[str, Any]]:
    """Fallback parser for legacy Gemini response formats."""
    medications = []
    lines = analysis_text.split('\n')
    current_medication = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for explicit medication names in structured output
        if ("medication name" in line.lower() or "drug name" in line.lower() or "item name" in line.lower()) and ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                drug_info = parts[1].strip()
                
                # Clean up the drug name and extract dosage
                dosage_match = re.search(r'(\d+(?:\.\d+)?)\s*(mg|ml|g|mcg|iu|gram|grams?)', drug_info, re.I)
                extracted_dosage = dosage_match.group(0) if dosage_match else None
                
                drug_name = re.sub(r'\(.*?\)', '', drug_info).strip()
                drug_name = re.sub(r'brand.*|likely.*', '', drug_name, flags=re.I).strip()
                drug_name = re.sub(r'\d+(?:\.\d+)?\s*(?:mg|ml|g|mcg|iu|gram|grams?)', '', drug_name, re.I).strip()
                
                if drug_name and len(drug_name) > 2:
                    fuzzy_result = fuzzy_match_drug(drug_name)
                    
                    if fuzzy_result.get("score", 0) > 50:
                        current_medication = {
                            "raw_line": line,
                            "name_candidate": drug_name,
                            "matched_name": fuzzy_result.get("match"),
                            "match_score": fuzzy_result.get("score", 0),
                            "match_candidates": fuzzy_result.get("candidates", []),
                            "dosage": extracted_dosage or _extract_dosage(drug_info),
                            "complete_dosage": drug_info,  # Store complete text
                            "frequency": _extract_frequency(drug_info),
                            "duration": _extract_duration(drug_info),
                            "confidence": "high" if fuzzy_result.get("score", 0) >= 80 else "medium" if fuzzy_result.get("score", 0) >= 60 else "low",
                            "method": "gemini_vision_legacy",
                            "notes": None,
                            "warnings": []
                        }
                        medications.append(current_medication)
        
        # Look for dosage, frequency, duration lines that follow medication names
        elif current_medication and ("dosage" in line.lower() or "frequency" in line.lower() or "duration" in line.lower()) and ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                field_name = parts[0].strip().lower()
                field_value = parts[1].strip()
                
                if medications and field_value.lower() not in ["not specified", "none", ""]:
                    last_med = medications[-1]
                    clean_value = re.sub(r'\*+\s*', '', field_value).strip()
                    
                    if "dosage" in field_name and not last_med.get("dosage"):
                        last_med["dosage"] = _extract_dosage(clean_value)
                        last_med["complete_dosage"] = clean_value
                    elif "frequency" in field_name and not last_med.get("frequency"):
                        last_med["frequency"] = clean_value
                    elif "duration" in field_name and not last_med.get("duration"):
                        last_med["duration"] = clean_value
    
    return medications


def analyze_prescription_image(image_bytes: bytes) -> Dict[str, Any]:
    """Main function to analyze prescription image with comprehensive error handling."""
    import time
    start_time = time.time()
    
    try:
        # Validate input
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Empty image data provided")
        
        # Try Gemini Vision API first (best for handwritten prescriptions)
        try:
            from . import gemini
            if hasattr(gemini, 'analyze_prescription_image') and gemini.available():
                print("Using Gemini Vision API for prescription analysis...")
                gemini_result = gemini.analyze_prescription_image(image_bytes)
                
                if gemini_result and "analysis" in gemini_result:
                    # Parse Gemini's response
                    parsed_result = parse_gemini_response(gemini_result["analysis"])
                    medications = parsed_result.get("medications", [])
                    prescription_header = parsed_result.get("prescription_header", {})
                    
                    if medications:  # If Gemini found medications, use its results
                        processing_time = time.time() - start_time
                        return {
                            "ocr_text": gemini_result["analysis"],
                            "medications": medications,
                            "prescription_header": prescription_header,
                            "status": "success",
                            "method": "gemini_vision",
                            "processing_time": processing_time,
                            "metadata": {
                                "total_medications_found": len(medications),
                                "high_confidence_matches": sum(1 for med in medications if med.get("confidence") == "high"),
                                "medium_confidence_matches": sum(1 for med in medications if med.get("confidence") == "medium"),
                                "overall_confidence": "high" if len([m for m in medications if m.get("confidence") == "high"]) >= len(medications) * 0.6 else "medium",
                                "ocr_engine": "gemini_vision",
                                "preprocessing_used": False,
                                "prescription_header_extracted": bool(prescription_header)
                            }
                        }
                    else:
                        print("Gemini Vision found no medications, falling back to traditional OCR...")
                else:
                    print("Gemini Vision API failed, falling back to traditional OCR...")
        except Exception as e:
            print(f"Gemini Vision API error: {str(e)}, falling back to traditional OCR...")
        
        # Fallback to traditional OCR
        print("Using traditional OCR for prescription analysis...")
        
        # If neither OCR method works, return basic info
        if not TESSERACT_AVAILABLE:
            print("Warning: Tesseract OCR not available, using basic text analysis...")
            # Try to do basic image analysis without OCR
            processing_time = time.time() - start_time
            return {
                "ocr_text": "OCR not available - please install Tesseract or ensure Gemini Vision API is working",
                "medications": [],
                "status": "limited_success",
                "method": "no_ocr_available",
                "processing_time": processing_time,
                "error": "Neither Gemini Vision nor Tesseract OCR is available",
                "metadata": {
                    "total_medications_found": 0,
                    "high_confidence_matches": 0,
                    "medium_confidence_matches": 0,
                    "overall_confidence": "none",
                    "ocr_engine": "unavailable",
                    "preprocessing_used": CV2_AVAILABLE
                }
            }
        
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
            processing_time = time.time() - start_time
            return {
                "ocr_text": text,
                "medications": [],
                "error": f"Text parsing failed: {str(e)}",
                "status": "partial_success",
                "method": "traditional_ocr",
                "processing_time": processing_time
            }
        
        # Calculate overall confidence metrics
        total_medications = len(parsed)
        high_confidence = sum(1 for med in parsed if med.get("confidence") == "high")
        medium_confidence = sum(1 for med in parsed if med.get("confidence") == "medium")
        
        overall_confidence = "high" if high_confidence >= total_medications * 0.8 else \
                           "medium" if (high_confidence + medium_confidence) >= total_medications * 0.6 else \
                           "low"
        
        processing_time = time.time() - start_time
        return {
            "ocr_text": text,
            "medications": parsed,
            "status": "success",
            "method": "traditional_ocr",
            "processing_time": processing_time,
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
        import time
        processing_time = time.time() - start_time if 'start_time' in locals() else 0
        return {
            "ocr_text": "",
            "medications": [],
            "error": str(e),
            "status": "error",
            "method": "error",
            "processing_time": processing_time,
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
