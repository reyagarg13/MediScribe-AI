"""
Advanced Prescription OCR System with Multi-Stage Processing
"""
import io
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    np = None
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    TESSERACT_AVAILABLE = False

from PIL import Image, ImageEnhance, ImageFilter
from .prescription_ocr import DRUG_LIST, fuzzy_match_drug


class ProcessingStage(Enum):
    """Different stages of prescription processing"""
    IMAGE_PREPROCESSING = "image_preprocessing"
    TEXT_EXTRACTION = "text_extraction"
    MEDICAL_ENTITY_RECOGNITION = "medical_entity_recognition"
    DOSAGE_PARSING = "dosage_parsing"
    VALIDATION = "validation"
    CONFIDENCE_SCORING = "confidence_scoring"


@dataclass
class ProcessingResult:
    """Result from each processing stage"""
    stage: ProcessingStage
    success: bool
    data: Any
    confidence: float
    processing_time: float
    errors: List[str]
    metadata: Dict[str, Any]


@dataclass
class MedicationEntry:
    """Enhanced medication entry with validation"""
    name: str
    generic_name: Optional[str]
    dosage: Optional[str]
    dosage_value: Optional[float]
    dosage_unit: Optional[str]
    frequency: Optional[str]
    frequency_per_day: Optional[float]
    duration: Optional[str]
    duration_days: Optional[int]
    route: Optional[str]  # oral, IV, topical, etc.
    instructions: Optional[str]
    
    # Confidence and validation
    name_confidence: float
    dosage_confidence: float
    frequency_confidence: float
    overall_confidence: float
    
    # Medical validation
    is_valid_drug: bool
    drug_interactions: List[str]
    contraindications: List[str]
    warnings: List[str]
    
    # Source information
    extracted_from: str
    raw_text: str
    bounding_box: Optional[Tuple[int, int, int, int]]


class AdvancedPrescriptionOCR:
    """Advanced prescription OCR with multi-stage processing"""
    
    def __init__(self):
        self.processing_stages = []
        self.total_processing_time = 0
        self.load_medical_knowledge()
    
    def load_medical_knowledge(self):
        """Load enhanced medical knowledge base"""
        # Enhanced drug database with generic names, dosage forms, etc.
        self.enhanced_drug_db = self._build_enhanced_drug_db()
        self.dosage_patterns = self._compile_dosage_patterns()
        self.frequency_patterns = self._compile_frequency_patterns()
        self.duration_patterns = self._compile_duration_patterns()
        self.route_patterns = self._compile_route_patterns()
    
    def _build_enhanced_drug_db(self) -> Dict[str, Dict]:
        """Build enhanced drug database with additional information"""
        enhanced_db = {}
        
        # Sample enhanced drug entries (in production, load from comprehensive database)
        drug_data = [
            {
                "brand_name": "amoxicillin",
                "generic_name": "amoxicillin",
                "category": "antibiotic",
                "common_dosages": ["250mg", "500mg", "875mg"],
                "common_frequencies": ["twice daily", "three times daily"],
                "typical_duration": ["7-10 days", "10-14 days"],
                "routes": ["oral"],
                "contraindications": ["penicillin allergy"],
                "interactions": ["warfarin", "methotrexate"]
            },
            {
                "brand_name": "lipitor",
                "generic_name": "atorvastatin",
                "category": "statin",
                "common_dosages": ["10mg", "20mg", "40mg", "80mg"],
                "common_frequencies": ["once daily"],
                "typical_duration": ["ongoing"],
                "routes": ["oral"],
                "contraindications": ["pregnancy", "liver disease"],
                "interactions": ["digoxin", "warfarin"]
            }
        ]
        
        for drug in drug_data:
            enhanced_db[drug["brand_name"].lower()] = drug
            enhanced_db[drug["generic_name"].lower()] = drug
        
        return enhanced_db
    
    def _compile_dosage_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for dosage extraction"""
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(mg|ml|g|mcg|iu|units?|tablets?|capsules?|drops?)',
            r'(\d+(?:\.\d+)?)\s*(milligrams?|milliliters?|grams?|micrograms?)',
            r'(\d+)/(\d+)\s*(mg)',  # compound dosages like 5/500 mg
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def _compile_frequency_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for frequency extraction"""
        patterns = [
            r'(\d+)\s*times?\s*(daily|a day|per day)',
            r'(once|twice|thrice)\s*(daily|a day|per day)',
            r'every\s*(\d+)\s*(hours?|hrs?)',
            r'(q\d+h|qid|tid|bid|qd)',
            r'(morning|evening|night|bedtime)',
            r'(before|after|with)\s*(meals?|food)',
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def _compile_duration_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for duration extraction"""
        patterns = [
            r'for\s*(\d+)\s*(days?|weeks?|months?)',
            r'(\d+)\s*(days?|weeks?|months?)',
            r'until\s*(symptoms\s*improve|completed)',
            r'(ongoing|continuous|indefinitely)',
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def _compile_route_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for route extraction"""
        patterns = [
            r'(oral|orally|by mouth|po)',
            r'(intravenous|iv|intravenously)',
            r'(intramuscular|im)',
            r'(subcutaneous|sc|subcut)',
            r'(topical|apply|rub)',
            r'(inhale|inhalation|nebulize)',
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def stage_1_advanced_preprocessing(self, image_bytes: bytes) -> ProcessingResult:
        """Stage 1: Advanced image preprocessing"""
        start_time = time.time()
        
        try:
            # Load image
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            if not CV2_AVAILABLE:
                return ProcessingResult(
                    stage=ProcessingStage.IMAGE_PREPROCESSING,
                    success=False,
                    data=None,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    errors=["OpenCV not available"],
                    metadata={}
                )
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Multiple preprocessing approaches
            preprocessed_variants = []
            
            # Variant 1: Standard preprocessing
            standard = self._standard_preprocessing(cv_image.copy())
            preprocessed_variants.append(("standard", standard))
            
            # Variant 2: High contrast for faded text
            high_contrast = self._high_contrast_preprocessing(cv_image.copy())
            preprocessed_variants.append(("high_contrast", high_contrast))
            
            # Variant 3: Optimized for handwriting
            handwriting = self._handwriting_preprocessing(cv_image.copy())
            preprocessed_variants.append(("handwriting", handwriting))
            
            # Variant 4: Morphological operations for printed text
            morphological = self._morphological_preprocessing(cv_image.copy())
            preprocessed_variants.append(("morphological", morphological))
            
            return ProcessingResult(
                stage=ProcessingStage.IMAGE_PREPROCESSING,
                success=True,
                data=preprocessed_variants,
                confidence=1.0,
                processing_time=time.time() - start_time,
                errors=[],
                metadata={
                    "original_size": cv_image.shape[:2],
                    "variants_created": len(preprocessed_variants)
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                stage=ProcessingStage.IMAGE_PREPROCESSING,
                success=False,
                data=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                errors=[str(e)],
                metadata={}
            )
    
    def _standard_preprocessing(self, image):
        """Standard preprocessing pipeline"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize if too large
        h, w = gray.shape
        if max(h, w) > 2000:
            scale = 2000 / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )
        
        return thresh
    
    def _high_contrast_preprocessing(self, image):
        """High contrast preprocessing for faded text"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Binary threshold
        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _handwriting_preprocessing(self, image):
        """Specialized preprocessing for handwritten text"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gaussian blur to smooth handwriting
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Morphological gradient to enhance edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
        
        # Adaptive threshold with larger neighborhood
        thresh = cv2.adaptiveThreshold(
            gradient, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 10
        )
        
        return thresh
    
    def _morphological_preprocessing(self, image):
        """Morphological preprocessing for printed text"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Binary threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove noise with morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Close gaps in characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def stage_2_multi_engine_text_extraction(self, preprocessed_variants) -> ProcessingResult:
        """Stage 2: Multi-engine text extraction"""
        start_time = time.time()
        
        if not TESSERACT_AVAILABLE:
            return ProcessingResult(
                stage=ProcessingStage.TEXT_EXTRACTION,
                success=False,
                data=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                errors=["Tesseract not available"],
                metadata={}
            )
        
        extraction_results = []
        
        # Try different OCR configurations on each variant
        ocr_configs = [
            "--oem 3 --psm 6",  # Uniform block of text
            "--oem 3 --psm 3",  # Fully automatic page segmentation
            "--oem 3 --psm 8",  # Single word
            "--oem 3 --psm 4",  # Single column of text
        ]
        
        for variant_name, variant_image in preprocessed_variants:
            for config_idx, config in enumerate(ocr_configs):
                try:
                    # Extract text with confidence scores
                    data = pytesseract.image_to_data(
                        variant_image, config=config, output_type=pytesseract.Output.DICT
                    )
                    
                    # Combine text with confidence scores
                    words = []
                    for i in range(len(data['text'])):
                        if int(data['conf'][i]) > 30:  # Filter low confidence
                            words.append({
                                'text': data['text'][i].strip(),
                                'confidence': int(data['conf'][i]),
                                'bbox': (data['left'][i], data['top'][i], 
                                        data['width'][i], data['height'][i])
                            })
                    
                    full_text = ' '.join([w['text'] for w in words if w['text']])
                    avg_confidence = np.mean([w['confidence'] for w in words]) if words else 0
                    
                    extraction_results.append({
                        'variant': variant_name,
                        'config': config,
                        'text': full_text,
                        'words': words,
                        'confidence': avg_confidence,
                        'word_count': len(words)
                    })
                    
                except Exception as e:
                    extraction_results.append({
                        'variant': variant_name,
                        'config': config,
                        'text': '',
                        'words': [],
                        'confidence': 0,
                        'error': str(e)
                    })
        
        # Select best extraction result
        best_result = max(extraction_results, key=lambda x: x.get('confidence', 0))
        
        return ProcessingResult(
            stage=ProcessingStage.TEXT_EXTRACTION,
            success=len(extraction_results) > 0,
            data=best_result,
            confidence=best_result.get('confidence', 0) / 100,
            processing_time=time.time() - start_time,
            errors=[],
            metadata={
                'total_attempts': len(extraction_results),
                'best_variant': best_result.get('variant'),
                'best_config': best_result.get('config')
            }
        )
    
    def stage_3_medical_entity_recognition(self, text_data) -> ProcessingResult:
        """Stage 3: Advanced medical entity recognition"""
        start_time = time.time()
        
        text = text_data.get('text', '')
        words = text_data.get('words', [])
        
        # Extract medical entities using multiple approaches
        entities = {
            'medications': self._extract_medication_entities(text, words),
            'dosages': self._extract_dosage_entities(text, words),
            'frequencies': self._extract_frequency_entities(text, words),
            'durations': self._extract_duration_entities(text, words),
            'routes': self._extract_route_entities(text, words),
            'instructions': self._extract_instruction_entities(text, words)
        }
        
        return ProcessingResult(
            stage=ProcessingStage.MEDICAL_ENTITY_RECOGNITION,
            success=True,
            data=entities,
            confidence=self._calculate_entity_confidence(entities),
            processing_time=time.time() - start_time,
            errors=[],
            metadata={'entities_found': sum(len(v) for v in entities.values())}
        )
    
    def _extract_medication_entities(self, text: str, words: List[Dict]) -> List[Dict]:
        """Extract medication names using LLM-powered intelligent reasoning for challenging handwritten prescriptions"""
        medications = []
        
        # Use LLM to intelligently identify medications from the raw text
        llm_medications = self._llm_medication_extraction(text)
        
        # Combine LLM results with traditional fuzzy matching for robustness
        traditional_medications = self._traditional_medication_extraction(text, words)
        
        # Merge and deduplicate results
        all_candidates = llm_medications + traditional_medications
        medications = self._merge_medication_candidates(all_candidates)
        
        return medications
    
    def _llm_medication_extraction(self, text: str) -> List[Dict]:
        """Use LLM reasoning to extract medications from prescription text"""
        try:
            from . import gemini
            if not hasattr(gemini, 'analyze_with_gemini') or not gemini.available():
                return []
            
            # Intelligent prompt for medication extraction
            extraction_prompt = f"""
You are an expert pharmacist with 20+ years of experience reading doctor's handwritten prescriptions. 
Analyze this prescription text and identify ONLY actual medications/drugs that patients should take.

Prescription text: "{text}"

Instructions:
1. Look for medication names (generic or brand names)
2. IGNORE medical procedures, lab tests, solutions like "IV dextrose", "ORS sachets", "adequate fluid intake"
3. IGNORE instructions like "till stat", "continue", "maintain"
4. Handle illegible/partial words using pharmaceutical knowledge
5. For unclear handwriting, suggest the most likely medication based on context

For each medication found, provide:
- medication_name: The identified drug name
- confidence: How certain you are (1-100)
- reasoning: Why you think this is a medication
- alternative_names: Other possible interpretations if handwriting is unclear

Format as JSON array:
[
  {{
    "medication_name": "drug_name",
    "confidence": 85,
    "reasoning": "Found dosage indicators and matches common medication pattern",
    "alternative_names": ["alternative1", "alternative2"]
  }}
]

If no medications found, return: []
"""
            
            llm_result = gemini.analyze_with_gemini(extraction_prompt)
            
            if llm_result and isinstance(llm_result, str):
                # Parse JSON response
                import json
                import re
                
                # Extract JSON from response
                json_match = re.search(r'\[.*?\]', llm_result, re.DOTALL)
                if json_match:
                    try:
                        medications_data = json.loads(json_match.group(0))
                        
                        medications = []
                        for med_data in medications_data:
                            if isinstance(med_data, dict) and 'medication_name' in med_data:
                                medications.append({
                                    'text': med_data['medication_name'],
                                    'matched_drug': med_data['medication_name'],
                                    'confidence': med_data.get('confidence', 50) / 100,
                                    'bbox': None,
                                    'method': 'llm_extraction',
                                    'reasoning': med_data.get('reasoning', ''),
                                    'alternatives': med_data.get('alternative_names', []),
                                    'validation_passed': True
                                })
                        
                        return medications
                    except json.JSONDecodeError:
                        pass
            
        except Exception as e:
            print(f"LLM medication extraction failed: {e}")
        
        return []
    
    def _traditional_medication_extraction(self, text: str, words: List[Dict]) -> List[Dict]:
        """Fallback traditional extraction with minimal rules"""
        medications = []
        
        # Only apply minimal filtering for obvious non-medications
        obvious_non_meds = {
            'patient', 'doctor', 'clinic', 'hospital', 'date', 'time', 'address',
            'phone', 'email', 'signature', 'signed', 'refill', 'quantity'
        }
        
        for word_data in words:
            word = word_data['text'].lower().strip()
            
            # Basic filtering - only skip obvious non-medications
            if (len(word) < 3 or 
                word.isdigit() or 
                word in obvious_non_meds or
                not any(c.isalpha() for c in word)):
                continue
            
            # Use fuzzy matching with lower threshold for challenging handwriting
            match_result = fuzzy_match_drug(word)
            if match_result.get('score', 0) > 40:  # Lower threshold for handwritten text
                medications.append({
                    'text': word_data['text'],
                    'matched_drug': match_result.get('match'),
                    'confidence': match_result.get('score', 0) / 100,
                    'bbox': word_data['bbox'],
                    'method': 'traditional_fuzzy',
                    'candidates': match_result.get('candidates', []),
                    'validation_passed': True
                })
        
        return medications
    
    def _merge_medication_candidates(self, all_candidates: List[Dict]) -> List[Dict]:
        """Intelligently merge medication candidates from different methods"""
        if not all_candidates:
            return []
        
        # Group similar medications
        merged = []
        used_indices = set()
        
        for i, candidate in enumerate(all_candidates):
            if i in used_indices:
                continue
            
            # Find similar candidates
            similar_candidates = [candidate]
            candidate_name = candidate.get('matched_drug', candidate.get('text', '')).lower()
            
            for j, other in enumerate(all_candidates[i+1:], i+1):
                if j in used_indices:
                    continue
                
                other_name = other.get('matched_drug', other.get('text', '')).lower()
                
                # Check if they're the same or very similar medication
                if (candidate_name == other_name or 
                    self._are_similar_medications(candidate_name, other_name)):
                    similar_candidates.append(other)
                    used_indices.add(j)
            
            # Choose the best candidate from similar ones
            best_candidate = max(similar_candidates, key=lambda x: x.get('confidence', 0))
            
            # Enhance with information from other methods
            if len(similar_candidates) > 1:
                best_candidate['multiple_methods'] = True
                best_candidate['all_methods'] = [c.get('method', 'unknown') for c in similar_candidates]
                
                # If LLM provided reasoning, include it
                for candidate in similar_candidates:
                    if candidate.get('reasoning'):
                        best_candidate['llm_reasoning'] = candidate['reasoning']
                    if candidate.get('alternatives'):
                        best_candidate['alternatives'] = candidate['alternatives']
            
            merged.append(best_candidate)
            used_indices.add(i)
        
        # Sort by confidence
        merged.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return merged
    
    def _are_similar_medications(self, name1: str, name2: str) -> bool:
        """Check if two medication names refer to the same drug"""
        if not name1 or not name2:
            return False
        
        # Exact match
        if name1 == name2:
            return True
        
        # One contains the other
        if name1 in name2 or name2 in name1:
            return True
        
        # Check if they're common brand/generic name pairs
        common_pairs = {
            ('tylenol', 'acetaminophen'),
            ('advil', 'ibuprofen'),
            ('motrin', 'ibuprofen'),
            ('lipitor', 'atorvastatin'),
            ('zocor', 'simvastatin'),
            ('norvasc', 'amlodipine'),
            ('glucophage', 'metformin'),
            ('zoloft', 'sertraline'),
            ('prozac', 'fluoxetine'),
            ('lexapro', 'escitalopram')
        }
        
        for pair in common_pairs:
            if (name1 in pair and name2 in pair):
                return True
        
        return False
    
    def _get_word_context(self, word: str, text: str, context_length: int = 20) -> str:
        """Get context around a word in text"""
        word_index = text.lower().find(word.lower())
        if word_index == -1:
            return ""
        
        start = max(0, word_index - context_length)
        end = min(len(text), word_index + len(word) + context_length)
        return text[start:end].lower()
    
    def _extract_dosage_entities(self, text: str, words: List[Dict]) -> List[Dict]:
        """Extract dosage information"""
        dosages = []
        
        for pattern in self.dosage_patterns:
            for match in pattern.finditer(text):
                dosages.append({
                    'text': match.group(0),
                    'value': float(match.group(1)) if match.group(1).replace('.','').isdigit() else None,
                    'unit': match.group(2) if len(match.groups()) >= 2 else None,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9  # High confidence for regex matches
                })
        
        return dosages
    
    def _extract_frequency_entities(self, text: str, words: List[Dict]) -> List[Dict]:
        """Extract frequency information"""
        frequencies = []
        
        for pattern in self.frequency_patterns:
            for match in pattern.finditer(text):
                freq_text = match.group(0)
                per_day = self._convert_frequency_to_daily(freq_text)
                
                frequencies.append({
                    'text': freq_text,
                    'times_per_day': per_day,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8
                })
        
        return frequencies
    
    def _extract_duration_entities(self, text: str, words: List[Dict]) -> List[Dict]:
        """Extract duration information"""
        durations = []
        
        for pattern in self.duration_patterns:
            for match in pattern.finditer(text):
                duration_text = match.group(0)
                days = self._convert_duration_to_days(duration_text)
                
                durations.append({
                    'text': duration_text,
                    'days': days,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8
                })
        
        return durations
    
    def _extract_route_entities(self, text: str, words: List[Dict]) -> List[Dict]:
        """Extract route of administration"""
        routes = []
        
        for pattern in self.route_patterns:
            for match in pattern.finditer(text):
                routes.append({
                    'text': match.group(0),
                    'standardized': self._standardize_route(match.group(0)),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9
                })
        
        return routes
    
    def _extract_instruction_entities(self, text: str, words: List[Dict]) -> List[Dict]:
        """Extract special instructions"""
        instruction_patterns = [
            r'(before|after|with)\s*(meals?|food)',
            r'(empty stomach|full stomach)',
            r'(do not crush|swallow whole)',
            r'(refrigerate|store at room temperature)',
            r'(if needed|as needed|prn)',
        ]
        
        instructions = []
        for pattern_str in instruction_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for match in pattern.finditer(text):
                instructions.append({
                    'text': match.group(0),
                    'category': self._categorize_instruction(match.group(0)),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.7
                })
        
        return instructions
    
    def _convert_frequency_to_daily(self, freq_text: str) -> Optional[float]:
        """Convert frequency text to times per day"""
        freq_lower = freq_text.lower()
        
        # Common patterns
        if 'once' in freq_lower or '1 time' in freq_lower:
            return 1.0
        elif 'twice' in freq_lower or '2 time' in freq_lower:
            return 2.0
        elif 'thrice' in freq_lower or '3 time' in freq_lower:
            return 3.0
        elif 'four time' in freq_lower or '4 time' in freq_lower:
            return 4.0
        
        # Extract number + times pattern
        match = re.search(r'(\d+)\s*times?', freq_lower)
        if match:
            return float(match.group(1))
        
        # Every X hours
        match = re.search(r'every\s*(\d+)\s*hours?', freq_lower)
        if match:
            hours = int(match.group(1))
            return 24.0 / hours if hours > 0 else None
        
        return None
    
    def _convert_duration_to_days(self, duration_text: str) -> Optional[int]:
        """Convert duration text to days"""
        duration_lower = duration_text.lower()
        
        # Extract number and unit
        match = re.search(r'(\d+)\s*(days?|weeks?|months?)', duration_lower)
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            
            if 'day' in unit:
                return value
            elif 'week' in unit:
                return value * 7
            elif 'month' in unit:
                return value * 30  # Approximate
        
        return None
    
    def _standardize_route(self, route_text: str) -> str:
        """Standardize route of administration"""
        route_lower = route_text.lower()
        
        if any(term in route_lower for term in ['oral', 'by mouth', 'po']):
            return 'oral'
        elif any(term in route_lower for term in ['iv', 'intravenous']):
            return 'intravenous'
        elif any(term in route_lower for term in ['im', 'intramuscular']):
            return 'intramuscular'
        elif any(term in route_lower for term in ['sc', 'subcutaneous']):
            return 'subcutaneous'
        elif any(term in route_lower for term in ['topical', 'apply']):
            return 'topical'
        
        return route_text
    
    def _categorize_instruction(self, instruction_text: str) -> str:
        """Categorize special instructions"""
        instruction_lower = instruction_text.lower()
        
        if any(term in instruction_lower for term in ['meal', 'food']):
            return 'food_related'
        elif any(term in instruction_lower for term in ['crush', 'swallow']):
            return 'administration'
        elif any(term in instruction_lower for term in ['store', 'refrigerate']):
            return 'storage'
        elif any(term in instruction_lower for term in ['needed', 'prn']):
            return 'as_needed'
        
        return 'general'
    
    def _calculate_entity_confidence(self, entities: Dict) -> float:
        """Calculate overall confidence for entity extraction"""
        total_entities = sum(len(v) for v in entities.values())
        if total_entities == 0:
            return 0.0
        
        total_confidence = 0
        for entity_list in entities.values():
            for entity in entity_list:
                total_confidence += entity.get('confidence', 0)
        
        return total_confidence / total_entities
    
    def analyze_prescription_advanced(self, image_bytes: bytes) -> Dict[str, Any]:
        """Main advanced analysis function"""
        start_time = time.time()
        
        try:
            # Stage 1: Advanced preprocessing
            stage1_result = self.stage_1_advanced_preprocessing(image_bytes)
            self.processing_stages.append(stage1_result)
            
            if not stage1_result.success:
                return self._create_error_response("Preprocessing failed", self.processing_stages)
            
            # Stage 2: Multi-engine text extraction
            stage2_result = self.stage_2_multi_engine_text_extraction(stage1_result.data)
            self.processing_stages.append(stage2_result)
            
            if not stage2_result.success:
                return self._create_error_response("Text extraction failed", self.processing_stages)
            
            # Stage 3: Medical entity recognition
            stage3_result = self.stage_3_medical_entity_recognition(stage2_result.data)
            self.processing_stages.append(stage3_result)
            
            # Combine results and create final response
            total_time = time.time() - start_time
            
            return {
                "status": "success",
                "method": "advanced_ocr",
                "processing_time": total_time,
                "stages": [
                    {
                        "stage": stage.stage.value,
                        "success": stage.success,
                        "confidence": stage.confidence,
                        "processing_time": stage.processing_time,
                        "errors": stage.errors,
                        "metadata": stage.metadata
                    }
                    for stage in self.processing_stages
                ],
                "extracted_text": stage2_result.data.get('text', ''),
                "entities": stage3_result.data,
                "overall_confidence": sum(s.confidence for s in self.processing_stages) / len(self.processing_stages),
                "medications_found": len(stage3_result.data.get('medications', [])),
                "advanced_features": {
                    "multi_variant_preprocessing": True,
                    "medical_entity_recognition": True,
                    "confidence_scoring": True,
                    "bounding_box_detection": True
                }
            }
            
        except Exception as e:
            return self._create_error_response(f"Analysis failed: {str(e)}", self.processing_stages)
    
    def _create_error_response(self, error_message: str, stages: List[ProcessingResult]) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "status": "error",
            "error": error_message,
            "method": "advanced_ocr",
            "processing_time": sum(s.processing_time for s in stages),
            "stages": [
                {
                    "stage": s.stage.value,
                    "success": s.success,
                    "errors": s.errors
                }
                for s in stages
            ]
        }