from fastapi import UploadFile, File, APIRouter
from fastapi.responses import JSONResponse
from PIL import Image
from typing import Optional
import io
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except Exception:
    transforms = None
    TORCHVISION_AVAILABLE = False
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    np = None
    CV2_AVAILABLE = False
import base64
from typing import Dict, Any
from .prescription_ocr import analyze_prescription_image
from .custom_trocr import analyze_prescription_with_custom_trocr, TROCR_AVAILABLE

router = APIRouter()

# Dummy model for demonstration (replace with real model)
class DummyImageModel:
    def predict(self, image):
        # Simulate prediction
        return {"diagnosis": "No abnormality detected", "confidence": 0.98}

model = DummyImageModel()

@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Image Analysis - Showcase all implemented image processing techniques.
    Demonstrates: Sampling & Quantization, CLAHE, Gaussian Blur, Edge Detection,
    Morphological Operations, Histogram Equalization, Noise Reduction
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Process image with all techniques and return visual results
    try:
        results = showcase_image_processing_techniques(contents)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "error": str(e),
            "message": "Failed to showcase image processing techniques"
        }, status_code=500)


@router.post("/prescription-ocr")
async def prescription_ocr(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        result = analyze_prescription_image(contents)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/prescription-ocr-advanced")
async def prescription_ocr_advanced(
    file: UploadFile = File(...),
    validation_level: str = "standard",
    include_safety_report: bool = True,
    patient_age: Optional[int] = None,
    patient_conditions: Optional[str] = None,
    is_pregnant: Optional[bool] = None
):
    """Advanced prescription OCR with comprehensive validation and safety checking"""
    try:
        from .advanced_prescription_ocr import AdvancedPrescriptionOCR
        from .medical_validation import MedicalValidator, ValidationLevel
        import time
        
        start_time = time.time()
        contents = await file.read()
        
        # Initialize advanced OCR system
        advanced_ocr = AdvancedPrescriptionOCR()
        
        # Run advanced analysis
        ocr_result = advanced_ocr.analyze_prescription_advanced(contents)
        
        if ocr_result["status"] != "success":
            return JSONResponse(content=ocr_result, status_code=500)
        
        # Extract medications from entities
        medications = []
        entities = ocr_result.get("entities", {})
        
        # Combine entity information into medication objects
        med_entities = entities.get("medications", [])
        dosage_entities = entities.get("dosages", [])
        frequency_entities = entities.get("frequencies", [])
        duration_entities = entities.get("durations", [])
        route_entities = entities.get("routes", [])
        
        for med_entity in med_entities:
            # Find associated dosage, frequency, etc. by proximity
            medication = {
                "name_candidate": med_entity["text"],
                "matched_name": med_entity.get("matched_drug"),
                "confidence": med_entity["confidence"],
                "enhanced_info": med_entity.get("enhanced_info", {}),
                "dosage": None,
                "frequency": None,
                "duration": None,
                "route": None,
                "bounding_box": med_entity.get("bbox")
            }
            
            # Find closest dosage (simple proximity matching)
            for dosage in dosage_entities:
                medication["dosage"] = dosage["text"]
                break  # Take first for now - could be improved with better proximity matching
            
            # Find closest frequency
            for frequency in frequency_entities:
                medication["frequency"] = frequency["text"]
                break
            
            # Find closest duration
            for duration in duration_entities:
                medication["duration"] = duration["text"]
                break
            
            # Find route
            for route in route_entities:
                medication["route"] = route["standardized"]
                break
            
            medications.append(medication)
        
        # Get prescription header information and fallback medications if needed
        prescription_header = {}
        
        if not medications:
            from .prescription_ocr import analyze_prescription_image
            fallback_result = analyze_prescription_image(contents)
            medications = fallback_result.get("medications", [])
            prescription_header = fallback_result.get("prescription_header", {})
        else:
            # Extract header from entities if available
            header_entities = entities.get("headers", {})
            prescription_header = header_entities
        
        # Medical validation if requested
        validation_results = None
        safety_report = None
        
        if include_safety_report and medications:
            validator = MedicalValidator()
            
            # Prepare patient info
            patient_info = {}
            if patient_age is not None:
                patient_info["age"] = patient_age
            if is_pregnant is not None:
                patient_info["is_pregnant"] = is_pregnant
            if patient_conditions:
                patient_info["conditions"] = [c.strip() for c in patient_conditions.split(",")]
            
            # Use batch validation for better performance
            validation_level_enum = ValidationLevel.COMPREHENSIVE if validation_level == "comprehensive" else ValidationLevel.STANDARD
            validation_results = validator.batch_validate_medications(medications, patient_info, validation_level_enum)
            
            # Generate safety report using batch validation results
            safety_report = validator.generate_safety_report_from_validations(validation_results, medications)
        
        # Combine all results
        total_time = time.time() - start_time
        
        # Calculate overall confidence from medication matches
        if medications:
            medication_scores = [med.get("match_score", 0) for med in medications if med.get("match_score")]
            overall_confidence = sum(medication_scores) / len(medication_scores) if medication_scores else 75
        else:
            overall_confidence = 50
        
        return JSONResponse(content={
            "status": "success",
            "method": "advanced_ocr_with_validation",
            "processing_time": total_time,
            "file_info": {
                "filename": file.filename,
                "size": len(contents),
                "content_type": file.content_type
            },
            
            # OCR Results
            "ocr_analysis": {
                "extracted_text": ocr_result.get("extracted_text", ""),
                "overall_confidence": round(overall_confidence, 1),
                "advanced_features": ocr_result.get("advanced_features", {})
            },
            
            # Extracted Entities
            "entities": entities,
            
            # Medications
            "medications": medications,
            "medications_found": len(medications),
            
            # Prescription Header Information  
            "prescription_header": prescription_header,
            
            # Medical Validation
            "validation_results": validation_results,
            "safety_report": safety_report,
            
            # Enhanced Metadata
            "metadata": {
                "validation_level": validation_level,
                "patient_specific_validation": bool(patient_age or patient_conditions or is_pregnant),
                "safety_analysis_included": include_safety_report,
                "total_entities_found": sum(len(v) for v in entities.values()) if entities else 0,
                "processing_method": "multi_stage_advanced_ocr"
            }
        })
        
    except Exception as e:
        import traceback
        return JSONResponse(content={
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "method": "advanced_ocr_with_validation"
        }, status_code=500)


@router.post("/prescription-custom-trocr")
async def prescription_custom_trocr(file: UploadFile = File(...)):
    """
    Custom TrOCR prescription analysis using the fine-tuned model.
    This is your trained model on Kaggle prescription dataset!
    """
    
    if not TROCR_AVAILABLE:
        return JSONResponse(content={
            "status": "error",
            "error": "Custom TrOCR not available - transformers library missing",
            "method": "custom_trocr"
        }, status_code=500)
    
    try:
        import time
        import tempfile
        import os
        
        start_time = time.time()
        contents = await file.read()
        
        # Save uploaded file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        
        try:
            # Run custom TrOCR analysis
            result = analyze_prescription_with_custom_trocr(tmp_file_path)
            
            # Add processing metadata
            total_time = time.time() - start_time
            result.update({
                "status": "success" if result["success"] else "error",
                "processing_time": round(total_time, 3),
                "file_info": {
                    "filename": file.filename,
                    "size": len(contents),
                    "content_type": file.content_type
                },
                "model_info": {
                    "type": "custom_trained_trocr",
                    "training_dataset": "kaggle_prescription_images",
                    "training_samples": 20,
                    "epochs": 30,
                    "base_model": "microsoft/trocr-base-handwritten"
                }
            })
            
            return JSONResponse(content=result)
            
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
    except Exception as e:
        import traceback
        return JSONResponse(content={
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "method": "custom_trocr"
        }, status_code=500)


def showcase_image_processing_techniques(image_bytes: bytes) -> Dict[str, Any]:
    """
    Showcase all implemented image processing techniques with visual outputs.
    Returns base64 encoded results for each technique.
    """
    if not CV2_AVAILABLE or cv2 is None or np is None:
        return {
            "status": "error",
            "error": "OpenCV not available - image processing techniques cannot be demonstrated"
        }
    
    try:
        # Load original image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Resize for processing if too large
        h, w = original_img.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            original_img = cv2.resize(original_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        results = {
            "status": "success",
            "original_size": {"width": w, "height": h},
            "techniques": {}
        }
        
        # Convert original to base64
        results["original_image"] = image_to_base64(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        
        # 1. Sampling & Quantization
        try:
            quantized_img = apply_quantization(original_img)
            results["techniques"]["sampling_quantization"] = {
                "title": "Sampling & Quantization",
                "description": "Reduces color levels to simulate lower bit-depth images",
                "image": image_to_base64(cv2.cvtColor(quantized_img, cv2.COLOR_BGR2RGB)),
                "success": True
            }
        except Exception as e:
            results["techniques"]["sampling_quantization"] = {
                "title": "Sampling & Quantization", 
                "error": str(e),
                "success": False
            }
        
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        try:
            clahe_img = apply_clahe(original_img)
            results["techniques"]["clahe"] = {
                "title": "CLAHE (Contrast Limited Adaptive Histogram Equalization)",
                "description": "Enhances contrast while preventing noise amplification",
                "image": image_to_base64(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)),
                "success": True
            }
        except Exception as e:
            results["techniques"]["clahe"] = {
                "title": "CLAHE",
                "error": str(e),
                "success": False
            }
        
        # 3. Gaussian Blur
        try:
            blur_img = apply_gaussian_blur(original_img)
            results["techniques"]["gaussian_blur"] = {
                "title": "Gaussian Blur",
                "description": "Smooths image and reduces noise using Gaussian kernel",
                "image": image_to_base64(cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)),
                "success": True
            }
        except Exception as e:
            results["techniques"]["gaussian_blur"] = {
                "title": "Gaussian Blur",
                "error": str(e),
                "success": False
            }
        
        # 4. Edge Detection
        try:
            edges_img = apply_edge_detection(original_img)
            results["techniques"]["edge_detection"] = {
                "title": "Edge Detection (Canny)",
                "description": "Detects edges and boundaries in the image",
                "image": image_to_base64(edges_img),
                "success": True
            }
        except Exception as e:
            results["techniques"]["edge_detection"] = {
                "title": "Edge Detection",
                "error": str(e),
                "success": False
            }
        
        # 5. Morphological Operations
        try:
            morph_img = apply_morphological_operations(original_img)
            results["techniques"]["morphological_operations"] = {
                "title": "Morphological Operations",
                "description": "Erosion and dilation to clean up image structure",
                "image": image_to_base64(cv2.cvtColor(morph_img, cv2.COLOR_BGR2RGB)),
                "success": True
            }
        except Exception as e:
            results["techniques"]["morphological_operations"] = {
                "title": "Morphological Operations",
                "error": str(e),
                "success": False
            }
        
        # 6. Histogram Equalization
        try:
            hist_eq_img = apply_histogram_equalization(original_img)
            results["techniques"]["histogram_equalization"] = {
                "title": "Histogram Equalization",
                "description": "Improves contrast by spreading out intensity distribution",
                "image": image_to_base64(cv2.cvtColor(hist_eq_img, cv2.COLOR_BGR2RGB)),
                "success": True
            }
        except Exception as e:
            results["techniques"]["histogram_equalization"] = {
                "title": "Histogram Equalization",
                "error": str(e),
                "success": False
            }
        
        # 7. Noise Reduction
        try:
            denoised_img = apply_noise_reduction(original_img)
            results["techniques"]["noise_reduction"] = {
                "title": "Noise Reduction",
                "description": "Removes noise while preserving important details",
                "image": image_to_base64(cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)),
                "success": True
            }
        except Exception as e:
            results["techniques"]["noise_reduction"] = {
                "title": "Noise Reduction",
                "error": str(e),
                "success": False
            }
        
        return results
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to process image with showcase techniques"
        }

def image_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy image array to base64 string."""
    if len(img_array.shape) == 3:
        # Color image
        pil_img = Image.fromarray(img_array)
    else:
        # Grayscale image
        pil_img = Image.fromarray(img_array, mode='L')
    
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def apply_quantization(img: np.ndarray, levels: int = 32) -> np.ndarray:
    """Apply color quantization to reduce bit depth."""
    # Quantize each channel
    quantized = img.copy()
    quantized = (quantized // (256 // levels)) * (256 // levels)
    return quantized

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE for contrast enhancement."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def apply_gaussian_blur(img: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur for noise reduction."""
    return cv2.GaussianBlur(img, (15, 15), 0)

def apply_edge_detection(img: np.ndarray) -> np.ndarray:
    """Apply Canny edge detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges  # Return grayscale edges

def apply_morphological_operations(img: np.ndarray) -> np.ndarray:
    """Apply morphological operations (erosion followed by dilation)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5), np.uint8)
    
    # Erosion followed by dilation (opening)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    
    # Convert back to BGR for consistency
    return cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)

def apply_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Apply histogram equalization in LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply histogram equalization to L channel
    l_eq = cv2.equalizeHist(l)
    
    # Merge channels and convert back
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

def apply_noise_reduction(img: np.ndarray) -> np.ndarray:
    """Apply noise reduction using Non-local Means Denoising."""
    return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

def enhance_image(image_bytes: bytes):
    """Return a BytesIO buffer with enhanced image and metadata dict.

    Enhancements: resize if large, denoise, CLAHE (contrast limited adaptive histogram equalization),
    unsharp mask for edge enhancement.
    """
    buf = io.BytesIO()
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    metadata: Dict[str, Any] = {}

    if CV2_AVAILABLE and cv2 is not None and np is not None:
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # Resize to reasonable max dimension for processing
        h, w = img.shape[:2]
        metadata["original_size"] = {"width": w, "height": h}
        max_dim = 1600
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            metadata["resized_to"] = img.shape[:2][::-1]

        # Convert to LAB and apply CLAHE on L channel for contrast enhancement
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            metadata["clahe_applied"] = True

            # Denoise
            img_denoised = cv2.fastNlMeansDenoisingColored(img_clahe, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
            metadata["denoised"] = True

            # Unsharp mask (sharpen)
            gaussian = cv2.GaussianBlur(img_denoised, (0, 0), sigmaX=3)
            img_sharp = cv2.addWeighted(img_denoised, 1.5, gaussian, -0.5, 0)
            metadata["sharpened"] = True

            # Convert back to PIL and save to buffer
            enhanced = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2RGB)
            pil_out = Image.fromarray(enhanced)
            pil_out.save(buf, format="JPEG", quality=90)
            buf.seek(0)
            return buf, metadata
        except Exception as e:
            # Fallback to simple save below
            metadata["cv2_error"] = str(e)

    # Fallback when cv2 not available or enhancement failed: save the original PIL image
    pil.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    metadata.setdefault("note", "cv2 not available or enhancement failed; returning original image")
    return buf, metadata
