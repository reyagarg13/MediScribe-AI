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

router = APIRouter()

# Dummy model for demonstration (replace with real model)
class DummyImageModel:
    def predict(self, image):
        # Simulate prediction
        return {"diagnosis": "No abnormality detected", "confidence": 0.98}

model = DummyImageModel()

@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run model prediction on original (dummy for now)
    result = model.predict(image)

    # Enhance image for debugging/visualization
    try:
        enhanced_buf, metadata = enhance_image(contents)
        enhanced_b64 = base64.b64encode(enhanced_buf.getvalue()).decode("utf-8")
    except Exception as e:
        enhanced_b64 = None
        metadata = {"error": str(e)}

    response: Dict[str, Any] = {
        "model": result,
        "enhanced_image_base64": enhanced_b64,
        "preprocessing": metadata,
    }

    return JSONResponse(content=response)


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
        
        # If no medications found through entities, fall back to original method
        if not medications:
            from .prescription_ocr import analyze_prescription_image
            fallback_result = analyze_prescription_image(contents)
            medications = fallback_result.get("medications", [])
        
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
            
            # Validate each medication
            validation_results = []
            for med in medications:
                validation = validator.validate_medication(
                    med, medications, patient_info, 
                    ValidationLevel.COMPREHENSIVE if validation_level == "comprehensive" else ValidationLevel.STANDARD
                )
                validation_results.append(validation)
            
            # Generate safety report
            safety_report = validator.generate_safety_report(medications, patient_info)
        
        # Combine all results
        total_time = time.time() - start_time
        
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
                "processing_stages": ocr_result.get("stages", []),
                "overall_confidence": ocr_result.get("overall_confidence", 0),
                "advanced_features": ocr_result.get("advanced_features", {})
            },
            
            # Extracted Entities
            "entities": entities,
            
            # Medications
            "medications": medications,
            "medications_found": len(medications),
            
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
