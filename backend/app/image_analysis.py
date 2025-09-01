from fastapi import UploadFile, File, APIRouter
from fastapi.responses import JSONResponse
from PIL import Image
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
