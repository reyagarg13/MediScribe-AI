"""
Custom TrOCR model integration for MediScribe AI Image Analysis.
Integrates the fine-tuned prescription OCR model trained on Kaggle dataset.
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Dict, Any
import cv2

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    print("Warning: transformers not available, custom TrOCR disabled")

class CustomPrescriptionOCR:
    """Custom TrOCR model for prescription text extraction."""
    
    def __init__(self, model_path: str = None):
        if not TROCR_AVAILABLE:
            raise ImportError("transformers library not available")
            
        # Default model path
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "trocr_gemini_teacher_demo")
        
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.loaded = False
        
        print(f"🤖 Custom TrOCR initialized - Device: {self.device}")
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    
    def load_model(self):
        """Load the custom trained TrOCR model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            print(f"📥 Loading custom TrOCR model from {self.model_path}")
            
            # Load processor and model
            self.processor = TrOCRProcessor.from_pretrained(self.model_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            self.loaded = True
            print("✅ Custom TrOCR model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load custom TrOCR model: {e}")
            raise
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """Preprocess image for better OCR results."""
        try:
            # Load image if it's a path
            if isinstance(image, (str, Path)):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply same preprocessing as training
            img_array = np.array(image)
            
            # Apply CLAHE for better contrast (medical documents often have poor contrast)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return Image.fromarray(enhanced)
            
        except Exception as e:
            print(f"⚠️  Error preprocessing image: {e}")
            # Return original image if preprocessing fails
            if isinstance(image, Image.Image):
                return image
            else:
                return Image.open(image) if isinstance(image, (str, Path)) else Image.fromarray(image)
    
    def extract_text(self, image: Union[str, Path, Image.Image, np.ndarray], 
                    max_length: int = 512, num_beams: int = 3) -> Dict[str, Any]:
        """Extract prescription text using custom TrOCR model."""
        
        if not self.loaded:
            self.load_model()
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Process with TrOCR processor
            inputs = self.processor(
                images=processed_image,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate prediction
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs['pixel_values'],
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False,
                    temperature=1.0
                )
            
            # Decode prediction
            predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Return structured result
            result = {
                "extracted_text": predicted_text,
                "method": "custom_trocr",
                "model": "trocr_gemini_teacher_demo", 
                "device": self.device,
                "confidence": 0.85,  # Placeholder confidence
                "preprocessing_applied": True,
                "success": True
            }
            
            print(f"✅ Custom TrOCR extraction successful: {len(predicted_text)} characters")
            return result
            
        except Exception as e:
            print(f"❌ Custom TrOCR extraction failed: {e}")
            return {
                "extracted_text": "",
                "method": "custom_trocr",
                "model": "trocr_gemini_teacher_demo",
                "device": self.device,
                "confidence": 0.0,
                "preprocessing_applied": False,
                "success": False,
                "error": str(e)
            }

# Global instance
_custom_trocr = None

def get_custom_trocr_instance():
    """Get or create the global custom TrOCR instance."""
    global _custom_trocr
    if _custom_trocr is None and TROCR_AVAILABLE:
        try:
            _custom_trocr = CustomPrescriptionOCR()
        except Exception as e:
            print(f"Failed to initialize custom TrOCR: {e}")
            _custom_trocr = None
    return _custom_trocr

def analyze_prescription_with_custom_trocr(image_path: str) -> Dict[str, Any]:
    """
    Analyze prescription image with custom TrOCR model.
    Main function to be called by image_analysis.py
    """
    
    if not TROCR_AVAILABLE:
        return {
            "extracted_text": "",
            "method": "custom_trocr",
            "model": "unavailable",
            "success": False,
            "error": "transformers library not available"
        }
    
    try:
        # Get TrOCR instance
        custom_trocr = get_custom_trocr_instance()
        
        if custom_trocr is None:
            return {
                "extracted_text": "",
                "method": "custom_trocr", 
                "model": "initialization_failed",
                "success": False,
                "error": "Failed to initialize custom TrOCR model"
            }
        
        # Extract text
        result = custom_trocr.extract_text(image_path)
        
        # Add metadata
        result.update({
            "image_path": image_path,
            "timestamp": torch.cuda.current_device() if torch.cuda.is_available() else "cpu_timestamp"
        })
        
        return result
        
    except Exception as e:
        print(f"❌ Custom TrOCR analysis failed: {e}")
        return {
            "extracted_text": "",
            "method": "custom_trocr",
            "model": "error",
            "success": False,
            "error": str(e)
        }