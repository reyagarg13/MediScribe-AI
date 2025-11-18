#!/usr/bin/env python3
"""
Enhanced inference script for TrOCR prescription OCR.
Supports batch processing and integration with web APIs.
"""
import argparse
import torch
import json
import time
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pathlib import Path
import cv2


class PrescriptionOCRModel:
    """Wrapper class for trained TrOCR prescription OCR model."""
    
    def __init__(self, model_dir: str, device: str = "auto"):
        """Initialize the model."""
        print(f"Loading TrOCR model from {model_dir}")
        
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Verify GPU availability
        if self.device == "cuda":
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Load processor and model
        self.processor = TrOCRProcessor.from_pretrained(model_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Load training info if available
        training_info_path = self.model_dir / "training_info.json"
        self.training_info = {}
        if training_info_path.exists():
            with open(training_info_path) as f:
                self.training_info = json.load(f)
        
        print(f"Model loaded successfully!")
        if self.training_info:
            print(f"Model trained with: {self.training_info.get('args', {}).get('model_name', 'Unknown base model')}")
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """Preprocess image for better OCR results."""
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
        
        # CLAHE for better contrast
        try:
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_img = cv2.merge((cl, a, b))
            enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2RGB)
            
            # Light denoising
            enhanced_img = cv2.medianBlur(enhanced_img, 3)
            
            return Image.fromarray(enhanced_img)
        except:
            # Fallback to original image if preprocessing fails
            return image
    
    def predict_single(
        self, 
        image: Union[str, Path, Image.Image, np.ndarray],
        max_length: int = 256,
        num_beams: int = 4,
        early_stopping: bool = True
    ) -> Dict[str, Any]:
        """Predict text from a single image."""
        start_time = time.time()
        
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
                early_stopping=early_stopping,
                do_sample=False,
                temperature=1.0
            )
        
        # Decode prediction
        predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        processing_time = time.time() - start_time
        
        return {
            "text": predicted_text,
            "processing_time": processing_time,
            "model_info": {
                "model_dir": str(self.model_dir),
                "device": self.device,
                "generation_params": {
                    "max_length": max_length,
                    "num_beams": num_beams,
                    "early_stopping": early_stopping
                }
            }
        }
    
    def predict_batch(
        self, 
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 4,
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """Predict text from multiple images in batches."""
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_results = []
            
            for image in batch_images:
                result = self.predict_single(image, **generation_kwargs)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results


# Simple function for backward compatibility
def infer(image_path, model_dir):
    """Simple inference function for backward compatibility."""
    model = PrescriptionOCRModel(model_dir)
    result = model.predict_single(image_path)
    return result["text"]


def main():
    parser = argparse.ArgumentParser(description="TrOCR Prescription OCR Inference")
    
    # Required arguments
    parser.add_argument("--model_dir", required=True, help="Path to trained model directory")
    parser.add_argument("--image", help="Path to single image for inference")
    parser.add_argument("--images_dir", help="Directory containing multiple images")
    parser.add_argument("--image_list", help="Text file with image paths (one per line)")
    
    # Optional arguments
    parser.add_argument("--output", help="Output file for results (JSON format)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for multiple images")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    
    # Generation parameters
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--no_early_stopping", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize model
    model = PrescriptionOCRModel(args.model_dir, device=args.device)
    
    # Collect images to process
    images_to_process = []
    
    if args.image:
        images_to_process = [args.image]
    elif args.images_dir:
        images_dir = Path(args.images_dir)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            images_to_process.extend(images_dir.glob(ext))
            images_to_process.extend(images_dir.glob(ext.upper()))
    elif args.image_list:
        with open(args.image_list) as f:
            images_to_process = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Must specify --image, --images_dir, or --image_list")
    
    if not images_to_process:
        print("No images found to process!")
        return
    
    print(f"Processing {len(images_to_process)} image(s)...")
    
    # Run inference
    generation_kwargs = {
        "max_length": args.max_length,
        "num_beams": args.num_beams,
        "early_stopping": not args.no_early_stopping
    }
    
    if len(images_to_process) == 1:
        # Single image
        result = model.predict_single(images_to_process[0], **generation_kwargs)
        print(f"\nResult for {images_to_process[0]}:")
        print(f"Text: {result['text']}")
        print(f"Processing time: {result['processing_time']:.3f}s")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"results": [result]}, f, indent=2)
            print(f"Result saved to {args.output}")
    
    else:
        # Multiple images
        results = model.predict_batch(images_to_process, args.batch_size, **generation_kwargs)
        
        total_time = sum(r['processing_time'] for r in results)
        avg_time = total_time / len(results)
        
        print(f"\nProcessed {len(results)} images:")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average time per image: {avg_time:.3f}s")
        
        # Print individual results
        for i, result in enumerate(results):
            print(f"\n{i+1}. {images_to_process[i]}")
            print(f"   Text: {result['text']}")
            print(f"   Time: {result['processing_time']:.3f}s")
        
        # Save results
        if args.output:
            output_data = {
                "summary": {
                    "total_images": len(results),
                    "total_time": total_time,
                    "average_time": avg_time
                },
                "results": results
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
