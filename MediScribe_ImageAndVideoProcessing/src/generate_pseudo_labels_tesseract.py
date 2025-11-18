#!/usr/bin/env python3
"""
Alternative pseudo-labeling script using Tesseract OCR instead of EasyOCR.
Use this if EasyOCR has dependency issues.
"""
import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

def preprocess_image_for_ocr(image_path):
    """Preprocess image for better OCR results."""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.medianBlur(enhanced, 3)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None

def main(images_dir, out_file, use_tesseract=True):
    try:
        if use_tesseract:
            import pytesseract
            # Set Tesseract path explicitly for Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            print("Using Tesseract OCR")
        else:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False)
            print("Using EasyOCR")
    except ImportError as e:
        print(f"OCR library not available: {e}")
        print("Please install: pip install pytesseract")
        print("And install Tesseract binary from: https://github.com/tesseract-ocr/tesseract")
        return
    
    results = []
    image_files = []
    
    # Collect all image files
    for fn in os.listdir(images_dir):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
            image_files.append(fn)
    
    print(f"Found {len(image_files)} images to process")
    
    for fn in tqdm(sorted(image_files), desc="Processing images"):
        path = os.path.join(images_dir, fn)
        
        try:
            if use_tesseract:
                # Preprocess image
                processed_img = preprocess_image_for_ocr(path)
                if processed_img is None:
                    continue
                
                # Run Tesseract
                config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,()/-: '
                text = pytesseract.image_to_string(processed_img, config=config)
                
                # Simple confidence estimation (length-based)
                confidence = min(0.9, len(text.strip()) / 100.0) if text.strip() else 0.1
                
            else:
                # Use EasyOCR
                res = reader.readtext(path, detail=1)
                text = "\n".join([r[1] for r in res]) if res else ""
                confidence = float(sum([r[2] for r in res]) / max(1, len(res))) if res else 0.0
            
            # Clean up text
            text = text.strip()
            if len(text) < 3:  # Very short text likely OCR error
                text = ""
                confidence = 0.1
            
            results.append({
                "image_path": path,
                "text": text,
                "conf": confidence
            })
            
        except Exception as e:
            print(f'OCR failed for {path} -> {e}')
            results.append({
                "image_path": path,
                "text": "",
                "conf": 0.0
            })
    
    # Save results
    with open(out_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f'Saved {len(results)} pseudo-labels to {out_file}')
    
    # Print some statistics
    valid_results = [r for r in results if r['conf'] > 0.1]
    avg_conf = sum(r['conf'] for r in valid_results) / len(valid_results) if valid_results else 0
    print(f'Valid results: {len(valid_results)}/{len(results)}')
    print(f'Average confidence: {avg_conf:.3f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--out', dest='out_file', required=True)
    parser.add_argument('--use_easyocr', action='store_true', help='Use EasyOCR instead of Tesseract')
    args = parser.parse_args()
    
    main(args.images_dir, args.out_file, use_tesseract=not args.use_easyocr)