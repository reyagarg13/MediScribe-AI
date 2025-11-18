#!/usr/bin/env python3
"""
Generate high-quality pseudo-labels using Gemini Vision API for prescription images.
This creates proper training data for TrOCR fine-tuning.
"""

import os
import json
import time
import argparse
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def setup_gemini():
    """Setup Gemini API with key + model rotation for cost efficiency."""
    # Working API key + model combinations (from test results)
    api_configs = [
        ("AIzaSyBSj34wX1CsJuA74jsw-5ChbrUKCuoFNhE", "gemini-2.5-pro", "GEMINI 1.5 FLASH"),
        ("AIzaSyCGY51vvMSYwX_pY_EcLYWYXBiPMmguY64", "gemini-2.5-pro", "GEMINI 2.0 FLASH EXP"),
        ("AIzaSyAwDJSZ2dpGbnJq4NnAdHjUBdbVXIwiVIw", "gemini-2.5-flash", "GEMINI 2.5 FLASH"),  # Use cheaper flash
        ("AIzaSyCHF2oxoG43oN1vyqDCcUmKOU_Yxr-vEFI", "gemini-2.5-pro", "GEMINI 2.5 PRO"),
        ("AIzaSyBzx3IsaPl6NM-7R_qJ69Rf8hzwsci-Pf4", "gemini-2.5-flash", "PALLAV MAIN"),  # Use cheaper flash
        ("AIzaSyB9078EC0wXNRwtknr3IigQGU8US-xq4Oo", "gemini-2.5-flash", "PYTHON ACCOUNT"),  # Use cheaper flash
        ("AIzaSyBQtZ1eNBAPDMNBoKSUR7kYIQFncQvfoZo", "gemini-2.5-pro", "PALLAV FOZO")
    ]
    
    # Try each configuration until one works
    for i, (api_key, model_name, name) in enumerate(api_configs):
        try:
            print(f"🔑 Trying {name} + {model_name}...")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            print(f"✅ {name} + {model_name} working!")
            return model, api_key, model_name, i
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            continue
    
    raise ValueError("All API configurations exhausted or invalid")

def create_prescription_prompt():
    """Create an optimized prompt for prescription OCR."""
    return """
You are an expert medical transcriptionist. Please extract ALL text from this prescription image with high accuracy.

Focus on:
- Doctor name and credentials
- Patient information
- Medication names (brand/generic)
- Dosages and strengths
- Frequency instructions (e.g., "twice daily", "BID", "PRN")
- Duration (e.g., "for 7 days", "x14 days")
- Special instructions
- Pharmacy instructions
- Date and signature

Output ONLY the extracted text in a clean, readable format. Do not add explanations or formatting.
If text is unclear, make your best interpretation based on medical context.
Preserve the original structure and line breaks where possible.

Example format:
Dr. John Smith, MD
Patient: Mary Johnson
DOB: 01/15/1980

Rx:
Amoxicillin 500mg
Take 1 capsule twice daily for 7 days
Quantity: #14
Refills: 0

Date: 03/15/2024
Signature: [Doctor signature]
"""

def extract_text_with_gemini(model, image_path, prompt, max_retries=3, api_configs=None, config_index=0):
    """Extract text from prescription image using Gemini Vision."""
    try:
        # Load and prepare image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Retry logic for API stability
        for attempt in range(max_retries):
            try:
                response = model.generate_content([prompt, image])
                
                if response.text:
                    return response.text.strip(), 1.0  # High confidence for Gemini
                else:
                    print(f"Warning: Empty response for {image_path}")
                    return "", 0.0
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for {image_path}: {e}")
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    print(f"Failed to process {image_path} after {max_retries} attempts: {e}")
                    return "", 0.0
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return "", 0.0

def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels using Gemini Vision")
    parser.add_argument("--images_dir", required=True, help="Directory containing prescription images")
    parser.add_argument("--output_file", required=True, help="Output JSONL file for labels")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("--batch_delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    
    args = parser.parse_args()
    
    # Setup
    print("🚀 Setting up Gemini Vision API...")
    try:
        model, current_key, model_name, config_index = setup_gemini()
        print(f"✅ Gemini API ready with {model_name}")
    except Exception as e:
        print(f"❌ Failed to setup Gemini: {e}")
        return
    
    prompt = create_prescription_prompt()
    
    # Find images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    images_path = Path(args.images_dir)
    for ext in image_extensions:
        image_files.extend(images_path.glob(f"*{ext}"))
        image_files.extend(images_path.glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)
    
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images
    results = []
    successful = 0
    failed = 0
    
    print("🔥 Processing images with Gemini Vision...")
    
    for i, image_path in enumerate(tqdm(image_files, desc="Extracting text")):
        try:
            # Extract text with Gemini
            text, confidence = extract_text_with_gemini(model, image_path, prompt)
            
            if text and len(text.strip()) > 10:  # Minimum text length
                result = {
                    "image_path": str(image_path),
                    "text": text,
                    "conf": confidence,
                    "method": "gemini_vision",
                    "model": model_name
                }
                results.append(result)
                successful += 1
            else:
                print(f"⚠️  Empty/short text for {image_path.name}")
                failed += 1
            
            # Rate limiting
            if args.batch_delay > 0:
                time.sleep(args.batch_delay)
                
        except Exception as e:
            print(f"❌ Failed to process {image_path.name}: {e}")
            failed += 1
            continue
    
    # Save results
    print(f"\n💾 Saving {len(results)} results to {args.output_file}")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n🎉 Complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success rate: {successful/(successful+failed)*100:.1f}%")
    print(f"💾 Output: {args.output_file}")

if __name__ == "__main__":
    main()