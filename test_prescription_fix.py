#!/usr/bin/env python3
"""
Test script to verify prescription analysis is working with Gemini Vision integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.prescription_ocr import analyze_prescription_image
from PIL import Image, ImageDraw, ImageFont
import io

def create_test_prescription():
    """Create a test prescription image with handwritten-style text"""
    # Create a white background image
    img = Image.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a font that looks more handwritten
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Add prescription content
    y = 50
    lines = [
        "Dr. Smith Medical Clinic",
        "Patient: John Doe",
        "Date: Sept 23, 2025",
        "",
        "Prescription:",
        "1. Amoxicillin 500mg - Take twice daily for 7 days",
        "2. Ibuprofen 400mg - Take as needed for pain",
        "3. Vitamin D 1000 IU - Take once daily",
        "4. Hydroxychloroquine 200mg - Take once daily",
        "",
        "Dr. Smith, MD"
    ]
    
    for line in lines:
        draw.text((50, y), line, fill='black', font=font)
        y += 40
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()

def main():
    print("=== Testing Prescription Analysis Fix ===")
    
    # Check if Gemini is available
    try:
        from backend.app import gemini
        gemini_available = gemini.available()
        print(f"✅ Gemini available: {gemini_available}")
        if gemini_available:
            print(f"Gemini API Key configured: {bool(os.getenv('GEMINI_API_KEY'))}")
    except Exception as e:
        print(f"❌ Gemini import failed: {e}")
        gemini_available = False
    
    # Create test prescription
    print("Creating test prescription image...")
    image_bytes = create_test_prescription()
    print(f"Image size: {len(image_bytes)} bytes")
    
    # Test prescription analysis
    print("Analyzing prescription...")
    try:
        result = analyze_prescription_image(image_bytes)
        
        print("\n=== RESULTS ===")
        print(f"Status: {result.get('status')}")
        print(f"Method: {result.get('method')}")
        print(f"Processing time: {result.get('processing_time', 0):.2f}s")
        
        medications = result.get('medications', [])
        print(f"✅ Medications found: {len(medications)}")
        
        for i, med in enumerate(medications, 1):
            print(f"\n--- Medication {i} ---")
            print(f"Name: {med.get('matched_name', med.get('name_candidate', 'Unknown'))}")
            print(f"Raw line: {med.get('raw_line', 'N/A')}")
            print(f"Match score: {med.get('match_score', 0)}%")
            print(f"Dosage: {med.get('dosage', 'N/A')}")
            print(f"Frequency: {med.get('frequency', 'N/A')}")
            print(f"Duration: {med.get('duration', 'N/A')}")
            print(f"Method: {med.get('method', 'N/A')}")
        
        if result.get('error'):
            print(f"\n⚠️ Error: {result['error']}")
        
        # Show OCR text if available
        ocr_text = result.get('ocr_text', '')
        if ocr_text:
            print(f"\n=== OCR Text ===")
            print(ocr_text[:500] + "..." if len(ocr_text) > 500 else ocr_text)
    
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()