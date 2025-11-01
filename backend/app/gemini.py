"""
Gemini adapter using the official Google GenAI SDK.
Uses GEMINI_API_KEY environment variable.
"""
import os
from typing import Optional, Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GENAI_AVAILABLE = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def available() -> bool:
    """Check if Gemini is available."""
    return bool(GEMINI_API_KEY and GENAI_AVAILABLE)


def analyze_text(text: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Use Gemini to analyze text using the Google GenAI SDK.
    """
    if not available():
        raise RuntimeError("Gemini not configured (set GEMINI_API_KEY and install google-genai)")

    # Configure the client
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"Analyze this medical transcription and provide a clinical summary: {text}"
    
    # Try multiple models in order of preference
    models_to_try = [
        'gemini-2.5-flash',
        'gemini-2.0-flash-exp',
        'gemini-1.5-flash',
        'gemini-1.5-pro'
    ]
    
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt]
            )
            
            return {
                "analysis": response.text,
                "success": True,
                "model_used": model_name
            }
            
        except Exception as e:
            continue
    
    # If all models failed
    return {
        "analysis": f"Analysis failed: All Gemini models unavailable",
        "success": False,
        "error": "All models failed"
    }


def analyze_prescription_image(image_bytes: bytes) -> Dict[str, Any]:
    """Analyze prescription image using Gemini Vision API with the new SDK."""
    if not available():
        raise RuntimeError("Gemini not configured (set GEMINI_API_KEY and install google-genai)")
    
    # Configure the client
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Enhanced prescription analysis prompt that extracts complete prescription information
    prompt = """You are a medical document analysis expert. I need you to carefully examine this prescription image and extract COMPLETE prescription information including both header details and all prescribed items.

## TASK OVERVIEW:
Analyze this prescription image to extract:
1. **PRESCRIPTION HEADER INFORMATION** (patient details, doctor info, clinic info, dates)
2. **ALL PRESCRIBED ITEMS** (medications, therapies, recommendations)

## SECTION 1: PRESCRIPTION HEADER EXTRACTION
First, look for and extract prescription header information:

**PATIENT INFORMATION:**
- Patient Name: [Extract full name if visible]
- Patient Age: [Extract age if visible]
- Patient Sex/Gender: [Extract if visible - M/F/Male/Female]
- Date of Birth: [Extract if visible]

**DOCTOR/CLINIC INFORMATION:**
- Doctor Name: [Extract prescribing doctor's name]
- Clinic/Hospital Name: [Extract facility name if visible]
- Clinic Address: [Extract if visible]
- Phone Number: [Extract if visible]

**PRESCRIPTION DETAILS:**
- Prescription Date: [Extract date when prescription was written]
- Prescription Number: [Extract if visible]

## SECTION 2: PRESCRIBED ITEMS EXTRACTION
Then extract all prescribed items with complete dosage information:

**FOR EACH PRESCRIBED ITEM:**
- **Item Name**: The exact medication/therapy name as written (including TAB, SYP, CAP prefixes)
- **Complete Dosage**: Full dosage description exactly as written, including ALL instructions (e.g., "1 tab 3x a day after meals", "500mg twice daily before food", "2 caps morning and evening with meals")
- **Frequency**: How often (e.g., "3x a day", "twice daily", "every 8 hours", "as needed")
- **Duration**: How long (e.g., "7 days", "2 weeks", "until symptoms improve")
- **Meal Instructions**: Timing relative to meals (e.g., "before meals", "after meals", "with food", "empty stomach")

## OUTPUT FORMAT:
Structure your response exactly like this:

=== PRESCRIPTION HEADER ===
**Patient Name:** [Name or "Not visible"]
**Patient Age:** [Age or "Not visible"]
**Patient Sex:** [M/F or "Not visible"]
**Date of Birth:** [DOB or "Not visible"]
**Doctor Name:** [Doctor name or "Not visible"]
**Clinic Name:** [Clinic name or "Not visible"]
**Clinic Address:** [Address or "Not visible"]
**Phone:** [Phone or "Not visible"]
**Prescription Date:** [Date or "Not visible"]
**Prescription Number:** [Number or "Not visible"]

=== PRESCRIBED MEDICATIONS ===
1. **Item Name:** [Exact name as written, including TAB/SYP/CAP prefixes]
   **Complete Dosage:** [Full dosage exactly as written - preserve ALL instructions including meal timing]
   **Frequency:** [How often]
   **Duration:** [How long]
   **Meal Instructions:** [Before/after meals, with food, etc. or "Not specified"]

2. **Item Name:** [Next item...]
   **Complete Dosage:** [...]
   **Frequency:** [...]
   **Duration:** [...]
   **Meal Instructions:** [...]

## CRITICAL INSTRUCTIONS:
- **PRESERVE EXACT WORDING**: For dosages, copy the complete text exactly as written (e.g., "1 cap 3x a day" not "3x daily")
- **READ CAREFULLY**: Look at the entire prescription - header, body, margins, stamps
- **BE PRECISE**: Extract text exactly as it appears, including abbreviations and medical shorthand
- **HANDLE UNCLEAR TEXT**: If text is unclear, indicate with "[unclear: possibly XYZ]"
- **INCLUDE EVERYTHING**: Extract all visible patient info, doctor info, and prescribed items
- **MAINTAIN FORMAT**: Follow the exact output structure shown above

Now please analyze the prescription image and provide the complete structured extraction."""
    
    # Create the image part using the new SDK
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/png'
    )
    
    # Try multiple models in order of preference
    models_to_try = [
        'gemini-2.5-flash',
        'gemini-2.0-flash-exp', 
        'gemini-1.5-flash',
        'gemini-1.5-pro'
    ]
    
    for model_name in models_to_try:
        try:
            print(f"Trying Gemini model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    image_part,
                    prompt
                ]
            )
            
            print(f"✅ Successfully used model: {model_name}")
            return {
                "analysis": response.text,
                "success": True,
                "model_used": model_name
            }
            
        except Exception as e:
            print(f"❌ Model {model_name} failed: {str(e)}")
            continue
    
    # If all models failed
    raise RuntimeError(f"All Gemini models failed. Last error: {str(e)}")


def analyze_with_gemini(prompt: str) -> str:
    """
    Use Gemini to analyze text with a custom prompt using the Google GenAI SDK.
    Returns the raw text response for flexible parsing.
    """
    if not available():
        raise RuntimeError("Gemini not configured (set GEMINI_API_KEY and install google-genai)")

    # Configure the client
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Try multiple models in order of preference
    models_to_try = [
        'gemini-2.5-flash',
        'gemini-2.0-flash-exp',
        'gemini-1.5-flash',
        'gemini-1.5-pro'
    ]
    
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt]
            )
            
            return response.text
            
        except Exception as e:
            continue
    
    # If all models failed
    raise RuntimeError("All Gemini models failed for text analysis")