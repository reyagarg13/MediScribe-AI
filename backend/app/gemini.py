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
    
    # Prescription analysis prompt
    prompt = """You are a medical document analysis expert. I need you to carefully examine this prescription image and extract ALL prescribed items with complete accuracy.

## TASK OVERVIEW:
Analyze this prescription image which may contain:
- Traditional pharmaceutical medications (pills, tablets, liquids, injections)
- Over-the-counter medicines and supplements
- Alternative therapies and wellness recommendations  
- Lifestyle modifications and therapeutic activities
- Medical devices or equipment recommendations

## DETAILED INSTRUCTIONS:

1. **EXAMINE THE IMAGE CAREFULLY**: Look for handwritten text, printed text, checkboxes, symbols, and any medical terminology

2. **IDENTIFY ALL PRESCRIBED ITEMS**: Extract everything that appears to be prescribed, recommended, or checked off by the healthcare provider

3. **FOR EACH ITEM FOUND**, provide the following information:
   - **Item Name**: The exact name as written (medication name, activity, recommendation)
   - **Dosage/Amount**: Any strength, quantity, or measurement specified (e.g., "500mg", "twice", "10 minutes", "as needed")
   - **Frequency**: How often it should be taken/done (e.g., "twice daily", "every 8 hours", "as needed", "weekly")
   - **Duration**: How long to continue (e.g., "7 days", "2 weeks", "ongoing", "until symptoms improve")

## OUTPUT FORMAT:
Use this exact structure for each item:

1. **Item Name:** [Exact name as written on prescription]
   **Dosage:** [Strength/amount if specified, or "Not specified"]
   **Frequency:** [How often, or "Not specified"] 
   **Duration:** [How long, or "Not specified"]

2. **Item Name:** [Next item...]
   **Dosage:** [...]
   **Frequency:** [...]
   **Duration:** [...]

## IMPORTANT GUIDELINES:
- Be extremely precise - copy text exactly as it appears
- If handwriting is unclear, provide your best interpretation
- Include ANY item that appears to be prescribed or recommended
- If dosage/frequency/duration is not specified for an item, write "Not specified"
- If you cannot read certain text clearly, indicate this with "[unclear text]"
- Do not make assumptions - only extract what you can actually see
- Maintain the numbered list format for easy parsing

## EXAMPLES:
- Traditional: "Amoxicillin 500mg - Take twice daily for 10 days"
- Wellness: "Deep breathing exercises - Practice for 5 minutes daily"
- Activity: "Walk 30 minutes - Every day for 2 weeks"

Now please analyze the prescription image and provide the structured extraction following this format exactly."""
    
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