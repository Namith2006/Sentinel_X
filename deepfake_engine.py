import os
import mimetypes
import requests

# Grab the secure key from Render
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Modern Hugging Face Inference Router URL
API_URL = "https://router.huggingface.co/hf-inference/models/dima806/deepfake_vs_real_image_detection"

def analyze_image(image_path: str) -> dict:
    if not HF_API_TOKEN:
        return {"error": True, "reason": "ERROR: HF_API_TOKEN is missing in Render environment variables."}
    
    # Automatically detect if it's image/jpeg, image/png, etc. based on file extension
    content_type, _ = mimetypes.guess_type(image_path)
    if not content_type:
        content_type = "image/jpeg"  # Safe fallback
        
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": content_type
    }
    
    try:
        # Read the actual image bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        response = requests.post(API_URL, headers=headers, data=image_bytes)
        
        # Catch 503 errors when the free cloud AI is waking up
        if response.status_code == 503:
            return {"error": True, "reason": "The Cloud AI is waking up from sleep mode. Please wait 20 seconds and scan again."}
            
        response.raise_for_status()
        data = response.json()
        
        # Clean nested lists returned by Hugging Face Vision models
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            data = data[0]
            
        fake_score = 0.0
        real_score = 0.0
        
        for item in data:
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0)) * 100
            
            if "fake" in label:
                fake_score = score
            else:
                real_score = score
                
        return {
            "error": False,
            "is_fake": fake_score >= 50.0,
            "fake_confidence": fake_score,
            "real_confidence": real_score
        }
        
    except Exception as e:
        err_msg = str(e)
        if 'response' in locals() and hasattr(response, 'text'):
            err_msg += f" | {response.text}"
        return {"error": True, "reason": f"API Connection Error: {err_msg}"}