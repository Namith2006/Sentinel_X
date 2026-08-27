import os
import requests

# Grab the secure key from Render (DO NOT paste your hf_... key here)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Use the modern Hugging Face Inference Router URL
API_URL = "https://router.huggingface.co/hf-inference/models/dima806/deepfake_vs_real_image_detection"

def analyze_image(image_path: str) -> dict:
    if not HF_API_TOKEN:
        return {"error": True, "reason": "ERROR: HF_API_TOKEN is missing in Render environment variables."}
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        response = requests.post(API_URL, headers=headers, data=image_bytes)
        
        if response.status_code == 503:
            return {"error": True, "reason": "The Cloud AI is waking up from sleep mode. Please wait 20 seconds and scan again."}
            
        response.raise_for_status()
        data = response.json()
        
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