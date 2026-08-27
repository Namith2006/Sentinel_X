import os
import requests

# Grab the secure key from Render
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Use the updated Hugging Face Router endpoint to avoid 410 Gone errors
API_URL = "https://router.huggingface.co/models/prithivMLmods/Deep-Fake-Detector-v2-Model"

def analyze_image(image_path: str) -> dict:
    if not HF_API_TOKEN:
        return {"error": True, "reason": "Backend Error: HF_API_TOKEN is missing in Render environment variables."}
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    try:
        # THE FIX: Read the actual image bytes from the temporary file path
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        # Send the actual image bytes to the cloud server
        response = requests.post(API_URL, headers=headers, data=image_bytes)
        response.raise_for_status()
        data = response.json()
        
        fake_score = 0.0
        real_score = 0.0
        
        for item in data:
            label = item.get("label", "").lower()
            score = item.get("score", 0.0) * 100
            
            if "fake" in label:
                fake_score = score
            else:
                real_score = score
                
        is_fake = fake_score >= 50.0
        
        return {
            "error": False,
            "is_fake": is_fake,
            "fake_confidence": fake_score,
            "real_confidence": real_score
        }
        
    except Exception as e:
        error_details = str(e)
        if 'response' in locals() and hasattr(response, 'text'):
            error_details += f" | {response.text}"
            
        # Pass the explicit error flag back to main.py
        return {
            "error": True,
            "is_fake": False,
            "fake_confidence": 0.0,
            "real_confidence": 0.0,
            "reason": f"Hugging Face API Error: {error_details}",
            "signs": ["Failed to communicate with the cloud AI.", "Check your API token and network."]
        }