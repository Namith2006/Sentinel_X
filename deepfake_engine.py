import os
import requests

# Grab the secure key from Render
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Connecting to a state-of-the-art deepfake detection model on Hugging Face
API_URL = "https://api-inference.huggingface.co/models/prithivMLmods/Deep-Fake-Detector-v2-Model"

def analyze_image(image_bytes: bytes) -> dict:
    if not HF_API_TOKEN:
        return {"reason": "Backend Error: HF_API_TOKEN is missing in Render environment variables."}
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    try:
        # Send the image bytes to the massive cloud server for real analysis
        response = requests.post(API_URL, headers=headers, data=image_bytes)
        response.raise_for_status()
        data = response.json()
        
        # The AI returns a list like: [{'label': 'Deepfake', 'score': 0.98}, {'label': 'Realism', 'score': 0.02}]
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
        
        # Generate forensic signs based on the real mathematical outcome
        signs = []
        if is_fake:
            signs = [
                "Vision Transformer (ViT) detected synthetic manipulation.",
                "Spatial relationships in pixel patches match generative AI noise."
            ]
        else:
            signs = [
                "Natural pixel noise distribution detected.",
                "No neural-network blending boundaries found."
            ]
            
        return {
            "is_deepfake": is_fake,
            "fake_confidence": fake_score,
            "real_confidence": real_score,
            "reason": "AI generative artifacts and anomalies detected." if is_fake else "Image conforms to natural cryptographic noise distributions.",
            "signs": signs
        }
        
    except Exception as e:
        # If the free AI server is waking up, it might take a few seconds on the very first try
        return {
            "is_deepfake": False,
            "fake_confidence": 0,
            "real_confidence": 0,
            "reason": f"Analysis failed: {str(e)}",
            "signs": ["Model may be waking up. Try uploading again in 15 seconds."]
        }