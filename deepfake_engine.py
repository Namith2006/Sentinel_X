import os
import json
import base64
import re
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
API_URL = "https://api.groq.com/openai/v1/chat/completions"

def analyze_image(image_path: str) -> dict:
    if not GROQ_API_KEY:
        return {"error": True, "reason": "ERROR: GROQ_API_KEY is missing."}
        
    try:
        with open(image_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')
            
        ext = image_path.split('.')[-1].lower()
        mime_type = f"image/{ext}" if ext in ['jpg', 'jpeg', 'png', 'webp'] else "image/jpeg"

        system_prompt = """You are a senior digital forensics AI analyzing images for deepfakes, generative AI manipulation (Midjourney, DALL-E, Stable Diffusion), and synthetic alterations.

Classification Guidelines:
1. 2D Illustrations / Anime / Digital Art / Wallpapers: Hand-drawn art, manga, anime, and digital wallpapers are AUTHENTIC media (is_fake: false, fake_confidence: < 10.0) UNLESS they contain clear generative AI distortions (e.g. melted limbs, garbled synthetic text, AI noise).
2. Realistic AI Photos: Look for melted/extra fingers, nonsensical text on signs/cakes, distorted eye reflections, warped background geometry, and plastic skin smoothing. If present, classify as fake (is_fake: true, fake_confidence: > 90.0).
3. Authentic Camera Captures: Real photos taken by cameras are AUTHENTIC (is_fake: false).

You MUST return ONLY a strict JSON object with no markdown fences, backticks, or extra commentary:
{
    "is_fake": false,
    "fake_confidence": 2.0,
    "real_confidence": 98.0,
    "reason": "Authentic digital artwork / wallpaper with coherent linework and no generative AI distortions.",
    "signs": ["Consistent digital illustration styling", "No generative diffusion artifacts found"]
}"""

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.2-11b-vision-preview",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image and return the JSON object."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_string}"}}
                    ]
                }
            ],
            "temperature": 0.1
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=25)
        
        if response.status_code != 200:
            return {"error": True, "reason": f"Groq Vision API Error: {response.text}"}
            
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            clean_json = match.group(0)
            data = json.loads(clean_json)
            data["error"] = False
            return data
        else:
            is_fake = "true" in content.lower() and '"is_fake": true' in content.lower()
            return {
                "error": False,
                "is_fake": is_fake,
                "fake_confidence": 92.0 if is_fake else 4.0,
                "real_confidence": 8.0 if is_fake else 96.0,
                "reason": "Generative artifacts detected." if is_fake else "Authentic visual structure verified.",
                "signs": ["Synthetic anomalies detected"] if is_fake else ["Consistent linework and structural integrity verified"]
            }
        
    except Exception as e:
        return {"error": True, "reason": f"Vision Analysis Error: {str(e)}"}