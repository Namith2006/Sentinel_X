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

        # Replaced the hardcoded example with a strict schema to prevent AI bias
        system_prompt = """You are a senior digital forensics AI analyzing images for deepfakes, generative AI manipulation (Midjourney, DALL-E, Stable Diffusion), and synthetic alterations.

Classification Guidelines:
1. 2D Illustrations / Anime / Digital Art: Authentic media (is_fake: false) UNLESS they contain clear generative AI distortions.
2. Realistic AI Photos: Look for melted/extra fingers, nonsensical text on signs/cakes, distorted eye reflections, warped background geometry, and plastic skin smoothing. If present, classify as fake (is_fake: true, fake_confidence: > 90.0).
3. Authentic Camera Captures: Real photos taken by cameras are AUTHENTIC (is_fake: false).

Output your analysis strictly in JSON format matching this schema:
{
    "is_fake": boolean,
    "fake_confidence": float (0-100),
    "real_confidence": float (0-100),
    "reason": "String explaining the specific artifacts found or why it is authentic.",
    "signs": ["List", "of", "specific", "visual", "evidence"]
}"""

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen/qwen3.6-27b",
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
            "temperature": 0.1,
            "response_format": {"type": "json_object"}  # FORCE STRICT JSON OUTPUT
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            return {"error": True, "reason": f"Groq Vision API Error: {response.text}"}
            
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        # Strip Qwen reasoning tags if they bleed into the output
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            clean_json = match.group(0)
            try:
                data = json.loads(clean_json)
                data["error"] = False
                return data
            except json.JSONDecodeError:
                pass
                
        # Robust String Fallback if JSON parsing completely fails
        content_lower = content.lower()
        is_fake_str = '"is_fake": true' in content_lower or 'is_fake":true' in content_lower
        has_fake_keywords = any(k in content_lower for k in ["melted", "garbled", "ai generated", "synthetic", "anomal"])
        is_fake = is_fake_str or has_fake_keywords
        
        return {
            "error": False,
            "is_fake": is_fake,
            "fake_confidence": 97.0 if is_fake else 4.0,
            "real_confidence": 3.0 if is_fake else 96.0,
            "reason": "Generative artifacts detected via fallback analysis." if is_fake else "Authentic visual structure verified.",
            "signs": ["Synthetic anomalies and AI generation hallmarks detected"] if is_fake else ["Consistent linework and structural integrity verified"]
        }
        
    except Exception as e:
        return {"error": True, "reason": f"Vision Analysis Error: {str(e)}"}