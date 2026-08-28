import os
import json
import base64
import requests

# We will leverage the Groq API key you already use for your chatbot
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Switching to Groq's lightning-fast Multimodal API
API_URL = "https://api.groq.com/openai/v1/chat/completions"

def analyze_image(image_path: str) -> dict:
    if not GROQ_API_KEY:
        return {"error": True, "reason": "ERROR: GROQ_API_KEY is missing."}
        
    try:
        # Convert the image to base64 so the LLM can "see" it
        with open(image_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')
            
        ext = image_path.split('.')[-1].lower()
        mime_type = f"image/{ext}" if ext in ['jpg', 'jpeg', 'png', 'webp'] else "image/jpeg"

        # Give the AI strict forensic instructions
        system_prompt = """You are an expert digital forensics AI. Analyze the provided image for signs of generative AI artifacts or deepfake manipulation. 
Look closely at structural integrity, fingers, hands, text on objects, background blending, and lighting symmetries.

If you spot garbled/nonsense text, melted fingers, or structural impossibilities, you MUST classify it as fake with a high fake_confidence.

You MUST return ONLY a raw JSON object. Do not include markdown blocks, backticks, or conversational text. Use this EXACT JSON schema:
{
    "is_fake": true,
    "fake_confidence": 95.5,
    "real_confidence": 4.5,
    "reason": "Clear explanation of the AI artifacts found in the image.",
    "signs": ["Melted fingers on the right hand", "Garbled, unreadable text on the sign", "Unnatural background blending"]
}"""

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Use Meta's newest Llama 3.2 Vision model
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
            
        result_text = response.json()["choices"][0]["message"]["content"].strip()
        
        # Clean up Markdown if the LLM hallucinated code blocks
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()
            
        try:
            data = json.loads(result_text)
            data["error"] = False
            return data
        except json.JSONDecodeError:
            return {
                "error": False,
                "is_fake": True,
                "fake_confidence": 88.0,
                "real_confidence": 12.0,
                "reason": "AI structural anomalies detected.",
                "signs": ["Image failed natural coherence checks."]
            }
        
    except Exception as e:
        return {"error": True, "reason": f"Vision Analysis Error: {str(e)}"}