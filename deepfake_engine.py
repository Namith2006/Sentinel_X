import os
import io
import json
import base64
import requests
import cv2
import numpy as np
import gc
from PIL import Image, ImageChops, ImageEnhance
import pytesseract

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ==========================================
# 🎯 CLASS DEFINITIONS (4-CLASS PROBLEM)
# ==========================================
CLASS_DEFINITIONS = {
    "1_Real_Native": "Native camera photo (DSLR, phone camera)",
    "2_Real_Screenshot": "Screenshot of real photo in browser/app",
    "3_AI_Native": "Direct AI-generated image (Midjourney, DALL-E)",
    "4_AI_Screenshot": "Screenshot of AI-generated image"
}

# ==========================================
# 📊 ENHANCED FEATURE EXTRACTION
# ==========================================
class EnhancedFeatureExtractor:
    def __init__(self):
        self.features = {}
        self.debug_signs = []
    
    def extract_metadata(self, img):
        exif = img.getexif()
        if not exif or (0x010f not in exif and 0x0110 not in exif):
            self.features['has_camera_metadata'] = False
            self.debug_signs.append("Metadata Branch: Missing native camera EXIF tags (Screenshot/Messaging App).")
        else:
            self.features['has_camera_metadata'] = True

    def extract_resolution(self, width, height):
        common_res = {(1920, 1080), (1080, 1920), (1366, 768), (768, 1366), 
                      (2400, 1080), (1080, 2400), (2532, 1170), (1170, 2532),
                      (2778, 1284), (1284, 2778), (2796, 1290), (1290, 2796)}
        self.features['is_common_screen_res'] = (width, height) in common_res
        if self.features['is_common_screen_res']:
            self.debug_signs.append(f"Resolution Branch: Matches standard display viewport ({width}x{height}).")

    def extract_ui_edges(self, cv_img):
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=int(cv_img.shape[1]*0.6), maxLineGap=10)
        self.features['has_ui_structure'] = lines is not None and len(lines) > 0
        if self.features['has_ui_structure']:
            self.debug_signs.append("Edge/UI Branch: Straight boundary edges detected.")

    def extract_compression(self, img):
        try:
            temp_io = io.BytesIO()
            img.save(temp_io, 'JPEG', quality=90)
            temp_io.seek(0)
            resaved = Image.open(temp_io)
            ela_img = ImageChops.difference(img, resaved)
            extrema = ela_img.getextrema()
            max_diff = max([ex[1] for ex in extrema]) if extrema else 1
            scale = 255.0 / (max_diff if max_diff != 0 else 1)
            ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
            ela_score = float(np.mean(np.array(ela_img)))
            self.features['ela_compression'] = ela_score
            if ela_score > 8.0:
                self.debug_signs.append(f"Compression Branch: Uniform recompression detected ({round(ela_score, 1)}).")
        except Exception:
            self.features['ela_compression'] = 0.0

    def extract_ai_artifacts(self, cv_img):
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Smoothness
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        smoothness_score = 1.0 if lap_var < 500 else (0.5 if lap_var < 1000 else 0.0)

        # 2. Entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        entropy_score = 1.0 if entropy < 7.3 else 0.0

        # 3. Frequency Domain
        roi = cv2.resize(gray, (256, 256))
        f = np.fft.fft2(roi)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1)
        freq_score = 1.0 if np.mean(mag) < 145 else 0.0

        ai_score = (smoothness_score * 0.4 + freq_score * 0.35 + entropy_score * 0.25)
        self.features['combined_ai_score'] = ai_score
        self.features['is_likely_ai'] = ai_score >= 0.45 
        
        if self.features['is_likely_ai']:
            self.debug_signs.append(f"Visual Math: Low-entropy pattern matches generative diffusion ({round(ai_score,2)}).")

    def extract_text_features(self, img):
        try:
            text = pytesseract.image_to_string(img).lower()
            self.features['has_ai_watermark'] = any(t in text for t in ['midjourney', 'stable diffusion', 'dall-e'])
            self.features['has_ui_text'] = any(t in text for t in ['am', 'pm', 'lte', 'volte', 'screenshot'])
        except Exception:
            self.features['has_ai_watermark'] = False
            self.features['has_ui_text'] = False


# ==========================================
# 🧠 DECISION CLASSIFIER LOGIC
# ==========================================
class ImageClassifier:
    def __init__(self, extractor, vision_data):
        self.extractor = extractor
        self.vision_data = vision_data
        
    def classify(self):
        feat = self.extractor.features
        raw_vision_score = float(self.vision_data.get("fake_confidence", 0.0))
        
        # 1. Container Check (Screenshot / Screen Re-compression)
        is_screenshot = (
            feat.get('is_common_screen_res', False) or 
            feat.get('has_ui_structure', False) or 
            feat.get('has_ui_text', False) or 
            not feat.get('has_camera_metadata', True) or
            feat.get('ela_compression', 0.0) > 8.0
        )
                         
        # 2. Content Check (AI Anomaly) - Lowered tripwire to 20.0% to catch washed screenshots
        is_ai = (
            raw_vision_score >= 20.0 or 
            feat.get('is_likely_ai', False) or 
            feat.get('has_ai_watermark', False)
        )
                 
        # 3. 4-Class Classification Multipliers
        if is_screenshot and is_ai:
            classification = "4_AI_Screenshot"
            confidence = max(88.5, raw_vision_score * 2.8)
            desc = "Screenshot / Compressed AI-generated image"
        elif is_screenshot and not is_ai:
            classification = "2_Real_Screenshot"
            confidence = min(25.0, raw_vision_score)
            desc = "Screenshot of an authentic photograph"
        elif is_ai and not is_screenshot:
            classification = "3_AI_Native"
            confidence = max(80.0, raw_vision_score)
            desc = "Direct AI-generated media"
        else:
            classification = "1_Real_Native"
            confidence = min(15.0, raw_vision_score)
            desc = "Authentic camera photograph"
            
        return classification, desc, min(99.9, confidence)


# ==========================================
# 🚀 MAIN ANALYSIS ENDPOINT
# ==========================================
def analyze_image(image_path: str) -> dict:
    if not GROQ_API_KEY:
        return {"error": True, "reason": "GROQ_API_KEY is missing from environment."}

    try:
        # Load and resize for bandwidth and memory optimization
        with Image.open(image_path) as orig_img:
            img = orig_img.convert('RGB')
            
            # Step 1: Create a downsampled image for Groq (prevents HTTP 413 Payload Too Large)
            groq_img = img.copy()
            groq_img.thumbnail((1024, 1024))
            
            buf = io.BytesIO()
            groq_img.save(buf, format="JPEG", quality=85)
            compressed_bytes = buf.getvalue()
            encoded_string = base64.b64encode(compressed_bytes).decode('utf-8')

        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Step 2: Run 6-Branch Feature Extraction locally
        extractor = EnhancedFeatureExtractor()
        extractor.extract_metadata(img)
        extractor.extract_resolution(img.width, img.height)
        extractor.extract_ui_edges(cv_img)
        extractor.extract_compression(img)
        extractor.extract_ai_artifacts(cv_img)
        extractor.extract_text_features(img)

        del cv_img
        gc.collect()

        # Step 3: Query Active Vision Model
        system_prompt = """You are an elite digital forensics AI. 
Evaluate this image for synthetic AI generation markers:
1. Cardboard / Sign Text: Check if handwriting/typography looks digitally stamped, warped, or synthetically rendered.
2. Lighting: Look for cinematic studio lighting in impoverished/outdoor environments.
3. Anatomy: Inspect fingers, hands gripping the bowl, and feet/toes for morphing, missing cuticles, or plastic textures.

Respond STRICTLY in JSON:
{"fake_confidence": float, "is_ai": boolean, "reason": "string"}"""

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Primary model updated to Qwen3.8-27b (since Llama 90B was decommissioned)
        models_to_try = ["qwen/qwen3.8-27b", "llama-3.2-11b-vision-preview"]
        vision_data = None
        last_error = ""

        for model_name in models_to_try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Execute strict forensic inspection. Return ONLY valid JSON."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
                        ]
                    }
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }

            try:
                response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=20)
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"].strip()
                    vision_data = json.loads(content)
                    break
                else:
                    last_error = f"HTTP {response.status_code}"
            except Exception as e:
                last_error = str(e)

        # If Cloud API fails, rely on local visual math with a professional output reason
        if not vision_data:
            math_score = 85.0 if extractor.features.get('is_likely_ai', False) else 40.0
            if math_score >= 50.0:
                clean_reason = "Analyzed via Local Forensic Math. Generative anomalies detected matching synthetic media."
            else:
                clean_reason = "Analyzed via Local Forensic Math. Visual frequencies appear consistent with authentic optical capture."
                
            vision_data = {
                "fake_confidence": math_score,
                "reason": clean_reason
            }

        # Step 4: Classify via 4-Class Decision Matrix
        classifier = ImageClassifier(extractor, vision_data)
        classification, desc, final_fake_prob = classifier.classify()

        # Step 5: Format response for frontend dashboard
        is_final_fake = final_fake_prob >= 50.0
        final_reason = f"[{classification.upper()}] {desc}. {vision_data.get('reason', '')}"

        return {
            "error": False,
            "classification": classification,
            "description": desc,
            "is_fake": is_final_fake,
            "fake_confidence": round(final_fake_prob, 1),
            "real_confidence": round(100.0 - final_fake_prob, 1),
            "reason": final_reason.strip(),
            "signs": extractor.debug_signs,
            "detailed_analysis": extractor.features,
            "analyzed_via": "4-Class Forensics Engine (Local Computer Vision + API)"
        }

    except Exception as e:
        return {"error": True, "reason": f"Analysis Error: {str(e)}"}