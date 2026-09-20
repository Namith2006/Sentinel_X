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
    """
    Improved 6-branch feature extraction optimized for 4-class classification.
    Memory-efficient for Render deployment.
    """
    def __init__(self):
        self.features = {}
        self.debug_signs = []
    
    def extract_metadata(self, img):
        exif = img.getexif()
        if not exif or (0x010f not in exif and 0x0110 not in exif):
            self.features['has_camera_metadata'] = False
            self.debug_signs.append("Metadata Branch: Missing native camera EXIF tags (Screenshot indicator).")
        else:
            self.features['has_camera_metadata'] = True

    def extract_resolution(self, width, height):
        common_res = {(1920, 1080), (1080, 1920), (1366, 768), (768, 1366), 
                      (2400, 1080), (1080, 2400), (2532, 1170), (1170, 2532),
                      (2778, 1284), (1284, 2778), (2796, 1290), (1290, 2796)}
        self.features['is_common_screen_res'] = (width, height) in common_res
        if self.features['is_common_screen_res']:
            self.debug_signs.append(f"Resolution Branch: Dimensions match display viewport ({width}x{height}).")

    def extract_ui_edges(self, cv_img):
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=cv_img.shape[1]*0.6, maxLineGap=10)
        self.features['has_ui_structure'] = lines is not None and len(lines) > 0
        if self.features['has_ui_structure']:
            self.debug_signs.append("Edge/UI Branch: Detected straight lines indicative of UI borders/letterboxing.")

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
            ela_score = np.mean(np.array(ela_img))
            self.features['ela_compression'] = ela_score
            if ela_score > 8.0:
                self.debug_signs.append(f"Compression Branch: ELA uniform recompression detected ({round(ela_score, 1)}).")
        except Exception:
            self.features['ela_compression'] = 0.0

    def extract_ai_artifacts(self, cv_img):
        # AI artifacts persist through screenshots via smoothness, low entropy, and frequency[cite: 11]
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Smoothness (Laplacian variance)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        smoothness_score = 1.0 if lap_var < 500 else (0.5 if lap_var < 1000 else 0.0)

        # 2. Entropy (Uniformity)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        entropy_score = 1.0 if entropy < 7.2 else 0.0

        # 3. Frequency Domain (FFT reduced for memory optimization)
        roi = cv2.resize(gray, (256, 256))
        f = np.fft.fft2(roi)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1)
        freq_score = 1.0 if np.mean(mag) < 140 else 0.0

        # Weighted combination from architectural spec
        ai_score = (smoothness_score * 0.4 + freq_score * 0.35 + entropy_score * 0.25)
        self.features['combined_ai_score'] = ai_score
        self.features['is_likely_ai'] = ai_score > 0.55
        
        if self.features['is_likely_ai']:
            self.debug_signs.append(f"Visual Branch: Generative artifacts found via frequency/smoothness math (Score: {round(ai_score,2)}).")

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
        groq_fake_prob = float(self.vision_data.get("fake_confidence", 30.0))
        
        # Combine screenshot indicators 
        is_screenshot = (feat.get('is_common_screen_res', False) or 
                         feat.get('has_ui_structure', False) or 
                         feat.get('has_ui_text', False) or 
                         not feat.get('has_camera_metadata', True) or
                         feat.get('ela_compression', 0.0) > 8.0)
                         
        # Combine AI indicators
        is_ai = (feat.get('is_likely_ai', False) or 
                 feat.get('has_ai_watermark', False) or 
                 groq_fake_prob >= 50.0)
                 
        # 4-Class Decision Tree
        if is_screenshot and is_ai:
            classification = "4_AI_Screenshot"
            confidence = max(85.0, groq_fake_prob * 2.8) # Force critical threat tier
            desc = "Screenshot of an AI-generated image"
        elif is_screenshot and not is_ai:
            classification = "2_Real_Screenshot"
            confidence = min(34.0, groq_fake_prob) # Cap safely below alert tier
            desc = "Screenshot of an authentic photograph"
        elif is_ai and not is_screenshot:
            classification = "3_AI_Native"
            confidence = max(50.0, groq_fake_prob)
            desc = "Native AI-generated output"
        else:
            classification = "1_Real_Native"
            confidence = min(25.0, groq_fake_prob)
            desc = "Native authentic photograph"
            
        return classification, desc, min(99.9, confidence)


# ==========================================
# 🚀 MAIN ANALYSIS ENDPOINT
# ==========================================
def analyze_image(image_path: str) -> dict:
    if not GROQ_API_KEY:
        return {"error": True, "reason": "ERROR: GROQ_API_KEY is missing."}

    try:
        # Load and encode image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            encoded_string = base64.b64encode(image_bytes).decode('utf-8')

        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Memory Optimization for Render[cite: 11]
        if max(img.size) > 2048:
            img.thumbnail((2048, 2048))
            
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 1. Run Enhanced Feature Extraction
        extractor = EnhancedFeatureExtractor()
        extractor.extract_metadata(img)
        extractor.extract_resolution(img.width, img.height)
        extractor.extract_ui_edges(cv_img)
        extractor.extract_compression(img)
        extractor.extract_ai_artifacts(cv_img)
        extractor.extract_text_features(img)

        # Cleanup memory
        del cv_img
        gc.collect()

        # 2. Run Vision LLM
        ext = image_path.split('.')[-1].lower()
        mime_type = f"image/{ext}" if ext in ['jpg', 'jpeg', 'png', 'webp'] else "image/jpeg"
        
        system_prompt = """Evaluate this image for generative AI anomalies (melted anatomy, gibberish text, cinematic studio lighting in impoverished settings). 
        Respond STRICTLY in JSON: {"fake_confidence": float, "reason": "string"}"""

        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.2-90b-vision-preview",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": "Execute forensic audit. Output JSON only."}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_string}"}}]}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }
        
        try:
            # Increased timeout for Groq API[cite: 11]
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30) 
            content = response.json()["choices"][0]["message"]["content"].strip()
            vision_data = json.loads(content)
        except Exception:
            vision_data = {"fake_confidence": 30.0, "reason": "Fallback visual assessment applied."}

        # 3. Classify via 4-Class Tree
        classifier = ImageClassifier(extractor, vision_data)
        classification, desc, final_fake_prob = classifier.classify()

        # 4. Map back to UI JSON Schema (Ensures Dashboard Compatibility)
        is_final_fake = final_fake_prob >= 50.0
        
        # Combine Vision AI reasoning with Structural Class
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
            "analyzed_via": "4-Class Classification Engine (Computer Vision + Llama-3.2)"
        }

    except Exception as e:
        return {"error": True, "reason": f"Analysis Error: {str(e)}"}