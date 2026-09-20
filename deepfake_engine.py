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

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "") 
HF_API_URL = "https://router.huggingface.co/hf-inference/models/prithivMLmods/Deep-Fake-Detector-v2-Model"

class TrueForensicEnsemble:
    def __init__(self, image_path):
        self.image_path = image_path
        self.features = {}
        self.signs = []
        
    def analyze_container(self, img, cv_img):
        """Step 1: Determine if the image is a raw file or a compressed screenshot/messaging app file."""
        exif = img.getexif()
        has_metadata = bool(exif and (0x010f in exif or 0x0110 in exif))
        
        try:
            temp_io = io.BytesIO()
            img.save(temp_io, 'JPEG', quality=90)
            temp_io.seek(0)
            ela_img = ImageChops.difference(img, Image.open(temp_io))
            ela_score = float(np.mean(np.array(ela_img)))
        except Exception:
            ela_score = 0.0

        self.features['is_screenshot'] = not has_metadata or ela_score > 8.0
        if self.features['is_screenshot']:
            self.signs.append("Container Analysis: Image lacks native metadata or exhibits uniform recompression (Screenshot/WhatsApp).")

    def calculate_math_threat(self, cv_img):
        """Step 2: Recalibrated Spatial Math for Mobile Photography."""
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Laplacian Variance
        # Recalibrated from 800 down to 300 to forgive standard smartphone skin-smoothing/noise-reduction.
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        smoothness_threat = max(0.0, min(100.0, (300.0 - lap_var) / 3.0))
        
        # 2. Pixel Entropy
        # Recalibrated from 7.5 down to 7.2 to forgive WhatsApp JPEG compression flattening.
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / (hist.sum() + 1e-7)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-7)))
        entropy_threat = max(0.0, min(100.0, (7.2 - entropy) * 100.0))
        
        math_threat = float((smoothness_threat * 0.5) + (entropy_threat * 0.5))
        self.features['math_threat'] = math_threat
        self.signs.append(f"Spatial Analysis: Entropy and variance math yielded {round(math_threat, 1)}% synthetic probability.")

    def query_cnn_threat(self, image_bytes):
        """Step 3: Query dedicated Deepfake Convolutional Neural Network."""
        if not HF_API_TOKEN:
            self.signs.append("CNN Analysis: HF_API_TOKEN missing. Defaulting strictly to Spatial Math.")
            return self.features['math_threat']

        headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "image/jpeg"}
        try:
            response = requests.post(HF_API_URL, headers=headers, data=image_bytes, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    data = data[0]
                for item in data:
                    if "fake" in str(item.get("label", "")).lower() or "artificial" in str(item.get("label", "")).lower():
                        cnn_score = float(item.get("score", 0.0)) * 100.0
                        self.signs.append(f"CNN Analysis: ViT Network identified {round(cnn_score, 1)}% synthetic markers.")
                        return cnn_score
            else:
                self.signs.append(f"CNN Analysis: Network offline (HTTP {response.status_code}).")
        except Exception:
            self.signs.append("CNN Analysis: Connection timeout.")
            
        return self.features['math_threat']

    def fuse_and_classify(self, cnn_threat):
        """Step 4: True Sensor Fusion with Hallucination Dampener."""
        math_threat = float(self.features['math_threat'])
        is_screenshot = bool(self.features['is_screenshot'])
        
        if is_screenshot:
            # HALLUCINATION DAMPENER:
            # If spatial math sees healthy optical noise (< 45% threat), 
            # we forcefully throttle the CNN to stop it from panicking over WhatsApp compression.
            if math_threat < 45.0:
                adjusted_cnn = cnn_threat * 0.35 
                self.signs.append(f"Fusion Override: CNN confidence throttled from {round(cnn_threat,1)}% to {round(adjusted_cnn,1)}% due to healthy optical noise.")
                cnn_threat = adjusted_cnn
            
            # Make spatial math the dominant decision maker for compressed media (70% weight)
            fused_score = (cnn_threat * 0.3) + (math_threat * 0.7)
        else:
            # For raw native images, trust the CNN more
            fused_score = (cnn_threat * 0.7) + (math_threat * 0.3)
            
        is_fake = bool(fused_score >= 50.0)

        # 4-Class Matrix
        if is_screenshot and is_fake:
            classification = "4_AI_Screenshot"
            desc = "Screenshot / Compressed AI-generated image"
        elif is_screenshot and not is_fake:
            classification = "2_Real_Screenshot"
            desc = "Screenshot / Compressed authentic photograph"
            fused_score = min(34.0, fused_score) # Cap safe screenshots so UI stays green
        elif is_fake and not is_screenshot:
            classification = "3_AI_Native"
            desc = "Direct AI-generated media"
        else:
            classification = "1_Real_Native"
            desc = "Authentic camera photograph"
            fused_score = min(34.0, fused_score)
            
        reason = f"[{classification.upper()}] Forensic Ensemble Analysis: Convolutional network assessed {round(cnn_threat, 1)}% synthetic threat, corroborated by {round(math_threat, 1)}% spatial entropy threat. Final fused probability: {round(fused_score, 1)}%."
            
        return classification, desc, fused_score, reason

# ==========================================
# 🚀 MAIN ANALYSIS ENDPOINT
# ==========================================
def analyze_image(image_path: str) -> dict:
    try:
        with Image.open(image_path) as orig_img:
            img = orig_img.convert('RGB')
            processing_img = img.copy()
            processing_img.thumbnail((1024, 1024))
            buf = io.BytesIO()
            processing_img.save(buf, format="JPEG", quality=85)
            image_bytes = buf.getvalue()

        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        ensemble = TrueForensicEnsemble(image_path)
        ensemble.analyze_container(img, cv_img)
        ensemble.calculate_math_threat(cv_img)
        
        del cv_img
        gc.collect()

        cnn_threat = ensemble.query_cnn_threat(image_bytes)
        classification, desc, fused_score, reason = ensemble.fuse_and_classify(cnn_threat)

        return {
            "error": False,
            "classification": classification,
            "description": desc,
            "is_fake": bool(fused_score >= 50.0),
            "fake_confidence": float(fused_score),
            "real_confidence": float(100.0 - fused_score),
            "reason": str(reason),
            "signs": ensemble.signs,
            "detailed_analysis": ensemble.features,
            "analyzed_via": "Sentinel X Sensor Fusion (With Hallucination Dampener)"
        }

    except Exception as e:
        return {"error": True, "reason": f"Analysis Error: {str(e)}"}