import os
import io
import json
import base64
import requests
import cv2
import numpy as np
import gc
from PIL import Image, ImageChops

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "") 
HF_API_URL = "https://router.huggingface.co/hf-inference/models/prithivMLmods/Deep-Fake-Detector-v2-Model"

class AdaptiveForensicEnsemble:
    def __init__(self, image_path):
        self.image_path = image_path
        self.features = {}
        self.signs = []
        
    def estimate_quality(self, img):
        """Step 1: Dynamic Quality Estimator & Container Labeling"""
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

        is_screenshot = not has_metadata or ela_score > 8.0
        self.features['is_screenshot'] = is_screenshot

        if is_screenshot and ela_score > 12.0:
            quality = "Low"
            self.signs.append("Quality Estimator: LOW (Aggressive compression/Screenshot detected).")
        elif is_screenshot or ela_score > 5.0:
            quality = "Medium"
            self.signs.append("Quality Estimator: MEDIUM (Standard compression detected).")
        else:
            quality = "High"
            self.signs.append("Quality Estimator: HIGH (Native capture characteristics detected).")
            
        self.features['image_quality'] = quality

    def calculate_math_threat(self, cv_img):
        """Step 2: Adaptive Signal Processing (With Baseline Floors)"""
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. Multi-Scale Analysis
        lap_1x = cv2.Laplacian(gray, cv2.CV_64F).var()
        gray_half = cv2.resize(gray, (w // 2, h // 2))
        lap_half = cv2.Laplacian(gray_half, cv2.CV_64F).var()
        gray_quarter = cv2.resize(gray_half, (w // 4, h // 4))
        lap_quarter = cv2.Laplacian(gray_quarter, cv2.CV_64F).var()
        
        scale_variance = float(np.std([lap_1x, lap_half, lap_quarter]))
        multi_scale_threat = max(5.0, min(100.0, (250.0 - scale_variance) / 2.0))
        
        # 2. Edge Density & Coherence 
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        
        edge_density = float(np.sum(sobel_mag > 30) / (h * w))
        edge_threat = max(5.0, min(100.0, (0.15 - edge_density) * 800.0))
        
        # 3. Frequency Domain Fingerprinting (FFT Spikes)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-7)
        
        mean_mag = np.mean(mag)
        std_mag = np.std(mag)
        spike_threshold = mean_mag + (3.5 * std_mag)
        spikes = np.sum(mag > spike_threshold)
        
        spike_density = (spikes / (w * h)) * 10000.0
        fft_threat = max(5.0, min(100.0, spike_density * 15.0))
        
        math_threat = float((multi_scale_threat + edge_threat + fft_threat) / 3.0)
        self.features['math_threat'] = math_threat
        self.signs.append(f"Adaptive Math: Pyramid ({round(multi_scale_threat,1)}%), Edge Density ({round(edge_threat,1)}%), Spectral Spikes ({round(fft_threat,1)}%) -> Threat: {round(math_threat, 1)}%")

    def query_cnn_threat(self, image_bytes):
        """Step 3: Query dedicated Deepfake CNN"""
        if not HF_API_TOKEN:
            self.signs.append("CNN Analysis: HF_API_TOKEN missing. Defaulting strictly to Spatial Math.")
            return self.features.get('math_threat', 0.0)

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
            
        return self.features.get('math_threat', 0.0)

    def fuse_and_classify(self, cnn_threat):
        """Step 4: Balanced Cross-Verification Matrix"""
        math_threat = float(self.features['math_threat'])
        quality = self.features.get('image_quality', 'Medium')
        is_screenshot = self.features.get('is_screenshot', False)

        # 1. Balanced Confidence Matrix (Reduced AI dominance)
        if quality == "High":
            ai_w, math_w = 0.70, 0.30
        elif quality == "Medium":
            ai_w, math_w = 0.60, 0.40
        else:
            ai_w, math_w = 0.50, 0.50

        fused_score = (cnn_threat * ai_w) + (math_threat * math_w)
        self.signs.append(f"Balanced Matrix: AI ({int(ai_w*100)}%) / Math ({int(math_w*100)}%).")

        # 2. THE MATH VETO
        if math_threat < 25.0:
            self.signs.append(f"Math Veto: Physical signals ({round(math_threat, 1)}%) are too low for a deepfake. Reducing final probability.")
            fused_score = fused_score * 0.6

        fused_score = max(0.0, min(100.0, fused_score))
        is_fake = bool(fused_score >= 50.0)

        # 4-Class Classification Mapping
        if is_screenshot and is_fake:
            classification = "4_AI_Screenshot"
            desc = "Screenshot / Compressed AI-generated image"
        elif is_screenshot and not is_fake:
            classification = "2_Real_Screenshot"
            desc = "Screenshot / Compressed authentic photograph"
        elif is_fake and not is_screenshot:
            classification = "3_AI_Native"
            desc = "Direct AI-generated media"
        else:
            classification = "1_Real_Native"
            desc = "Authentic camera photograph"
            
        reason = f"[{classification.upper()}] Balanced Fusion: CNN ({round(cnn_threat, 1)}%) fused with Math ({round(math_threat, 1)}%) under {quality.upper()} matrix. Final probability: {round(fused_score, 1)}%."
            
        return classification, desc, fused_score, reason

# ==========================================
# 🚀 MAIN ANALYSIS ENDPOINT
# ==========================================
def analyze_image(image_path: str) -> dict:
    try:
        with Image.open(image_path) as orig_img:
            img = orig_img.convert('RGB')
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # --- PRE-PROCESSING: CNN Input (Raw/Unaltered) ---
            cnn_pil = img.copy()
            cnn_pil.thumbnail((1024, 1024))
            buf = io.BytesIO()
            cnn_pil.save(buf, format="JPEG", quality=85)
            cnn_image_bytes = buf.getvalue()

        ensemble = AdaptiveForensicEnsemble(image_path)
        
        ensemble.estimate_quality(img)
        ensemble.calculate_math_threat(cv_img)
        
        del cv_img
        gc.collect()

        cnn_threat = ensemble.query_cnn_threat(cnn_image_bytes)
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
            "analyzed_via": "Sentinel X Balanced Cross-Verification Matrix"
        }

    except Exception as e:
        return {"error": True, "reason": f"Analysis Error: {str(e)}"}