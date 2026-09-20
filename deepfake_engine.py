import os
import io
import json
import base64
import requests
import cv2
import numpy as np
import gc
from PIL import Image, ImageChops, ImageEnhance

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "") 
HF_API_URL = "https://router.huggingface.co/hf-inference/models/prithivMLmods/Deep-Fake-Detector-v2-Model"

class TrueForensicEnsemble:
    def __init__(self, image_path):
        self.image_path = image_path
        self.features = {}
        self.signs = []
        
    def analyze_container(self, img):
        """Step 1: Container Labeling"""
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
        """Step 2: Sensitive Spatial Math (Fixed 0.0% Blind Spot)"""
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Laplacian Variance - Widened gradient to stop 0.0% clamping
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        smoothness_threat = max(0.0, min(100.0, (600.0 - lap_var) / 6.0))
        
        # 2. Pixel Entropy - Softened multiplier
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / (hist.sum() + 1e-7)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-7)))
        entropy_threat = max(0.0, min(100.0, (7.8 - entropy) * 40.0))
        
        # 3. FFT High-Frequency Energy Ratio - Increased baseline sensitivity
        roi = cv2.resize(gray, (256, 256))
        f = np.fft.fft2(roi)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        
        total_energy = float(np.sum(mag)) + 1e-7
        low_freq_energy = float(np.sum(mag[128-15:128+15, 128-15:128+15]))
        high_freq_energy = total_energy - low_freq_energy
        
        hf_ratio = (high_freq_energy / total_energy) * 100.0
        fft_threat = max(0.0, min(100.0, (12.0 - hf_ratio) * 8.0))
        
        math_threat = float((smoothness_threat + entropy_threat + fft_threat) / 3.0)
        self.features['math_threat'] = math_threat
        self.signs.append(f"Mathematical Analysis: Spatial/FFT metrics returned {round(math_threat, 1)}% synthetic baseline.")

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
        """Step 4: The Weighted Veto (Soft Veto) Fusion"""
        math_threat = float(self.features['math_threat'])
        is_screenshot = bool(self.features['is_screenshot'])

        # AI is the baseline
        base_score = cnn_threat
        multiplier = 1.0

        if is_screenshot:
            if math_threat > 50.0:
                multiplier = 1.0
                self.signs.append(f"Weighted Veto: Math confirms AI ({round(math_threat, 1)}%). Multiplier: 1.0x.")
            elif math_threat > 35.0:
                # Moderate threshold: Math is unsure. Slight penalty to AI.
                multiplier = 0.85
                self.signs.append(f"Weighted Veto: Math is moderate ({round(math_threat, 1)}%). Multiplier: 0.85x.")
            else:
                # Very Low threshold (< 35%): Math strongly suggests REAL. Heavy penalty to counter hallucination.
                multiplier = 0.5
                self.signs.append(f"Weighted Veto: Math strongly suggests REAL ({round(math_threat, 1)}%). Throttling CNN hallucination. Multiplier: 0.5x.")
        else:
            # Native image. Trust the AI heavily, but apply a tiny sanity check if math is completely clean.
            if math_threat < 20.0:
                multiplier = 0.9
                self.signs.append(f"Weighted Veto: Native image with very low math threat. Multiplier: 0.9x.")

        # Apply multiplier and lock bounds
        fused_score = max(0.0, min(100.0, base_score * multiplier))
        is_fake = bool(fused_score >= 50.0)

        # 4-Class Matrix
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
            
        reason = f"[{classification.upper()}] Weighted Veto Fusion: CNN Base ({round(base_score, 1)}%) applied with {multiplier}x Multiplier based on Spatial Math ({round(math_threat, 1)}%). Final probability: {round(fused_score, 1)}%."
            
        return classification, desc, fused_score, reason

# ==========================================
# 🚀 MAIN ANALYSIS ENDPOINT
# ==========================================
def analyze_image(image_path: str) -> dict:
    try:
        with Image.open(image_path) as orig_img:
            img = orig_img.convert('RGB')
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # --- PRE-PROCESSING: CLAHE Edge Enhancement ---
            lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            
            merged = cv2.merge((cl, a_channel, b_channel))
            enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            
            enhanced_pil = Image.fromarray(enhanced_rgb)
            enhanced_pil.thumbnail((1024, 1024))
            buf = io.BytesIO()
            enhanced_pil.save(buf, format="JPEG", quality=85)
            cnn_image_bytes = buf.getvalue()

        ensemble = TrueForensicEnsemble(image_path)
        
        ensemble.analyze_container(img)
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
            "analyzed_via": "Sentinel X Weighted Veto Fusion"
        }

    except Exception as e:
        return {"error": True, "reason": f"Analysis Error: {str(e)}"}