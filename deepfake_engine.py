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
        """Step 2: Calibrated Spatial & Frequency Math (Laplacian, Entropy, FFT Ratio)."""
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Laplacian Variance (Texture/Smoothness) - Calibrated for mobile
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        smoothness_threat = max(0.0, min(100.0, (300.0 - lap_var) / 3.0))
        
        # 2. Pixel Entropy (Information Density) - Calibrated for compression
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / (hist.sum() + 1e-7)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-7)))
        entropy_threat = max(0.0, min(100.0, (7.0 - entropy) * 100.0))
        
        # 3. Fast Fourier Transform (High-Frequency Energy Ratio)
        roi = cv2.resize(gray, (256, 256))
        f = np.fft.fft2(roi)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        
        # Calculate total energy vs low-frequency energy (center 30x30 pixels)
        total_energy = float(np.sum(mag)) + 1e-7
        low_freq_energy = float(np.sum(mag[128-15:128+15, 128-15:128+15]))
        high_freq_energy = total_energy - low_freq_energy
        
        # Natural images have a healthy distribution of high-frequency noise.
        # Deepfakes are unnaturally smooth in the frequency domain.
        hf_ratio = (high_freq_energy / total_energy) * 100.0
        fft_threat = max(0.0, min(100.0, (8.0 - hf_ratio) * 12.5))
        
        # Average the three spatial/frequency components
        math_threat = float((smoothness_threat + entropy_threat + fft_threat) / 3.0)
        self.features['math_threat'] = math_threat
        self.signs.append(f"Mathematical Analysis: Spatial/FFT metrics yielded {round(math_threat, 1)}% synthetic probability.")

    def query_cnn_threat(self, image_bytes):
        """Step 3: Query dedicated Deepfake Convolutional Neural Network."""
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
        """Step 4: True Uncapped Sensor Fusion."""
        math_threat = float(self.features['math_threat'])
        is_screenshot = bool(self.features['is_screenshot'])
        
        # Dynamic Fusion Weights based on Container
        if is_screenshot:
            # Screenshots destroy optical pixel noise. Rely on the semantic CNN (80/20 split).
            fused_score = (cnn_threat * 0.8) + (math_threat * 0.2)
        else:
            # Boosted CNN weight for Native images (85/15 split).
            fused_score = (cnn_threat * 0.85) + (math_threat * 0.15)
            
        is_fake = bool(fused_score >= 50.0)

        # 4-Class Matrix (Uncapped)
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
            
        reason = f"[{classification.upper()}] Forensic Ensemble Analysis: Convolutional network assessed {round(cnn_threat, 1)}% synthetic threat, corroborated by {round(math_threat, 1)}% spatial/FFT threat. Final fused probability: {round(fused_score, 1)}%."
            
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
            
            # Apply Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            
            merged = cv2.merge((cl, a_channel, b_channel))
            enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            
            # Downsample enhanced image to avoid memory exhaustion
            enhanced_pil = Image.fromarray(enhanced_rgb)
            enhanced_pil.thumbnail((1024, 1024))
            buf = io.BytesIO()
            enhanced_pil.save(buf, format="JPEG", quality=85)
            cnn_image_bytes = buf.getvalue()

        # Initialize Ensemble
        ensemble = TrueForensicEnsemble(image_path)
        
        ensemble.analyze_container(img)
        ensemble.calculate_math_threat(cv_img)
        
        del cv_img
        gc.collect()

        # Query CNN using the CLAHE-enhanced image bytes
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
            "analyzed_via": "Sentinel X Calibrated Sensor Fusion"
        }

    except Exception as e:
        return {"error": True, "reason": f"Analysis Error: {str(e)}"}