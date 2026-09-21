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

class ForensicPipeline:
    def __init__(self, image_path):
        self.image_path = image_path
        self.results = {
            "preprocessing": {},
            "features": {},
            "ai_models": {},
            "decision": {}
        }
        self.signs = []

    # --- LAYER 1: PREPROCESSING ---
    def preprocess(self, img):
        """Preprocessing: Extract metadata and run ELA"""
        # EXIF Analysis
        exif = img.getexif()
        self.results['preprocessing']['has_metadata'] = bool(exif and (0x010f in exif or 0x0110 in exif))

        # Error Level Analysis (ELA) - Professional Forensic Tool
        try:
            temp_io = io.BytesIO()
            img.save(temp_io, 'JPEG', quality=90)
            temp_io.seek(0)
            ela_img = ImageChops.difference(img, Image.open(temp_io))
            ela_score = float(np.mean(np.array(ela_img)))
            self.results['preprocessing']['ela_score'] = ela_score
        except Exception:
            ela_score = 0.0
            self.results['preprocessing']['ela_score'] = ela_score

        is_screenshot = not self.results['preprocessing']['has_metadata'] or ela_score > 8.0
        self.results['preprocessing']['is_screenshot'] = is_screenshot

        # Determine Quality Tier
        if ela_score > 12.0 and is_screenshot: 
            quality = "Low"
        elif ela_score > 5.0 or is_screenshot: 
            quality = "Medium"
        else: 
            quality = "High"
            
        self.results['preprocessing']['quality'] = quality
        self.signs.append(f"[LAYER 1] Preprocessing: {quality} quality container detected via ELA.")

    # --- LAYER 2: FEATURE EXTRACTION ---
    def extract_features(self, cv_img):
        """Feature Extraction: Spectral, Spatial, and Noise"""
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Spectral Analysis (FFT) - Looking for AI checkerboards
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-7)
        
        mean_mag = np.mean(mag)
        std_mag = np.std(mag)
        spike_density = (np.sum(mag > (mean_mag + 3.5 * std_mag)) / (w * h)) * 10000.0
        spectral_score = max(0.0, min(100.0, spike_density * 15.0))

        # 2. Spatial Analysis (Edges/Blur)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        spatial_score = max(0.0, min(100.0, (500.0 - laplacian_var) / 5.0))

        self.results['features']['spectral'] = float(spectral_score)
        self.results['features']['spatial'] = float(spatial_score)
        self.signs.append(f"[LAYER 2] Features: Spectral ({round(spectral_score,1)}%), Spatial ({round(spatial_score,1)}%)")

    # --- LAYER 3: AI DETECTION MODELS ---
    def run_ai_models(self, image_bytes):
        """AI Detection: Transformer-based CNN"""
        if not HF_API_TOKEN:
            self.signs.append("[LAYER 3] AI Detection: Standby (API Token Missing).")
            return 0.0

        headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "image/jpeg"}
        try:
            response = requests.post(HF_API_URL, headers=headers, data=image_bytes, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    item = data[0] if isinstance(data[0], list) else data[0]
                    for entry in (item if isinstance(item, list) else [item]):
                        if "fake" in str(entry.get("label", "")).lower() or "artificial" in str(entry.get("label", "")).lower():
                            score = float(entry.get("score", 0.0)) * 100.0
                            self.results['ai_models']['cnn'] = score
                            self.signs.append(f"[LAYER 3] AI Detection: Vision Transformer identified {round(score, 1)}% synthetic markers.")
                            return score
        except Exception:
            pass
            
        self.signs.append("[LAYER 3] AI Detection: Fallback activated.")
        return 0.0

    # --- LAYER 4: DECISION LAYER (The Brain) ---
    def decide(self, cnn_score):
        """Decision Layer: Weighted Confidence Scoring"""
        spec = float(self.results['features']['spectral'])
        spat = float(self.results['features']['spatial'])
        qual = self.results['preprocessing']['quality']
        is_screenshot = bool(self.results['preprocessing'].get('is_screenshot', False))

        # Weighted Sum based on Quality
        if qual == "Low":
            weights = {'cnn': 0.3, 'spec': 0.4, 'spat': 0.3}
        elif qual == "Medium":
            weights = {'cnn': 0.5, 'spec': 0.3, 'spat': 0.2}
        else:
            weights = {'cnn': 0.7, 'spec': 0.2, 'spat': 0.1}

        final_score = (cnn_score * weights['cnn']) + (spec * weights['spec']) + (spat * weights['spat'])

        # --- THE laPlace VETO ---
        if spec < 20.0 and spat < 20.0:
            final_score = min(final_score, 40.0)
            self.signs.append(f"[LAYER 4] Decision: Strong authenticity markers found. Overriding AI high-score (laPlace Veto).")

        final_score = max(0.0, min(100.0, float(final_score)))
        is_fake = bool(final_score >= 60.0)

        # UI Classification Mapping
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

        return is_fake, final_score, classification, desc

# ==========================================
# MAIN ANALYSIS ENDPOINT
# ==========================================
def analyze_image(image_path: str) -> dict:
    try:
        with Image.open(image_path) as orig_img:
            img = orig_img.convert('RGB')
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Preprocess for AI
            cnn_pil = img.copy()
            cnn_pil.thumbnail((1024, 1024))
            buf = io.BytesIO()
            cnn_pil.save(buf, format="JPEG", quality=85)
            cnn_bytes = buf.getvalue()

        pipeline = ForensicPipeline(image_path)
        pipeline.preprocess(img)
        pipeline.extract_features(cv_img)
        
        del cv_img
        gc.collect()
        
        cnn_score = pipeline.run_ai_models(cnn_bytes)
        is_fake, final_score, classification, desc = pipeline.decide(cnn_score)

        return {
            "error": False,
            "classification": classification,
            "description": desc,
            "is_fake": bool(is_fake),
            "fake_confidence": float(round(final_score, 2)),
            "real_confidence": float(round(100.0 - final_score, 2)),
            "reason": f"[{classification.upper()}] Final probability: {round(final_score, 1)}% based on Hybrid Pipeline Analysis.",
            "signs": pipeline.signs,
            "detailed_analysis": pipeline.results,
            "analyzed_via": "Sentinel X 4-Layer Forensic Pipeline"
        }
    except Exception as e:
        return {"error": True, "reason": f"Analysis Error: {str(e)}"}