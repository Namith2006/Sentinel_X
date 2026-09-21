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

class DoubleGateForensicEnsemble:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.features = {}
        self.signs = []
        self.results = {}

    def gate_1_authenticity_check(self, img: Image.Image):
        """Gate 1: The Real Baseline (Metadata + Compression Noise Floor)"""
        exif = img.getexif()
        has_native_metadata = bool(exif and (0x010f in exif or 0x0110 in exif))
        
        try:
            temp_io = io.BytesIO()
            img.save(temp_io, 'JPEG', quality=90)
            temp_io.seek(0)
            ela_img = ImageChops.difference(img, Image.open(temp_io))
            ela_score = float(np.mean(np.array(ela_img)))
        except Exception:
            ela_score = 0.0

        if has_native_metadata and ela_score <= 8.0:
            auth_tier = "Strongly Real"
            self.signs.append("[GATE 1] Native camera EXIF and low compression detected. Establishing 'Strongly Real' baseline.")
        elif not has_native_metadata and ela_score <= 12.0:
            auth_tier = "Moderate Real"
            self.signs.append("[GATE 1] Stripped metadata and moderate compression (WhatsApp/Social Media). Establishing 'Moderate Real' baseline.")
        else:
            auth_tier = "Low Real"
            self.signs.append("[GATE 1] High compression variance or screenshot detected. Authenticity baseline lowered.")
            
        self.features['auth_tier'] = auth_tier
        self.features['is_screenshot'] = (auth_tier != "Strongly Real")

    def gate_2_synthetic_check(self, cv_img: np.ndarray, image_bytes: bytes) -> float:
        """Gate 2: The AI Baseline (Spectral FFT + CNN)"""
        
        # --- Spectral FFT (Pixel-Level Artifacts) ---
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-7)
        
        mean_mag = float(np.mean(mag))
        std_mag = float(np.std(mag))
        spike_threshold = mean_mag + (3.5 * std_mag)
        spikes = int(np.sum(mag > spike_threshold))
        
        spike_density = float((spikes / (w * h)) * 10000.0)
        fft_threat = max(0.0, min(100.0, spike_density * 9.0))
        self.features['fft_threat'] = fft_threat
        self.signs.append(f"[GATE 2] Spectral FFT calculated a {round(fft_threat, 1)}% structural synthetic threat.")

        # --- Neural Opinion (CNN) ---
        cnn_score = 0.0
        if HF_API_TOKEN:
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "image/jpeg"}
            try:
                response = requests.post(HF_API_URL, headers=headers, data=image_bytes, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        item = data[0] if isinstance(data[0], list) else data[0]
                        for entry in (item if isinstance(item, list) else [item]):
                            label = str(entry.get("label", "")).lower()
                            if "fake" in label or "artificial" in label:
                                cnn_score = float(entry.get("score", 0.0)) * 100.0
                                self.signs.append(f"[GATE 2] Vision Transformer detected {round(cnn_score, 1)}% synthetic markers.")
                                break
            except Exception:
                self.signs.append("[GATE 2] Neural API timeout.")
        
        self.features['cnn_threat'] = cnn_score
        
        # Formulate Gate 2 synthetic score (Semantic CNN backed by Pixel-Level FFT)
        synthetic_score = (cnn_score * 0.75) + (fft_threat * 0.25)
        self.features['synthetic_score'] = synthetic_score
        return synthetic_score

    def evaluate_intersection(self):
        """The Decision Layer: Intersection Logic (Double-Key Lock)"""
        auth_tier = self.features['auth_tier']
        synthetic_score = self.features['synthetic_score']
        is_screenshot = self.features['is_screenshot']
        
        # --- THE DOUBLE-KEY LOCK IMPLEMENTATION ---
        if auth_tier == "Strongly Real":
            if synthetic_score > 98.0:
                final_score = synthetic_score
                verdict = "FAKE"
                reason_text = "Gate 1 established Strong Authenticity, but Gate 2 provided absolute (98%+) synthetic proof. Overriding to FAKE."
            else:
                final_score = min(synthetic_score, 35.0)  # Suppress hallucination
                verdict = "REAL"
                reason_text = "Gate 1 verified Strong Authenticity. Gate 2 synthetic signals rejected as environmental noise."

        elif auth_tier == "Moderate Real":  # The WhatsApp Profile
            if synthetic_score > 95.0:
                final_score = synthetic_score
                verdict = "FAKE"
                reason_text = "WhatsApp/Compressed image detected. Gate 2 provided overwhelming (>95%) synthetic proof. FAKE confirmed."
            elif synthetic_score > 75.0:
                final_score = 65.0  # Push to UNCERTAIN zone
                verdict = "UNCERTAIN"
                reason_text = "WhatsApp/Compressed image detected. Gate 2 is highly suspicious but lacks absolute certainty. Marking UNCERTAIN."
            else:
                final_score = min(synthetic_score, 45.0)
                verdict = "REAL"
                reason_text = "WhatsApp/Compressed image detected. Gate 2 lacks extreme certainty. Forcing REAL classification to prevent CNN hallucination."

        else:  # Low Real (Screenshots / Aggressive compression)
            if synthetic_score > 90.0:
                final_score = synthetic_score
                verdict = "FAKE"
                reason_text = "Low authenticity container. Gate 2 synthetic signals exceeded 90%. FAKE confirmed."
            elif synthetic_score > 60.0:
                final_score = 70.0
                verdict = "UNCERTAIN"
                reason_text = "Low authenticity container with moderate-high AI flags. Marking UNCERTAIN."
            else:
                final_score = min(synthetic_score, 49.0)
                verdict = "REAL"
                reason_text = "Low authenticity container, but Gate 2 lacks sufficient synthetic evidence. Defaulting to REAL."

        self.signs.append(f"[DECISION LAYER] {reason_text}")
        
        is_fake = (verdict == "FAKE")
        
        # UI Classification Mapping
        if is_screenshot and verdict == "FAKE":
            classification = "4_AI_Screenshot"
            desc = "Screenshot / Compressed AI-generated media"
        elif is_screenshot and verdict == "REAL":
            classification = "2_Real_Screenshot"
            desc = "Screenshot / Compressed authentic photograph"
        elif verdict == "FAKE" and not is_screenshot:
            classification = "3_AI_Native"
            desc = "Direct AI-generated diffusion/GAN media"
        elif verdict == "UNCERTAIN":
            classification = "5_Uncertain"
            desc = "Inconclusive / Heavy Compression Masking"
        else:
            classification = "1_Real_Native"
            desc = "Authentic camera photograph"

        return classification, desc, final_score, f"[{classification.upper()}] Intersection Verdict: {reason_text}", is_fake

# ==========================================
# MAIN ANALYSIS ENDPOINT
# ==========================================
def analyze_image(image_path: str) -> dict:
    try:
        with Image.open(image_path) as orig_img:
            img = orig_img.convert('RGB')
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Send clean, unmanipulated pixels to ViT to preserve diffusion noise
            cnn_pil = img.copy()
            cnn_pil.thumbnail((1024, 1024))
            buf = io.BytesIO()
            cnn_pil.save(buf, format="JPEG", quality=85)
            cnn_image_bytes = buf.getvalue()

        ensemble = DoubleGateForensicEnsemble(image_path)
        
        # Execute Gate 1
        ensemble.gate_1_authenticity_check(img)
        
        # Execute Gate 2
        ensemble.gate_2_synthetic_check(cv_img, cnn_image_bytes)
        
        del cv_img
        gc.collect()

        # Execute Intersection Logic
        classification, desc, fused_score, reason, is_fake = ensemble.evaluate_intersection()

        return {
            "error": False,
            "classification": classification,
            "description": desc,
            "is_fake": bool(is_fake),
            "fake_confidence": float(round(fused_score, 2)),
            "real_confidence": float(round(100.0 - fused_score, 2)),
            "reason": str(reason),
            "signs": ensemble.signs,
            "detailed_analysis": {
                "auth_tier": ensemble.features.get("auth_tier", "Unknown"),
                "synthetic_score": float(round(ensemble.features.get("synthetic_score", 0.0), 2)),
                "cnn_threat": float(round(ensemble.features.get("cnn_threat", 0.0), 2)),
                "fft_threat": float(round(ensemble.features.get("fft_threat", 0.0), 2)),
                "is_screenshot": bool(ensemble.features.get("is_screenshot", False))
            },
            "analyzed_via": "Sentinel X Double-Gate Intersection Matrix"
        }

    except Exception as e:
        return {"error": True, "reason": f"Analysis Error: {str(e)}"}