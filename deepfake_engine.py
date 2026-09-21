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

class PriorityDecisionEnsemble:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.features = {}
        self.signs = []
        self.results = {}

    def extract_container_context(self, img: Image.Image):
        """Step 1: Container Context (Metadata & Compression)"""
        exif = img.getexif()
        metadata_present = bool(exif and (0x010f in exif or 0x0110 in exif))
        self.features['metadata_present'] = metadata_present
        
        try:
            temp_io = io.BytesIO()
            img.save(temp_io, 'JPEG', quality=90)
            temp_io.seek(0)
            ela_img = ImageChops.difference(img, Image.open(temp_io))
            ela_score = float(np.mean(np.array(ela_img)))
        except Exception:
            ela_score = 0.0

        # Compression is confirmed if native EXIF is stripped or Error Level Analysis shows high variance
        compression_detected = not metadata_present or ela_score > 8.0
        self.features['compression_detected'] = compression_detected
        
        meta_str = "Present" if metadata_present else "Missing/Stripped"
        comp_str = "Detected" if compression_detected else "Not Detected"
        self.signs.append(f"Context Extraction: Metadata: {meta_str} | Compression: {comp_str}")

    def extract_synthetic_markers(self, cv_img: np.ndarray, image_bytes: bytes) -> float:
        """Step 2 & 3: Spectral Math & Neural Opinion Fusion"""
        
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
        self.signs.append(f"Spectral Math: FFT calculated a {round(fft_threat, 1)}% synthetic threat.")

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
                                self.signs.append(f"Neural Model: Vision Transformer detected {round(cnn_score, 1)}% synthetic markers.")
                                break
            except Exception:
                self.signs.append("Neural Model: API timeout.")
        
        # Calculate unified Synthetic Score
        synthetic_score = (cnn_score * 0.75) + (fft_threat * 0.25)
        self.features['synthetic_score'] = synthetic_score
        return synthetic_score

    def execute_decision_tree(self):
        """Step 4: Priority-Based Decision Tree Logic"""
        synthetic_score = float(self.features['synthetic_score'])
        metadata_present = bool(self.features['metadata_present'])
        compression_detected = bool(self.features['compression_detected'])
        
        # --- RULE 1: The Synthetic Override ---
        if synthetic_score > 85.0:
            verdict = "FAKE"
            final_score = max(synthetic_score, 86.0)
            reason_text = "Synthetic Override: Score exceeds 85% absolute synthetic threshold."
            
        # --- RULE 2: The Real Confidence Rule ---
        elif synthetic_score < 60.0 and metadata_present:
            verdict = "REAL"
            final_score = min(synthetic_score, 40.0)
            reason_text = "Real Confidence Rule: Native metadata verified and synthetic markers are low."
            
        # --- RULE 3: The Compression Gray Zone ---
        elif 60.0 <= synthetic_score <= 85.0 and compression_detected:
            verdict = "UNCERTAIN"
            final_score = 65.0  # Lock at the threshold of uncertainty
            reason_text = "Compression Gray Zone: Moderate synthetic signals masked by compression artifacts."
            
        # --- RULE 4: Default Safety ---
        else:
            verdict = "REAL"
            final_score = min(synthetic_score, 49.0)
            reason_text = "Default Safety: Ambiguous state defaults to authentic to prevent False Positives."

        self.signs.append(f"Decision Tree: {reason_text}")
        is_fake = (verdict == "FAKE")
        
        # UI Classification Mapping
        if compression_detected and verdict == "FAKE":
            classification = "4_AI_Screenshot"
            desc = "Screenshot / Compressed AI-generated media"
        elif compression_detected and verdict == "REAL":
            classification = "2_Real_Screenshot"
            desc = "Screenshot / Compressed authentic photograph"
        elif verdict == "FAKE" and not compression_detected:
            classification = "3_AI_Native"
            desc = "Direct AI-generated diffusion/GAN media"
        elif verdict == "UNCERTAIN":
            classification = "5_Uncertain"
            desc = "Inconclusive / Heavy Compression Masking"
        else:
            classification = "1_Real_Native"
            desc = "Authentic camera photograph"

        return classification, desc, final_score, f"[{classification.upper()}] Decision Tree Verdict: {reason_text}", is_fake

# ==========================================
# MAIN ANALYSIS ENDPOINT
# ==========================================
def analyze_image(image_path: str) -> dict:
    try:
        with Image.open(image_path) as orig_img:
            img = orig_img.convert('RGB')
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            cnn_pil = img.copy()
            cnn_pil.thumbnail((1024, 1024))
            buf = io.BytesIO()
            cnn_pil.save(buf, format="JPEG", quality=85)
            cnn_image_bytes = buf.getvalue()

        ensemble = PriorityDecisionEnsemble(image_path)
        
        ensemble.extract_container_context(img)
        ensemble.extract_synthetic_markers(cv_img, cnn_image_bytes)
        
        del cv_img
        gc.collect()

        classification, desc, final_score, reason, is_fake = ensemble.execute_decision_tree()

        return {
            "error": False,
            "classification": classification,
            "description": desc,
            "is_fake": bool(is_fake),
            "fake_confidence": float(round(final_score, 2)),
            "real_confidence": float(round(100.0 - final_score, 2)),
            "reason": str(reason),
            "signs": ensemble.signs,
            "detailed_analysis": {
                "metadata_present": ensemble.features.get("metadata_present", False),
                "compression_detected": ensemble.features.get("compression_detected", False),
                "synthetic_score": float(round(ensemble.features.get("synthetic_score", 0.0), 2))
            },
            "analyzed_via": "Sentinel X Priority-Based Decision Tree"
        }

    except Exception as e:
        return {"error": True, "reason": f"Analysis Error: {str(e)}"}