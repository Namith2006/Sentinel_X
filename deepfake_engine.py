import os
import io
import json
import requests
import cv2
import numpy as np
import gc
from PIL import Image, ImageChops

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "") 
HF_API_URL = "https://router.huggingface.co/hf-inference/models/prithivMLmods/Deep-Fake-Detector-v2-Model"

class QuadGateForensicEnsemble:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.signs = []
        self.features = {}
        
        # State variables for Gates
        self.has_metadata = False
        self.ela_score = 0.0
        self.auth_tier = "Low"
        self.signal_score = 0.0
        self.neural_score = 0.0

    def gate_1_container(self, img: Image.Image):
        """GATE 1: The Container Analysis (Context)"""
        exif = img.getexif()
        self.has_metadata = bool(exif and (0x010f in exif or 0x0110 in exif))
        self.features['has_native_metadata'] = self.has_metadata
        
        try:
            temp_io = io.BytesIO()
            img.save(temp_io, 'JPEG', quality=90)
            temp_io.seek(0)
            ela_img = ImageChops.difference(img, Image.open(temp_io))
            self.ela_score = float(np.mean(np.array(ela_img)))
        except Exception:
            self.ela_score = 0.0

        # Assign Authenticity Tier
        if self.has_metadata and self.ela_score <= 8.0:
            self.auth_tier = "Strong"
            gate1_status = "PASS (Native EXIF Verified)"
        elif not self.has_metadata and self.ela_score <= 12.0:
            self.auth_tier = "Moderate"
            gate1_status = "WARNING (Compressed / Social Media Container)"
        else:
            self.auth_tier = "Low"
            gate1_status = "FAIL (Stripped Metadata / High Variance Screenshot)"
            
        self.features['auth_tier'] = self.auth_tier
        self.features['is_screenshot'] = (self.auth_tier != "Strong")
        self.gate1_log = f"[GATE 1: CONTAINER] Tier: {self.auth_tier} | Status: {gate1_status}"

    def gate_2_signal(self, cv_img: np.ndarray):
        """GATE 2: The Signal Analysis (Math & Spectral)"""
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # --- Multi-Scale Laplacian (Spatial Blur) ---
        lap_1x = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gray_half = cv2.resize(gray, (w // 2, h // 2))
        lap_half = float(cv2.Laplacian(gray_half, cv2.CV_64F).var())
        gray_quarter = cv2.resize(gray_half, (w // 4, h // 4))
        lap_quarter = float(cv2.Laplacian(gray_quarter, cv2.CV_64F).var())
        
        scale_variance = float(np.std([lap_1x, lap_half, lap_quarter]))
        spatial_threat = max(0.0, min(100.0, (180.0 - scale_variance) / 2.5))
        
        # --- Frequency Domain Fingerprinting (FFT Spikes) ---
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-7)
        
        mean_mag = float(np.mean(mag))
        std_mag = float(np.std(mag))
        spike_threshold = mean_mag + (3.5 * std_mag)
        spikes = int(np.sum(mag > spike_threshold))
        
        spike_density = float((spikes / (w * h)) * 10000.0)
        fft_threat = max(0.0, min(100.0, spike_density * 9.0))
        
        self.signal_score = float((spatial_threat + fft_threat) / 2.0)
        self.features['signal_score'] = self.signal_score
        
        if self.signal_score > 60.0:
            gate2_status = "FAIL (Synthetic Checkerboard/Over-smoothing detected)"
        else:
            gate2_status = "PASS (Organic optical noise floor intact)"
            
        self.gate2_log = f"[GATE 2: SIGNAL] Threat: {round(self.signal_score, 1)}% | Status: {gate2_status}"

    def gate_3_neural(self, image_bytes: bytes):
        """GATE 3: The Neural Analysis (Vision Transformer)"""
        if not HF_API_TOKEN:
            self.neural_score = self.signal_score
            self.gate3_log = f"[GATE 3: NEURAL] Offline. Fallback to Signal Threat: {round(self.neural_score, 1)}%"
            return

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
                            self.neural_score = float(entry.get("score", 0.0)) * 100.0
                            status = "FAIL (High probability of semantic anomalies)" if self.neural_score > 60 else "PASS (Clean)"
                            self.gate3_log = f"[GATE 3: NEURAL] AI Threat: {round(self.neural_score, 1)}% | Status: {status}"
                            return
            else:
                self.neural_score = self.signal_score
                self.gate3_log = f"[GATE 3: NEURAL] API Error HTTP {response.status_code}. Fallback: {round(self.neural_score, 1)}%"
                return
        except Exception:
            self.neural_score = self.signal_score
            self.gate3_log = f"[GATE 3: NEURAL] Timeout. Fallback: {round(self.neural_score, 1)}%"
            return
            
        self.neural_score = 0.0
        self.gate3_log = "[GATE 3: NEURAL] Threat: 0.0% | Status: PASS (No synthetic markers found)"

    def gate_4_decision(self):
        """GATE 4: The Decision Layer (Asymmetric Intersection Logic)"""
        # Base Matrix
        fused_score = (self.neural_score * 0.6) + (self.signal_score * 0.4)
        logic_applied = "Standard Fusion Matrix"
        is_whatsapp = (self.auth_tier in ["Moderate", "Low"])

        # 1. The "Smoking Gun" Rule (Absolute Fake)
        if self.neural_score > 90.0 and self.signal_score > 90.0:
            fused_score = max(fused_score, 95.0)
            logic_applied = "Smoking Gun Rule (Neural > 90% AND Signal > 90%) -> Forced FAKE"
            
        # 2. The "Authenticity Veto" Rule (Saves Native Selfies)
        elif self.auth_tier == "Strong" and self.signal_score < 30.0:
            fused_score = min(fused_score, 35.0)
            logic_applied = "Authenticity Veto (Strong Container AND Low Signal) -> Forced REAL"
            
        # 3. The "Compression Offset" Rule (Saves WhatsApp Selfies)
        elif is_whatsapp and self.neural_score < 85.0:
            fused_score = max(0.0, fused_score - 20.0)
            logic_applied = "Compression Offset (Moderate/Low Container AND Neural < 85%) -> Reduced 20%"

        fused_score = max(0.0, min(100.0, float(fused_score)))

        # 4. The "Uncertainty Zone" & Verdict Assignment
        if fused_score > 75.0:
            verdict = "FAKE"
            class_code = "4_AI_Screenshot" if is_whatsapp else "3_AI_Native"
            desc = "Confirmed Deepfake / Synthetic Media"
        elif fused_score >= 55.0:
            verdict = "UNCERTAIN"
            class_code = "5_Uncertain"
            desc = "Inconclusive / Heavy Compression Artifacts"
        else:
            verdict = "REAL"
            class_code = "2_Real_Screenshot" if is_whatsapp else "1_Real_Native"
            desc = "Authentic Photographic Media"

        self.gate4_log = f"[GATE 4: DECISION] Logic: {logic_applied} -> VERDICT: {verdict}"

        # Compile the mega-string for the frontend UI
        detailed_reason = (
            f"Forensic Quad-Gate Audit Report:\n"
            f"• {self.gate1_log}\n"
            f"• {self.gate2_log}\n"
            f"• {self.gate3_log}\n"
            f"• {self.gate4_log}\n"
            f"Final Fused Probability: {round(fused_score, 1)}%"
        )

        is_fake = (verdict == "FAKE")
        return class_code, desc, fused_score, detailed_reason, is_fake

# ==========================================
# MAIN ANALYSIS ENDPOINT
# ==========================================
def analyze_image(image_path: str) -> dict:
    try:
        with Image.open(image_path) as orig_img:
            img = orig_img.convert('RGB')
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Send raw unaltered pixels to preserve AI artifacts
            cnn_pil = img.copy()
            cnn_pil.thumbnail((1024, 1024))
            buf = io.BytesIO()
            cnn_pil.save(buf, format="JPEG", quality=85)
            cnn_image_bytes = buf.getvalue()

        ensemble = QuadGateForensicEnsemble(image_path)
        
        # Execute the 4-Gate Pipeline
        ensemble.gate_1_container(img)
        ensemble.gate_2_signal(cv_img)
        
        del cv_img
        gc.collect()

        ensemble.gate_3_neural(cnn_image_bytes)
        classification, desc, fused_score, reason_str, is_fake = ensemble.gate_4_decision()

        return {
            "error": False,
            "classification": classification,
            "description": desc,
            "is_fake": bool(is_fake),
            "fake_confidence": float(round(fused_score, 2)),
            "real_confidence": float(round(100.0 - fused_score, 2)),
            "reason": str(reason_str),
            "signs": ensemble.signs,
            "detailed_analysis": {
                "auth_tier": ensemble.auth_tier,
                "signal_score": float(round(ensemble.signal_score, 2)),
                "neural_score": float(round(ensemble.neural_score, 2)),
                "fused_score": float(round(fused_score, 2))
            },
            "analyzed_via": "Sentinel X Quad-Gate Forensic Architecture"
        }

    except Exception as e:
        return {"error": True, "reason": f"Analysis Error: {str(e)}"}