import os
import io
import requests
import cv2
import numpy as np
import gc
from PIL import Image, ImageChops

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "") 
HF_API_URL = "https://router.huggingface.co/hf-inference/models/prithivMLmods/Deep-Fake-Detector-v2-Model"

class QuadGateForensicSuite:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.features = {}
        
        # State variables for Gates
        self.has_metadata = False
        self.auth_tier = "Low"
        self.container_log = ""
        
        self.signal_score = 0.0
        self.signal_log = ""
        
        self.neural_score = 0.0
        self.neural_log = ""
        
        self.bio_score = 0.0
        self.bio_log = ""

    def _resize_inter_area(self, cv_img: np.ndarray) -> np.ndarray:
        """Memory Cap: Adaptive resize using INTER_AREA to prevent aliasing artifacts."""
        h, w = cv_img.shape[:2]
        if max(h, w) > 1024:
            scale = 1024.0 / max(h, w)
            return cv2.resize(cv_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return cv_img

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
            ela_score = float(np.mean(np.array(ela_img)))
        except Exception:
            ela_score = 0.0

        if self.has_metadata and ela_score <= 8.0:
            self.auth_tier = "Strong"
            self.container_log = "PASS (Native EXIF Verified)"
        elif not self.has_metadata and ela_score <= 12.0:
            self.auth_tier = "Moderate"
            self.container_log = "WARNING (WhatsApp / Social Media Compression)"
        else:
            self.auth_tier = "Low"
            self.container_log = "FAIL (Stripped Metadata / High Variance Screenshot)"
            
        self.features['auth_tier'] = self.auth_tier
        self.features['is_compressed'] = (self.auth_tier != "Strong")

    def gate_2_signal(self, cv_img: np.ndarray):
        """GATE 2: The Signal Analysis (Math & Spectral) - Float32 RAM Discipline"""
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = gray.shape
            
            # --- Multi-Scale Laplacian (Spatial Blur) ---
            lap_1x = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            gray_half = cv2.resize(gray, (w // 2, h // 2))
            lap_half = float(cv2.Laplacian(gray_half, cv2.CV_32F).var())
            gray_quarter = cv2.resize(gray_half, (w // 4, h // 4))
            lap_quarter = float(cv2.Laplacian(gray_quarter, cv2.CV_32F).var())
            
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
                self.signal_log = f"FAIL (Synthetic Checkerboard/Over-smoothing detected - {round(self.signal_score,1)}%)"
            else:
                self.signal_log = f"PASS (Organic optical noise floor intact - {round(self.signal_score,1)}%)"
                
        finally:
            del gray
            if 'f' in locals(): del f
            if 'fshift' in locals(): del fshift
            if 'mag' in locals(): del mag
            gc.collect()

    def gate_3_neural(self, image_bytes: bytes):
        """GATE 3: The Neural Analysis (ViT Transformer) - 10s Timeout limit"""
        if not HF_API_TOKEN:
            self.neural_score = self.signal_score
            self.neural_log = "API Offline. Fallback to Signal Threat."
            return

        headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "image/jpeg"}
        try:
            response = requests.post(HF_API_URL, headers=headers, data=image_bytes, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    item = data[0] if isinstance(data[0], list) else data[0]
                    for entry in (item if isinstance(item, list) else [item]):
                        label = str(entry.get("label", "")).lower()
                        if "fake" in label or "artificial" in label:
                            self.neural_score = float(entry.get("score", 0.0)) * 100.0
                            status = "FAIL (High probability of semantic anomalies)" if self.neural_score > 60 else "PASS (Clean)"
                            self.neural_log = f"{status} - Confidence {round(self.neural_score,1)}%"
                            return
            else:
                self.neural_score = self.signal_score
                self.neural_log = f"API Error (HTTP {response.status_code}). Fallback applied."
                return
        except Exception:
            self.neural_score = self.signal_score
            self.neural_log = "Timeout Exception (10s cap). Fallback applied."
            return
            
        self.neural_score = 0.0
        self.neural_log = "PASS (No synthetic markers found) - Confidence 0.0%"

    def gate_4_biological(self, cv_img: np.ndarray):
        """GATE 4: The Biological Layer (LAB/Sobel) - Surgical 50% ROI Crop"""
        try:
            h, w = cv_img.shape[:2]
            
            # ROI Crop: Center 50% reduces memory footprint by 75%
            start_y, end_y = h // 4, 3 * h // 4
            start_x, end_x = w // 4, 3 * w // 4
            roi = cv_img[start_y:end_y, start_x:end_x]
            
            # LAB Variance (Skin Smoothness)
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)
            
            lab_var = float(np.var(l_channel))
            
            # Sobel Noise Floor (Detecting unnaturally smooth gradients)
            sobelx = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(l_channel, cv2.CV_32F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobelx**2 + sobely**2)
            sobel_mean = float(np.mean(sobel_mag))
            
            smoothness_threat = max(0.0, min(100.0, (800.0 - lab_var) / 8.0))
            gradient_threat = max(0.0, min(100.0, (25.0 - sobel_mean) * 4.0))
            
            self.bio_score = (smoothness_threat + gradient_threat) / 2.0
            
            if self.bio_score > 60.0:
                self.bio_log = f"FAIL (LAB Variance Low: Plastic Skin / Synthetic Texture - {round(self.bio_score,1)}%)"
            else:
                self.bio_log = f"PASS (Organic Micro-Texture Verified - {round(self.bio_score,1)}%)"
                
        finally:
            if 'roi' in locals(): del roi
            if 'lab' in locals(): del lab
            if 'l_channel' in locals(): del l_channel
            if 'sobelx' in locals(): del sobelx
            if 'sobely' in locals(): del sobely
            if 'sobel_mag' in locals(): del sobel_mag
            gc.collect()

    def execute_grand_jury(self):
        """THE GRAND JURY: Intersection Logic & Forensic Audit Reporting"""
        is_compressed = self.features.get('is_compressed', False)
        
        # Base Matrix
        fused_score = (self.neural_score * 0.5) + (self.signal_score * 0.25) + (self.bio_score * 0.25)
        logic_applied = "Standard Fusion Matrix"

        # 1. The "Smoking Gun" Rule (Absolute Fake)
        if self.neural_score > 90.0 and self.signal_score > 90.0:
            fused_score = max(fused_score, 95.0)
            logic_applied = "Smoking Gun Rule (Neural > 90% AND Signal > 90%) -> Forced FAKE"
            
        # 2. The "Compression Offset" Rule (Saves WhatsApp Selfies)
        elif is_compressed and self.neural_score < 85.0:
            fused_score = max(0.0, fused_score - 20.0)
            logic_applied = "Compression Offset (Moderate/Low Container AND Neural < 85%) -> Reduced 20%"

        # 3. The "Authenticity Veto" Rule (Saves Native Selfies)
        elif self.auth_tier == "Strong" and self.signal_score < 30.0:
            fused_score = min(fused_score, 35.0)
            logic_applied = "Authenticity Veto (Strong Container AND Low Signal) -> Forced REAL"

        fused_score = max(0.0, min(100.0, float(fused_score)))

        # 4. The "Uncertainty Zone" & Verdict Assignment
        if fused_score > 75.0:
            verdict = "🚨 DEEPFAKE DETECTED"
            class_code = "4_AI_Screenshot" if is_compressed else "3_AI_Native"
            desc = "Confirmed Deepfake / Synthetic Media"
        elif fused_score >= 55.0:
            verdict = "⚠️ UNCERTAIN"
            class_code = "5_Uncertain"
            desc = "Inconclusive / Heavy Compression Artifacts"
        else:
            verdict = "✅ AUTHENTIC"
            class_code = "2_Real_Screenshot" if is_compressed else "1_Real_Native"
            desc = "Authentic Photographic Media"

        # Forensic Audit Report Generation
        audit_report = (
            f"▎ Verdict: {verdict} ({round(fused_score, 1)}%)\n"
            f"▎ Forensic Audit:\n"
            f"▎ - [CONTAINER]: {self.container_log}\n"
            f"▎ - [SIGNAL]: {self.signal_log}\n"
            f"▎ - [NEURAL]: {self.neural_log}\n"
            f"▎ - [BIOLOGICAL]: {self.bio_log}\n"
            f"▎ Decision: {logic_applied}"
        )

        is_fake = (fused_score > 75.0)
        return class_code, desc, fused_score, audit_report, is_fake

# ==========================================
# MAIN ANALYSIS ENDPOINT
# ==========================================
def analyze_image(image_path: str) -> dict:
    try:
        ensemble = QuadGateForensicSuite(image_path)
        
        with Image.open(image_path) as orig_img:
            img = orig_img.convert('RGB')
            # Gate 1
            ensemble.gate_1_container(img)
            
            # Pre-allocate for OpenCV operations
            raw_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv_img = ensemble._resize_inter_area(raw_cv)
            del raw_cv
            
            # Compress strict bytes for HF API
            buf = io.BytesIO()
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            img.save(buf, format="JPEG", quality=85)
            cnn_image_bytes = buf.getvalue()
            
        # Gate 2 & 4
        ensemble.gate_2_signal(cv_img)
        ensemble.gate_4_biological(cv_img)
        del cv_img
        gc.collect()

        # Gate 3 & Grand Jury
        ensemble.gate_3_neural(cnn_image_bytes)
        classification, desc, fused_score, audit_report, is_fake = ensemble.execute_grand_jury()

        return {
            "error": False,
            "classification": classification,
            "description": desc,
            "is_fake": bool(is_fake),
            "fake_confidence": float(round(fused_score, 2)),
            "real_confidence": float(round(100.0 - fused_score, 2)),
            "reason": str(audit_report),
            "detailed_analysis": {
                "auth_tier": ensemble.auth_tier,
                "signal_score": float(round(ensemble.signal_score, 2)),
                "neural_score": float(round(ensemble.neural_score, 2)),
                "bio_score": float(round(ensemble.bio_score, 2)),
                "fused_score": float(round(fused_score, 2))
            },
            "analyzed_via": "Sentinel X Quad-Gate Digital Forensic Suite"
        }

    except Exception as e:
        return {"error": True, "reason": f"System Exception: {str(e)}"}