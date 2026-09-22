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
        
        # State variables
        self.auth_tier = "Low"
        self.container_log = ""
        
        # Scores initialized as None to distinguish between 0.0 and "Not Available"
        self.signal_score = None
        self.signal_log = ""
        
        self.neural_score = None
        self.neural_log = ""
        
        self.bio_score = None
        self.bio_log = ""
        
        self.metadata_score = None
        self.metadata_log = ""

    def _resize_inter_area(self, cv_img: np.ndarray) -> np.ndarray:
        h, w = cv_img.shape[:2]
        if max(h, w) > 1024:
            scale = 1024.0 / max(h, w)
            return cv2.resize(cv_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return cv_img

    def gate_1_container(self, img: Image.Image):
        """GATE 1: Container & Compression Analysis"""
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

        # Define Heavy Compression based on ELA
        self.features['is_heavily_compressed'] = (ela_score > 20.0)
        
        if has_native_metadata and ela_score <= 8.0:
            self.auth_tier = "Strong"
            self.container_log = "PASS (Native EXIF Verified)"
        elif not has_native_metadata and ela_score <= 12.0:
            self.auth_tier = "Moderate"
            self.container_log = "WARNING (Standard Compression)"
        else:
            self.auth_tier = "Low"
            self.container_log = "FAIL (Heavy Artifacts/Stripped)"
            
        # Metadata is neutral unless a conflict is found
        self.metadata_score = None 
        self.metadata_log = "Neutral (Stripped or Standard)"

    def gate_2_signal(self, cv_img: np.ndarray):
        """GATE 2: Signal Analysis"""
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = gray.shape
            
            lap_1x = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            gray_half = cv2.resize(gray, (w // 2, h // 2))
            lap_half = float(cv2.Laplacian(gray_half, cv2.CV_32F).var())
            gray_quarter = cv2.resize(gray_half, (w // 4, h // 4))
            lap_quarter = float(cv2.Laplacian(gray_quarter, cv2.CV_32F).var())
            
            scale_variance = float(np.std([lap_1x, lap_half, lap_quarter]))
            spatial_threat = max(0.0, min(100.0, (130.0 - scale_variance) / 3.0))
            
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
            if self.signal_score > 60.0:
                self.signal_log = f"FAIL (Synthetic markers: {round(self.signal_score,1)}%)"
            else:
                self.signal_log = f"PASS (Organic noise: {round(self.signal_score,1)}%)"
                
        finally:
            gc.collect()

    def gate_3_neural(self, image_bytes: bytes):
        """GATE 3: Neural Analysis"""
        if not HF_API_TOKEN:
            self.neural_score = None
            self.neural_log = "API Offline"
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
                        score = float(entry.get("score", 0.0)) * 100.0
                        if "fake" in label or "artificial" in label:
                            self.neural_score = score
                            self.neural_log = f"FAIL (Neural: {round(score,1)}%)"
                        elif "real" in label or "authentic" in label:
                            self.neural_score = 0.0
                            self.neural_log = f"PASS (Neurally Real: {round(score,1)}%)"
                        return
        except Exception:
            self.neural_score = None
            self.neural_log = "Timeout/Error"

    def gate_4_biological(self, cv_img: np.ndarray):
        """GATE 4: Biological Layer with Face Detection"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                self.features['bio_analysis_available'] = False
                self.bio_score = None
                self.bio_log = "Bypassed (No face detected)"
                return

            self.features['bio_analysis_available'] = True
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            roi = cv_img[y:y+h, x:x+w]
            
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)
            lab_var = float(np.var(l_channel))
            
            sobelx = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(l_channel, cv2.CV_32F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobelx**2 + sobely**2)
            sobel_mean = float(np.mean(sobel_mag))
            
            smoothness_threat = max(0.0, min(100.0, (600.0 - lab_var) / 10.0))
            gradient_threat = max(0.0, min(100.0, (25.0 - sobel_mean) * 4.0))
            
            self.bio_score = (smoothness_threat + gradient_threat) / 2.0
            self.bio_log = f"FAIL (Texture: {round(self.bio_score,1)}%)" if self.bio_score > 60 else f"PASS (Organic: {round(self.bio_score,1)}%)"
                
        except Exception:
            self.features['bio_analysis_available'] = False
            self.bio_score = None
            self.bio_log = "Error in bio-analysis"

    def execute_grand_jury(self):
        """THE GRAND JURY: Weighted Evidence Fusion"""
        evidence, weights = [], []

        if self.neural_score is not None:
            evidence.append(self.neural_score)
            weights.append(0.55)

        if self.signal_score is not None:
            w = 0.20 * (0.5 if self.features.get('is_heavily_compressed', False) else 1.0)
            evidence.append(self.signal_score)
            weights.append(w)

        if self.features.get('bio_analysis_available', False) and self.bio_score is not None:
            evidence.append(self.bio_score)
            weights.append(0.15)

        if self.metadata_score is not None:
            evidence.append(self.metadata_score)
            weights.append(0.10)

        total_w = sum(weights)
        if total_w == 0:
            fused_score, verdict = 50.0, "UNCERTAIN"
        else:
            fused_score = sum(s*w for s,w in zip(evidence, weights)) / total_w
            fused_score = max(0.0, min(100.0, fused_score))

        forensic_reliable = not self.features.get('is_heavily_compressed', False)
        smoking_gun = (
            self.neural_score is not None and self.signal_score is not None and
            self.neural_score >= 70 and self.signal_score >= 70 and forensic_reliable
        )

        disagreement = False
        if self.neural_score is not None and self.signal_score is not None:
            if (self.neural_score >= 80 and self.signal_score <= 30) or \
               (self.neural_score <= 20 and self.signal_score >= 70):
                disagreement = True

        if smoking_gun:
            verdict = "AI-GENERATED"
        elif disagreement:
            verdict = "UNCERTAIN"
        elif fused_score >= 70:
            verdict = "AI-GENERATED"
        elif fused_score <= 30:
            verdict = "LIKELY REAL"
        else:
            verdict = "UNCERTAIN"

        audit_report = (
            f"▎ Verdict: {verdict} ({round(fused_score, 1)}%)\\n"
            f"▎ Forensic Audit:\\n"
            f"▎ - [CONTAINER]: {self.container_log}\\n"
            f"▎ - [SIGNAL]: {self.signal_log}\\n"
            f"▎ - [NEURAL]: {self.neural_log}\\n"
            f"▎ - [BIOLOGICAL]: {self.bio_log}\\n"
            f"▎ Evidence Fusion: {round(fused_score, 1)}%"
        )

        return verdict, fused_score, audit_report

def analyze_image(image_path: str) -> dict:
    try:
        ensemble = SentinelXForensicEngine(image_path)
        
        with Image.open(image_path) as orig_img:
            img = orig_img.convert('RGB')
            
            # FIX: Shrink the image using PIL *before* converting to heavy NumPy arrays
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # Now this array is tiny and perfectly safe for 512MB RAM limits
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            ensemble.gate_0_sentinel_guards(img, cv_img)
            
            if ensemble.insufficient_quality:
                classification, desc, fused_score, audit_report, is_fake = ensemble.execute_grand_jury()
                return {
                    "error": True,
                    "classification": classification,
                    "description": desc,
                    "reason": audit_report
                }

            ensemble.gate_1_metadata(img)
            
            buf = io.BytesIO()
            # Image is already max 1024px, so this byte compression is instant
            img.save(buf, format="JPEG", quality=85)
            cnn_image_bytes = buf.getvalue()
            
        ensemble.gate_2_signal(cv_img)
        ensemble.gate_4_biological()
        
        del cv_img
        gc.collect()

        ensemble.gate_3_neural(cnn_image_bytes)
        classification, desc, fused_score, audit_report, is_fake = ensemble.execute_grand_jury()

        return {
            "error": False,
            "verdict": verdict,
            "fused_score": float(round(fused_score, 2)),
            "reason": str(audit_report),
            "detailed_analysis": {
                "auth_tier": ensemble.auth_tier,
                "signal_score": ensemble.signal_score,
                "neural_score": ensemble.neural_score,
                "bio_score": ensemble.bio_score,
                "fused_score": fused_score
            }
        }
    except Exception as e:
        return {"error": True, "reason": f"System Exception: {str(e)}"}
