import os
import io
import requests
import cv2
import numpy as np
import gc
from PIL import Image, ImageChops

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "") 
HF_API_URL = "https://router.huggingface.co/hf-inference/models/prithivMLmods/Deep-Fake-Detector-v2-Model"

class SentinelXForensicEngine:
    def __init__(self, image_path: str):
        self.image_path = image_path
        
        # Threat Probabilities (0-100)
        self.meta_threat = 0.0
        self.signal_threat = 0.0
        self.neural_threat = 0.0
        self.bio_threat = 0.0
        
        # State & Logging
        self.is_compressed = False
        self.bio_valid = False
        self.face_roi = None
        self.audit_logs = {}
        
        # Guardrails
        self.insufficient_quality = False

    def _resize_inter_area(self, cv_img: np.ndarray) -> np.ndarray:
        h, w = cv_img.shape[:2]
        if max(h, w) > 1024:
            scale = 1024.0 / max(h, w)
            return cv2.resize(cv_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return cv_img

    def gate_0_sentinel_guards(self, img: Image.Image, cv_img: np.ndarray):
        """GATE 0: Resolution & Face Presence Guards (Anti-GIGO)"""
        w, h = img.size
        if w < 256 or h < 256:
            self.insufficient_quality = True
            self.audit_logs['sentinel'] = f"Resolution Guard Failed ({w}x{h}). Minimum 256x256 required."
            return

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if not face_cascade.empty():
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                self.bio_valid = True
                # Extract largest face for Biological Gate
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                self.face_roi = cv_img[y:y+h, x:x+w]
                self.audit_logs['sentinel'] = "Resolution & Face Guards Passed."
                return
                
        self.bio_valid = False
        self.audit_logs['sentinel'] = "Resolution Passed. No Face Detected -> Biological Gate Bypassed."

    def gate_1_metadata(self, img: Image.Image):
        """GATE 1: Advanced Metadata & Container Analysis"""
        exif = img.getexif()
        exif_str = str(exif).lower() if exif else ""
        
        # 1. AI Signature Detection
        ai_signatures = ["midjourney", "dall-e", "stable diffusion", "ai generated", "software: adobe photoshop"]
        has_ai_sig = any(sig in exif_str for sig in ai_signatures)
        
        has_metadata = bool(exif and (0x010f in exif or 0x0110 in exif))
        
        try:
            temp_io = io.BytesIO()
            img.save(temp_io, 'JPEG', quality=90)
            temp_io.seek(0)
            ela_img = ImageChops.difference(img, Image.open(temp_io))
            ela_score = float(np.mean(np.array(ela_img)))
        except Exception:
            ela_score = 0.0

        self.is_compressed = not has_metadata or ela_score > 10.0
        
        if has_ai_sig:
            self.meta_threat = 90.0
            self.audit_logs['metadata'] = "Generative AI/Editing signatures found in metadata (Threat: 90.0%)"
        elif has_metadata and not self.is_compressed:
            self.meta_threat = 10.0
            self.audit_logs['metadata'] = "Organic camera EXIF verified (Threat: 10.0%)"
        else:
            self.meta_threat = 75.0
            self.audit_logs['metadata'] = "Metadata stripped / Compressed container (Threat: 75.0%)"

    def gate_2_signal(self, cv_img: np.ndarray):
        """GATE 2: Forensic Signal Analysis (Symmetric Scoring)"""
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = gray.shape
            
            lap_1x = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            gray_half = cv2.resize(gray, (w // 2, h // 2))
            lap_half = float(cv2.Laplacian(gray_half, cv2.CV_32F).var())
            scale_variance = float(np.std([lap_1x, lap_half]))
            spatial_threat = max(0.0, min(100.0, (150.0 - scale_variance) / 1.5))
            
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            mag = 20 * np.log(np.abs(fshift) + 1e-7)
            
            spike_threshold = float(np.mean(mag)) + (3.5 * float(np.std(mag)))
            spikes = int(np.sum(mag > spike_threshold))
            spike_density = float((spikes / (w * h)) * 10000.0)
            fft_threat = max(0.0, min(100.0, spike_density * 8.5))
            
            self.signal_threat = float((spatial_threat + fft_threat) / 2.0)
            self.audit_logs['signal'] = f"Spectral/Spatial Analysis (Threat: {round(self.signal_threat,1)}%)"
        finally:
            del gray
            if 'f' in locals(): del f
            if 'fshift' in locals(): del fshift
            if 'mag' in locals(): del mag
            gc.collect()

    def gate_3_neural(self, image_bytes: bytes):
        """GATE 3: Neural Detector (Symmetric Scoring)"""
        if not HF_API_TOKEN:
            self.neural_threat = 50.0
            self.audit_logs['neural'] = "API Offline. Neutral Probability Assigned (50.0%)"
            return

        headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "image/jpeg"}
        try:
            response = requests.post(HF_API_URL, headers=headers, data=image_bytes, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    item = data[0] if isinstance(data[0], list) else data[0]
                    fake_prob = 0.0
                    for entry in (item if isinstance(item, list) else [item]):
                        label = str(entry.get("label", "")).lower()
                        score = float(entry.get("score", 0.0)) * 100.0
                        if "fake" in label or "artificial" in label:
                            fake_prob = score
                        elif "real" in label and fake_prob == 0.0:
                            fake_prob = 100.0 - score
                            
                    self.neural_threat = max(0.0, min(100.0, fake_prob))
                    self.audit_logs['neural'] = f"ViT Semantic Analysis (Threat: {round(self.neural_threat,1)}%)"
                    return
        except Exception:
            pass
            
        self.neural_threat = 50.0
        self.audit_logs['neural'] = "API Timeout/Error. Neutral Probability Assigned (50.0%)"

    def gate_4_biological(self):
        """GATE 4: Biological Texture (Executes strictly on Detected Face ROI)"""
        if not self.bio_valid or self.face_roi is None:
            self.bio_threat = 50.0
            self.audit_logs['biological'] = "Gate Bypassed (No Face Detected)."
            return
            
        try:
            lab = cv2.cvtColor(self.face_roi, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)
            
            lab_var = float(np.var(l_channel))
            self.bio_threat = max(0.0, min(100.0, (600.0 - lab_var) / 6.0))
            self.audit_logs['biological'] = f"Facial ROI Texture Analysis (Threat: {round(self.bio_threat,1)}%)"
        finally:
            del lab
            del l_channel
            gc.collect()

    def execute_grand_jury(self):
        """THE GRAND JURY: Dynamic Fusion & Safety Mechanisms"""
        if self.insufficient_quality:
            return "0_Error", "Insufficient Quality", 0.0, "Resolution Guard Failed. Image too small for forensic analysis.", False

        w_neural, w_signal, w_bio, w_meta = 0.55, 0.20, 0.15, 0.10
        logic_applied = "Standard Normalized Evidence Fusion"

        if self.is_compressed:
            w_signal *= 0.5  
            logic_applied = "Compression Detected -> Forensic Signal Weight Halved"
            
        if not self.bio_valid:
            w_bio = 0.0      
            
        total_w = w_neural + w_signal + w_bio + w_meta
        w_neural, w_signal, w_bio, w_meta = w_neural/total_w, w_signal/total_w, w_bio/total_w, w_meta/total_w

        fused_score = (
            (self.neural_threat * w_neural) +
            (self.signal_threat * w_signal) +
            (self.bio_threat * w_bio) +
            (self.meta_threat * w_meta)
        )

        # Safety Mechanisms
        active_threats = [self.neural_threat, self.signal_threat, self.meta_threat]
        if self.bio_valid: 
            active_threats.append(self.bio_threat)
            
        disagreement_gap = max(active_threats) - min(active_threats)

        if self.neural_threat >= 70.0 and self.signal_threat >= 70.0:
            fused_score = max(fused_score, 85.0)
            logic_applied = "Smoking Gun Override (Neural & Signal > 70%)"
        elif disagreement_gap > 55.0:
            fused_score = 50.0 
            logic_applied = f"Disagreement Detection (Gap: {round(disagreement_gap)}%) -> Forced UNCERTAIN"

        fused_score = max(0.0, min(100.0, float(fused_score)))

        # 3-Tier Thresholds
        if fused_score >= 70.0:
            verdict = "🚨 AI-GENERATED"
            class_code = "3_AI_Native" if not self.is_compressed else "4_AI_Screenshot"
            desc = "High Probability of Synthetic Media"
        elif fused_score >= 31.0:
            verdict = "⚠️ UNCERTAIN"
            class_code = "5_Uncertain"
            desc = "Conflicting Evidence / Heavy Compression"
        else:
            verdict = "✅ LIKELY REAL"
            class_code = "1_Real_Native" if not self.is_compressed else "2_Real_Screenshot"
            desc = "Organic Characteristics Verified"

        audit_report = (
            f"▎ Verdict: {verdict} ({round(fused_score, 1)}% Probability)\n"
            f"▎ Evidence Board:\n"
            f"▎ - [GUARDS]  {self.audit_logs.get('sentinel', 'N/A')}\n"
            f"▎ - [NEURAL]  (Weight {round(w_neural*100)}%): {self.audit_logs.get('neural', 'N/A')}\n"
            f"▎ - [SIGNAL]  (Weight {round(w_signal*100)}%): {self.audit_logs.get('signal', 'N/A')}\n"
            f"▎ - [BIO]     (Weight {round(w_bio*100)}%): {self.audit_logs.get('biological', 'N/A')}\n"
            f"▎ - [META]    (Weight {round(w_meta*100)}%): {self.audit_logs.get('metadata', 'N/A')}\n"
            f"▎ Decision Logic: {logic_applied}"
        )

        is_fake = (fused_score >= 70.0)
        return class_code, desc, fused_score, audit_report, is_fake

def analyze_image(image_path: str) -> dict:
    try:
        ensemble = SentinelXForensicEngine(image_path)
        
        with Image.open(image_path) as orig_img:
            img = orig_img.convert('RGB')
            
            raw_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv_img = ensemble._resize_inter_area(raw_cv)
            del raw_cv
            
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
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
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
            "classification": classification,
            "description": desc,
            "is_fake": bool(is_fake),
            "fake_confidence": float(round(fused_score, 2)),
            "real_confidence": float(round(100.0 - fused_score, 2)),
            "reason": str(audit_report),
            "detailed_analysis": {
                "neural_threat": float(round(ensemble.neural_threat, 2)),
                "signal_threat": float(round(ensemble.signal_threat, 2)),
                "bio_threat": float(round(ensemble.bio_threat, 2)),
                "meta_threat": float(round(ensemble.meta_threat, 2)),
                "fused_probability": float(round(fused_score, 2)),
                "face_detected": bool(ensemble.bio_valid)
            },
            "analyzed_via": "Sentinel X Master Probabilistic Engine"
        }

    except Exception as e:
        return {"error": True, "reason": f"System Exception: {str(e)}"}