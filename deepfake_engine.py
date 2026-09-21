import os
import io
import requests
import cv2
import numpy as np
import gc
from PIL import Image, ImageChops

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "") 
HF_API_URL = "https://router.huggingface.co/hf-inference/models/prithivMLmods/Deep-Fake-Detector-v2-Model"

class EvidenceFusionEngine:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.features = {}
        
        # Symmetrical Threat Probabilities (0-100)
        self.meta_threat = 0.0
        self.signal_threat = 0.0
        self.neural_threat = 0.0
        self.bio_threat = 0.0
        
        # State & Logging
        self.is_compressed = False
        self.face_detected = False
        self.audit_logs = {}

    def _resize_inter_area(self, cv_img: np.ndarray) -> np.ndarray:
        """Memory Cap: Adaptive resize using INTER_AREA to prevent aliasing."""
        h, w = cv_img.shape[:2]
        if max(h, w) > 1024:
            scale = 1024.0 / max(h, w)
            return cv2.resize(cv_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return cv_img

    def gate_1_metadata(self, img: Image.Image):
        """GATE 1: Metadata & Container (Weak Corroboration - 10% Base Weight)"""
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

        self.is_compressed = not has_metadata or ela_score > 10.0
        
        # Symmetrical Probability Assignment
        if has_metadata:
            self.meta_threat = 15.0  # Strong indicator of reality
            self.audit_logs['metadata'] = "Native EXIF Verified (Threat: 15.0%)"
        else:
            self.meta_threat = 75.0  # Suspicious, but common in social media
            self.audit_logs['metadata'] = "Metadata Stripped / Compressed Container (Threat: 75.0%)"

    def gate_2_signal(self, cv_img: np.ndarray):
        """GATE 2: Forensic Signal Analysis (20% Base Weight)"""
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = gray.shape
            
            # Spatial Blur (Laplacian)
            lap_1x = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            gray_half = cv2.resize(gray, (w // 2, h // 2))
            lap_half = float(cv2.Laplacian(gray_half, cv2.CV_32F).var())
            scale_variance = float(np.std([lap_1x, lap_half]))
            spatial_threat = max(0.0, min(100.0, (150.0 - scale_variance) / 1.5))
            
            # Spectral Spikes (FFT)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            mag = 20 * np.log(np.abs(fshift) + 1e-7)
            
            spike_threshold = float(np.mean(mag)) + (3.5 * float(np.std(mag)))
            spikes = int(np.sum(mag > spike_threshold))
            spike_density = float((spikes / (w * h)) * 10000.0)
            fft_threat = max(0.0, min(100.0, spike_density * 8.5))
            
            self.signal_threat = float((spatial_threat + fft_threat) / 2.0)
            self.audit_logs['signal'] = f"Spectral/Spatial Anomalies (Threat: {round(self.signal_threat,1)}%)"
        finally:
            del gray
            if 'f' in locals(): del f
            if 'fshift' in locals(): del fshift
            if 'mag' in locals(): del mag
            gc.collect()

    def gate_3_neural(self, image_bytes: bytes):
        """GATE 3: Neural Detector (55% Base Weight)"""
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

    def gate_4_biological(self, cv_img: np.ndarray):
        """GATE 4: Biological Texture (Conditional 15% Base Weight)"""
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Fast Haar Cascade Face Detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if not face_cascade.empty():
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            else:
                faces = ()
                
            if len(faces) > 0:
                self.face_detected = True
                # Use the largest face found
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                roi = cv_img[y:y+h, x:x+w]
            else:
                # Face-dependent condition not met
                self.face_detected = False
                self.bio_threat = 50.0
                self.audit_logs['biological'] = "No Face Detected -> Bio-Weight Redistributed."
                return
            
            # LAB Variance (Skin Smoothness) on Face ROI
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)
            
            lab_var = float(np.var(l_channel))
            smoothness_threat = max(0.0, min(100.0, (600.0 - lab_var) / 6.0))
            
            self.bio_threat = smoothness_threat
            self.audit_logs['biological'] = f"Facial ROI Texture Analysis (Threat: {round(self.bio_threat,1)}%)"
                
        finally:
            if 'gray' in locals(): del gray
            if 'roi' in locals(): del roi
            if 'lab' in locals(): del lab
            if 'l_channel' in locals(): del l_channel
            gc.collect()

    def execute_grand_jury(self):
        """THE GRAND JURY: Weighted Evidence Fusion & Safety Mechanisms"""
        
        # 1. Base Weight Initialization
        w_neural = 0.55
        w_signal = 0.20
        w_bio = 0.15
        w_meta = 0.10

        logic_applied = "Standard Normalized Evidence Fusion"

        # 2. Dynamic Weighting Logic
        if self.is_compressed:
            w_signal *= 0.5  # Reduce forensic signal reliability heavily
            logic_applied = "Compression Detected -> Forensic Weight Reduced"
            
        if not self.face_detected:
            w_bio = 0.0      # Remove bio influence entirely
            
        # Normalize weights so they equal 1.0
        total_w = w_neural + w_signal + w_bio + w_meta
        w_neural /= total_w
        w_signal /= total_w
        w_bio /= total_w
        w_meta /= total_w

        # 3. Calculate Fused Probability
        fused_score = (
            (self.neural_threat * w_neural) +
            (self.signal_threat * w_signal) +
            (self.bio_threat * w_bio) +
            (self.meta_threat * w_meta)
        )

        # 4. Safety Mechanisms (Preventing Overconfidence)
        active_threats = [self.neural_threat, self.signal_threat, self.meta_threat]
        if self.face_detected: 
            active_threats.append(self.bio_threat)
            
        disagreement_gap = max(active_threats) - min(active_threats)

        if self.neural_threat >= 70.0 and self.signal_threat >= 70.0:
            fused_score = max(fused_score, 85.0)
            logic_applied = "Smoking Gun Override (Neural & Signal > 70%)"
        elif disagreement_gap > 65.0:
            # Force score toward the uncertain center (50.0) if evidence wildly conflicts
            fused_score = 50.0 
            logic_applied = f"Disagreement Detection (Gap: {round(disagreement_gap)}%) -> Forced UNCERTAIN"

        fused_score = max(0.0, min(100.0, float(fused_score)))

        # 5. Threshold Refinement
        if fused_score >= 70.0:
            verdict = "🚨 AI-GENERATED"
            class_code = "3_AI_Native" if not self.is_compressed else "4_AI_Screenshot"
            desc = "High Probability of Synthetic Media"
        elif fused_score > 30.0:
            verdict = "⚠️ UNCERTAIN"
            class_code = "5_Uncertain"
            desc = "Conflicting Evidence / Heavy Compression"
        else:
            verdict = "✅ LIKELY REAL"
            class_code = "1_Real_Native" if not self.is_compressed else "2_Real_Screenshot"
            desc = "Organic Characteristics Verified"

        # Forensic Audit Report
        audit_report = (
            f"▎ Verdict: {verdict} ({round(fused_score, 1)}% Probability)\n"
            f"▎ Evidence Board:\n"
            f"▎ - [NEURAL]  (Weight {round(w_neural*100)}%): {self.audit_logs.get('neural', 'N/A')}\n"
            f"▎ - [SIGNAL]  (Weight {round(w_signal*100)}%): {self.audit_logs.get('signal', 'N/A')}\n"
            f"▎ - [BIO]     (Weight {round(w_bio*100)}%): {self.audit_logs.get('biological', 'N/A')}\n"
            f"▎ - [META]    (Weight {round(w_meta*100)}%): {self.audit_logs.get('metadata', 'N/A')}\n"
            f"▎ Decision Logic: {logic_applied}"
        )

        is_fake = (fused_score >= 70.0)
        return class_code, desc, fused_score, audit_report, is_fake

# ==========================================
# MAIN ANALYSIS ENDPOINT
# ==========================================
def analyze_image(image_path: str) -> dict:
    try:
        ensemble = EvidenceFusionEngine(image_path)
        
        with Image.open(image_path) as orig_img:
            img = orig_img.convert('RGB')
            # Gate 1
            ensemble.gate_1_metadata(img)
            
            raw_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv_img = ensemble._resize_inter_area(raw_cv)
            del raw_cv
            
            buf = io.BytesIO()
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            img.save(buf, format="JPEG", quality=85)
            cnn_image_bytes = buf.getvalue()
            
        # Gate 2 & 4
        ensemble.gate_2_signal(cv_img)
        ensemble.gate_4_biological(cv_img)
        del cv_img
        gc.collect()

        # Gate 3 & Grand Jury Fusion
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
                "face_detected": bool(ensemble.face_detected)
            },
            "analyzed_via": "Sentinel X Weighted Evidence Fusion"
        }

    except Exception as e:
        return {"error": True, "reason": f"System Exception: {str(e)}"}