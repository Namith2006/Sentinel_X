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

class MultiPillarForensicEnsemble:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.features = {}
        self.signs = []
        self.pillars = {}
        
    def analyze_container(self, img: Image.Image):
        """Pillar 1: Metadata Verification & Error Level Compression"""
        exif = img.getexif()
        has_native_metadata = bool(exif and (0x010f in exif or 0x0110 in exif))
        self.features['has_native_metadata'] = has_native_metadata
        
        try:
            temp_io = io.BytesIO()
            img.save(temp_io, 'JPEG', quality=90)
            temp_io.seek(0)
            ela_img = ImageChops.difference(img, Image.open(temp_io))
            ela_score = float(np.mean(np.array(ela_img)))
        except Exception:
            ela_score = 0.0

        is_screenshot = not has_native_metadata or ela_score > 8.0
        self.features['is_screenshot'] = is_screenshot

        if is_screenshot and ela_score > 12.0:
            quality = "Low"
        elif is_screenshot or ela_score > 5.0:
            quality = "Medium"
        else:
            quality = "High"
            
        self.features['image_quality'] = quality
        
        meta_status = "Verified Authentic Camera Metadata" if has_native_metadata else "Stripped / Uniform Compression (WhatsApp/Screenshot)"
        self.pillars['metadata'] = meta_status
        self.signs.append(f"[PILLAR 1: CONTAINER] Status: {meta_status} (Quality Tier: {quality})")

    def calculate_math_threat(self, cv_img: np.ndarray):
        """Pillars 2 & 3: Spectral FFT Spikes & Spatial Edge Gradient Coherence"""
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. Multi-Scale Laplacian Analysis (Upscaling Seams)
        lap_1x = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gray_half = cv2.resize(gray, (w // 2, h // 2))
        lap_half = float(cv2.Laplacian(gray_half, cv2.CV_64F).var())
        gray_quarter = cv2.resize(gray_half, (w // 4, h // 4))
        lap_quarter = float(cv2.Laplacian(gray_quarter, cv2.CV_64F).var())
        
        scale_variance = float(np.std([lap_1x, lap_half, lap_quarter]))
        multi_scale_threat = max(0.0, min(100.0, (180.0 - scale_variance) / 2.5))
        
        # 2. Edge Density & Boundary Coherence (Sobel Gradient)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        
        edge_density = float(np.sum(sobel_mag > 30) / (h * w))
        edge_threat = max(0.0, min(100.0, (0.09 - edge_density) * 600.0))
        edge_status = "Anomalous Blurring / Low Grain" if edge_threat > 50.0 else "Organic Optical Grain & Sharp Edge Transitions"
        self.pillars['edge_coherence'] = edge_status
        self.signs.append(f"[PILLAR 2: EDGE COHERENCE] {edge_status} (Threat: {round(edge_threat, 1)}%)")
        
        # 3. Frequency Domain Fingerprinting (FFT Spectral Spikes)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-7)
        
        mean_mag = float(np.mean(mag))
        std_mag = float(np.std(mag))
        spike_threshold = mean_mag + (3.5 * std_mag)
        spikes = int(np.sum(mag > spike_threshold))
        
        spike_density = float((spikes / (w * h)) * 10000.0)
        fft_threat = max(0.0, min(100.0, spike_density * 9.0))
        spectral_status = "Positive (Periodic High-Frequency Checkerboard Spikes)" if fft_threat > 45.0 else "Negative (Natural Exponential Decay)"
        self.pillars['spectral_analysis'] = spectral_status
        self.signs.append(f"[PILLAR 3: SPECTRAL FFT] {spectral_status} (Threat: {round(fft_threat, 1)}%)")
        
        math_threat = float((multi_scale_threat + edge_threat + fft_threat) / 3.0)
        self.features['math_threat'] = math_threat
        self.features['fft_threat'] = fft_threat
        self.features['edge_threat'] = edge_threat

    def query_cnn_threat(self, image_bytes: bytes) -> float:
        """Pillar 4: Deep Neural Network / Vision Transformer"""
        if not HF_API_TOKEN:
            self.signs.append("[PILLAR 4: NEURAL CLASSIFIER] Standby: HF_API_TOKEN missing, operating in local mathematical mode.")
            self.pillars['neural_model'] = "Engine Offline (Math Fallback)"
            return float(self.features.get('math_threat', 0.0))

        headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "image/jpeg"}
        try:
            response = requests.post(HF_API_URL, headers=headers, data=image_bytes, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    data = data[0]
                for item in data:
                    label = str(item.get("label", "")).lower()
                    if "fake" in label or "artificial" in label:
                        cnn_score = float(item.get("score", 0.0)) * 100.0
                        neural_desc = f"{round(cnn_score, 1)}% Synthetic Marker Probability"
                        self.pillars['neural_model'] = neural_desc
                        self.signs.append(f"[PILLAR 4: NEURAL CLASSIFIER] Vision Transformer: {neural_desc}")
                        return cnn_score
            else:
                self.signs.append(f"[PILLAR 4: NEURAL CLASSIFIER] API Offline (HTTP {response.status_code}).")
        except Exception:
            self.signs.append("[PILLAR 4: NEURAL CLASSIFIER] Request timeout.")
            
        self.pillars['neural_model'] = f"Local Fallback ({round(float(self.features.get('math_threat', 0.0)), 1)}%)"
        return float(self.features.get('math_threat', 0.0))

    def fuse_and_classify(self, cnn_threat: float):
        """Step 5: Compression-Aware Forensic Fusion"""
        math_threat = float(self.features['math_threat'])
        quality = str(self.features.get('image_quality', 'Medium'))
        is_screenshot = bool(self.features.get('is_screenshot', False))
        has_metadata = bool(self.features.get('has_native_metadata', False))

        # --- THE COMPRESSION OFFSET (The Fix for WhatsApp Images) ---
        compression_offset = 0.0
        if quality == "Medium":
            compression_offset = 15.0
        elif quality == "Low":
            compression_offset = 25.0

        adjusted_cnn = max(0.0, cnn_threat - compression_offset)
        adjusted_math = max(0.0, math_threat - compression_offset)
        
        if compression_offset > 0:
            self.signs.append(f"Compression Filter: Offset of -{compression_offset}% applied to mitigate synthetic hallucinations.")

        # 1. Balanced Weights
        ai_w, math_w = 0.60, 0.40
        fused_score = (adjusted_cnn * ai_w) + (adjusted_math * math_w)

        # 2. Native Camera Whitelist Bias
        if has_metadata and not is_screenshot:
            self.signs.append("Forensic Whitelist: Native camera metadata detected. Applying authenticity bias (-50%).")
            fused_score *= 0.5

        # 3. The "Real-World" Veto
        if adjusted_math < 20.0:
            self.signs.append(f"Compression Filter: Spectral spikes attributed to JPEG compression rather than AI. Vetoing threat.")
            fused_score = min(fused_score, adjusted_math + 20.0)

        # 4. The "Smoking Gun" Logic
        if cnn_threat > 80.0 and math_threat > 30.0:
            fused_score = max(fused_score, 75.0)
            self.signs.append("Forensic Trigger: High raw AI confidence combined with spatial anomalies creates a 'Smoking Gun' signature.")

        fused_score = max(0.0, min(100.0, float(fused_score)))
        
        # Professional Threshold
        is_fake = bool(fused_score >= 60.0)

        # Forensic Tier Categorization
        if fused_score >= 75.0:
            verdict_label = "HIGH-CONFIDENCE DEEPFAKE"
        elif fused_score >= 60.0:
            verdict_label = "SUSPICIOUS / SYNTHETIC ARTIFACTS DETECTED"
        elif fused_score >= 40.0:
            verdict_label = "INCONCLUSIVE / COMPRESSED ASSET"
        else:
            verdict_label = "AUTHENTIC MEDIA ASSET"

        if is_screenshot and is_fake:
            classification = "4_AI_Screenshot"
            desc = "Compressed/Screenshot AI-generated media"
        elif is_screenshot and not is_fake:
            classification = "2_Real_Screenshot"
            desc = "Compressed authentic mobile photograph"
        elif is_fake and not is_screenshot:
            classification = "3_AI_Native"
            desc = "Direct generative diffusion / GAN media"
        else:
            classification = "1_Real_Native"
            desc = "Authentic camera photograph"
            
        reason = (
            f"[{verdict_label}] Compression-Aware Fusion: Adjusted CNN ({round(adjusted_cnn, 1)}%) "
            f"fused with Adjusted Math ({round(adjusted_math, 1)}%). "
            f"Final Fused Probability: {round(fused_score, 1)}%."
        )
            
        return classification, desc, fused_score, reason, is_fake

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

        ensemble = MultiPillarForensicEnsemble(image_path)
        
        ensemble.analyze_container(img)
        ensemble.calculate_math_threat(cv_img)
        
        del cv_img
        gc.collect()

        cnn_threat = ensemble.query_cnn_threat(cnn_image_bytes)
        classification, desc, fused_score, reason, is_fake = ensemble.fuse_and_classify(cnn_threat)

        return {
            "error": False,
            "classification": classification,
            "description": desc,
            "is_fake": bool(is_fake),
            "fake_confidence": float(round(fused_score, 2)),
            "real_confidence": float(round(100.0 - fused_score, 2)),
            "reason": str(reason),
            "signs": ensemble.signs,
            "pillars": ensemble.pillars,
            "detailed_analysis": {
                "math_threat": float(round(ensemble.features.get("math_threat", 0.0), 2)),
                "fft_threat": float(round(ensemble.features.get("fft_threat", 0.0), 2)),
                "edge_threat": float(round(ensemble.features.get("edge_threat", 0.0), 2)),
                "image_quality": ensemble.features.get("image_quality", "Medium"),
                "has_native_metadata": bool(ensemble.features.get("has_native_metadata", False)),
                "is_screenshot": bool(ensemble.features.get("is_screenshot", False))
            },
            "analyzed_via": "Sentinel X Multi-Pillar Evidence Ensemble"
        }

    except Exception as e:
        return {"error": True, "reason": f"Analysis Error: {str(e)}"}