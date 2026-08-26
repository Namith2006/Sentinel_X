import io
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

# OpenCV is preferred for ELA / Laplacian work because it's substantially
# faster and supports JPEG re-encoding via cv2.imencode. Fall back to a pure
# PIL + NumPy path when cv2 is unavailable so the engine still works on
# minimal installs.
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# Compression heuristics
# ---------------------------------------------------------------------------
# Substrings in the original filename that almost always mean the image has
# been re-encoded by a messaging service (WhatsApp / Telegram) or has been
# downloaded and re-saved by a browser.
COMPRESSION_KEYWORDS = ("whatsapp", "telegram", "download", "compressed")

# Images larger than this on the longest side are unlikely to be the heavily
# down-scaled output of a chat client. Anything smaller triggers the
# resolution-based compression heuristic.
COMPRESSION_RESOLUTION_CAP = 1600

# How aggressively to boost the PyTorch model's fake probability when we
# detect compression artifacts. 1.8x means a 40% raw score becomes 72% —
# enough to flip borderline cases without drowning the model in noise.
COMPRESSION_SENSITIVITY = 1.8

# Combined ELA + tensor score above this percentage is reported as fake.
COMBINED_FAKE_THRESHOLD = 50.0

# ---------------------------------------------------------------------------
# Secondary analysis: Error Level Analysis + Laplacian variance
# ---------------------------------------------------------------------------
def _ela_score(image_path: str) -> float:
    """
    Error Level Analysis (ELA): re-encode the image at a known JPEG quality
    and measure the per-channel absolute difference against the original.

    A larger mean difference means the image has more high-frequency energy
    than a re-encode would normally produce — typical of synthetic content
    or heavy compression. Returned value is normalized to roughly [0, 1].
    """
    try:
        if _HAS_CV2:
            img = cv2.imread(image_path)
            if img is None:
                return 0.0
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            ok, enc = cv2.imencode(".jpg", img, encode_param)
            if not ok:
                return 0.0
            rec = cv2.imdecode(np.frombuffer(enc.tobytes(), np.uint8), cv2.IMREAD_COLOR)
            diff = cv2.absdiff(img, rec)
            mean_diff = float(np.mean(diff)) / 255.0
        else:
            # Pure PIL / NumPy fallback.
            with Image.open(image_path) as src:
                src_rgb = src.convert("RGB")
                arr = np.asarray(src_rgb, dtype=np.int16)
                buf = io.BytesIO()
                src_rgb.save(buf, format="JPEG", quality=75)
                buf.seek(0)
                rec_img = Image.open(buf).convert("RGB")
                rec = np.asarray(rec_img, dtype=np.int16)
            diff = np.abs(arr - rec)
            mean_diff = float(np.mean(diff)) / 255.0
        # Amplify so a small mean diff still produces a usable signal.
        return float(min(1.0, mean_diff * 4.0))
    except Exception:
        # If ELA itself fails we don't want to abort the whole pipeline.
        return 0.0


def _laplacian_variance(image_path: str) -> float:
    """
    Laplacian variance: a real photograph has rich high-frequency content;
    many AI generators produce an over-smoothed image whose Laplacian
    variance is much lower. Returned value is normalized to roughly [0, 1]
    where 1.0 means "looks like a real photograph" and 0.0 means "looks
    suspiciously smooth / synthetic".
    """
    try:
        if _HAS_CV2:
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return 0.5
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            var = float(np.var(lap))
        else:
            with Image.open(image_path) as src:
                gray_img = src.convert("L")
            arr = np.asarray(gray_img, dtype=np.float32)
            # 3x3 Laplacian (centre -4, neighbours +1) applied manually so we
            # don't require SciPy. Built via shifted views of a reflected pad.
            padded = np.pad(arr, 1, mode="reflect")
            # Build shifted views for the four neighbours and the centre.
            up    = padded[:-2, 1:-1]
            down  = padded[2:,  1:-1]
            left  = padded[1:-1, :-2]
            right = padded[1:-1, 2:]
            centre = padded[1:-1, 1:-1]
            lap = (up + down + left + right) - 4.0 * centre
            var = float(np.var(lap))
        # Empirical scaling: real photos routinely exceed 100–500, while
        # heavily smoothed synthetic frames often sit below 50.
        return float(min(1.0, max(0.0, var / 500.0)))
    except Exception:
        return 0.5


def _compression_signals(filename: str | None, image_path: str) -> tuple[bool, list[str]]:
    """
    Inspect the filename and the on-disk image to decide whether the file
    has been mangled by a messaging app or a generic re-compression pass.

    Returns (compressed, reasons). `reasons` is a list of human-readable
    strings that explain why we flagged the image; the dashboard shows
    these in the analysis details.
    """
    reasons: list[str] = []

    if filename:
        lowered = filename.lower()
        for kw in COMPRESSION_KEYWORDS:
            if kw in lowered:
                reasons.append(f"filename contains '{kw}'")
                break

    try:
        with Image.open(image_path) as im:
            w, h = im.size
        if max(w, h) < COMPRESSION_RESOLUTION_CAP:
            reasons.append(
                f"resolution {w}x{h} suggests chat-app compression"
            )
    except Exception:
        pass

    return (len(reasons) > 0, reasons)


# ---------------------------------------------------------------------------
# Neural model
# ---------------------------------------------------------------------------
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        # Load pre-trained ResNet-18 using modern PyTorch weights syntax
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Freeze base layers to retain feature extraction weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer for Real vs. Fake binary output
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def analyze_image(image_path: str, filename: str | None = None):
    """
    Run the full Neural Tensor + Compression ELA Pipeline.

    Pipeline:
        1. ResNet-18 tensor forward pass -> tensor_fake_prob
        2. ELA re-encode diff -> ela_score
        3. Laplacian variance -> lap_real_score
        4. Compression heuristics -> sensitivity multiplier
        5. Combine the three signals into a single fake_confidence

    The combined fake_confidence is compared against COMBINED_FAKE_THRESHOLD
    (50.0%) to decide `is_fake`.
    """
    # Automatically leverage your RTX 5060 if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeDetector().to(device)
    model.eval()

    # Standard ImageNet normalization for ResNet-18
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    effective_filename = filename or os.path.basename(image_path)

    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            tensor_fake_prob = float(output.item())

        # Secondary ELA + Laplacian pass.
        ela = _ela_score(image_path)
        lap_real = _laplacian_variance(image_path)         # 1.0 = real, 0.0 = smooth/synthetic
        lap_fake_prob = 1.0 - lap_real                     # invert for fusion
        ela_fake_score = (ela + lap_fake_prob) / 2.0       # combined spatial score in [0, 1]

        # Compression-aware weighting.
        compressed, compression_reasons = _compression_signals(effective_filename, image_path)
        if compressed:
            # Apply the 1.8x sensitivity multiplier to the tensor probability
            # and bias the blend toward ELA — compression artifacts hurt the
            # tensor more than they hurt pixel-domain heuristics.
            boosted_tensor = min(1.0, tensor_fake_prob * COMPRESSION_SENSITIVITY)
            combined_fake_prob = min(
                1.0,
                0.55 * boosted_tensor + 0.45 * ela_fake_score,
            )
            weighting_mode = f"compression_boosted_{COMPRESSION_SENSITIVITY:.1f}x"
        else:
            # Clean source: trust the tensor but still sanity-check with ELA.
            combined_fake_prob = min(
                1.0,
                0.70 * tensor_fake_prob + 0.30 * ela_fake_score,
            )
            weighting_mode = "standard"

        fake_confidence = combined_fake_prob * 100.0
        is_fake = fake_confidence >= COMBINED_FAKE_THRESHOLD

        return {
            "status": "success",
            "is_fake": bool(is_fake),
            "fake_confidence": f"{fake_confidence:.2f}%",
            "real_confidence": f"{(100.0 - fake_confidence):.2f}%",
            "analyzed_via": "Neural Tensor + Compression ELA Pipeline",
            "decision_threshold": f"{COMBINED_FAKE_THRESHOLD:.2f}%",
            "tensor_fake_prob": f"{tensor_fake_prob * 100:.2f}%",
            "ela_score": f"{ela:.4f}",
            "laplacian_real_score": f"{lap_real:.4f}",
            "ela_fake_score": f"{ela_fake_score * 100:.2f}%",
            "combined_fake_prob": f"{fake_confidence:.2f}%",
            "weighting_mode": weighting_mode,
            "compression_detected": bool(compressed),
            "compression_signals": compression_reasons,
            "cv2_backend": _HAS_CV2,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
