import re
import ssl
import socket
import warnings
import joblib
from urllib.parse import urlparse
from trusted_domains import is_trusted_domain

warnings.filterwarnings("ignore")

# --- ML MODEL INITIALIZATION ---
MODEL_PATH = "phishing_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Sentinel X AI Engine Active (ML + Hybrid Mode)")
except Exception as e:
    model = None
    print(f"⚠️ ML Model not found or failed to load. Falling back to Heuristics & SSL only: {e}")


# --- HEURISTIC OVERRIDES ---
SUSPICIOUS_KEYWORDS = ["apk", "mod", "crack", "hack", "nulled", "free-robux", "torrent"]
KEYWORD_OVERRIDE_RISK = 92.5
KEYWORD_OVERRIDE_REASON = "High-Risk Software/Malware Distribution"

THIRD_PARTY_DISTRIBUTORS = ["softonic", "uptodown", "apkpure", "apkmirror", "malavida", "mediafire", "zippyshare"]
DISTRIBUTOR_OVERRIDE_RISK = 88.5
DISTRIBUTOR_OVERRIDE_REASON = "Unverified Third-Party App Distributor / Adware Risk"

def _keyword_override(url: str) -> dict | None:
    lowered = url.lower()
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in lowered:
            return {
                "is_phishing": True,
                "phishing_risk_percent": KEYWORD_OVERRIDE_RISK,
                "status": "PHISHING",
                "reason": f"{KEYWORD_OVERRIDE_REASON} (Matched: {keyword})"
            }
    return None

def _distributor_override(url: str) -> dict | None:
    lowered = url.lower()
    for host in THIRD_PARTY_DISTRIBUTORS:
        if host in lowered:
            return {
                "is_phishing": True,
                "phishing_risk_percent": DISTRIBUTOR_OVERRIDE_RISK,
                "status": "SUSPICIOUS",
                "reason": f"{DISTRIBUTOR_OVERRIDE_REASON} (Matched: {host})"
            }
    return None

def extract_features(url: str):
    parsed = urlparse(url)
    domain = parsed.netloc.split(":")[0]
    return [[
        len(url),                              # 1. Total length
        url.count("-"),                        # 2. Hyphens
        url.count("@"),                        # 3. '@' symbols
        url.count("?"),                        # 4. Query parameters
        1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else 0  # 5. IP Address
    ]]


def analyze_url(url: str) -> dict:
    """True Hybrid Pipeline: Whitelist -> Heuristics -> ML Model -> Dynamic SSL"""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # PHASE 1: Core Trusted Domain Whitelist
    if is_trusted_domain(url):
        return {
            "is_phishing": False,
            "phishing_risk_percent": 0.0,
            "status": "SAFE",
            "reason": "Verified legitimate enterprise domain."
        }

    # PHASE 2: Hardcoded Heuristic Overrides
    kw_override = _keyword_override(url)
    if kw_override:
        return kw_override

    dist_override = _distributor_override(url)
    if dist_override:
        return dist_override

    # PHASE 3: Machine Learning Base Score
    ml_risk_score = 0.0
    reasons = []
    
    if model:
        features = extract_features(url)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        ml_risk_score = probabilities[1] * 100
        if prediction == 1:
            reasons.append("ML Model classified URL features as malicious")
        else:
            reasons.append("ML Model classified URL as benign")
    else:
        reasons.append("ML engine offline; relying on dynamic heuristics")

    # PHASE 4: Dynamic SSL/TLS Validation
    ssl_risk_penalty = 0.0
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=3.0) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                reasons.append("Valid SSL Certificate verified")
    except ssl.SSLCertVerificationError:
        ssl_risk_penalty += 40.0
        reasons.append("Invalid, expired, or self-signed SSL/TLS certificate")
    except socket.timeout:
        ssl_risk_penalty += 15.0
        reasons.append("Secure connection timed out")
    except Exception:
        ssl_risk_penalty += 25.0
        reasons.append("Domain lacks a verifiable HTTPS connection")

    # PHASE 5: Composite Scoring
    # Combine ML probability and SSL penalties
    total_risk = min(100.0, ml_risk_score + ssl_risk_penalty)
    is_phish = total_risk >= 65.0

    if is_phish:
        status = "PHISHING"
    elif total_risk >= 35.0:
        status = "SUSPICIOUS"
    else:
        status = "SAFE"

    return {
        "is_phishing": is_phish,
        "phishing_risk_percent": total_risk,
        "status": status,
        "reason": " | ".join(reasons)
    }

# --- TEST INFERENCE LOCALLY ---
if __name__ == "__main__":
    print("\n=== SENTINEL X: LIVE HYBRID URL SCANNER TEST ===")
    test_urls = [
        "https://www.google.com/search?q=cybersecurity",
        "http://verify-account-update-paypal.com/login.php?user=123",
        "https://softonic.com/download-apk"
    ]
    for test_url in test_urls:
        res = analyze_url(test_url)
        print(f"\nURL: {test_url}")
        print(f"Result: {res['status']} | Risk Score: {res['phishing_risk_percent']:.2f}%")
        print(f"Details: {res['reason']}")