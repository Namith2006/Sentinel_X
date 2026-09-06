import re
import ssl
import socket
import warnings
import joblib
import requests
from urllib.parse import urlparse

try:
    from trusted_domains import is_trusted_domain
except ImportError:
    def is_trusted_domain(url: str) -> bool:
        return False

warnings.filterwarnings("ignore")

# --- ML MODEL INITIALIZATION ---
MODEL_PATH = "phishing_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

SUSPICIOUS_KEYWORDS = ["apk", "mod", "crack", "hack", "nulled", "free-robux", "torrent"]
KEYWORD_OVERRIDE_RISK = 92.5

THIRD_PARTY_DISTRIBUTORS = ["softonic", "uptodown", "apkpure", "apkmirror", "malavida", "mediafire", "zippyshare"]
DISTRIBUTOR_OVERRIDE_RISK = 88.5

def extract_features(url: str):
    parsed = urlparse(url)
    domain = parsed.netloc.split(":")[0] if parsed.netloc else ""
    return [[
        len(url),
        url.count("-"),
        url.count("@"),
        url.count("?"),
        1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else 0
    ]]

def check_urlhaus(url: str) -> dict | None:
    """Query the URLhaus OSINT database for known malware distribution."""
    try:
        data = {'url': url}
        # 2.0 second timeout ensures the UI doesn't hang during a live presentation
        response = requests.post('https://urlhaus-api.abuse.ch/v1/url/', data=data, timeout=2.0)
        if response.status_code == 200:
            json_resp = response.json()
            if json_resp.get('query_status') == 'ok' and json_resp.get('url_status') == 'online':
                tags = json_resp.get('tags', [])
                tag_str = f" (Tags: {', '.join(tags)})" if tags else ""
                return {
                    "is_phishing": True,
                    "phishing_risk_percent": 99.0,
                    "status": "CRITICAL THREAT",
                    "reason": f"OSINT Alert: URL flagged as active malware by URLhaus database{tag_str}."
                }
    except Exception:
        pass  # Fail silently and let the local ML/SSL pipeline take over
    return None

def analyze_url(url: str) -> dict:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # 1. CORE WHITELIST
    if is_trusted_domain(url):
        return {
            "is_phishing": False,
            "phishing_risk_percent": 0.0,
            "status": "SAFE",
            "reason": "Verified legitimate enterprise domain."
        }
        
    # 2. OSINT THREAT INTELLIGENCE (Crowdsourced Datasets)
    osint_override = check_urlhaus(url)
    if osint_override:
        return osint_override

    # 3. KEYWORD OVERRIDES
    lowered = url.lower()
    for kw in SUSPICIOUS_KEYWORDS:
        if kw in lowered:
            return {
                "is_phishing": True,
                "phishing_risk_percent": KEYWORD_OVERRIDE_RISK,
                "status": "PHISHING",
                "reason": f"High-Risk Software/Malware Distribution (Matched: {kw})"
            }

    for dist in THIRD_PARTY_DISTRIBUTORS:
        if dist in lowered:
            return {
                "is_phishing": True,
                "phishing_risk_percent": DISTRIBUTOR_OVERRIDE_RISK,
                "status": "SUSPICIOUS",
                "reason": f"Unverified Third-Party App Distributor / Adware Risk (Matched: {dist})"
            }

    reasons = []
    ml_risk_score = 0.0

    # 4. MACHINE LEARNING INFERENCE (GUARDED)
    # 4. MACHINE LEARNING INFERENCE (GUARDED)
    if model:
        try:
            features = extract_features(url)
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            ml_risk_score = float(probabilities[1] * 100)
            
            if prediction == 1:
                reasons.append("ML Model flagged suspicious URL structure")
                # Ensure the score is high enough to trigger the Phishing UI
                ml_risk_score = max(ml_risk_score, 65.0) 
            else:
                reasons.append("ML Model classified URL as benign")
                # Cap the score at 20% so it doesn't trigger "Suspicious" unless the SSL check fails
                ml_risk_score = min(ml_risk_score, 20.0)
        except Exception as ml_err:
            ml_risk_score = 10.0
            reasons.append(f"ML evaluation fallback: {str(ml_err)[:30]}")
    else:
        reasons.append("Rule-based heuristics active")
    # 5. DYNAMIC SSL/TLS VALIDATION (GUARDED)
    ssl_risk_penalty = 0.0
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    if hostname and not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname):
        try:
            context = ssl.create_default_context()
            with socket.create_connection((hostname, 443), timeout=2.5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    ssock.getpeercert()
                    reasons.append("Valid SSL Certificate verified")
        except ssl.SSLCertVerificationError:
            ssl_risk_penalty += 40.0
            reasons.append("Invalid, expired, or self-signed SSL/TLS certificate")
        except socket.timeout:
            ssl_risk_penalty += 15.0
            reasons.append("Secure connection timed out")
        except Exception:
            ssl_risk_penalty += 20.0
            reasons.append("Domain lacks a verifiable HTTPS connection")
    elif re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname):
        ssl_risk_penalty += 45.0
        reasons.append("Raw IP address used instead of domain name")

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
        "reason": " | ".join(reasons) if reasons else "Domain analysis complete."
    }