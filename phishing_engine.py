import re
from urllib.parse import urlparse
import warnings
import joblib

warnings.filterwarnings("ignore")
MODEL_PATH = "phishing_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Sentinel X AI Engine Active (Hybrid Mode)")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# --- THE RULE ENGINE (WHITELIST) ---
VERIFIED_DOMAINS = [
    "google.com", "www.google.com",
    "github.com", "www.github.com",
    "wikipedia.org", "www.wikipedia.org",
    "vercel.app", "smart-ai-tutor-mauve.vercel.app","https://legacy-ledger-nine.vercel.app/"
]

# --- THE HEURISTIC KEYWORD FILTER ---
# Substrings that strongly correlate with cracked/nulled software, piracy,
# or unsolicited APK distributions. A match here overrides the AI score and
# forces the URL into the high-risk malware-distribution bucket.
SUSPICIOUS_KEYWORDS = [
    "apk", "mod", "crack", "hack", "nulled", "free-robux", "torrent",
]

KEYWORD_OVERRIDE_RISK = 92.5
KEYWORD_OVERRIDE_REASON = "High-Risk Software/Malware Distribution"

# --- UNVERIFIED THIRD-PARTY APP DISTRIBUTORS / ADWARE RISK ---
# Third-party APK and freeware hosts that bundle adware or repackaged
# binaries. Hits here override the AI score with a slightly lower (but
# still high) risk, since these domains are not strictly phishing but
# routinely distribute unwanted software.
THIRD_PARTY_DISTRIBUTORS = [
    "softonic", "uptodown", "apkpure", "apkmirror", "malavida",
    "mediafire", "zippyshare",
]

DISTRIBUTOR_OVERRIDE_RISK = 88.5
DISTRIBUTOR_OVERRIDE_REASON = "Unverified Third-Party App Distributor / Adware Risk"


def _keyword_override(url: str) -> dict | None:
    """
    Heuristic keyword filter — returns an override result dict if the URL
    contains a suspicious keyword, otherwise None.

    Matching is case-insensitive and substring-based so it catches the
    keyword whether it appears in the path, subdomain, query string, etc.
    """
    lowered = url.lower()
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in lowered:
            return {
                "url": url,
                "is_phishing": True,
                "phishing_risk_percent": f"{KEYWORD_OVERRIDE_RISK}%",
                "matched_keyword": keyword,
                "status": KEYWORD_OVERRIDE_REASON,
            }
    return None


def _distributor_override(url: str) -> dict | None:
    """
    Third-party app/freeware distributor filter — returns an override
    result dict if the URL points at a known adware-prone APK/file host,
    otherwise None.

    Matching is case-insensitive and substring-based so it catches the
    host whether it appears in the domain, subdomain, or path.
    """
    lowered = url.lower()
    for host in THIRD_PARTY_DISTRIBUTORS:
        if host in lowered:
            return {
                "url": url,
                "is_phishing": True,
                "phishing_risk_percent": f"{DISTRIBUTOR_OVERRIDE_RISK}%",
                "matched_distributor": host,
                "status": DISTRIBUTOR_OVERRIDE_REASON,
            }
    return None

def extract_features(url: str):
    """Extract the exact 5 features the Colab model was trained on."""
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
    """Hybrid Pipeline: Whitelist Check -> Keyword Override -> Distributor Override -> AI Prediction"""
    parsed = urlparse(url)
    domain = parsed.netloc.split(":")[0]

    # PHASE 1: Rule-Based Whitelist
    if domain in VERIFIED_DOMAINS:
        return {
            "url": url,
            "is_phishing": False,
            "phishing_risk_percent": "0.0%",
            "status": "SAFE (Verified Domain)",
        }

    # PHASE 1.5: Heuristic Keyword Filter — overrides the AI score for known
    # cracked/pirated/malware-distribution patterns (apk, mod, crack, hack,
    # nulled, free-robux, torrent, etc.).
    override = _keyword_override(url)
    if override is not None:
        return override

    # PHASE 1.6: Third-Party App Distributor Filter — overrides the AI score
    # for known adware-prone APK/freeware hosts (softonic, uptodown, apkpure,
    # apkmirror, malavida, mediafire, zippyshare, etc.).
    distributor_override = _distributor_override(url)
    if distributor_override is not None:
        return distributor_override

    # PHASE 2: Machine Learning Analysis
    features = extract_features(url)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0] 
    phishing_risk = round(probabilities[1] * 100, 2)

    return {
        "url": url,
        "is_phishing": bool(prediction == 1),
        "phishing_risk_percent": f"{phishing_risk}%",
        "status": "MALICIOUS / PHISHING" if prediction == 1 else "SAFE (AI Analyzed)",
    }

# --- TEST INFERENCE LOCALLY ---
if __name__ == "__main__":
    print("\n=== SENTINEL X: LIVE URL SCANNER TEST ===")

    test_urls = [
        "https://www.google.com/search?q=cybersecurity",
        "http://verify-account-update-paypal.com/login.php?user=123",
        "http://192.168.1.1/admin/auth.html",
    ]

    for test_url in test_urls:
        res = analyze_url(test_url)
        print(f"\nURL: {res['url']}")
        print(f"Result: {res['status']} | Risk Score: {res['phishing_risk_percent']}")