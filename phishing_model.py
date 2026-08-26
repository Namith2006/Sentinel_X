import math
import re
from urllib.parse import urlparse

# Sample training data: (URL, Label) where 1 = Phishing, 0 = Legitimate
dataset = [
    ("http://login-update-bank-security-check.com/verify", 1),
    ("http://192.168.1.1/paypal/login.php", 1),
    ("https://www.google.com/search?q=python", 0),
    ("https://github.com/torvalds/linux", 0),
    ("http://secure-verify-account-update.xyz/login", 1),
    ("https://en.wikipedia.org/wiki/Main_Page", 0),
]


def calculate_entropy(text: str) -> float:
    """Calculate Shannon Entropy (randomness of characters in string)."""
    if not text:
        return 0.0
    prob = [float(text.count(c)) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)


def extract_features(url: str) -> dict:
    """Extract numerical features from a URL using standard Python."""
    parsed = urlparse(url)

    # 1. Length of URL
    url_length = len(url)

    # 2. Count of suspicious special characters
    special_char_count = len(re.findall(r"[@\-_?=%]", url))

    # 3. Check if domain uses IP address instead of domain name
    domain = parsed.netloc.split(":")[0]
    has_ip = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else 0

    # 4. HTTPS check (0 if https present, 1 if missing)
    is_http = 1 if parsed.scheme == "http" else 0

    # 5. String randomness (Entropy)
    entropy = calculate_entropy(url)

    return {
        "url_length": url_length,
        "special_chars": special_char_count,
        "has_ip": has_ip,
        "is_http": is_http,
        "entropy": round(entropy, 2),
    }


print("=== SENTINEL X: PURE-PYTHON FEATURE EXTRACTOR ===")
for url, label in dataset:
    features = extract_features(url)
    status = "MALICIOUS" if label == 1 else "SAFE"
    print(f"\nURL: {url}")
    print(f"Status: {status}")
    print(f"Features: {features}")