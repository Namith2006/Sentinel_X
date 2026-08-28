from urllib.parse import urlparse

# Curated global dataset of trusted enterprise domains, CDNs, platforms, and services
TRUSTED_DOMAINS = {
    # Tech Giants & Cloud
    "google.com", "google.co.in", "gstatic.com", "googleapis.com", "youtube.com", "youtubekids.com",
    "microsoft.com", "live.com", "office.com", "azure.com", "windows.com", "msn.com", "bing.com",
    "apple.com", "icloud.com", "amazon.com", "aws.amazon.com", "cloudflare.com", "akamai.com",
    "meta.com", "facebook.com", "instagram.com", "whatsapp.com", "threads.net",

    # Developer, Hosting & Infrastructure
    "github.com", "github.io", "gitlab.com", "bitbucket.org", "stackoverflow.com",
    "render.com", "vercel.app", "vercel.com", "netlify.app", "heroku.com", "supabase.com",
    "huggingface.co", "groq.com", "openai.com", "anthropic.com", "python.org", "pypi.org",
    "npmjs.com", "docker.com", "postman.com", "mongodb.com", "postgresql.org",

    # Streaming, Media & Knowledge
    "netflix.com", "spotify.com", "disneyplus.com", "primevideo.com", "twitch.tv",
    "wikipedia.org", "wikimedia.org", "medium.com", "reddit.com", "linkedin.com", "x.com", "twitter.com",

    # Global E-Commerce & Finance
    "paypal.com", "stripe.com", "razorpay.com", "visa.com", "mastercard.com",
    "ebay.com", "flipkart.com", "walmart.com", "target.com",

    # Major Tech & Academic
    "mit.edu", "stanford.edu", "harvard.edu", "w3schools.com", "geeksforgeeks.org", "coursera.org", "udemy.com"
}

def extract_root_domain(url: str) -> str:
    """Safely extracts the hostname and removes common prefixes."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
        
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    
    if hostname.startswith("www."):
        hostname = hostname[4:]
        
    return hostname

def is_trusted_domain(url: str) -> bool:
    """
    Evaluates whether the URL belongs to a verified domain or legitimate subdomain.
    Blocks spoofing attempts like: 'microsoft.com.malicious-site.ru'.
    """
    hostname = extract_root_domain(url)
    if not hostname:
        return False

    # 1. Exact match (e.g., 'microsoft.com')
    if hostname in TRUSTED_DOMAINS:
        return True

    # 2. Legitimate subdomain match (e.g., 'login.microsoft.com' or 'docs.github.com')
    parts = hostname.split(".")
    for i in range(1, len(parts) - 1):
        root_candidate = ".".join(parts[i:])
        if root_candidate in TRUSTED_DOMAINS:
            return True

    # 3. Government / Educational Top-Level Domains (High inherent trust)
    if hostname.endswith(".gov") or hostname.endswith(".gov.in") or hostname.endswith(".edu"):
        return True

    return False