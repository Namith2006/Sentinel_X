"""
Sentinel X — Mitigation Expert (Groq Cloud LLM)
==============================================

Client that interfaces with Groq's high-speed Cloud API to generate
real-time, actionable 3-step incident mitigation plans for detected threats.

If Groq is unreachable, times out, or encounters an API error, the module
automatically fails over to threat-specific static mitigation procedures.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_MODEL = "mixtral-8x7b-32768"
REQUEST_TIMEOUT_SECONDS = 10

# ---------------------------------------------------------------------------
# Static fallback responses
# ---------------------------------------------------------------------------

_FALLBACK_STEPS: Dict[str, List[Dict[str, str]]] = {
    "phishing": [
        {"title": "Network Isolation", "detail": "Do not open or interact with the link. Block the domain at the DNS/firewall level."},
        {"title": "Session Termination", "detail": "Purge browser cache, cookies, and active session tokens across endpoints."},
        {"title": "Credential Rotation", "detail": "Enforce an immediate password reset and verify hardware MFA keys on associated accounts."},
    ],
    "deepfake": [
        {"title": "Asset Quarantine", "detail": "Halt distribution of the media asset and transfer it to an isolated forensic storage vault."},
        {"title": "Origin Verification", "detail": "Cross-reference asset hashes and run reverse image checks against known authentic source repositories."},
        {"title": "Channel Authentication", "detail": "Verify communication claims through an out-of-band, cryptographically authenticated channel."},
    ],
    "weak_password": [
        {"title": "Immediate Revocation", "detail": "Revoke the exposed credential across all linked services and directory services."},
        {"title": "Entropy Enforcement", "detail": "Generate a unique 16+ character passphrase using a zero-knowledge password manager."},
        {"title": "Hardware Authentication", "detail": "Enforce FIDO2/WebAuthn hardware token verification for all administrative logins."},
    ],
}

_DEFAULT_FALLBACK_STEPS: List[Dict[str, str]] = [
    {"title": "Isolate Endpoint", "detail": "Disconnect the affected device from the local network to contain lateral threat movement."},
    {"title": "Preserve Artifacts", "detail": "Retain system logs and forensic telemetry for incident investigation."},
    {"title": "Remediate & Recover", "detail": "Restore operations from a verified SHA-256 backup state and update firewall rules."},
]


def _get_fallback_plan(threat_type: str, risk_score: float) -> Dict[str, Any]:
    """Return a structured fallback recovery plan when the LLM is unreachable."""
    normalized_type = threat_type.lower().replace(" ", "_")
    steps = _FALLBACK_STEPS.get(normalized_type, _DEFAULT_FALLBACK_STEPS)
    
    severity = (
        "Critical" if risk_score >= 85 else
        "High" if risk_score >= 65 else
        "Medium" if risk_score >= 35 else
        "Low"
    )
    
    return {
        "status": "fallback",
        "threat_level": severity,
        "steps": steps,
        "source": "static_fallback",
    }


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def generate_mitigation_plan(threat_type: str, risk_score: float) -> Dict[str, Any]:
    """
    Query Groq API for a 3-step mitigation plan in structured JSON.
    Falls back gracefully if the network request fails or times out.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"""You are Sentinel X's automated incident response engine.
A {threat_type} threat was detected with a risk score of {risk_score}%.
Return a 3-step mitigation plan in strictly valid JSON format.
Example format:
{{
  "threat_level": "High",
  "steps": [
    {{"title": "Step 1 Title", "detail": "Actionable detail for step 1"}},
    {{"title": "Step 2 Title", "detail": "Actionable detail for step 2"}},
    {{"title": "Step 3 Title", "detail": "Actionable detail for step 3"}}
  ]
}}
"""

    payload = {
        "model": "openai/gpt-oss-20b",  # ✨ Updated here too!
        "messages": [
            {"role": "system", "content": "You output only valid JSON. Do not include markdown formatting or intro text."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}, 
        "max_tokens": 300,
        "temperature": 0.2
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            if "steps" in parsed and isinstance(parsed["steps"], list):
                parsed["source"] = "llm"
                return parsed

        # If response was non-200 or missing required keys, fallback
        return _get_fallback_plan(threat_type, risk_score)

    except (requests.exceptions.RequestException, ValueError, KeyError):
        return _get_fallback_plan(threat_type, risk_score)