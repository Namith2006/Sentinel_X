"""
Sentinel X — FastAPI Backend
============================
Single-file backend that powers every endpoint consumed by `index.html`.
"""

from __future__ import annotations
import hashlib
import json
import math
import os
import re
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Project-local modules
from crypto_ledger import SecurityLedger
from deepfake_engine import analyze_image
from llm_expert import generate_mitigation_plan
from phishing_engine import analyze_url
from trusted_domains import is_trusted_domain

LEDGER_FILE = os.path.join(tempfile.gettempdir(), "sentinel_x_ledger.json")
ledger = SecurityLedger()


def _persist_ledger() -> None:
    try:
        with open(LEDGER_FILE, "w", encoding="utf-8") as fh:
            json.dump(ledger.chain, fh, indent=2)
    except Exception:
        pass


def _restore_ledger() -> None:
    if not os.path.exists(LEDGER_FILE):
        return
    try:
        with open(LEDGER_FILE, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        if isinstance(saved, list) and saved:
            ledger.chain = saved
    except Exception:
        pass


_restore_ledger()


def log_event(threat_type: str, risk_score: float, details: dict) -> str:
    entry_hash = ledger.log_threat(threat_type, risk_score, details)
    _persist_ledger()
    return entry_hash


def compute_security_score() -> int:
    base = 100.0
    now = datetime.now()

    events = ledger.chain[1:] if len(ledger.chain) > 1 else []
    if not events:
        return 88

    for block in events:
        risk = float(block.get("risk_score", 0) or 0)
        penalty = min(risk * 25.0, 25.0)

        try:
            ts = datetime.fromisoformat(str(block.get("timestamp", "")))
            age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
        except Exception:
            age_hours = 24.0
        recency = max(0.25, 1.0 - (age_hours / 24.0))

        base -= penalty * recency

    return max(0, min(100, int(round(base))))


COMMON_PASSWORDS = {
    "password", "123456", "12345678", "qwerty", "abc123", "letmein",
    "welcome", "monkey", "iloveyou", "admin", "passw0rd", "sunshine",
    "princess", "dragon", "football", "baseball", "111111", "000000",
    "superman", "trustno1", "shadow", "master", "michael", "jordan",
}


def password_entropy_bits(password: str) -> float:
    if not password:
        return 0.0
    pool = 0
    if re.search(r"[a-z]", password): pool += 26
    if re.search(r"[A-Z]", password): pool += 26
    if re.search(r"[0-9]", password): pool += 10
    if re.search(r"[^A-Za-z0-9]", password): pool += 33
    if pool == 0:
        return 0.0
    return len(password) * math.log2(pool)


def password_complexity_score(password: str) -> int:
    if not password:
        return 0
    length = len(password)
    entropy = password_entropy_bits(password)
    length_pts = min(40, length * 3)
    classes = sum(bool(re.search(p, password))
                  for p in [r"[a-z]", r"[A-Z]", r"[0-9]", r"[^A-Za-z0-9]"])
    diversity_pts = classes * 7.5
    entropy_pts = min(30.0, entropy / 80.0 * 30.0)
    total = length_pts + diversity_pts + entropy_pts
    return max(0, min(100, int(round(total))))


def hibp_pwned_count(password: str) -> int:
    try:
        sha1 = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
        prefix, suffix = sha1[:5], sha1[5:]
        resp = requests.get(
            f"https://api.pwnedpasswords.com/range/{prefix}",
            timeout=4,
            headers={"User-Agent": "Sentinel-X/1.0"},
        )
        if resp.status_code != 200:
            return 0
        for line in resp.text.splitlines():
            h, count = line.split(":")
            if h.strip().upper() == suffix:
                return int(count.strip())
        return 0
    except Exception:
        return 0


def password_strength_label(score: int, breached: bool, common: bool) -> str:
    if breached or common:
        return "WEAK (Compromised)"
    if score < 30:  return "WEAK"
    if score < 60:  return "MODERATE"
    if score < 80:  return "STRONG"
    return "EXCELLENT"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if len(ledger.chain) <= 1:
        log_event("system", 0.05, {"event": "Backend online — ledger initialized"})
        
    # The "Wake Up" Call: Pings HF in a background thread to prevent cold starts
    def wake_hf_model():
        hf_token = os.getenv("HF_API_TOKEN", "")
        if hf_token:
            try:
                requests.post(
                    "https://router.huggingface.co/hf-inference/models/prithivMLmods/Deep-Fake-Detector-v2-Model",
                    headers={"Authorization": f"Bearer {hf_token}"},
                    json={"inputs": "wake up"},
                    timeout=5
                )
            except Exception:
                pass
                
    threading.Thread(target=wake_hf_model, daemon=True).start()
    
    yield


app = FastAPI(title="Sentinel X API", version="2.0", lifespan=lifespan)

origins = [
    "https://sentinel-x-navy.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception):
    origin = request.headers.get("origin", "*")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status": "SYSTEM MESSAGE",
            "reason": f"Internal Server Error: {str(exc)}",
            "signs": ["Backend encountered an unhandled exception.", str(exc)],
        },
        headers={
            "Access-Control-Allow-Origin": origin if origin else "*",
            "Access-Control-Allow-Credentials": "true",
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    origin = request.headers.get("origin", "*")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status": "HTTP ERROR",
            "reason": exc.detail,
            "signs": [f"HTTP Error Status: {exc.status_code}"],
        },
        headers={
            "Access-Control-Allow-Origin": origin if origin else "*",
            "Access-Control-Allow-Credentials": "true",
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    origin = request.headers.get("origin", "*")
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "status": "HTTP 422",
            "reason": "Data Validation Error (Missing python-multipart or incorrect payload structure).",
            "signs": [str(err) for err in exc.errors()],
        },
        headers={
            "Access-Control-Allow-Origin": origin if origin else "*",
            "Access-Control-Allow-Credentials": "true",
        },
    )


if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    message: str


def to_json_serializable(val: Any) -> Any:
    if isinstance(val, dict):
        return {k: to_json_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple, set)):
        return [to_json_serializable(x) for x in val]
    elif hasattr(val, "item"): 
        return val.item()
    elif hasattr(val, "tolist"):
        return val.tolist()
    return val


def _coerce_url_payload(payload: dict | None, form_url: str | None) -> str:
    if payload and isinstance(payload, dict) and payload.get("url"):
        return str(payload["url"]).strip()
    if form_url:
        return str(form_url).strip()
    return ""


def _normalize_phishing(raw: dict, url: str) -> dict:
    is_phishing = bool(raw.get("is_phishing", False))
    
    risk_raw = raw.get("phishing_risk_percent", 0.0)
    if isinstance(risk_raw, str):
        risk_score = float(risk_raw.rstrip("%"))
    else:
        risk_score = float(risk_raw)
        
    risk_score = max(0.0, min(100.0, risk_score))
    status = str(raw.get("status", "SAFE")).upper()
    reason = str(raw.get("reason", "Domain analysis complete."))

    return {
        "url": url,
        "is_phishing": is_phishing,
        "phishing_risk_percent": f"{risk_score:.2f}%",
        "risk_score": risk_score,
        "status": status,
        "reason": reason,
        "details": raw,
    }


def _normalize_image(raw: dict, filename: str) -> dict:
    if raw.get("error"):
        return {
            "filename": filename,
            "is_fake": False,
            "fake_confidence": "0.00",
            "real_confidence": "0.00",
            "risk_score": 0.0,
            "status": "SYSTEM MESSAGE",
            "reason": raw.get("reason", "An unknown error occurred."),
            "signs": raw.get("signs", ["Please check server connection.", "Awaiting AI activation."]),
            "analyzed_via": "Error Handler",
            "details": raw,
        }

    is_fake = bool(raw.get("is_fake", False))
    fake_score = float(raw.get("fake_confidence", 0.0))
    real_score = float(raw.get("real_confidence", 100.0 - fake_score))
    
    reason = str(raw.get("reason") or "Analysis complete.")
    signs = raw.get("signs", [])
    if not isinstance(signs, list):
        signs = [str(signs)]

    verdict = "fake" if is_fake else "real"
    analyzed_via = raw.get("analyzed_via", "Sentinel X True Sensor Fusion Ensemble")

    return {
        "filename": filename,
        "classification": raw.get("classification", ""),
        "description": raw.get("description", ""),
        "is_fake": is_fake,
        "fake_confidence": f"{fake_score:.2f}",
        "real_confidence": f"{real_score:.2f}",
        "risk_score": fake_score, 
        "status": verdict.upper(),
        "reason": reason, 
        "signs": signs,  
        "analyzed_via": analyzed_via,
        "details": raw,
    }


@app.get("/")
def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "Sentinel X API is operational."}


@app.get("/api/health")
def api_health():
    return {"status": "online", "service": "sentinel-x", "version": app.version}


@app.post("/api/scan-url")
async def api_scan_url(
    request: Request,
    url: str | None = Form(default=None),
):
    json_payload: dict | None = None
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        try:
            json_payload = await request.json()
            if not isinstance(json_payload, dict):
                json_payload = None
        except Exception:
            json_payload = None
    elif "application/x-www-form-urlencoded" in content_type or "multipart/" in content_type:
        try:
            body = await request.form()
            for key in ("payload", "json", "body"):
                if key in body:
                    maybe = body.get(key)
                    if isinstance(maybe, str):
                        json_payload = json.loads(maybe)
                    break
        except Exception:
            json_payload = None

    target = _coerce_url_payload(json_payload, url)
    if not target:
        return {
            "url": "",
            "is_phishing": False,
            "phishing_risk_percent": "0.00%",
            "risk_score": 0.0,
            "status": "SAFE",
            "reason": "Missing or empty URL provided.",
            "details": {},
        }

    try:
        # Offload synchronous execution to threadpool
        raw = await run_in_threadpool(analyze_url, target)
    except Exception as exc:
        raw = {
            "is_phishing": False,
            "phishing_risk_percent": 25.0,
            "status": "SUSPICIOUS",
            "reason": f"Fallback Heuristics (Engine warning: {str(exc)})",
        }

    result = _normalize_phishing(raw, target)
    result = to_json_serializable(result)

    try:
        log_event(
            threat_type="phishing",
            risk_score=result["risk_score"] / 100.0,
            details={
                "url": target,
                "is_phishing": result["is_phishing"],
                "engine_status": result["reason"],
            },
        )
    except Exception:
        pass

    return result


@app.post("/api/scan-image")
async def api_scan_image(request: Request, file: UploadFile | None = File(None)):
    if file is None:
        try:
            form = await request.form()
            file = form.get("file") or form.get("image")
        except Exception:
            pass

    if not file or not hasattr(file, "filename"):
        return JSONResponse(status_code=400, content={
            "error": True,
            "reason": "Missing 'file' payload in form data.",
            "signs": ["Request parsing failed at the API gateway."]
        })

    original_filename = file.filename
    suffix = os.path.splitext(original_filename)[1] or ".bin"
    tmp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Offload heavy CV2/NumPy synchronous execution to threadpool
        raw = await run_in_threadpool(analyze_image, tmp_path)
        
    except Exception as exc:
        raw = {
            "error": True, 
            "reason": f"Analysis engine error: {str(exc)}",
            "signs": ["Error occurred during image processing.", str(exc)]
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    result = _normalize_image(raw, original_filename)
    result = to_json_serializable(result)
    
    try:
        log_event(
            threat_type="deepfake",
            risk_score=float(result["risk_score"]) / 100.0,
            details={
                "filename": result["filename"],
                "is_fake": bool(result["is_fake"]),
            },
        )
    except Exception:
        pass
        
    return result


@app.post("/api/check-password")
async def api_check_password(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body required.")
    password = str(payload.get("password", ""))
    if not password:
        raise HTTPException(status_code=400, detail="Missing 'password' field.")

    is_common = password.lower() in COMMON_PASSWORDS
    # Offload network-bound request
    breach_count = await run_in_threadpool(hibp_pwned_count, password) if not is_common else 10_000_000
    complexity = password_complexity_score(password)
    entropy = password_entropy_bits(password)
    is_breached = breach_count > 0 or is_common

    if is_breached:
        risk_score = 100.0
    else:
        risk_score = max(0.0, min(100.0, 100.0 - complexity))

    status_text = password_strength_label(complexity, is_breached, is_common)

    suggestions = []
    if is_common:
        suggestions.append("This password is on a common-password list.")
    if breach_count > 0:
        suggestions.append(f"Found in {breach_count:,} known data breaches.")
    if len(password) < 12:
        suggestions.append("Use at least 12 characters.")
    if not re.search(r"[A-Z]", password):
        suggestions.append("Add uppercase letters.")
    if not re.search(r"[0-9]", password):
        suggestions.append("Add numbers.")
    if not re.search(r"[^A-Za-z0-9]", password):
        suggestions.append("Add a symbol (!, @, #, etc.).")

    result = {
        "password_length": len(password),
        "strength": complexity,
        "entropy": round(entropy, 2),
        "is_breached": is_breached,
        "breach_count": breach_count,
        "is_common": is_common,
        "risk_score": risk_score,
        "status": status_text,
        "suggestion": " ".join(suggestions) or "Looks good — keep using a unique password manager.",
    }

    result = to_json_serializable(result)

    try:
        log_event(
            threat_type="weak_password",
            risk_score=risk_score / 100.0,
            details={
                "length": len(password),
                "is_breached": is_breached,
                "breach_count": breach_count,
                "strength": complexity,
                "entropy": entropy,
                "password_sha256": hashlib.sha256(password.encode("utf-8")).hexdigest(),
            },
        )
    except Exception:
        pass

    return result


@app.post("/api/mitigate")
async def api_mitigate(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body required.")

    threat_type = str(payload.get("threat_type") or payload.get("type") or "unknown")
    try:
        risk_score = float(payload.get("risk_score", 0) or 0)
    except (TypeError, ValueError):
        risk_score = 0.0
    risk_score = max(0.0, min(100.0, risk_score))

    plan = None
    try:
        # Offload generation mapping to prevent blocking
        plan = await run_in_threadpool(generate_mitigation_plan, threat_type, risk_score)
    except Exception as exc:
        plan = {"error": str(exc)}

    severity = (
        "critical" if risk_score >= 85 else
        "high"     if risk_score >= 65 else
        "medium"   if risk_score >= 35 else
        "low"
    )

    if isinstance(plan, dict) and "steps" in plan and plan["steps"]:
        raw_steps = plan["steps"]
        structured_steps = []
        titles = {
            "phishing":       ["Disconnect & Quarantine", "Rotate Credentials", "Harden Defenses"],
            "deepfake":       ["Quarantine the Asset", "Trace the Origin", "Notify Stakeholders"],
            "weak_password":  ["Rotate the Secret", "Enable MFA Everywhere", "Adopt a Password Manager"],
        }
        defaults = titles.get(threat_type, ["Isolate", "Eradicate", "Recover"])
        for i, step in enumerate(raw_steps[:3]):
            if isinstance(step, dict):
                structured_steps.append({
                    "title": step.get("title") or defaults[i],
                    "detail": step.get("detail") or step.get("description") or str(step),
                })
            else:
                structured_steps.append({
                    "title": defaults[i] if i < len(defaults) else f"Step {i+1}",
                    "detail": str(step),
                })
        threat_level = plan.get("threat_level") or severity
    else:
        structured_steps = [
            {
                "title": "Isolate",
                "detail": (
                    f"Disconnect the affected endpoint, revoke active sessions, "
                    f"and disable shared credentials related to this {threat_type} incident."
                ),
            },
            {
                "title": "Eradicate",
                "detail": (
                    "Remove the malicious artifact, rotate exposed secrets, "
                    "and apply the latest vendor patches and IOC blocklists."
                ),
            },
            {
                "title": "Recover",
                "detail": (
                    "Restore from the most recent SHA-256-verified backup, "
                    "re-enable continuous monitoring, and document lessons learned."
                ),
            },
        ]
        threat_level = severity

    return {
        "status": "ok",
        "threat_type": threat_type,
        "risk_score": risk_score,
        "severity": threat_level,
        "steps": structured_steps,
        "source": "llm" if isinstance(plan, dict) and "steps" in plan and plan["steps"] else "fallback",
    }


@app.post("/api/chat")
async def security_copilot_chat(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "").strip()
        user_logs = data.get("logs", [])

        threat_count = 0
        suspicious_count = 0
        safe_count = 0
        recent_events = []

        if user_logs:
            for entry in user_logs[:25]:
                status = str(entry.get("status", "")).upper()
                vector = str(entry.get("vector", entry.get("threat_type", "UNKNOWN"))).upper()
                ts = str(entry.get("ts", entry.get("timestamp", "N/A")))[:19]

                if any(k in status for k in ["THREAT", "COMPROMISED", "MALICIOUS", "PHISHING"]):
                    threat_count += 1
                    status_label = "THREAT"
                elif any(k in status for k in ["SUSPICIOUS", "MODERATE", "ADWARE"]):
                    suspicious_count += 1
                    status_label = "SUSPICIOUS"
                else:
                    safe_count += 1
                    status_label = "SAFE"

                recent_events.append(f"- {vector} ({status_label}) at {ts}")

            total_scans = len(recent_events)
            log_context = (
                f"Total Scans: {total_scans}\n"
                f"- High-Risk Threats: {threat_count}\n"
                f"- Suspicious Items: {suspicious_count}\n"
                f"- Safe/Verified: {safe_count}\n\n"
                f"Chronological Event Summary:\n" + "\n".join(recent_events[:15])
            )
        else:
            log_context = "No system logs recorded yet."

        if user_message.lower() in {"hi", "hello", "hey", "help"}:
            return {"reply": "Operator online. How can I assist with your security analysis?"}

        system_prompt = f"""You are the Sentinel X Security Copilot, an elite cybersecurity AI. 
Your tone must be helpful, direct, and clear. Avoid dense corporate jargon and long run-on sentences.

CURRENT TELEMETRY DATA:
{log_context}

INSTRUCTION ROUTING:
Evaluate the user's request:
A) If the user asks for a report, to "analyze logs", "summarize", or asks about their current dashboard status, you MUST use this EXACT strict format:

**Posture Summary**
- **Total scans analyzed:** [Count]
- **Threats/Compromised:** [Count] ([Percentage]%)
- **Suspicious/Pending:** [Count] ([Percentage]%)
- **Safe/Verified:** [Count] ([Percentage]%)
- **Overall health:** [Calculated Score]% secure

**Critical Findings**
1. **Deepfake Vectors:** [Brief, clear summary of what was found.]
2. **Phishing Operations:** [Brief, clear summary of malicious URLs.]
3. **Identity & Passwords:** [Brief, clear summary of password exposures.]

**Actionable Remediation**
1. **Immediate Action:** [One short, direct sentence on what to quarantine or block right now.]
2. **Threat Eradication:** [One short sentence on how to remove the active threat.]
3. **Future Hardening:** [One short sentence on how to prevent this next time.]

B) If the user asks a general security question, asks for advice (e.g., "how to avoid risks", "what is a deepfake?"), or makes conversation, DO NOT use the Posture Summary layout. Instead, answer them directly and naturally. 
CRITICAL UI RULE: NEVER use Markdown tables. Always use standard bullet points and short paragraphs."""

        GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
        if not GROQ_API_KEY:
            return {"reply": "Security Copilot is currently offline (GROQ_API_KEY missing)."}

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Request: {user_message}\n\nSystem Logs:\n{log_context}"}
            ],
            "temperature": 0.2,
            "max_tokens": 2048
        }

        # Offload Groq request to threadpool to prevent UI chat freezing
        def fetch_chat():
            return requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )

        response = await run_in_threadpool(fetch_chat)

        if response.status_code == 200:
            return {"reply": response.json()["choices"][0]["message"]["content"]}
        return {"reply": f"Cloud AI Error: {response.status_code} - {response.text}"}

    except Exception as e:
        return {"reply": f"Internal System Error: {str(e)}"}


@app.get("/api/score")
def api_score():
    score = compute_security_score()
    return {
        "score": score,
        "label": (
            "Excellent — system hardened." if score >= 80 else
            "Moderate — review findings."     if score >= 50 else
            "At risk — take action now."
        ),
        "events_analyzed": max(0, len(ledger.chain) - 1),
        "computed_at": str(datetime.now()),
    }


@app.get("/api/ledger")
def api_ledger():
    return {
        "length": len(ledger.chain),
        "integrity_ok": ledger.verify_chain_integrity(),
        "blocks": ledger.chain,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)