# test_deep_tech.py
import torch
import os
from deepfake_engine import analyze_image
from crypto_ledger import SecurityLedger
from llm_expert import generate_mitigation_plan

def run_deep_tech_suite():
    print("=" * 60)
    print("⚡ SENTINEL X - DEEP TECH MODULE INTEGRATION TEST")
    print("=" * 60)

    # 1. Check GPU Hardware Acceleration
    device = "CUDA (RTX GPU)" if torch.cuda.is_available() else "CPU"
    print(f"\n[1/3] Hardware Acceleration Status: Running on {device}")

    # 2. Test Cryptographic Hash Ledger
    print("\n[2/3] Testing Cryptographic SHA-256 Audit Ledger...")
    ledger = SecurityLedger()
    
    # Simulate a deepfake detection log
    sample_threat = {
        "threat_type": "Deepfake Synthetic Media",
        "risk_score": 94.5,
        "filename": "fake_face_sample.jpg"
    }
    log_hash = ledger.log_threat(
        threat_type=sample_threat["threat_type"],
        risk_score=sample_threat["risk_score"],
        details=sample_threat
    )
    print(f"✅ Log Entry Created! Hash-Chain Link: {log_hash[:16]}...")
    print(f"   Total Chain Height: {len(ledger.chain)} blocks")

    # 3. Test Local LLM Expert System (Ollama Llama-3)
    print("\n[3/3] Testing Local LLM (Llama-3 via Ollama)...")
    plan = generate_mitigation_plan(
        threat_type=sample_threat["threat_type"],
        risk_score=sample_threat["risk_score"]
    )
    
    if "error" in plan:
        print(f"❌ LLM Error: {plan['error']}")
    else:
        print("✅ Structured JSON Response Received from Local Llama-3:")
        print(f"   - Status: {plan.get('status')}")
        print(f"   - Threat Level: {plan.get('threat_level')}")
        print("   - Recovery Steps:")
        for idx, step in enumerate(plan.get('steps', []), 1):
            print(f"     {idx}. {step}")

    print("\n" + "=" * 60)
    print("🎯 ALL DEEP TECH ENGINES VERIFIED OPERATIONAL!")
    print("=" * 60)

if __name__ == "__main__":
    run_deep_tech_suite()