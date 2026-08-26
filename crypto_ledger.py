import hashlib
import json
from datetime import datetime

class SecurityLedger:
    def __init__(self):
        self.chain = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        # Genesis block has a zero-hash precedent
        genesis_log = {
            "timestamp": str(datetime.now()),
            "event": "Sentinel X - Ledger Initialized",
            "previous_hash": "0" * 64
        }
        genesis_log["hash"] = self._hash_entry(genesis_log)
        self.chain.append(genesis_log)

    def _hash_entry(self, entry: dict) -> str:
        # Sort keys to ensure consistent hashing regardless of dictionary order
        entry_string = json.dumps(entry, sort_keys=True).encode('utf-8')
        return hashlib.sha256(entry_string).hexdigest()

    def log_threat(self, threat_type: str, risk_score: float, details: dict):
        last_log = self.chain[-1]
        
        new_log = {
            "timestamp": str(datetime.now()),
            "threat_type": threat_type,
            "risk_score": risk_score,
            "details": details,
            "previous_hash": last_log["hash"]
        }
        
        new_log["hash"] = self._hash_entry(new_log)
        self.chain.append(new_log)

        # Returning the hash to be committed to PostgreSQL by the backend team
        return new_log["hash"]

    def verify_chain_integrity(self) -> bool:
        """Walk the entire chain and confirm every block's hash matches its
        recomputed hash and every previous_hash points to the prior block."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # 1. Recompute this block's hash from its content
            block_for_hashing = {
                "timestamp": current["timestamp"],
                "threat_type": current["threat_type"],
                "risk_score": current["risk_score"],
                "details": current["details"],
                "previous_hash": current["previous_hash"],
            }
            if self._hash_entry(block_for_hashing) != current["hash"]:
                return False

            # 2. Confirm the link to the previous block
            if current["previous_hash"] != previous["hash"]:
                return False

        return True


def mock_threat_append_test():
    """Mock test: simulate Module 1's three threat detection engines appending
    alerts to the ledger, then verify the chain is intact."""
    print("=" * 60)
    print("Sentinel X - Module 1: Cryptographic Ledger Mock Test")
    print("=" * 60)

    ledger = SecurityLedger()
    print(f"[Genesis] Block 0 hash: {ledger.chain[0]['hash'][:16]}...")

    # Simulated outputs from the three detection engines described in Section 1
    mock_threats = [
        {
            "threat_type": "phishing",
            "risk_score": 0.87,
            "details": {"url": "http://secure-login-paypa1.tk/auth", "tokens": ["login", "paypal"]},
        },
        {
            "threat_type": "deepfake",
            "risk_score": 0.92,
            "details": {"source": "uploaded_image.png", "fake_confidence": "92.31%"},
        },
        {
            "threat_type": "cryptojacking",
            "risk_score": 0.74,
            "details": {"miner_pool": "pool.minexmr.com:4444", "cpu_usage": "93%"},
        },
    ]

    for i, threat in enumerate(mock_threats, start=1):
        block_hash = ledger.log_threat(
            threat_type=threat["threat_type"],
            risk_score=threat["risk_score"],
            details=threat["details"],
        )
        print(f"[Block {i}] {threat['threat_type']:<14} -> hash: {block_hash[:16]}...")

    print("-" * 60)
    print(f"Chain length: {len(ledger.chain)} blocks")
    print(f"Integrity check: {'PASSED ✓' if ledger.verify_chain_integrity() else 'FAILED ✗'}")

    # Tamper test: mutate a historical block and confirm verification catches it
    ledger.chain[1]["risk_score"] = 0.10  # attacker rewrites a past alert
    print(f"After tampering with block 1 -> Integrity check: "
          f"{'PASSED ✓' if ledger.verify_chain_integrity() else 'FAILED ✗ (tamper detected)'}")
    print("=" * 60)


if __name__ == "__main__":
    mock_threat_append_test()