import os
import hmac
import hashlib
import requests
from dotenv import load_dotenv

load_dotenv()

# MINER_URL = "http://localhost:8000/v1/inference"
MINER_URL = "http://0.0.0.0:8000/v1/inference"

# Read key from environment or fallback to default
SECRET_KEY = os.getenv("MINER_SECRET_KEY", "telegraph_secret_key_node_01_change_in_production").encode("utf-8")

def verify_miner_signature(data: dict, secret_key: bytes) -> bool:
    """Recomputes HMAC-SHA256 signature locally and compares with miner's signature."""
    task_id = str(data["task_id"])
    miner_id = str(data["miner_id"])
    output = str(data["output"])
    latency_ms = float(data["latency_ms"])
    received_sig = str(data["signature"])

    # Ensure latency_ms is formatted with exactly 2 decimal places (.2f)
    payload = f"{task_id}:{miner_id}:{output}:{latency_ms:.2f}".encode("utf-8")
    expected_sig = hmac.new(secret_key, payload, hashlib.sha256).hexdigest()

    return hmac.compare_digest(expected_sig, received_sig)

def test_inference_and_verification():
    payload = {
        "task_id": "task_validator_99",
        "prompt": "Summarize the primary benefits of zero-knowledge proofs in decentralized inference networks.",
        "max_tokens": 128
    }

    print(f"[*] Sending Task '{payload['task_id']}' to Miner Node...")
    try:
        response = requests.post(MINER_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"[!] Request failed: {e}")
        return
    
    if response.status_code != 200:
        print(f"[!] Error from Miner Node: {response.status_code} - {response.text}")
        return

    res_json = response.json()
    print(f"[+] Response Received from [{res_json['miner_id']}] in {res_json['latency_ms']} ms")
    print(f"[+] Output: {res_json['output'][:80]}...\n")

    print("[*] Verifying Cryptographic Signature...")
    is_valid = verify_miner_signature(res_json, SECRET_KEY)

    if is_valid:
        print("✅ VERIFICATION SUCCESS: Signature matches! Payload is authentic and untampered.")
    else:
        print("❌ VERIFICATION FAILED: Invalid signature or payload tampered!")

if __name__ == "__main__":
    test_inference_and_verification()
