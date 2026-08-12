import requests
import json

MINER_URL = "http://127.0.0.1:8000/v1/chat/completions"

def test_chat_completion():
    payload = {
        "messages": [
            {"role": "user", "content": "Explain decentralized compute in one sentence."}
        ],
        "temperature": 0.7
    }

    print(f"[*] Sending CHAT_COMPLETION request to {MINER_URL}...")
    try:
        response = requests.post(MINER_URL, json=payload, timeout=15)
        print(f"[+] HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("[+] Output Signal:", data.get("choices", [{}])[0].get("message", {}).get("content"))
            print("[+] Confidence/Tokens:", data.get("usage", {}).get("total_tokens"))
            print("✅ TEST SUCCESS: Endpoint conforms to Telegraph schema.")
        else:
            print("❌ TEST FAILED:", response.text)
    except Exception as e:
        print("❌ CONNECTION FAILED:", str(e))

if __name__ == "__main__":
    test_chat_completion()
