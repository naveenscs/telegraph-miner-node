# Telegraph Protocol Miner Node (Track 1)

A high-performance, asynchronous miner node engineered for Telegraph Protocol's verified inference network.

## Track
**Track 1: Miners (Network Supply & Compute Layer)**

## System Architecture
- **Control Plane:** Python 3.11+ using `asyncio`, `FastAPI`, & `uvloop` for low-latency network I/O and HTTP REST interface.
- **Data Plane:** Groq LPU API backend running `llama-3.1-8b-instant` (compatible with `vLLM` / local `llama.cpp` wrappers).
- **Resilience Engine:** Dual-mode HTTP client supporting automatic SSL validation fallback for restricted proxy / VPN network environments.
- **Verification Layer:** Deterministic HMAC-SHA256 signal signature and response timing tracking (`latency_ms`).

---

## Features
- **Sub-Second Latency:** Asynchronous execution pipeline delivering verified inference responses in ~200–900ms.
- **Cryptographic Attestation:** Every response payload is HMAC-SHA256 signed for protocol validator verification.
- **Dual Runtime Options:** Can be executed natively via Python or containerized via Docker/Podman.
- **Network Resilience:** Handles corporate firewalls, VPN proxies, and missing root CA certificate stores smoothly.
- **Interactive OpenAPI Docs:** Built-in Swagger UI for testing endpoints at `/docs`.

---

## Repository Structure

```text
telegraph-miner-node/
├── src/
│   ├── miner.py          # Core FastAPI server & Async Groq client engine
│   └── validator_test.py # Standalone verification script for network validators
├── Dockerfile            # Container build instructions
├── .env.example          # Environment variables template
├── .gitignore            # Ignores local secrets and cache files
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
