# Telegraph Protocol Miner Node (Track 1)

A high-performance, asynchronous miner node engineered for Telegraph Protocol's verified inference network.

## Track
**Track 1: Miners (Network Supply & Compute Layer)**

## System Architecture
- **Control Plane:** Python 3.11+ using `asyncio`, `FastAPI`, & `uvloop` for low-latency network I/O and HTTP REST interface.
- **Data Plane:** Groq LPU API backend (compatible with `vLLM` / local `llama.cpp` model wrappers) running `llama-3.1-8b-instant`.
- **Resilience Engine:** Dual-mode HTTP client supporting automatic SSL validation fallback for restricted proxy / VPN network environments.
- **Verification Layer:** Deterministic signal signature and response timing tracking (`latency_ms`).

---

## Features
- **Low Latency:** Asynchronous execution pipeline delivering sub-second response times (~200–900ms).
- **Dual Runtime:** Can be run as a standalone CLI script or exposed as an OpenAPI-compliant REST API microservice.
- **Network Resilience:** Handles corporate firewalls, VPN proxies, and missing root CA certificate stores without crashing.
- **Interactive OpenAPI Docs:** Built-in Swagger UI for testing endpoints at `/docs`.

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Groq API Key ([Get a free key here](https://console.groq.com))

### Installation
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/telegraph-miner-node.git](https://github.com/YOUR_GITHUB_USERNAME/telegraph-miner-node.git)
cd telegraph-miner-node
pip install -r requirements.txt
