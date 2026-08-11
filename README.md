# Telegraph Protocol — Track 1 Miner Node

An asynchronous, verified AI inference node built for the **Telegraph Protocol** network. This service processes inbound LLM tasks via FastAPI and Groq, returning cryptographic HMAC-SHA256 signatures to ensure verifiable compute execution.

---

## Key Features

- **FastAPI Core**: Lightweight, asynchronous web backend providing `/v1/inference` and `/health` endpoints.
- **Groq LPU Acceleration**: Utilizes `llama-3.1-8b-instant` for ultra-low latency response generation.
- **Cryptographic Attestation**: Signs task output and execution latency with HMAC-SHA256 to ensure data authenticity.
- **Network & Proxy Compatibility**: Configurable SSL verification (`SSL_VERIFY`) for seamless deployment across WSL2, local enterprise firewalls, and proxy setups.
- **Modern Lifecycle**: Configured with FastAPI `lifespan` context handlers for clean connection management.

---

## Prerequisites

- Python 3.10 or higher
- A [Groq API Key](https://console.groq.com/)
- Docker (optional, for containerized execution)

---

## Environment Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
MINER_SECRET_KEY=telegraph_secret_key_node_01_change_in_production
SSL_VERIFY=false
GROQ_TIMEOUT=60.0
