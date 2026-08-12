# Telegraph Protocol — Track 1 Miner Node

High-throughput Groq-backed miner for the Telegraph Protocol **Track 1 (Network Supply & Compute Layer)**.

This node exposes a public HTTPS inference API that Telegraph routes to after on-chain YAML registration.

## Intent

- `CHAT_COMPLETION`

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness / monitoring |
| POST | `/v1/chat/completions` | OpenAI-compatible chat (Telegraph `CHAT_COMPLETION`) |

### Request

```json
{
  "messages": [{"role": "user", "content": "What is zkTLS?"}],
  "temperature": 0.7,
  "max_tokens": 1024
}
```

### Response (includes Telegraph `signal_mapping` fields)

```json
{
  "output": "...",
  "confidence": 0.95,
  "reason": "Groq LPU inference completed successfully",
  "choices": [{"message": {"role": "assistant", "content": "..."}}],
  "usage": {"total_tokens": 42}
}
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env — set GROQ_API_KEY
uvicorn src.miner:app --host 0.0.0.0 --port 8000
```

Docker:

```bash
cp .env.example .env
docker compose up -d --build
curl http://127.0.0.1:8000/health
```

Local smoke test:

```bash
python src/validator_test.py
```

## Track 1 integration

1. Deploy this API on a stable **HTTPS** URL
2. Set `base_url` in `miner.yaml` to that URL
3. Validate + register via [integrate.telegraphprotocol.com](https://integrate.telegraphprotocol.com)
4. Follow `docs/REGISTRATION.md`
5. Confirm your slug appears in:
   `GET http://13.237.89.59:7044/miner-dispatcher/integrations`
6. Keep the miner live through Track 3
7. Post progress on X tagged [@Telegraphprotoc](https://x.com/Telegraphprotoc)

Key files:

- `miner.yaml` — Telegraph Miner Standard config
- `docs/REGISTRATION.md` — deploy / hash / register checklist

## Security

- Never commit `.env` or real API keys
- Rotate any key previously exposed in git history
- Prefer `SSL_VERIFY=true` in production

## References

- Hackathon: https://hackathon.telegraphprotocol.com/
- Rules: https://hackathon.telegraphprotocol.com/rules
- Miner overview: https://docs.telegraphprotocol.com/docs/miners/miner-overview
- YAML config: https://docs.telegraphprotocol.com/docs/miners/yaml-config
- Registration: https://docs.telegraphprotocol.com/docs/miners/miner-registration
