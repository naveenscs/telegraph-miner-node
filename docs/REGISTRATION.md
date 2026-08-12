# Telegraph Protocol Miner Registration Guide

## Prerequisites

- Public HTTPS endpoint for this miner (ngrok, VPS, or cloud deploy)
- Base Sepolia wallet with ETH for gas
- Fee address for MACHINA payouts
- Validated `miner.yaml`

---

## Step 1: Deploy the node

```bash
cp .env.example .env
# set GROQ_API_KEY in .env
docker compose up -d --build
curl http://127.0.0.1:8000/health
```

Expose HTTPS, for example:

```bash
ngrok http 8000
```

Copy the `https://...` URL.

---

## Step 2: Update `miner.yaml`

Set:

```yaml
base_url: "https://YOUR_PUBLIC_HTTPS_HOST"
```

Commit/push the updated YAML (or host it separately on IPFS/HTTPS).

Public raw URL example:

`https://raw.githubusercontent.com/naveenscs/telegraph-miner-node/main/miner.yaml`

---

## Step 3: Validate YAML (required)

1. Open https://integrate.telegraphprotocol.com
2. Paste `miner.yaml`
3. Sandbox-test `/v1/chat/completions` against your live HTTPS API
4. Fix any schema/auth/endpoint failures before registering

---

## Step 4: Compute YAML hash

```bash
sha256sum miner.yaml | awk '{print "0x"$1}'
```

If the YAML changes after hashing, recompute before registration.

---

## Step 5: Register on-chain (Base Sepolia)

Preferred: complete registration in the integrate UI (pins YAML + sends `registerMiner`).

Manual path: follow  
https://docs.telegraphprotocol.com/docs/miners/miner-registration

Use at least:

- Intent: `CHAT_COMPLETION`
- Floor price: `10000` (= $0.01 USDC minimum)

---

## Step 6: Confirm activation

Wait a few minutes, then:

```bash
curl http://13.237.89.59:7044/miner-dispatcher/integrations
```

Your slug `groq-llama31-instant-miner` should appear.

Also check the Intelligence Terminal:  
https://terminal.telegraphprotocol.com/

---

## Step 7: Hackathon process

- Stay live through Track 3
- Join official Hackathon Discord
- Post updates on X tagged `@Telegraphprotoc`
