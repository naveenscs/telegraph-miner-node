# Telegraph Protocol Miner Registration Guide

This guide walks you through deploying, validating, and registering your Telegraph Groq Miner for Track 1.

---

## Prerequisites
- A running miner node on local port `8000`.
- A publicly accessible HTTPS URL pointing to port `8000` (e.g., via `ngrok`, Cloudflare Tunnel, or VPS with Caddy/Nginx).
- A Base Sepolia EVM wallet funded with testnet ETH (for gas) and testnet USDC.

---

## Step 1: Deploy Node & Public HTTPS Endpoint

1. Start your local miner node:
   ```bash
   python3 -m uvicorn src.miner:app --host 0.0.0.0 --port 8000
