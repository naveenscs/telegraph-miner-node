# Telegraph Protocol Miner Registration Guide

## Prerequisites
- Publicly accessible HTTPS endpoint (via cloud deployment, Ngrok, or Caddy/Nginx reverse proxy).
- Base Sepolia wallet funded with ETH (for gas) and testnet USDC.

## Step 1: Compute Manifest Hash
Generate the SHA-256 hash of your validated `miner.yaml`:

```bash
sha256sum miner.yaml
