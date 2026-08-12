# Telegraph Protocol Miner Registration Guide

## Prerequisites
- A publicly accessible HTTPS endpoint (via ngrok, VPS, or cloud deploy).
- A Base Sepolia wallet funded with ETH (for gas).

---

## Step 1: Deploy Node and Update `miner.yaml`
1. Expose your node running on port 8000:
   ```bash
   ngrok http 8000
