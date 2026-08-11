import os
import sys
import hmac
import hashlib
import asyncio
import logging
import urllib3
import httpx
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from groq import AsyncGroq
from fastapi import FastAPI, HTTPException

# Load environment variables from .env
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Request Schema
class InferenceRequest(BaseModel):
    task_id: str
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)

# Response Schema with Protocol Signature
class InferenceResponse(BaseModel):
    task_id: str
    output: str
    latency_ms: float
    model_used: str
    miner_id: str
    signature: str  # HMAC-SHA256 signature for validator verification

class TelegraphMiner:
    def __init__(self, miner_id: str = "node-01", model_name: str = "llama-3.1-8b-instant"):
        self.miner_id = miner_id
        self.model_name = model_name
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logging.error("GROQ_API_KEY not found! Set it in your .env file.")
            sys.exit(1)

        self.secret_key = os.getenv("MINER_SECRET_KEY", "telegraph_secret_key_node_01_change_in_production").encode("utf-8")

        ssl_env = os.getenv("SSL_VERIFY", "true").lower()
        self.verify_ssl = ssl_env in ("true", "1", "yes")
        timeout_sec = float(os.getenv("GROQ_TIMEOUT", "60.0"))

        if not self.verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.http_client = httpx.AsyncClient(
            verify=self.verify_ssl,
            timeout=httpx.Timeout(timeout_sec, connect=15.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

        self.client = AsyncGroq(
            api_key=api_key,
            http_client=self.http_client
        )
        
        mode = "Strict SSL" if self.verify_ssl else "Permissive SSL (Bypass)"
        logging.info(f"Initialized Telegraph Miner [{self.miner_id}] | Model: {self.model_name} | Mode: {mode}")

    def generate_signature(self, task_id: str, output: str, latency_ms: float) -> str:
        """Generates an HMAC-SHA256 signature forcing exact 2 decimal places on latency."""
        payload = f"{task_id}:{self.miner_id}:{output}:{latency_ms:.2f}".encode("utf-8")
        return hmac.new(self.secret_key, payload, hashlib.sha256).hexdigest()

    async def process_inference(self, request: InferenceRequest) -> InferenceResponse:
        start_time = asyncio.get_running_loop().time()
        logging.info(f"[{self.miner_id}] Processing Task '{request.task_id}': {request.prompt[:40]}...")
        
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a specialized AI inference node serving verified signals to the Telegraph Protocol network. Keep answers concise, factual, and analytical."},
                    {"role": "user", "content": request.prompt}
                ],
                max_tokens=request.max_tokens,
                temperature=0.2,
            )
            
            output_text = completion.choices[0].message.content
            latency = round((asyncio.get_running_loop().time() - start_time) * 1000, 2)
            
            # Sign output with deterministic 2-decimal formatting
            signature = self.generate_signature(request.task_id, output_text, latency)

            return InferenceResponse(
                task_id=request.task_id,
                output=output_text,
                latency_ms=latency,
                model_used=self.model_name,
                miner_id=self.miner_id,
                signature=signature
            )
        except Exception as e:
            logging.error(f"Inference error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Miner failure: {str(e)}")

    async def close(self):
        await self.http_client.aclose()

# Global miner instance
miner_node = TelegraphMiner()

# Lifespan Context Manager (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await miner_node.close()

# FastAPI App Setup
app = FastAPI(
    title="Telegraph Miner Node API",
    version="1.1.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {
        "status": "active",
        "miner_id": miner_node.miner_id,
        "model": miner_node.model_name,
        "signing_enabled": True
    }

@app.post("/v1/inference", response_model=InferenceResponse)
async def handle_inference(request: InferenceRequest):
    return await miner_node.process_inference(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("miner:app", host="0.0.0.0", port=8000, reload=True)
