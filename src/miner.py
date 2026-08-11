import os
import sys
import asyncio
import logging
import urllib3
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel
from groq import AsyncGroq
from fastapi import FastAPI, HTTPException

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class InferenceRequest(BaseModel):
    task_id: str
    prompt: str
    max_tokens: int = 512

class InferenceResponse(BaseModel):
    task_id: str
    output: str
    latency_ms: float
    model_used: str

class TelegraphMiner:
    def __init__(self, miner_id: str = "node-01", model_name: str = "llama-3.1-8b-instant"):
        self.miner_id = miner_id
        self.model_name = model_name
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logging.error("GROQ_API_KEY not found! Set it in your .env file.")
            sys.exit(1)

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
            execution_time = (asyncio.get_running_loop().time() - start_time) * 1000
            
            return InferenceResponse(
                task_id=request.task_id,
                output=output_text,
                latency_ms=round(execution_time, 2),
                model_used=self.model_name
            )
        except Exception as e:
            logging.error(f"Inference error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def close(self):
        await self.http_client.aclose()

# FastAPI Server Initialization
app = FastAPI(title="Telegraph Miner Node API", version="1.0.0")
miner_node = TelegraphMiner()

@app.get("/health")
async def health_check():
    return {"status": "active", "miner_id": miner_node.miner_id, "model": miner_node.model_name}

@app.post("/v1/inference", response_model=InferenceResponse)
async def handle_inference(request: InferenceRequest):
    return await miner_node.process_inference(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("miner:app", host="0.0.0.0", port=8000, reload=True)
