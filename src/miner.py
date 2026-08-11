import os
import asyncio
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from groq import AsyncGroq

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    def __init__(self, miner_id: str, model_name: str = "llama-3.1-8b-instant"):
        self.miner_id = miner_id
        self.model_name = model_name
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment. Please set it in .env")
            
        self.client = AsyncGroq(api_key=api_key)
        logging.info(f"Initialized Telegraph Miner [{self.miner_id}] using Groq ({self.model_name})")

    async def process_inference(self, request: InferenceRequest) -> InferenceResponse:
        start_time = asyncio.get_running_loop().time()
        
        logging.info(f"[{self.miner_id}] Processing Task '{request.task_id}': {request.prompt[:40]}...")
        
        # Query Groq asynchronously
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

async def main():
    miner = TelegraphMiner(miner_id="node-01")
    req = InferenceRequest(
        task_id="task_102", 
        prompt="Analyze the network risk score for transaction batch #88412."
    )
    res = await miner.process_inference(req)
    logging.info(f"Completed Response:\n{res.model_dump_json(indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
