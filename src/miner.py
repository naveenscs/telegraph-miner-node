import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from groq import AsyncGroq

app = FastAPI(
    title="Telegraph Miner Node",
    description="Track 1 Telegraph Protocol Miner powered by Groq LPU",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
SSL_VERIFY = os.getenv("SSL_VERIFY", "true").lower() not in ("false", "0", "no")
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "60.0"))

http_client = httpx.AsyncClient(verify=SSL_VERIFY, timeout=GROQ_TIMEOUT)
client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http_client)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = GROQ_MODEL
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


@app.get("/health")
async def health_check():
    """Health endpoint pinged by monitoring and deployment checks."""
    return {
        "status": "active",
        "protocol": "telegraph",
        "track": 1,
        "supported_intents": ["CHAT_COMPLETION"],
        "model": GROQ_MODEL
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """Canonical Telegraph Intent: CHAT_COMPLETION."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on miner")

    try:
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in req.messages]

        completion = await client.chat.completions.create(
            messages=formatted_messages,
            model=req.model or GROQ_MODEL,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            top_p=req.top_p,
            stream=False
        )

        content = completion.choices[0].message.content if completion.choices else ""

        # Returns OpenAI response structure with top-level output for signal_mapping
        return {
            "id": completion.id,
            "object": "chat.completion",
            "created": completion.created,
            "model": completion.model,
            "output": content,
            "confidence": 0.95,
            "reason": "Groq LPU inference completed successfully",
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content
                    },
                    "finish_reason": choice.finish_reason
                } for choice in completion.choices
            ],
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                "completion_tokens": completion.usage.completion_tokens if completion.usage else 0,
                "total_tokens": completion.usage.total_tokens if completion.usage else 0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("miner:app", host="0.0.0.0", port=8000, reload=False)
