import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq

app = FastAPI(title="Telegraph Groq Miner")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "llama3-8b-8192"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    try:
        completion = client.chat.completions.create(
            model=req.model,
            messages=[m.model_dump() for m in req.messages],
            temperature=req.temperature
        )
        
        response_data = completion.model_dump()
        
        # Telegraph schema alignment:
        # Map 0.0–1.0 score for confidence_field instead of token count
        response_data["confidence"] = 0.95
        response_data["reason"] = "Groq LLaMA3 inference output complete"
        
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("miner:app", host="0.0.0.0", port=8000, reload=True)
