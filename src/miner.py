import os
import itertools
import logging
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telegraph-miner")

app = FastAPI(
    title="Telegraph Miner Node",
    description="Track 1 Telegraph Protocol Miner powered by Groq LPU",
    version="1.1.3",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq shut down llama-3.1-8b-instant on 2026-08-16 (free/dev tier).
# Default replacement per https://console.groq.com/docs/deprecations
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
SSL_VERIFY = os.getenv("SSL_VERIFY", "true").lower() not in ("false", "0", "no")
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "60.0"))
# Concise answers tend to score better on Telegraph CHAT_COMPLETION evals.
SHORT_ANSWERS = os.getenv("SHORT_ANSWERS", "true").lower() not in ("false", "0", "no")
SHORT_ANSWER_MAX_TOKENS = int(os.getenv("SHORT_ANSWER_MAX_TOKENS", "384"))
_DEFAULT_SYSTEM_PROMPT = (
    "You are a concise Telegraph miner. "
    "For forecast or yes/no questions (e.g. 'Will …?'): "
    "start with exactly one of: Yes / No / Lean yes / Lean no / Uncertain. "
    "Then give at most 5 short sentences of support. "
    "No markdown tables, no long essays, no bullet lists longer than 5 items. "
    "Be decisive; avoid hedging filler."
)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT).strip()


def _load_groq_keys() -> List[str]:
    keys: List[str] = []
    multi = os.getenv("GROQ_API_KEYS", "")
    if multi:
        keys.extend(k.strip() for k in multi.split(",") if k.strip())
    for name in (
        "GROQ_API_KEY",
        "GROQ_API_KEY_2",
        "GROQ_API_KEY_3",
        "GROQ_API_KEY_4",
        "GROQ_API_KEY_5",
    ):
        value = os.getenv(name, "").strip()
        if value and value not in keys:
            keys.append(value)
    return keys


GROQ_API_KEYS = _load_groq_keys()
_http_client = httpx.AsyncClient(verify=SSL_VERIFY, timeout=GROQ_TIMEOUT)
_key_cycle = itertools.cycle(range(max(len(GROQ_API_KEYS), 1)))


def _is_retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "401",
        "invalid api key",
        "invalid_api_key",
        "authentication",
        "unauthorized",
        "429",
        "rate limit",
        "rate_limit",
        "too many requests",
        "quota",
        "over capacity",
        "timeout",
        "temporar",
        "503",
        "502",
        "cloudflare",
    )
    return any(m in text for m in markers)


def _with_system_prompt(messages: List[dict]) -> List[dict]:
    """Prepend concise system guidance unless the client already sent a system message."""
    if not SHORT_ANSWERS or not SYSTEM_PROMPT:
        return messages
    if messages and str(messages[0].get("role", "")).lower() == "system":
        return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}, *messages]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = GROQ_MODEL
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = Field(default=512, ge=1)
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


@app.get("/health")
async def health_check():
    """Health endpoint pinged by monitoring and deployment checks."""
    return {
        "status": "active",
        "protocol": "telegraph",
        "track": 1,
        "supported_intents": ["CHAT_COMPLETION", "LANGUAGE_GENERATION", "TEXT_GENERATION"],
        "model": GROQ_MODEL,
        "groq_keys_configured": len(GROQ_API_KEYS),
        "short_answers": SHORT_ANSWERS,
    }


async def _chat_completions_impl(req: ChatCompletionRequest):
    """Canonical Telegraph Intent: CHAT_COMPLETION."""
    if not GROQ_API_KEYS:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on miner")

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")

    formatted_messages = _with_system_prompt(
        [{"role": msg.role, "content": msg.content} for msg in req.messages]
    )
    requested_model = (req.model or GROQ_MODEL).strip()
    # Always call Groq with the configured model. Telegraph often sends the miner
    # slug (e.g. groq-llama31-instant-miner) or placeholders in `model`.
    model = GROQ_MODEL
    if requested_model != GROQ_MODEL:
        logger.info("Ignoring client model %r; using %s", requested_model, GROQ_MODEL)

    # gpt-oss can spend early tokens on reasoning; avoid empty visible output.
    # Short-answer mode also caps completion length to reduce essay-style replies.
    requested_max = req.max_tokens or 512
    if SHORT_ANSWERS:
        max_tokens = max(min(requested_max, SHORT_ANSWER_MAX_TOKENS), 256)
    else:
        max_tokens = max(requested_max, 256)

    start = next(_key_cycle)
    last_error: Optional[Exception] = None

    for attempt in range(len(GROQ_API_KEYS)):
        key = GROQ_API_KEYS[(start + attempt) % len(GROQ_API_KEYS)]
        client = AsyncGroq(api_key=key, http_client=_http_client)
        try:
            completion = await client.chat.completions.create(
                messages=formatted_messages,
                model=model,
                temperature=req.temperature,
                max_tokens=max_tokens,
                top_p=req.top_p,
                stream=False,
            )

            content = completion.choices[0].message.content if completion.choices else ""

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
                            "content": choice.message.content,
                        },
                        "finish_reason": choice.finish_reason,
                    }
                    for choice in completion.choices
                ],
                "usage": {
                    "prompt_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                    "completion_tokens": completion.usage.completion_tokens if completion.usage else 0,
                    "total_tokens": completion.usage.total_tokens if completion.usage else 0,
                },
            }
        except Exception as exc:
            last_error = exc
            if _is_retryable(exc) and attempt + 1 < len(GROQ_API_KEYS):
                logger.warning(
                    "Groq call failed on key index %s (%s); rotating key",
                    (start + attempt) % len(GROQ_API_KEYS),
                    exc,
                )
                continue
            logger.exception("Inference failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Inference execution failed: {exc}") from exc

    raise HTTPException(
        status_code=500,
        detail=f"Inference execution failed after key rotation: {last_error}",
    )


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    return await _chat_completions_impl(req)


@app.post("/chat")
async def chat_alias(req: ChatCompletionRequest):
    """YAML path alias — some Telegraph node calls hit /chat instead of external_path."""
    return await _chat_completions_impl(req)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("miner:app", host="0.0.0.0", port=8000, reload=False)
