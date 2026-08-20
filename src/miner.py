import os
import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple

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
    version="1.1.4",
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
# Option 3: floor at 512, escalate on empty/truncated (gpt-oss reasoning burns budget).
_DEFAULT_TOKEN_LADDER = "512,768,1024"
TOKEN_LADDER = [
    int(x.strip())
    for x in os.getenv("TOKEN_LADDER", _DEFAULT_TOKEN_LADDER).split(",")
    if x.strip().isdigit()
] or [512, 768, 1024]
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


def _token_budgets(requested_max: Optional[int]) -> List[int]:
    """Option 3: never start below ladder floor (512); escalate up the ladder."""
    floor = TOKEN_LADDER[0]
    ceiling = TOKEN_LADDER[-1]
    start = max(floor, requested_max or floor)
    begin: Optional[int] = None
    for step in TOKEN_LADDER:
        if step >= start:
            begin = step
            break
    if begin is None:
        return [ceiling]
    return [step for step in TOKEN_LADDER if step >= begin]


def _completion_unsatisfactory(content: Optional[str], finish_reason: Optional[str]) -> bool:
    text = (content or "").strip()
    if not text:
        return True
    if (finish_reason or "").lower() == "length":
        return True
    return False


def _pack_response(completion: Any, content: str) -> Dict[str, Any]:
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
        "token_ladder": TOKEN_LADDER,
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

    budgets = _token_budgets(req.max_tokens)
    logger.info("Completion token ladder for this request: %s (client max_tokens=%s)", budgets, req.max_tokens)

    start = next(_key_cycle)
    last_error: Optional[Exception] = None
    last_weak: Optional[Tuple[Any, str, Optional[str]]] = None  # completion, content, finish

    for budget_i, max_tokens in enumerate(budgets):
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

                choice0 = completion.choices[0] if completion.choices else None
                content = choice0.message.content if choice0 else ""
                finish = choice0.finish_reason if choice0 else None

                if _completion_unsatisfactory(content, finish):
                    last_weak = (completion, content or "", finish)
                    logger.warning(
                        "Unsatisfactory completion at max_tokens=%s finish=%r out_len=%s; escalating",
                        max_tokens,
                        finish,
                        len((content or "").strip()),
                    )
                    # Don't burn other keys on the same too-small budget — escalate tokens.
                    break

                if budget_i > 0:
                    logger.info("Accepted completion after escalate to max_tokens=%s", max_tokens)
                return _pack_response(completion, content or "")
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
        else:
            # exhausted keys without unsatisfactory break — continue outer only if needed
            continue

    # Return best-effort last weak answer rather than 500 if Groq succeeded but truncated.
    if last_weak is not None:
        completion, content, finish = last_weak
        logger.warning(
            "Returning truncated/weak answer after ladder exhausted (finish=%r len=%s)",
            finish,
            len(content.strip()),
        )
        return _pack_response(completion, content)

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
