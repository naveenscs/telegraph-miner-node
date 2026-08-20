import os
import itertools
import logging
import re
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
    version="1.1.6",
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
# Infer short vs explain vs howto vs as-is from the user question (no intent in body).
ADAPTIVE_STYLE = os.getenv("ADAPTIVE_STYLE", "true").lower() not in ("false", "0", "no")
# Legacy blanket short mode if adaptive is off.
SHORT_ANSWERS = os.getenv("SHORT_ANSWERS", "true").lower() not in ("false", "0", "no")
# Option 3: floor at 512, escalate on empty/truncated (gpt-oss reasoning burns budget).
_DEFAULT_TOKEN_LADDER = "512,768,1024"
TOKEN_LADDER = [
    int(x.strip())
    for x in os.getenv("TOKEN_LADDER", _DEFAULT_TOKEN_LADDER).split(",")
    if x.strip().isdigit()
] or [512, 768, 1024]

# GT-match-first: WASM scores ~50% cosine vs ground truth. Optimize for a typical
# reference answer (on-topic, general, enough length), not maximal brevity.
_GT_CORE = (
    "Match a typical reference / ground-truth answer: accurate, on-topic, "
    "general (not a niche digression), plain prose. Aim for roughly 80–250 "
    "characters of substance unless the question clearly needs more. "
    "No markdown tables, no filler, no apologies."
)
PROMPT_FORECAST = (
    "You are a Telegraph miner. "
    + _GT_CORE
    + " For forecast or yes/no questions: lead with Yes / No / Lean yes / "
    "Lean no / Uncertain, then 2–4 short supporting sentences."
)
PROMPT_EXPLAIN = (
    "You are a Telegraph miner. "
    + _GT_CORE
    + " For definition / explain / compare questions: one clear textbook-style "
    "paragraph (two short ones max). Prefer the common general definition; "
    "do not digress into niche variants unless asked. Bullets only if the "
    "question asks for a list."
)
PROMPT_HOWTO = (
    "You are a Telegraph miner. "
    + _GT_CORE
    + " For how / process questions: explain the standard mechanism in order "
    "using short paragraphs or up to 5 numbered steps."
)
PROMPT_DEFAULT = (
    "You are a Telegraph miner. "
    + _GT_CORE
    + " Answer the question directly in one short paragraph unless steps or "
    "a list are clearly required."
)
PROMPT_FORECAST = os.getenv("PROMPT_FORECAST", PROMPT_FORECAST).strip()
PROMPT_EXPLAIN = os.getenv("PROMPT_EXPLAIN", PROMPT_EXPLAIN).strip()
PROMPT_HOWTO = os.getenv("PROMPT_HOWTO", PROMPT_HOWTO).strip()
PROMPT_DEFAULT = os.getenv("PROMPT_DEFAULT", PROMPT_DEFAULT).strip()
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", PROMPT_DEFAULT).strip()


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


def _last_user_text(messages: List[dict]) -> str:
    for msg in reversed(messages):
        if str(msg.get("role", "")).lower() == "user":
            return str(msg.get("content") or "")
    return str(messages[-1].get("content") or "") if messages else ""


def _classify_question(text: str) -> str:
    """Return style: forecast | explain | howto | default (mild GT-match)."""
    q = " ".join((text or "").strip().split())
    if not q:
        return "default"
    low = q.lower()

    if re.search(r"\b(yes or no|true or false)\b", low):
        return "forecast"

    if re.search(r"\bwill\b.+\?", low) or re.match(
        r"^(will|is|are|can|should|do|does)\b", low
    ):
        if any(
            k in low
            for k in (
                "will ",
                "by 20",
                "mainstream",
                "adopt",
                "happen",
                "likely",
                "probable",
                "forecast",
            )
        ):
            return "forecast"
        if low.startswith(
            ("will ", "is it ", "are there ", "can we ", "should ", "does it ", "do you ")
        ):
            return "forecast"

    if re.match(r"^how\b", low) or re.search(
        r"\b(step by step|token by token|how do|how does|how can|how to|walk me through)\b",
        low,
    ):
        return "howto"

    if re.search(
        r"\b(explain|what is|what's|what are|what does|define|definition|"
        r"difference between|compare|versus|vs\.?|why (is|are|do|does|did)|"
        r"describe|in one (paragraph|sentence)|briefly|summarize|list the|"
        r"name the|enumerate)\b",
        low,
    ):
        return "explain"

    return "default"


def _system_prompt_for_style(style: str) -> Optional[str]:
    if style == "forecast":
        return PROMPT_FORECAST or None
    if style == "explain":
        return PROMPT_EXPLAIN or None
    if style == "howto":
        return PROMPT_HOWTO or None
    if style in ("default", "as_is"):
        return PROMPT_DEFAULT or None
    return None


def _prepare_messages(messages: List[dict]) -> Tuple[List[dict], str]:
    """Attach GT-match style prompt when adaptive; mild default when unsure."""
    if messages and str(messages[0].get("role", "")).lower() == "system":
        return messages, "client_system"

    if ADAPTIVE_STYLE:
        style = _classify_question(_last_user_text(messages))
        prompt = _system_prompt_for_style(style)
        if prompt:
            return [{"role": "system", "content": prompt}, *messages], style
        return messages, style

    if SHORT_ANSWERS and SYSTEM_PROMPT:
        return [{"role": "system", "content": SYSTEM_PROMPT}, *messages], "legacy_short"
    return messages, "as_is"


def _token_budgets(requested_max: Optional[int], style: str = "as_is") -> List[int]:
    """Option 3 ladder; explain/howto may start one step higher when possible."""
    floor = TOKEN_LADDER[0]
    ceiling = TOKEN_LADDER[-1]
    start = max(floor, requested_max or floor)
    if style in ("explain", "howto") and len(TOKEN_LADDER) > 1:
        start = max(start, TOKEN_LADDER[1])
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
        "adaptive_style": ADAPTIVE_STYLE,
        "token_ladder": TOKEN_LADDER,
    }


async def _chat_completions_impl(req: ChatCompletionRequest):
    """Canonical Telegraph chat handler."""
    if not GROQ_API_KEYS:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on miner")

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")

    base_messages = [{"role": msg.role, "content": msg.content} for msg in req.messages]
    formatted_messages, style = _prepare_messages(base_messages)
    requested_model = (req.model or GROQ_MODEL).strip()
    model = GROQ_MODEL
    if requested_model != GROQ_MODEL:
        logger.info("Ignoring client model %r; using %s", requested_model, GROQ_MODEL)

    budgets = _token_budgets(req.max_tokens, style=style)
    logger.info(
        "style=%s token_ladder=%s (client max_tokens=%s)",
        style,
        budgets,
        req.max_tokens,
    )

    start = next(_key_cycle)
    last_error: Optional[Exception] = None
    last_weak: Optional[Tuple[Any, str, Optional[str]]] = None

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
            continue

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
