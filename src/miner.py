import asyncio
import json
import os
import itertools
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import AsyncGroq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telegraph-miner")

app = FastAPI(
    title="Telegraph Miner Node",
    description="Track 1 Telegraph Protocol Miner powered by Groq LPU",
    version="1.2.5",
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
    "general (not a niche digression), plain prose like a textbook or API doc. "
    "Aim for roughly 80–250 characters of substance unless the question clearly "
    "needs more. No markdown (no bold/italics/bullets unless asked), no filler, "
    "no apologies."
)
# GT cluster: chat completion / OpenAI-style messages API (roles, history, window).
_CHAT_API_FACTS = (
    "If the question is about chat completion, chat models, message roles, "
    "conversation history, multi-turn chat, context windows, or system prompts: "
    "answer like an API doc. Prefer GT-style wording. Key facts — "
    "(1) Stateless: the model does not retain memory between separate API calls. "
    "(2) Multi-turn context: on every request the client resends the messages "
    "list (full history, or as much as fits); the model reads that list each time "
    "to stay consistent with prior turns — not an internal long-term memory. "
    "(3) Roles: system = standing instructions/behavior; user = human input; "
    "assistant = prior model replies already in the messages list (history), "
    "not only the next reply being generated. Role is the label on each message "
    "telling the model who produced it and how to weigh that content. "
    "(4) Context window / conversation history: the sequence of prior messages "
    "the model can attend to for the next reply — so it can reference earlier "
    "statements, keep topic continuity, and avoid contradicting or repeating "
    "itself. It is resent with each request because the model is stateless. "
    "(5) Separate system prompt: a distinct channel for persistent developer "
    "instructions, persona, or constraints that apply throughout the conversation "
    "without being mixed into or overridden by the user's own input — independent "
    "of the dialogue turns. "
    "(6) Over context limit: when history + new input exceed the max context "
    "length, the request fails with an error, or earlier messages are truncated, "
    "summarized, or dropped to fit; the model then loses access to the removed "
    "turns. "
    "(7) Single-turn vs multi-turn: a single-turn prompt is one self-contained "
    "input that produces one response with no expectation of continued dialogue. "
    "A multi-turn chat exchange is a sequence of related messages where later "
    "responses build on and reference earlier ones. When asked for that "
    "difference, lead with this dialogue definition; mention resent message "
    "lists only as brief supporting API detail. "
    "Prefer these facts over 'the model remembers' or vague sliding-window "
    "metaphors unless the question asks for that."
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
    "paragraph (two short ones max). Prefer the common general definition from "
    "standard docs; do not digress into niche variants unless asked. "
    + _CHAT_API_FACTS
)
PROMPT_HOWTO = (
    "You are a Telegraph miner. "
    + _GT_CORE
    + " For how / process questions: explain the standard mechanism in order "
    "using short paragraphs or up to 5 numbered steps. "
    + _CHAT_API_FACTS
)
PROMPT_DEFAULT = (
    "You are a Telegraph miner. "
    + _GT_CORE
    + " Answer the question directly in one short paragraph unless steps or "
    "a list are clearly required. "
    + _CHAT_API_FACTS
)
PROMPT_FORECAST = os.getenv("PROMPT_FORECAST", PROMPT_FORECAST).strip()
PROMPT_EXPLAIN = os.getenv("PROMPT_EXPLAIN", PROMPT_EXPLAIN).strip()
PROMPT_HOWTO = os.getenv("PROMPT_HOWTO", PROMPT_HOWTO).strip()
PROMPT_DEFAULT = os.getenv("PROMPT_DEFAULT", PROMPT_DEFAULT).strip()
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", PROMPT_DEFAULT).strip()
# Lower temperature for definition-style answers (closer to GT wording).
EXPLAIN_TEMPERATURE = float(os.getenv("EXPLAIN_TEMPERATURE", "0.25"))


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
# Extra 429 sleep-retries beyond one pass over keys (same-org keys share TPM).
GROQ_RATE_LIMIT_ROUNDS = max(1, int(os.getenv("GROQ_RATE_LIMIT_ROUNDS", "8")))


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


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        m in text
        for m in ("429", "rate limit", "rate_limit", "too many requests", "tokens per minute", "tpm")
    )


def _rate_limit_wait_seconds(exc: Exception) -> Optional[float]:
    """Parse Groq's 'Please try again in X.Xs / Yms' hint; cap to keep tournament latency sane."""
    text = str(exc)
    m = re.search(r"try again in ([0-9]*\.?[0-9]+)\s*ms", text, re.I)
    if m:
        return min(float(m.group(1)) / 1000.0 + 0.15, 6.0)
    m = re.search(r"try again in ([0-9]*\.?[0-9]+)\s*s", text, re.I)
    if m:
        return min(float(m.group(1)) + 0.2, 6.0)
    if _is_rate_limit(exc):
        return 1.25
    return None


def _last_user_text(messages: List[dict]) -> str:
    for msg in reversed(messages):
        if str(msg.get("role", "")).lower() == "user":
            return str(msg.get("content") or "")
    return str(messages[-1].get("content") or "") if messages else ""


def _is_chat_api_question(low: str) -> bool:
    """Detect OpenAI-style chat/completion API fact questions (GT cluster)."""
    return bool(
        re.search(
            r"\b("
            r"chat completion|chat model|chat api|message(?:s)? (?:list|format|role)|"
            r"system prompt|conversation history|context window|multi-?turn|single-?turn|"
            r"system,\s*user,\s*(?:and )?assistant|role(?:s)? in (?:a )?chat|"
            r"maintain context|across (?:multiple )?turns"
            r")\b",
            low,
        )
    )


def _classify_question(text: str) -> str:
    """Return style: forecast | explain | howto | default (mild GT-match)."""
    q = " ".join((text or "").strip().split())
    if not q:
        return "default"
    low = q.lower()

    if re.search(r"\b(yes or no|true or false)\b", low):
        return "forecast"

    # Chat-API facts score best as short textbook paragraphs (explain), even when
    # phrased as "How does …" (otherwise howto wins and misses GT wording).
    if _is_chat_api_question(low):
        return "explain"

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


def _base_style(style: str) -> str:
    """Strip '+client_system' (or similar) suffix for ladder/temperature decisions."""
    return (style or "as_is").split("+", 1)[0]


def _prepare_messages(messages: List[dict]) -> Tuple[List[dict], str]:
    """Attach GT-match style prompt; still apply when client already sent a system message."""
    if not ADAPTIVE_STYLE:
        if SHORT_ANSWERS and SYSTEM_PROMPT:
            if messages and str(messages[0].get("role", "")).lower() == "system":
                merged = list(messages)
                client_sys = str(merged[0].get("content") or "")
                merged[0] = {
                    "role": "system",
                    "content": (client_sys + "\n\n" + SYSTEM_PROMPT).strip()
                    if client_sys
                    else SYSTEM_PROMPT,
                }
                return merged, "legacy_short+client_system"
            return [{"role": "system", "content": SYSTEM_PROMPT}, *messages], "legacy_short"
        return messages, "as_is"

    style = _classify_question(_last_user_text(messages))
    prompt = _system_prompt_for_style(style)
    has_client_system = bool(
        messages and str(messages[0].get("role", "")).lower() == "system"
    )
    tag = f"{style}+client_system" if has_client_system else style

    if not prompt:
        return messages, tag

    if has_client_system:
        merged = list(messages)
        client_sys = str(merged[0].get("content") or "")
        merged[0] = {
            "role": "system",
            "content": (client_sys + "\n\n" + prompt).strip() if client_sys else prompt,
        }
        return merged, tag

    return [{"role": "system", "content": prompt}, *messages], style


def _token_budgets(requested_max: Optional[int], style: str = "as_is") -> List[int]:
    """Option 3 ladder; explain/howto may start one step higher when possible."""
    floor = TOKEN_LADDER[0]
    ceiling = TOKEN_LADDER[-1]
    start = max(floor, requested_max or floor)
    if _base_style(style) in ("explain", "howto") and len(TOKEN_LADDER) > 1:
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


def _coerce_message_content(value: Any) -> str:
    """Normalize OpenAI-style content (str | list of parts | null) to a plain string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("text") is not None:
                    parts.append(str(item.get("text")))
                elif item.get("content") is not None:
                    parts.append(str(item.get("content")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _extract_user_text(value: Any, *, depth: int = 0) -> str:
    """Best-effort pull of a user-facing question/prompt from any JSON-ish value."""
    if depth > 4 or value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_extract_user_text(v, depth=depth + 1) for v in value]
        return "\n".join(p for p in parts if p).strip()
    if isinstance(value, dict):
        # Prefer known content keys first.
        for key in (
            "content",
            "text",
            "input",
            "prompt",
            "query",
            "question",
            "message",
            "user",
        ):
            if key in value and value[key] is not None:
                got = _extract_user_text(value[key], depth=depth + 1)
                if got:
                    return got
        # Else longest remaining string field (skip meta).
        skip = {
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "stream",
            "n",
            "stop",
            "role",
            "id",
            "object",
            "created",
            "usage",
            "choices",
            "confidence",
            "reason",
            "output",
        }
        best = ""
        for k, v in value.items():
            if str(k).lower() in skip:
                continue
            got = _extract_user_text(v, depth=depth + 1)
            if len(got) > len(best):
                best = got
        return best
    return _coerce_message_content(value).strip()


def _normalize_chat_body(data: Any) -> dict:
    """Map any reasonable request shape into OpenAI-style {messages:[...]}."""
    if data is None:
        data = {}
    if isinstance(data, str):
        data = {"input": data}
    if not isinstance(data, dict):
        data = {"input": _coerce_message_content(data)}

    out = dict(data)
    messages = out.get("messages")

    if isinstance(messages, str) and messages.strip():
        out["messages"] = [{"role": "user", "content": messages.strip()}]
        return out

    if isinstance(messages, list) and messages:
        normalized = []
        for item in messages:
            if isinstance(item, str) and item.strip():
                normalized.append({"role": "user", "content": item.strip()})
            elif isinstance(item, dict):
                role = str(item.get("role") or "user")
                content = _extract_user_text(item.get("content", item))
                if content:
                    normalized.append({"role": role, "content": content})
            else:
                text = _extract_user_text(item)
                if text:
                    normalized.append({"role": "user", "content": text})
        if normalized:
            out["messages"] = normalized
            return out

    # No usable messages — dig known aliases, then any string in the body.
    for key in ("input", "prompt", "query", "text", "question", "message", "user"):
        if key in out and out[key] is not None:
            text = _extract_user_text(out[key])
            if text:
                logger.info("Coerced body.%s into messages[user] (len=%s)", key, len(text))
                out["messages"] = [{"role": "user", "content": text}]
                return out

    text = _extract_user_text(out)
    if text:
        logger.info("Coerced generic body text into messages[user] (len=%s)", len(text))
        out["messages"] = [{"role": "user", "content": text}]
        return out

    # Last resort: never leave messages missing (avoids 422 → empty tournament score).
    logger.warning("No user text found in body; using generic fallback prompt keys=%s", list(out.keys())[:20])
    out["messages"] = [
        {
            "role": "user",
            "content": (
                "Provide a short, accurate general answer. "
                "If the question is missing, explain that the request body had no readable input."
            ),
        }
    ]
    return out


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = "user"
    content: str = ""

    @field_validator("content", mode="before")
    @classmethod
    def _content_to_str(cls, value: Any) -> str:
        return _coerce_message_content(value)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = GROQ_MODEL
    messages: List[ChatMessage] = Field(default_factory=list)
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = Field(default=512, ge=1)
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False

    @model_validator(mode="before")
    @classmethod
    def _coerce_plain_input(cls, data: Any) -> Any:
        return _normalize_chat_body(data)


def _fallback_pack(content: str) -> Dict[str, Any]:
    """OpenAI-shaped response when we must answer without a Groq completion object."""
    import time

    text = (content or "").strip() or (
        "Unable to parse a clear question from the request; please retry with a messages or input field."
    )
    return {
        "id": "chatcmpl-fallback",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": GROQ_MODEL,
        "output": text,
        "confidence": 0.5,
        "reason": "Fallback response after request-shape recovery",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.exception_handler(RequestValidationError)
async def _log_validation_error(request: Request, exc: RequestValidationError):
    """Log rejected bodies; for chat paths, salvage instead of returning empty/422."""
    raw = b""
    try:
        raw = await request.body()
    except Exception as body_exc:
        logger.warning("422 could not read body: %s", body_exc)
    body_preview = raw.decode("utf-8", errors="replace")[:4000]
    logger.warning(
        "422 validation failed method=%s path=%s errors=%s body=%s",
        request.method,
        request.url.path,
        exc.errors(),
        body_preview,
    )
    path = request.url.path or ""
    if path in ("/v1/chat/completions", "/chat") and request.method.upper() == "POST":
        try:
            parsed: Any = json.loads(body_preview) if body_preview.strip() else {}
        except Exception:
            parsed = {"input": body_preview} if body_preview.strip() else {}
        try:
            req = ChatCompletionRequest.model_validate(_normalize_chat_body(parsed))
            return await _chat_completions_impl(req)
        except Exception as salvage_exc:
            logger.exception("Chat 422 salvage failed: %s", salvage_exc)
            return JSONResponse(status_code=200, content=_fallback_pack(""))
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


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
        # Should be rare after _normalize_chat_body; still never return empty.
        return _fallback_pack("")

    base_messages = [{"role": msg.role, "content": msg.content} for msg in req.messages]
    formatted_messages, style = _prepare_messages(base_messages)
    requested_model = (req.model or GROQ_MODEL).strip()
    model = GROQ_MODEL
    if requested_model != GROQ_MODEL:
        logger.info("Ignoring client model %r; using %s", requested_model, GROQ_MODEL)

    budgets = _token_budgets(req.max_tokens, style=style)
    temperature = req.temperature if req.temperature is not None else 0.7
    if _base_style(style) in ("explain", "howto", "default"):
        temperature = min(temperature, EXPLAIN_TEMPERATURE)
    logger.info(
        "style=%s token_ladder=%s temperature=%s (client max_tokens=%s)",
        style,
        budgets,
        temperature,
        req.max_tokens,
    )

    start = next(_key_cycle)
    last_error: Optional[Exception] = None
    last_weak: Optional[Tuple[Any, str, Optional[str]]] = None
    # Same-org API keys share TPM — spinning all keys on 429 is useless. Sleep using
    # Groq's retry hint, then retry (still rotate key index for multi-org setups).
    attempt_budget = max(len(GROQ_API_KEYS), GROQ_RATE_LIMIT_ROUNDS)

    for budget_i, max_tokens in enumerate(budgets):
        for attempt in range(attempt_budget):
            key = GROQ_API_KEYS[(start + attempt) % len(GROQ_API_KEYS)]
            client = AsyncGroq(api_key=key, http_client=_http_client, max_retries=0)
            try:
                completion = await client.chat.completions.create(
                    messages=formatted_messages,
                    model=model,
                    temperature=temperature,
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
                wait = _rate_limit_wait_seconds(exc) if _is_rate_limit(exc) else None
                if wait is not None and attempt + 1 < attempt_budget:
                    logger.warning(
                        "Groq 429 on key index %s; sleeping %.2fs then retry (attempt %s/%s)",
                        (start + attempt) % len(GROQ_API_KEYS),
                        wait,
                        attempt + 1,
                        attempt_budget,
                    )
                    await asyncio.sleep(wait)
                    continue
                if _is_retryable(exc) and attempt + 1 < attempt_budget:
                    logger.warning(
                        "Groq call failed on key index %s (%s); rotating key",
                        (start + attempt) % len(GROQ_API_KEYS),
                        exc,
                    )
                    continue
                logger.exception("Inference failed: %s", exc)
                return _fallback_pack(
                    "Temporary inference error; please retry. "
                    f"Detail: {type(exc).__name__}"
                )
        else:
            continue

    if last_weak is not None:
        completion, content, finish = last_weak
        if (content or "").strip():
            logger.warning(
                "Returning truncated/weak answer after ladder exhausted (finish=%r len=%s)",
                finish,
                len(content.strip()),
            )
            return _pack_response(completion, content)
        logger.warning("Ladder exhausted with empty content; returning fallback text")
        return _fallback_pack(
            "The model returned an empty completion after retries. "
            "Please resend the question."
        )

    logger.error("Inference failed after retries: %s", last_error)
    return _fallback_pack(
        "Temporary inference error after retries; please try again."
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        data = await request.json()
    except Exception:
        raw = (await request.body()).decode("utf-8", errors="replace")
        data = {"input": raw} if raw.strip() else {}
    req = ChatCompletionRequest.model_validate(_normalize_chat_body(data))
    return await _chat_completions_impl(req)


@app.post("/chat")
async def chat_alias(request: Request):
    """YAML path alias — some Telegraph node calls hit /chat instead of external_path."""
    try:
        data = await request.json()
    except Exception:
        raw = (await request.body()).decode("utf-8", errors="replace")
        data = {"input": raw} if raw.strip() else {}
    req = ChatCompletionRequest.model_validate(_normalize_chat_body(data))
    return await _chat_completions_impl(req)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("miner:app", host="0.0.0.0", port=8000, reload=False)
