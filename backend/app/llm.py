"""
LangChain wiring for the Document Analyzer.

- Chat / answer generation: an ordered chain of providers (Groq, OpenRouter,
  Claude by default). Each request walks the chain and uses the first provider
  that answers; if one is rate limited or out of credit, the next one takes
  over. Configure the order with LLM_PROVIDERS.
- Embeddings: a small local sentence-transformers model, via
  langchain-huggingface. None of the chat providers is used for retrieval, so
  indexing keeps working even when every chat provider is exhausted.

Groq and OpenRouter both expose OpenAI-compatible APIs, so both are driven
through ChatOpenAI with a provider-specific base_url - no extra client library
per provider.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from .config import settings

log = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = (
    "You are a document analysis assistant. Answer the user's question "
    "using ONLY the context excerpts provided below. If the answer is not "
    "contained in the context, say you don't have enough information in "
    "the ingested documents. Keep answers concise and cite the source file "
    "name when relevant."
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

KNOWN_PROVIDERS = ("groq", "openrouter", "anthropic")


@dataclass(frozen=True)
class Answer:
    text: str
    provider: str
    model: str


class NoProviderConfigured(RuntimeError):
    """No provider in the chain has an API key set."""


class AllProvidersFailed(RuntimeError):
    """Every configured provider refused or errored."""

    def __init__(self, failures: list[tuple[str, str]]):
        self.failures = failures
        detail = "; ".join(f"{name}: {reason}" for name, reason in failures)
        super().__init__(f"All answer providers failed ({detail})")

    @property
    def all_rate_limited(self) -> bool:
        return bool(self.failures) and all(
            "rate limit" in reason.lower() or "quota" in reason.lower()
            or "credit" in reason.lower() or "429" in reason
            for _, reason in self.failures
        )


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    """Cached so the model is loaded into memory once, not per-request."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def provider_chain() -> list[str]:
    """The configured order, filtered to providers that actually have a key."""
    requested = [p.strip().lower() for p in settings.llm_providers.split(",") if p.strip()]

    chain: list[str] = []
    for name in requested:
        if name not in KNOWN_PROVIDERS:
            log.warning("[llm] ignoring unknown provider %r in LLM_PROVIDERS", name)
            continue
        if name in chain:
            continue
        if not _api_key(name):
            log.info("[llm] skipping %s - no API key configured", name)
            continue
        chain.append(name)
    return chain


def _api_key(name: str) -> str:
    return {
        "groq": settings.groq_api_key,
        "openrouter": settings.openrouter_api_key,
        "anthropic": settings.anthropic_api_key,
    }.get(name, "")


def model_name(name: str) -> str:
    return {
        "groq": settings.groq_model,
        "openrouter": settings.openrouter_model,
        "anthropic": settings.chat_model,
    }[name]


@lru_cache
def get_chat_model(provider: str) -> BaseChatModel:
    """One cached client per provider.

    max_retries=0 on purpose: the SDK's own backoff would sit on a 429 for
    seconds before giving up, when the whole point here is to hand straight
    over to the next provider in the chain.
    """
    common = dict(
        temperature=settings.llm_temperature,
        max_retries=0,
        timeout=settings.llm_timeout_seconds,
    )

    if provider == "anthropic":
        return ChatAnthropic(
            model=settings.chat_model,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.llm_max_tokens,
            **common,
        )

    if provider == "groq":
        return ChatOpenAI(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            base_url=GROQ_BASE_URL,
            max_tokens=settings.llm_max_tokens,
            **common,
        )

    if provider == "openrouter":
        return ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            max_tokens=settings.llm_max_tokens,
            **common,
        )

    raise ValueError(f"Unknown provider: {provider}")


def describe_failure(exc: Exception) -> str:
    """Short, human-readable reason - and the string the UI ends up showing."""
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    text = str(exc)

    if status == 429 or "rate_limit" in text or "rate limit" in text.lower():
        return "rate limit reached"
    if status == 402 or "credit balance" in text.lower() or "insufficient" in text.lower():
        return "out of credit/quota"
    if status == 401 or status == 403:
        return "API key rejected"
    if status == 404:
        return f"model not available ({text[:80]})"
    if status and 500 <= status < 600:
        return f"provider error {status}"
    return text[:160] or exc.__class__.__name__


async def embed(text: str) -> list[float]:
    return get_embeddings().embed_query(text)


async def embed_many(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)


def _system_content(context: str, cache: bool) -> str | list[dict]:
    """Plain string for the OpenAI-compatible providers (Groq, OpenRouter),
    which don't understand Anthropic's cache_control field.

    For Anthropic, when `cache` is set, the context is split into its own
    content block with a cache_control breakpoint - Anthropic caches
    everything up to and including that block for a few minutes, so a
    multi-turn conversation that keeps resending the same large context
    (whole-document mode, see main.py) only pays full price on the first
    turn. Not worth doing for normal search-mode context, since retrieved
    chunks usually differ question to question and would just pay a cache
    write for a prefix that's never read again.
    """
    if not cache:
        return f"{SYSTEM_INSTRUCTIONS}\n\nCONTEXT:\n{context}"
    return [
        {"type": "text", "text": SYSTEM_INSTRUCTIONS},
        {
            "type": "text",
            "text": f"CONTEXT:\n{context}",
            "cache_control": {"type": "ephemeral"},
        },
    ]


async def chat(
    question: str,
    context_chunks: list[str],
    history: list[dict] | None = None,
    cacheable: bool = False,
) -> Answer:
    """Answer `question` from `context_chunks`, failing over across providers.

    `history` is the earlier turns of this conversation, oldest first, as
    {"role": "user"|"assistant", "text": str} - replayed so follow-up questions
    like "what about the second one?" resolve against what was already said.

    `cacheable` marks the context as worth an Anthropic prompt-cache
    breakpoint (see _system_content) - set it when the same large context is
    likely to be resent on the next turn, e.g. whole-document mode.

    Raises NoProviderConfigured if nothing has a key, or AllProvidersFailed if
    every provider in the chain was tried and none answered.
    """
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no matching context found)"

    def build_messages(cache: bool) -> list:
        messages = [SystemMessage(content=_system_content(context, cache))]
        for turn in history or []:
            text = turn.get("text") or ""
            if not text:
                continue
            if turn.get("role") == "assistant":
                messages.append(AIMessage(content=text))
            else:
                messages.append(HumanMessage(content=text))
        messages.append(HumanMessage(content=question))
        return messages

    plain_messages = build_messages(cache=False)
    cached_messages = build_messages(cache=True) if cacheable else plain_messages

    chain = provider_chain()
    if not chain:
        raise NoProviderConfigured(
            "No answer provider is configured. Set GROQ_API_KEY, "
            "OPENROUTER_API_KEY, or ANTHROPIC_API_KEY in backend/.env"
        )

    failures: list[tuple[str, str]] = []
    for provider in chain:
        messages = cached_messages if provider == "anthropic" else plain_messages
        try:
            response = await get_chat_model(provider).ainvoke(messages)
        except Exception as exc:  # noqa: BLE001 - any failure means try the next one
            reason = describe_failure(exc)
            failures.append((provider, reason))
            log.warning(
                "[llm] %s (%s) failed: %s - falling back to next provider",
                provider,
                model_name(provider),
                reason,
            )
            continue

        text = response.content if isinstance(response.content, str) else str(response.content)
        if not text.strip():
            failures.append((provider, "returned an empty answer"))
            log.warning("[llm] %s returned an empty answer - trying next provider", provider)
            continue

        if failures:
            log.info(
                "[llm] answered by %s after %d failed provider(s)", provider, len(failures)
            )
        return Answer(text=text, provider=provider, model=model_name(provider))

    raise AllProvidersFailed(failures)
