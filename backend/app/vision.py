"""
Image-to-text via a vision-capable LLM, tried before falling back to local
OCR (see ocr.py).

Providers are tried in IMAGE_PROVIDERS order (Claude, then Groq's vision
model, by default) the same way llm.py fails over across chat providers -
each is asked to transcribe the image, and a rate limit, missing key, or
empty response just moves on to the next one. If every vision provider is
unavailable (no keys configured, all failed, or the file isn't a format any
of them accept inline), this falls back to ocr.extract_text() so image
uploads keep working with zero API keys set, exactly as before this existed.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from . import cooldown, ocr
from .config import settings
from .llm import GROQ_BASE_URL, _api_key, describe_failure

log = logging.getLogger(__name__)

KNOWN_IMAGE_PROVIDERS = ("anthropic", "groq")

TRANSCRIBE_PROMPT = (
    "Transcribe every piece of visible text in this image exactly as it "
    "appears, preserving reading order. Output only the transcribed text - "
    "no commentary, no description of the image itself. If the image has no "
    "legible text, output nothing."
)

# Inline image formats every vision provider here reliably accepts. bmp/tif
# aren't universally supported as inline vision input, so those two skip
# straight to local OCR rather than risk a confusing provider-specific error.
_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

_MAX_TOKENS = 2048


def _provider_chain() -> list[str]:
    requested = [p.strip().lower() for p in settings.image_providers.split(",") if p.strip()]
    chain: list[str] = []
    for name in requested:
        if name not in KNOWN_IMAGE_PROVIDERS:
            log.warning("[vision] ignoring unknown provider %r in IMAGE_PROVIDERS", name)
            continue
        if name in chain:
            continue
        if not _api_key(name):
            continue
        chain.append(name)
    return chain


def _get_model(provider: str):
    common = dict(max_tokens=_MAX_TOKENS, temperature=0, max_retries=0, timeout=settings.llm_timeout_seconds)
    if provider == "anthropic":
        # Every current Claude model is multimodal - reuse the chat model
        # rather than adding a separate vision-specific setting.
        return ChatAnthropic(model=settings.chat_model, api_key=settings.anthropic_api_key, **common)
    if provider == "groq":
        return ChatOpenAI(
            model=settings.groq_vision_model,
            api_key=settings.groq_api_key,
            base_url=GROQ_BASE_URL,
            **common,
        )
    raise ValueError(f"Unknown image provider: {provider}")


def extract_text(path: Path) -> str:
    """Transcribe an image's visible text, preferring a vision LLM over
    local OCR when one is configured and available.
    """
    mime = _MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        return ocr.extract_text(path)

    chain = _provider_chain()
    if not chain:
        return ocr.extract_text(path)
    chain = cooldown.filter_available(chain)

    data = base64.b64encode(path.read_bytes()).decode("ascii")
    message = HumanMessage(
        content=[
            {"type": "text", "text": TRANSCRIBE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
        ]
    )

    for provider in chain:
        try:
            response = _get_model(provider).invoke([message])
        except Exception as exc:  # noqa: BLE001 - any failure means try the next one
            reason = describe_failure(exc)
            cooldown.mark_failed(provider, reason)
            log.warning(
                "[vision] %s failed on %s: %s - falling back to next provider",
                provider,
                path.name,
                reason,
            )
            continue

        text = response.content if isinstance(response.content, str) else str(response.content)
        if text.strip():
            return text.strip()
        log.warning("[vision] %s returned no text for %s - trying next provider", provider, path.name)

    log.info("[vision] no vision provider produced text for %s - falling back to local OCR", path.name)
    return ocr.extract_text(path)
