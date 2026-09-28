"""Chat model construction.

DeepSeek speaks the OpenAI wire protocol, so ``ChatOpenAI`` pointed at
``https://api.deepseek.com`` is the whole integration. Verified against the
provider's current docs: the base URL carries no ``/v1`` suffix, and the
current model ids are ``deepseek-flash`` and ``deepseek-v4-pro``.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from .config import Settings


def build_chat_model(settings: Settings, **overrides: Any) -> ChatOpenAI:
    """Construct the model client.

    ``stream_usage=True`` matters: the graph is consumed through
    ``stream_mode="messages"``, and without it the streaming path omits token
    usage, which would leave every trajectory without cost data.
    """
    params: dict[str, Any] = {
        "model": settings.model,
        "base_url": settings.base_url,
        "api_key": settings.api_key,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "reasoning_effort": settings.reasoning_effort,
        "request_timeout": settings.request_timeout,
        "max_retries": settings.max_retries,
        "stream_usage": True,
    }
    params.update(overrides)
    return ChatOpenAI(**params)


def describe_target(settings: Settings) -> str:
    """One-line, secret-free description of where calls are going."""
    host = settings.base_url.split("://", 1)[-1].split("/", 1)[0]
    return f"{settings.model} @ {host} (key {settings.api_key_fingerprint})"
