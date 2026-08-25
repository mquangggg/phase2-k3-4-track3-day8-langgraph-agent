"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

import os
from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr


def get_llm(model: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    """Create an LLM client from environment configuration."""
    if os.getenv("DEEPSEEK_API_KEY"):
        try:
            from langchain_openai import (  # type: ignore[import-not-found]
                ChatOpenAI,
            )
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc

        deepseek_model = (
            model
            or os.getenv("DEEPSEEK_MODEL")
            or os.getenv("LLM_MODEL", "deepseek-v4-flash")
        )
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        api_key_str = os.getenv("DEEPSEEK_API_KEY", "")
        return ChatOpenAI(
            model=str(deepseek_model),
            api_key=SecretStr(api_key_str),
            base_url=base_url,
            temperature=temperature,
        )

    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import (  # type: ignore[import-not-found]
                ChatGoogleGenerativeAI,
            )
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        return cast(
            BaseChatModel,
            ChatGoogleGenerativeAI(
                model=model or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=temperature,
            ),
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import (  # type: ignore[import-not-found]
                ChatOpenAI,
            )
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=str(model or os.getenv("LLM_MODEL", "gpt-4o-mini")),
            api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
            temperature=temperature,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import (  # type: ignore[import-not-found]
                ChatAnthropic,
            )
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        return cast(
            BaseChatModel,
            ChatAnthropic(
                model=model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
                temperature=temperature,
            ),
        )


    raise RuntimeError(
        "No LLM API key found. Set DEEPSEEK_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, "
        "or ANTHROPIC_API_KEY in .env\nSee .env.example for configuration."
    )



