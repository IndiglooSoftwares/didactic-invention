"""Configuration for the Agentic Mentor."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """Settings for the language model provider."""

    # Provider: "openai" | "anthropic" | "ollama" | "custom"
    provider: str = "openai"

    # Model identifier (provider-specific)
    model: str = "gpt-4o"

    # API key — reads from environment variable if not set directly
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_API_KEY")
    )

    # Base URL override (useful for proxies or local models like Ollama)
    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL")
    )

    # Sampling parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0

    # Streaming
    stream: bool = False


@dataclass
class MentorConfig:
    """Top-level configuration for the Agentic Mentor."""

    llm: LLMConfig = field(default_factory=LLMConfig)

    # Conversation memory window
    max_history: int = 20

    # Logging
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    # Persistence: path to a JSON file for saving/loading session state.
    # Set to None to disable persistence.
    session_file: Optional[str] = field(
        default_factory=lambda: os.getenv("SESSION_FILE")
    )

    @classmethod
    def from_env(cls) -> "MentorConfig":
        """Build a :class:`MentorConfig` entirely from environment variables.

        Environment variables:
            LLM_PROVIDER   — LLM provider (default: openai)
            LLM_MODEL      — Model name (default: gpt-4o)
            LLM_API_KEY    — API key
            LLM_BASE_URL   — Optional base URL override
            LLM_TEMP       — Sampling temperature (default: 0.7)
            LLM_MAX_TOKENS — Max tokens (default: 2048)
            MAX_HISTORY    — Conversation window size (default: 20)
            LOG_LEVEL      — Logging level (default: INFO)
            SESSION_FILE   — Path for session persistence (optional)
        """
        llm = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=float(os.getenv("LLM_TEMP", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            stream=os.getenv("LLM_STREAM", "false").lower() == "true",
        )
        return cls(
            llm=llm,
            max_history=int(os.getenv("MAX_HISTORY", "20")),
            session_file=os.getenv("SESSION_FILE"),
        )
