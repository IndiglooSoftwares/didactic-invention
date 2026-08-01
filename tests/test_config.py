"""Tests for MentorConfig."""
import os

import pytest

from src.config import LLMConfig, MentorConfig


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048
        assert cfg.stream is False

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key-123")
        cfg = LLMConfig()
        assert cfg.api_key == "test-key-123"


class TestMentorConfig:
    def test_defaults(self):
        cfg = MentorConfig()
        assert cfg.max_history == 20
        assert cfg.llm.provider == "openai"

    def test_from_env_reads_all_variables(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_MODEL", "claude-3-opus")
        monkeypatch.setenv("LLM_API_KEY", "sk-ant-test")
        monkeypatch.setenv("LLM_TEMP", "0.5")
        monkeypatch.setenv("LLM_MAX_TOKENS", "1024")
        monkeypatch.setenv("LLM_STREAM", "true")
        monkeypatch.setenv("MAX_HISTORY", "30")
        monkeypatch.setenv("SESSION_FILE", "/tmp/session.json")

        cfg = MentorConfig.from_env()

        assert cfg.llm.provider == "anthropic"
        assert cfg.llm.model == "claude-3-opus"
        assert cfg.llm.api_key == "sk-ant-test"
        assert cfg.llm.temperature == 0.5
        assert cfg.llm.max_tokens == 1024
        assert cfg.llm.stream is True
        assert cfg.max_history == 30
        assert cfg.session_file == "/tmp/session.json"

    def test_from_env_uses_defaults_when_vars_absent(self, monkeypatch):
        # Remove all relevant env vars so defaults kick in
        for key in [
            "LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL",
            "LLM_TEMP", "LLM_MAX_TOKENS", "LLM_STREAM", "MAX_HISTORY",
            "SESSION_FILE",
        ]:
            monkeypatch.delenv(key, raising=False)

        cfg = MentorConfig.from_env()

        assert cfg.llm.provider == "openai"
        assert cfg.llm.model == "gpt-4o"
        assert cfg.llm.temperature == 0.7
        assert cfg.max_history == 20
        assert cfg.session_file is None
