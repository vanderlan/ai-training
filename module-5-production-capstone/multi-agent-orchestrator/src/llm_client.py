"""LLM Client — DeepSeek integration via OpenAI-compatible API."""
import logging
import os
from typing import Dict, List

logger = logging.getLogger("multi_agent.llm")


class LLMClient:
    """Base LLM client interface."""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class DeepSeekClient(LLMClient):
    """DeepSeek client (OpenAI-compatible API)."""

    def __init__(self, model: str = None):
        from openai import OpenAI
        api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        self.client = OpenAI(
            api_key=api_key or None,
            base_url="https://api.deepseek.com/v1",
        )
        self.model = (model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")).strip()

    def chat(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content


class MockClient(LLMClient):
    """Fallback client for local testing or missing credentials."""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        return (
            "FINAL: This is a mock response — no API key configured. "
            "Set DEEPSEEK_API_KEY to enable real LLM calls."
        )


def get_llm_client(provider: str = "deepseek") -> LLMClient:
    """Instantiate the DeepSeek LLM client, falling back to MockClient."""
    provider = (provider or "deepseek").strip().lower()
    if provider == "mock":
        return MockClient()
    if provider != "deepseek":
        logger.warning("Unknown provider '%s', using DeepSeek.", provider)
    try:
        return DeepSeekClient()
    except Exception as exc:
        logger.warning("Failed to init DeepSeek: %s. Falling back to mock.", exc)
        return MockClient()
