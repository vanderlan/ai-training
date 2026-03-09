"""LLM Client abstraction."""
from typing import List, Dict
import os


class LLMClient:
    """Base LLM client interface."""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send chat messages and get response."""
        raise NotImplementedError


class AnthropicClient(LLMClient):
    """Anthropic Claude client."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send chat messages to Claude."""
        # Separate system messages from user/assistant messages
        system_msg = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs = {"model": self.model, "max_tokens": 4096, "messages": chat_messages}
        if system_msg:
            kwargs["system"] = system_msg

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class OpenAIClient(LLMClient):
    """OpenAI GPT client."""

    def __init__(self, model: str = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send chat messages to GPT."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content


class DeepSeekClient(LLMClient):
    """DeepSeek client (OpenAI-compatible API)."""

    def __init__(self, model: str = None):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
        )
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send chat messages to DeepSeek."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content


class GeminiClient(LLMClient):
    """Google Gemini client."""

    def __init__(self, model: str = "gemini-pro"):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send chat messages to Gemini."""
        # Convert to Gemini format
        prompt_parts = []
        for msg in messages:
            role = "user" if msg["role"] in ("user", "system") else "model"
            prompt_parts.append(f"{role}: {msg['content']}")

        prompt = "\n\n".join(prompt_parts)
        response = self.model.generate_content(prompt)
        return response.text


class MockClient(LLMClient):
    """Local test client that returns deterministic responses."""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Return predictable outputs to support local testing."""
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_content = msg.get("content", "")
                break

        if "Return only a single number from 1 to 5" in last_content:
            return "4"

        return "[MOCK RESPONSE] Local testing mode is enabled."


def get_llm_client(provider: str = "anthropic") -> LLMClient:
    """Get LLM client based on provider."""
    providers = {
        "anthropic": AnthropicClient,
        "openai": OpenAIClient,
        "deepseek": DeepSeekClient,
        "gemini": GeminiClient,
        "mock": MockClient,
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(providers.keys())}")

    try:
        return providers[provider]()
    except Exception as exc:
        # Keep local development unblocked if credentials/dependencies are missing.
        print(f"Failed to initialize provider '{provider}', using mock client instead: {exc}")
        return MockClient()
