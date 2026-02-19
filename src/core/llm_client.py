"""LLM Client for Ollama/Qwen integration."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = structlog.get_logger()


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: dict = field(default_factory=dict)

    @property
    def as_flywheel_record(self) -> dict:
        """Convert to flywheel-compatible record."""
        return {
            "content": self.content,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class LLMClient:
    """Async client for Ollama LLM."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ):
        self.host = host or settings.ollama_host
        self.model = model or settings.default_model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "LLMClient":
        self._client = httpx.AsyncClient(
            base_url=self.host,
            timeout=httpx.Timeout(self.timeout),
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("LLMClient must be used as async context manager")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        start_time = asyncio.get_event_loop().time()

        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or settings.temperature,
                "num_predict": max_tokens or settings.max_tokens,
            },
        }

        if system:
            payload["system"] = system

        logger.debug("llm_request", model=payload["model"], prompt_length=len(prompt))

        response = await self.client.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()

        latency = (asyncio.get_event_loop().time() - start_time) * 1000

        result = LLMResponse(
            content=data.get("response", ""),
            model=data.get("model", self.model),
            tokens_used=data.get("eval_count", 0),
            latency_ms=latency,
            raw_response=data,
        )

        logger.info(
            "llm_response",
            model=result.model,
            tokens=result.tokens_used,
            latency_ms=round(result.latency_ms, 2),
        )

        return result

    async def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream response from the LLM."""
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": settings.temperature,
                "num_predict": settings.max_tokens,
            },
        }

        if system:
            payload["system"] = system

        async with self.client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    import json

                    data = json.loads(line)
                    if chunk := data.get("response"):
                        yield chunk

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Chat completion with message history."""
        start_time = asyncio.get_event_loop().time()

        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or settings.temperature,
                "num_predict": settings.max_tokens,
            },
        }

        response = await self.client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        latency = (asyncio.get_event_loop().time() - start_time) * 1000

        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=data.get("model", self.model),
            tokens_used=data.get("eval_count", 0),
            latency_ms=latency,
            raw_response=data,
        )

    async def embeddings(self, text: str, model: str | None = None) -> list[float]:
        """Generate embeddings for text."""
        payload = {
            "model": model or settings.embedding_model,
            "input": text,
        }

        response = await self.client.post("/api/embed", json=payload)
        response.raise_for_status()
        data = response.json()

        return data.get("embeddings", [[]])[0]

    async def health_check(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = await self.client.get("/api/tags")
            return response.status_code == 200
        except (httpx.HTTPError, OSError, ConnectionError):
            return False

    async def list_models(self) -> list[str]:
        """List available models."""
        response = await self.client.get("/api/tags")
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]
