"""Tests for the LLM client."""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from tenacity import RetryError

from src.core.llm_client import LLMClient, LLMResponse


@pytest.fixture
def mock_settings():
    with patch("src.core.llm_client.settings") as ms:
        ms.ollama_host = "http://localhost:11434"
        ms.default_model = "qwen2.5-coder:7b"
        ms.temperature = 0.7
        ms.max_tokens = 2048
        ms.embedding_model = "nomic-embed-text"
        yield ms


class TestLLMResponse:
    """Tests for the LLMResponse dataclass."""

    def test_fields(self):
        r = LLMResponse(content="Hello", model="test", tokens_used=10, latency_ms=100.0)
        assert r.content == "Hello"
        assert r.model == "test"
        assert r.tokens_used == 10
        assert r.latency_ms == 100.0
        assert isinstance(r.timestamp, datetime)

    def test_default_fields(self):
        r = LLMResponse(content="Hi", model="m")
        assert r.tokens_used == 0
        assert r.latency_ms == 0
        assert r.raw_response == {}
        assert isinstance(r.timestamp, datetime)

    def test_as_flywheel_record(self):
        r = LLMResponse(content="Hi", model="m", tokens_used=5, latency_ms=50.0)
        record = r.as_flywheel_record
        assert record["content"] == "Hi"
        assert record["model"] == "m"
        assert record["tokens_used"] == 5
        assert record["latency_ms"] == 50.0
        assert "timestamp" in record
        # timestamp should be ISO format string
        datetime.fromisoformat(record["timestamp"])

    def test_as_flywheel_record_has_all_keys(self):
        r = LLMResponse(content="x", model="y", tokens_used=1, latency_ms=2.0)
        record = r.as_flywheel_record
        expected_keys = {"content", "model", "tokens_used", "latency_ms", "timestamp"}
        assert set(record.keys()) == expected_keys

    def test_raw_response_stored(self):
        raw = {"response": "text", "model": "m", "eval_count": 10}
        r = LLMResponse(content="text", model="m", raw_response=raw)
        assert r.raw_response == raw


class TestLLMClientInit:
    """Tests for LLMClient initialization."""

    def test_default_settings(self, mock_settings):
        client = LLMClient()
        assert client.host == "http://localhost:11434"
        assert client.model == "qwen2.5-coder:7b"
        assert client.timeout == 120.0

    def test_custom_overrides(self):
        client = LLMClient(host="http://custom:1234", model="custom-model", timeout=60.0)
        assert client.host == "http://custom:1234"
        assert client.model == "custom-model"
        assert client.timeout == 60.0

    def test_partial_overrides(self, mock_settings):
        client = LLMClient(model="other-model")
        assert client.host == "http://localhost:11434"  # from settings
        assert client.model == "other-model"  # overridden
        assert client.timeout == 120.0  # default

    def test_client_property_raises_without_init(self, mock_settings):
        client = LLMClient()
        with pytest.raises(RuntimeError, match="async context manager"):
            _ = client.client

    def test_internal_client_none_at_init(self, mock_settings):
        client = LLMClient()
        assert client._client is None

    def test_context_manager(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                assert client._client is not None
                assert isinstance(client._client, httpx.AsyncClient)

        asyncio.run(_test())

    def test_context_manager_client_property_works(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                # Should not raise inside context
                c = client.client
                assert c is not None

        asyncio.run(_test())


class TestGenerate:
    """Tests for the generate method."""

    def test_basic_generate(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {
                    "response": "Generated text",
                    "model": "qwen2.5-coder:7b",
                    "eval_count": 42,
                }
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                result = await client.generate("Test prompt")
                assert result.content == "Generated text"
                assert result.model == "qwen2.5-coder:7b"
                assert result.tokens_used == 42
                assert result.latency_ms >= 0

                # Verify API call
                call_args = client._client.post.call_args
                assert call_args[0][0] == "/api/generate"
                payload = call_args[1]["json"]
                assert payload["prompt"] == "Test prompt"
                assert payload["stream"] is False

        asyncio.run(_test())

    def test_generate_with_system_prompt(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"response": "ok", "model": "m", "eval_count": 1}
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                await client.generate("prompt", system="You are helpful")
                payload = client._client.post.call_args[1]["json"]
                assert payload["system"] == "You are helpful"

        asyncio.run(_test())

    def test_generate_without_system_prompt(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"response": "ok", "model": "m", "eval_count": 1}
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                await client.generate("prompt")
                payload = client._client.post.call_args[1]["json"]
                assert "system" not in payload

        asyncio.run(_test())

    def test_generate_with_model_override(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"response": "ok", "model": "custom", "eval_count": 1}
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                await client.generate("prompt", model="custom")
                payload = client._client.post.call_args[1]["json"]
                assert payload["model"] == "custom"

        asyncio.run(_test())

    def test_generate_with_temperature_and_max_tokens(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"response": "ok", "model": "m", "eval_count": 1}
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                await client.generate("prompt", temperature=0.1, max_tokens=100)
                payload = client._client.post.call_args[1]["json"]
                assert payload["options"]["temperature"] == 0.1
                assert payload["options"]["num_predict"] == 100

        asyncio.run(_test())

    def test_generate_uses_default_options(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"response": "ok", "model": "m", "eval_count": 1}
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                await client.generate("prompt")
                payload = client._client.post.call_args[1]["json"]
                # Uses settings defaults when not overridden
                assert payload["options"]["temperature"] == mock_settings.temperature
                assert payload["options"]["num_predict"] == mock_settings.max_tokens

        asyncio.run(_test())

    def test_generate_missing_response_fields(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {}  # No fields
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                result = await client.generate("prompt")
                assert result.content == ""
                assert result.tokens_used == 0
                assert result.model == "qwen2.5-coder:7b"  # falls back to self.model

        asyncio.run(_test())

    def test_generate_stores_raw_response(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                raw_data = {
                    "response": "text",
                    "model": "qwen2.5-coder:7b",
                    "eval_count": 10,
                    "extra_field": "extra_value",
                }
                mock_resp = MagicMock()
                mock_resp.json.return_value = raw_data
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                result = await client.generate("prompt")
                assert result.raw_response == raw_data

        asyncio.run(_test())

    def test_generate_http_error_retries(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "Server Error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
                client._client.post = AsyncMock(return_value=mock_resp)

                # tenacity wraps the final exception in RetryError after 3 attempts
                with pytest.raises(RetryError):
                    await client.generate("prompt")
                assert client._client.post.call_count == 3

        asyncio.run(_test())

    def test_generate_connection_error_retries(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                client._client.post = AsyncMock(
                    side_effect=httpx.ConnectError("Connection refused")
                )

                # tenacity wraps the final exception in RetryError after 3 attempts
                with pytest.raises(RetryError):
                    await client.generate("prompt")
                assert client._client.post.call_count == 3

        asyncio.run(_test())


class TestGenerateStream:
    """Tests for the generate_stream method."""

    def test_basic_stream(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                chunks = [
                    json.dumps({"response": "Hello"}),
                    json.dumps({"response": " world"}),
                    json.dumps({"response": "!"}),
                    json.dumps({"done": True}),  # No response key
                ]

                mock_response = AsyncMock()
                mock_response.raise_for_status = MagicMock()
                mock_response.aiter_lines = _async_line_iter(chunks)

                # Mock the stream context manager
                mock_stream_cm = AsyncMock()
                mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
                mock_stream_cm.__aexit__ = AsyncMock(return_value=False)
                client._client.stream = MagicMock(return_value=mock_stream_cm)

                collected = []
                async for chunk in client.generate_stream("test prompt"):
                    collected.append(chunk)

                assert collected == ["Hello", " world", "!"]

        asyncio.run(_test())

    def test_stream_with_system_prompt(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                chunks = [json.dumps({"response": "ok"})]

                mock_response = AsyncMock()
                mock_response.raise_for_status = MagicMock()
                mock_response.aiter_lines = _async_line_iter(chunks)

                mock_stream_cm = AsyncMock()
                mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
                mock_stream_cm.__aexit__ = AsyncMock(return_value=False)
                client._client.stream = MagicMock(return_value=mock_stream_cm)

                async for _ in client.generate_stream("prompt", system="Be helpful"):
                    pass

                call_args = client._client.stream.call_args
                assert call_args[0][0] == "POST"
                assert call_args[0][1] == "/api/generate"
                payload = call_args[1]["json"]
                assert payload["system"] == "Be helpful"
                assert payload["stream"] is True

        asyncio.run(_test())

    def test_stream_skips_empty_lines(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                chunks = [
                    json.dumps({"response": "data"}),
                    "",  # empty line
                    json.dumps({"response": "more"}),
                ]

                mock_response = AsyncMock()
                mock_response.raise_for_status = MagicMock()
                mock_response.aiter_lines = _async_line_iter(chunks)

                mock_stream_cm = AsyncMock()
                mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
                mock_stream_cm.__aexit__ = AsyncMock(return_value=False)
                client._client.stream = MagicMock(return_value=mock_stream_cm)

                collected = []
                async for chunk in client.generate_stream("prompt"):
                    collected.append(chunk)

                assert collected == ["data", "more"]

        asyncio.run(_test())


def _async_line_iter(lines):
    """Create a callable that returns an async iterator over lines."""

    async def _iter():
        for line in lines:
            yield line

    return _iter


class TestChat:
    """Tests for the chat method."""

    def test_basic_chat(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {
                    "message": {"content": "Chat response"},
                    "model": "qwen2.5-coder:7b",
                    "eval_count": 30,
                }
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                messages = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "How are you?"},
                ]
                result = await client.chat(messages)
                assert result.content == "Chat response"
                assert result.model == "qwen2.5-coder:7b"
                assert result.tokens_used == 30

                # Verify API endpoint and payload
                call_args = client._client.post.call_args
                assert call_args[0][0] == "/api/chat"
                payload = call_args[1]["json"]
                assert payload["messages"] == messages
                assert payload["stream"] is False

        asyncio.run(_test())

    def test_chat_empty_message(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"message": {}, "model": "m", "eval_count": 0}
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                result = await client.chat([{"role": "user", "content": "hi"}])
                assert result.content == ""
                assert result.tokens_used == 0

        asyncio.run(_test())

    def test_chat_missing_message_key(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"model": "m", "eval_count": 5}
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                result = await client.chat([{"role": "user", "content": "hi"}])
                assert result.content == ""  # data.get("message", {}).get("content", "")

        asyncio.run(_test())

    def test_chat_with_model_override(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {
                    "message": {"content": "ok"},
                    "model": "other",
                    "eval_count": 1,
                }
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                await client.chat(
                    [{"role": "user", "content": "hi"}],
                    model="other",
                    temperature=0.2,
                )
                payload = client._client.post.call_args[1]["json"]
                assert payload["model"] == "other"
                assert payload["options"]["temperature"] == 0.2

        asyncio.run(_test())

    def test_chat_stores_raw_response(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                raw_data = {
                    "message": {"content": "response"},
                    "model": "m",
                    "eval_count": 5,
                }
                mock_resp = MagicMock()
                mock_resp.json.return_value = raw_data
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                result = await client.chat([{"role": "user", "content": "hi"}])
                assert result.raw_response == raw_data

        asyncio.run(_test())


class TestEmbeddings:
    """Tests for the embeddings method."""

    def test_get_embeddings(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                result = await client.embeddings("test text")
                assert result == [0.1, 0.2, 0.3]

                # Verify API call
                call_args = client._client.post.call_args
                assert call_args[0][0] == "/api/embed"
                payload = call_args[1]["json"]
                assert payload["input"] == "test text"
                assert payload["model"] == "nomic-embed-text"

        asyncio.run(_test())

    def test_embeddings_with_custom_model(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"embeddings": [[0.5, 0.6]]}
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                result = await client.embeddings("text", model="custom-embed")
                assert result == [0.5, 0.6]

                payload = client._client.post.call_args[1]["json"]
                assert payload["model"] == "custom-embed"

        asyncio.run(_test())

    def test_embeddings_empty_response_raises_index_error(self, mock_settings):
        """When embeddings list is empty, accessing [0] raises IndexError."""

        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"embeddings": []}
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                with pytest.raises(IndexError):
                    await client.embeddings("test")

        asyncio.run(_test())

    def test_embeddings_missing_key_returns_empty(self, mock_settings):
        """When 'embeddings' key is absent, default [[]] provides empty list at [0]."""

        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {}  # no embeddings key
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                result = await client.embeddings("test")
                assert result == []  # [[]][0] == []

        asyncio.run(_test())

    def test_embeddings_multiple_vectors_returns_first(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {
                    "embeddings": [[1.0, 2.0], [3.0, 4.0]]
                }
                mock_resp.raise_for_status = MagicMock()
                client._client.post = AsyncMock(return_value=mock_resp)

                result = await client.embeddings("test")
                assert result == [1.0, 2.0]  # returns first embedding only

        asyncio.run(_test())


class TestHealthCheck:
    """Tests for the health_check method."""

    def test_health_check_success(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                client._client.get = AsyncMock(return_value=mock_resp)

                result = await client.health_check()
                assert result is True

                # Verify endpoint
                client._client.get.assert_called_once_with("/api/tags")

        asyncio.run(_test())

    def test_health_check_failure_connection_error(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                client._client.get = AsyncMock(
                    side_effect=httpx.ConnectError("Connection refused")
                )

                result = await client.health_check()
                assert result is False

        asyncio.run(_test())

    def test_health_check_non_200(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.status_code = 503
                client._client.get = AsyncMock(return_value=mock_resp)

                result = await client.health_check()
                assert result is False

        asyncio.run(_test())

    def test_health_check_timeout(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                client._client.get = AsyncMock(
                    side_effect=httpx.ReadTimeout("Timeout")
                )

                result = await client.health_check()
                assert result is False

        asyncio.run(_test())

    def test_health_check_connection_error(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                client._client.get = AsyncMock(
                    side_effect=ConnectionError("Connection refused")
                )

                result = await client.health_check()
                assert result is False

        asyncio.run(_test())


class TestListModels:
    """Tests for the list_models method."""

    def test_list_models(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {
                    "models": [
                        {"name": "qwen2.5-coder:7b"},
                        {"name": "qwen2.5-coder:7b"},
                        {"name": "nomic-embed-text"},
                    ]
                }
                mock_resp.raise_for_status = MagicMock()
                client._client.get = AsyncMock(return_value=mock_resp)

                models = await client.list_models()
                assert models == ["qwen2.5-coder:7b", "qwen2.5-coder:7b", "nomic-embed-text"]

                # Verify endpoint
                client._client.get.assert_called_once_with("/api/tags")

        asyncio.run(_test())

    def test_list_models_empty(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"models": []}
                mock_resp.raise_for_status = MagicMock()
                client._client.get = AsyncMock(return_value=mock_resp)

                models = await client.list_models()
                assert models == []

        asyncio.run(_test())

    def test_list_models_missing_models_key(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {}  # no "models" key
                mock_resp.raise_for_status = MagicMock()
                client._client.get = AsyncMock(return_value=mock_resp)

                models = await client.list_models()
                assert models == []

        asyncio.run(_test())

    def test_list_models_http_error(self, mock_settings):
        async def _test():
            async with LLMClient() as client:
                mock_resp = MagicMock()
                mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "Error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
                client._client.get = AsyncMock(return_value=mock_resp)

                with pytest.raises(httpx.HTTPStatusError):
                    await client.list_models()

        asyncio.run(_test())
