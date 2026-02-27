# RAG Retry Logic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add retry with exponential backoff to `RAGEngine._get_embedding()` so transient Ollama failures don't crash indexing or retrieval.

**Architecture:** Single `@retry` decorator on `_get_embedding` covers all code paths. Same tenacity pattern as `LLMClient.generate()`. Logging via structlog.

**Tech Stack:** tenacity (already installed), structlog (already used), httpx

---

### Task 1: Add retry decorator to `_get_embedding`

**Files:**
- Modify: `src/knowledge/rag.py:1-28` (imports)
- Modify: `src/knowledge/rag.py:91-101` (`_get_embedding` method)

**Step 1: Add imports**

Add after `import httpx` (line 23):

```python
import logging

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
```

Add after `EMBEDDING_DIM` (line 30):

```python
logger = structlog.get_logger()
```

**Step 2: Add `@retry` decorator to `_get_embedding`**

Replace the method signature at line 91:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, OSError)),
    before_sleep_log(logger, logging.WARNING),
)
async def _get_embedding(self, text: str) -> list[float]:
```

**Step 3: Run existing tests**

Run: `python -m pytest tests/test_rag.py -v`
Expected: Most pass. `test_get_embedding_raises_on_http_error` may need update.

**Step 4: Commit**

```bash
git add src/knowledge/rag.py
git commit -m "FEAT: Add retry with exponential backoff to RAG embeddings"
```

---

### Task 2: Add RetryError handling in `index_datalake`

**Files:**
- Modify: `src/knowledge/rag.py:227-283` (`index_datalake` method)

**Step 1: Import RetryError**

Add to existing tenacity import:

```python
from tenacity import RetryError
```

**Step 2: Add RetryError to except clauses**

In `index_datalake`, each try/except block catches specific exceptions. Add
`RetryError` to each:

```python
except (OSError, httpx.HTTPError, json.JSONDecodeError, RetryError) as e:
    stats["errors"] += 1
```

And for the non-JSONL blocks:

```python
except (OSError, httpx.HTTPError, UnicodeDecodeError, RetryError):
    stats["errors"] += 1
```

**Step 3: Run tests**

Run: `python -m pytest tests/test_rag.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/knowledge/rag.py
git commit -m "FIX: Handle RetryError in index_datalake after retry exhaustion"
```

---

### Task 3: Write and adapt retry tests

**Files:**
- Modify: `tests/test_rag.py` (TestEmbeddings class)

**Step 1: Add imports**

```python
import httpx
from tenacity import RetryError
```

**Step 2: Adapt existing test**

`test_get_embedding_raises_on_http_error` should now expect `RetryError`
after 3 attempts:

```python
def test_get_embedding_raises_on_http_error(self):
    def _test():
        async def _run():
            engine = _make_engine()
            engine._http_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
            engine._http_client.post = AsyncMock(return_value=mock_response)

            with pytest.raises(RetryError):
                await engine._get_embedding("test text")
            assert engine._http_client.post.call_count == 3
        asyncio.run(_run())
    _test()
```

**Step 3: Add new retry tests**

```python
def test_get_embedding_retries_on_connect_error(self):
    def _test():
        async def _run():
            engine = _make_engine()
            engine._http_client = AsyncMock()
            engine._http_client.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            with pytest.raises(RetryError):
                await engine._get_embedding("test text")
            assert engine._http_client.post.call_count == 3
        asyncio.run(_run())
    _test()

def test_get_embedding_retries_on_os_error(self):
    def _test():
        async def _run():
            engine = _make_engine()
            engine._http_client = AsyncMock()
            engine._http_client.post = AsyncMock(
                side_effect=OSError("Network unreachable")
            )

            with pytest.raises(RetryError):
                await engine._get_embedding("test text")
            assert engine._http_client.post.call_count == 3
        asyncio.run(_run())
    _test()

def test_get_embedding_succeeds_after_transient_failure(self):
    def _test():
        async def _run():
            engine = _make_engine()
            engine._http_client = AsyncMock()
            mock_success = MagicMock()
            mock_success.json.return_value = {"embedding": _fake_embedding()}
            mock_success.raise_for_status = MagicMock()

            engine._http_client.post = AsyncMock(
                side_effect=[
                    httpx.ConnectError("Connection refused"),
                    mock_success,
                ]
            )

            result = await engine._get_embedding("test text")
            assert len(result) == EMBEDDING_DIM
            assert engine._http_client.post.call_count == 2
        asyncio.run(_run())
    _test()

def test_get_embedding_no_retry_on_success(self):
    def _test():
        async def _run():
            engine = _make_engine()
            _setup_mock_http(engine)

            result = await engine._get_embedding("test text")
            assert len(result) == EMBEDDING_DIM
            assert engine._http_client.post.call_count == 1
        asyncio.run(_run())
    _test()
```

**Step 4: Run all tests**

Run: `python -m pytest tests/test_rag.py -v`
Expected: ALL PASS

**Step 5: Run full suite**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS (793+)

**Step 6: Commit**

```bash
git add tests/test_rag.py
git commit -m "TEST: Add retry behavior tests for RAG embeddings"
```
