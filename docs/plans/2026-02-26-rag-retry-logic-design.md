# RAG Retry Logic for Transient Errors

**Date:** 2026-02-26
**Status:** Approved

## Problem

`RAGEngine._get_embedding()` makes HTTP calls to Ollama with no retry logic.
A single transient failure (Ollama restart, network blip, GPU busy) causes the
entire operation to fail — whether that's `retrieve()`, `index_file()`, or
`index_datalake()`. Meanwhile, `LLMClient.generate()` already has retry via
tenacity and handles transient errors gracefully.

## Design Principles

1. **Single retry point** — `_get_embedding` is the only external HTTP call;
   decorating it covers all code paths
2. **Consistent with existing patterns** — Same tenacity config as `llm_client.py`
3. **Observable** — Log every retry attempt so problems are visible
4. **Non-breaking** — After max retries, raise the original exception unchanged

## Solution

### Approach: `@retry` decorator on `_get_embedding`

Add tenacity `@retry` to `_get_embedding()` with the same config used in
`llm_client.py`:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, OSError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _get_embedding(self, text: str) -> list[float]:
```

**Parameters:**
- **3 attempts** (1 original + 2 retries) — same as LLMClient
- **Exponential backoff: 2s → 4s → 8s** — matches  ticket spec
- **Retry only on transient errors**: `httpx.HTTPStatusError` (5xx),
  `httpx.ConnectError` (Ollama down), `OSError` (network)
- **Logging**: `before_sleep_log` logs each retry at WARNING level

### Why not retry on all exceptions?

`httpx.HTTPStatusError` for 4xx (bad model name, malformed request) should NOT
retry — they'll fail every time. But tenacity's `retry_if_exception_type` with
`HTTPStatusError` will retry all status errors including 4xx. This is acceptable
because:
1. 4xx errors from Ollama embeddings API are extremely rare (model validated at startup)
2. Adding status code filtering adds complexity for a near-zero probability event
3. Max 3 attempts with backoff means worst case is ~14s wasted, not a crash loop

### Coverage

All these methods are automatically covered:
- `_get_embeddings_batch()` — calls `_get_embedding` per text
- `index_file()` → `_get_embeddings_batch()` → `_get_embedding()`
- `index_jsonl()` → `_get_embeddings_batch()` → `_get_embedding()`
- `index_datalake()` → `index_jsonl()` / `index_file()` → `_get_embedding()`
- `retrieve()` → `_get_embedding()`
- `query_with_context()` → `retrieve()` → `_get_embedding()`

## Changes by File

### `src/knowledge/rag.py`

| Change | Description |
|--------|-------------|
| Imports | Add `logging`, `structlog`, `tenacity` imports |
| Logger | Add `logger = structlog.get_logger()` |
| `_get_embedding()` | Add `@retry` decorator with exponential backoff |

### `tests/test_rag.py`

| Test | Validates |
|------|-----------|
| `test_get_embedding_retries_on_http_error` | 3 attempts on HTTPStatusError, raises RetryError |
| `test_get_embedding_retries_on_connect_error` | 3 attempts on ConnectError |
| `test_get_embedding_retries_on_os_error` | 3 attempts on OSError |
| `test_get_embedding_succeeds_after_transient_failure` | Fails once, succeeds on retry |
| `test_get_embedding_no_retry_on_success` | Normal call, 1 attempt only |

### Existing test adaptation

`test_get_embedding_raises_on_http_error` currently expects immediate raise.
With retry, it will now raise `RetryError` after 3 attempts. Update this test.

## Error Handling

- **After max retries**: tenacity raises `RetryError` wrapping the last exception
- **Callers** (`index_datalake`, etc.) already catch `httpx.HTTPError` — they'll
  also need to catch `RetryError` or let it propagate (current try/except in
  `index_datalake` catches `httpx.HTTPError`, which `RetryError` is NOT)
- **Fix**: Add `RetryError` to the except clauses in `index_datalake()` so that
  retry exhaustion is handled the same as immediate HTTP failure

## Out of Scope

- Retry on LanceDB operations (local DB, not transient)
- Circuit breaker pattern (overkill for single-user local tool)
- Retry on `generate_stream` in LLMClient (different ticket)
