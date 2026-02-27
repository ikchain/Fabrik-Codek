# Session Observer Watch Mode

**Date:** 2026-02-26
**Status:** Approved

## Problem

`fabrik learn process` must be invoked manually to extract training pairs from
Claude Code session transcripts. Sessions accumulate continuously as users work,
but there is no automatic processing. The session observer already handles
extraction, quality filtering, and idempotent tracking — only the continuous
monitoring layer is missing.

## Design Principles

1. **Polling over watchfiles** — Session transcripts are appended to continuously
   during active sessions. File-system watchers (inotify) would fire on every
   line write, causing excessive reprocessing. Polling with configurable interval
   avoids processing files still being written to.
2. **Reuse existing infrastructure** — `process_all_sessions()` already handles
   session discovery, pair extraction, quality filtering, and marker tracking.
   Watch mode is a loop around this existing function.
3. **Graceful shutdown** — Ctrl+C (SIGINT) must stop the loop cleanly without
   corrupting the processed marker file or leaving partial output.
4. **Minimal footprint** — No background daemon, no systemd service. Just a
   long-running CLI command that can be run in a terminal or tmux.

## Solution: Polling Watch Loop

### Architecture

```
fabrik learn watch [--interval 60]
    │
    └─ while not shutdown:
         ├─ process_all_sessions()    ← existing function, idempotent
         ├─ log stats (new sessions, new pairs)
         └─ await asyncio.sleep(interval)
```

### Implementation

New async function in `session_observer.py`:

```python
async def watch_sessions(
    interval_seconds: int = 60,
    min_quality: float = 0.4,
) -> None:
    """Continuously poll for new sessions and process them."""
    logger.info("watch_started", interval=interval_seconds)

    while True:
        stats = process_all_sessions(min_quality=min_quality)
        if stats["sessions_processed"] > 0:
            logger.info("watch_cycle_processed",
                        sessions=stats["sessions_processed"],
                        pairs=stats["pairs_extracted"])
        await asyncio.sleep(interval_seconds)
```

Signal handling is done at the CLI layer, not in the observer.

### CLI Integration

In `cli.py`, the `learn()` command adds a `watch` action:

```python
elif action == "watch":
    import signal

    async def run_watch():
        stop = asyncio.Event()

        def handle_signal():
            stop.set()

        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, handle_signal)
        loop.add_signal_handler(signal.SIGTERM, handle_signal)

        while not stop.is_set():
            stats = process_all_sessions(min_quality=0.4)
            if stats["sessions_processed"] > 0:
                console.print(f"[green]Processed {stats['sessions_processed']} sessions, "
                              f"{stats['pairs_extracted']} pairs extracted[/green]")
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass  # Normal: timeout means keep polling

    console.print(f"[bold]Watching sessions every {interval}s... (Ctrl+C to stop)[/bold]")
    asyncio.run(run_watch())
```

### CLI Flags

- `--interval INT` (default 60): Seconds between polling cycles
- No `--min-quality` flag (uses default 0.4, consistent with `process`)

## Changes by File

### `src/flywheel/session_observer.py`

| Change | Description |
|--------|-------------|
| `watch_sessions()` | New async function, polling loop with process_all_sessions |

### `src/interfaces/cli.py`

| Change | Description |
|--------|-------------|
| `learn()` | Add `watch` action with `--interval` flag |
| Signal handling | SIGINT/SIGTERM graceful shutdown via asyncio.Event |

### `tests/test_session_observer.py`

| Test | Validates |
|------|-----------|
| `test_watch_sessions_processes_and_sleeps` | Calls process_all_sessions, sleeps, repeats |
| `test_watch_sessions_logs_on_new_sessions` | Logs when sessions_processed > 0 |
| `test_watch_sessions_quiet_on_no_sessions` | No log when nothing new |

### `tests/test_cli.py`

| Test | Validates |
|------|-----------|
| `test_learn_watch_starts` | CLI starts watch mode |

## Error Handling

- **process_all_sessions failure**: Already handles individual session errors
  internally (try/except per file). Watch loop continues.
- **Disk full**: Pair output fails, logged as warning, loop continues.
- **SIGINT during sleep**: asyncio.Event.set() wakes immediately, exits cleanly.
- **SIGINT during processing**: process_all_sessions completes current session,
  then loop exits on next check.

## Out of Scope

- Background daemon / systemd service
- Incremental file processing (only new lines since last read)
- Watchfiles/inotify integration (too noisy for append-heavy files)
- Real-time processing (sub-second latency not needed for training data)
